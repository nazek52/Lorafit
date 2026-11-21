"""Match a saved dense PEFT adapter to the exact names in a local base."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from lorafit._metadata import (
    FLOAT_BYTES,
    MAX_CONFIG,
    Header,
    InputError,
    Tensor,
    read_header,
    read_index,
    read_json,
)

INDEX_SUFFIX = ".safetensors.index.json"
WRAPPER = "base_model.model."
UNSUPPORTED_CONFIG = (
    "alora_invocation_tokens",
    "alpha_pattern",
    "arrow_config",
    "fan_in_fan_out",
    "loftq_config",
    "lora_bias",
    "megatron_config",
    "modules_to_save",
    "target_parameters",
    "trainable_token_indices",
    "use_bdlora",
    "use_dora",
    "use_qalora",
    "use_rslora",
)
PAIR_SUFFIXES = ((".lora_A.weight", "A"), (".lora_B.weight", "B"))


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str
    module: str | None = None
    path: str | None = None
    expected_shape: tuple[int, ...] | None = None
    actual_shape: tuple[int, ...] | None = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    incomplete: bool = False
    modules: int = 0
    matched: int = 0
    base_shards: int = 0
    base_shards_read: int = 0
    metadata_bytes: int = 0
    declared_base: str | None = None

    @property
    def exit_code(self) -> int:
        if self.incomplete:
            return 2
        return int(bool(self.findings))

    @property
    def status(self) -> str:
        return ("match", "mismatch", "incomplete")[self.exit_code]

    def record_input_error(self, error: InputError, module: str | None = None) -> None:
        self.incomplete = True
        self.findings.append(Finding(error.code, str(error), module, str(error.path)))


def _rank(config: dict[str, object], path: Path) -> int:
    if config.get("peft_type") != "LORA":
        raise InputError("unsupported_config", path, "expected peft_type='LORA'")
    enabled = [key for key in UNSUPPORTED_CONFIG if config.get(key)]
    if config.get("bias", "none") != "none":
        enabled.append("bias")
    init = config.get("init_lora_weights", True)
    if type(init) is not bool and init != "gaussian":
        enabled.append("init_lora_weights")
    if enabled:
        raise InputError(
            "unsupported_config", path, f"outside supported layout: {', '.join(enabled)}"
        )
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise InputError("invalid_config", path, "expected a positive integer for rank 'r'")
    return rank


def _rank_pattern(config: dict[str, object], path: Path) -> list[tuple[str, int]]:
    raw = config.get("rank_pattern")
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise InputError("invalid_config", path, "expected a mapping for 'rank_pattern'")
    result: list[tuple[str, int]] = []
    for pattern, value in raw.items():
        if not isinstance(pattern, str) or not pattern:
            raise InputError("invalid_config", path, "expected non-empty string keys in 'rank_pattern'")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise InputError("invalid_config", path, f"invalid regex in 'rank_pattern': {pattern!r}") from exc
        if type(value) is not int or value < 1:
            raise InputError(
                "invalid_config",
                path,
                f"expected a positive integer rank for pattern {pattern!r}",
            )
        result.append((pattern, value))
    return result


def _module_rank(module: str, default_rank: int, rank_pattern: list[tuple[str, int]]) -> int:
    for pattern, value in rank_pattern:
        if pattern == module or re.match(rf"(?:.*\.)?{pattern}$", module):
            return value
    return default_rank


def _groups(header: Header, path: Path, report: Report) -> dict[str, dict[str, Tensor]]:
    result: dict[str, dict[str, Tensor]] = {}
    for name, tensor in sorted(header.tensors.items()):
        for suffix, side in PAIR_SUFFIXES:
            if name.startswith(WRAPPER) and name.endswith(suffix):
                module = name[len(WRAPPER) : -len(suffix)]
                if module and all(module.split(".")):
                    result.setdefault(module, {})[side] = tensor
                    break
        else:
            report.incomplete = True
            report.findings.append(
                Finding(
                    "unsupported_tensor",
                    f"outside saved PEFT A/B layout: {name!r}",
                    path=str(path),
                )
            )
    if not result and not report.findings:
        raise InputError("empty_adapter", path, "no LoRA tensor pairs found")
    return result


def _select_base(path: Path) -> Path:
    try:
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            candidates = [
                path / filename
                for filename in ("model.safetensors", "model" + INDEX_SUFFIX)
                if (path / filename).exists() or (path / filename).is_symlink()
            ]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                raise InputError(
                    "ambiguous_base", path, "both weight layouts exist; pass an explicit file"
                )
            raise InputError(
                "base_not_found", path, "expected model.safetensors or model.safetensors.index.json"
            )
    except FileNotFoundError as exc:
        raise InputError("file_missing", path, "base path is missing") from exc
    except (OSError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("file_unreadable", path, "could not inspect base path") from exc
    if not path.name.endswith((".safetensors", INDEX_SUFFIX)):
        raise InputError("unsupported_base", path, "pass a SafeTensors file or its sharded index")
    return path


def _matrix(tensor: Tensor, report: Report, module: str, name: str) -> bool:
    width = FLOAT_BYTES.get(tensor.dtype)
    if width is None or len(tensor.shape) != 2:
        report.incomplete = True
        report.findings.append(
            Finding(
                "unsupported_layout",
                f"{name} must be a dense floating-point matrix; "
                f"found {tensor.dtype!r} {tensor.shape}",
                module,
            )
        )
        return False
    if not all(tensor.shape) or prod(tensor.shape) * width != tensor.end - tensor.start:
        report.incomplete = True
        report.findings.append(
            Finding(
                "invalid_matrix",
                f"{name} has empty dimensions or its byte span disagrees with dtype and shape",
                module,
            )
        )
        return False
    return True


def _check_pair(
    module: str, pair: dict[str, Tensor], rank: int, base: Tensor | None, report: Report
) -> None:
    if len(pair) != 2:
        side = "B" if "A" in pair else "A"
        report.findings.append(Finding("missing_pair", f"missing lora_{side}.weight", module))
        return
    a, b = pair["A"], pair["B"]
    a_ok = _matrix(a, report, module, "LoRA A")
    b_ok = _matrix(b, report, module, "LoRA B")
    if not (a_ok and b_ok):
        return
    if a.shape[0] != b.shape[1] or a.shape[0] != rank:
        report.findings.append(
            Finding(
                "rank_mismatch",
                f"configured r={rank}; A rows={a.shape[0]}, B columns={b.shape[1]}",
                module,
            )
        )
        return
    if base is None or not _matrix(base, report, module, "base weight"):
        return
    implied = (b.shape[0], a.shape[1])
    if implied != base.shape:
        report.findings.append(
            Finding(
                "shape_mismatch",
                f"adapter B @ A gives {implied}; base weight is {base.shape}",
                module,
                expected_shape=base.shape,
                actual_shape=implied,
            )
        )
    else:
        report.matched += 1


def _match(
    path: Path,
    groups: dict[str, dict[str, Tensor]],
    rank: int,
    report: Report,
    rank_pattern: list[tuple[str, int]] | None = None,
) -> None:
    patterns = rank_pattern or []
    by_path: dict[Path, list[str]] = {}
    sharded = path.name.endswith(INDEX_SUFFIX)
    if sharded:
        mapping, size = read_index(path)
        report.metadata_bytes += size
        report.base_shards = len(set(mapping.values()))
        for module in groups:
            key = module + ".weight"
            shard = mapping.get(key)
            if shard is None:
                report.findings.append(
                    Finding("target_missing", f"{key!r} is absent from the base index", module)
                )
                _check_pair(
                    module, groups[module], _module_rank(module, rank, patterns), None, report
                )
            else:
                by_path.setdefault(path.parent / shard, []).append(module)
    else:
        report.base_shards = 1
        by_path[path] = list(groups)
    # Keep just one shard header resident at a time.
    for shard_path, modules in sorted(by_path.items()):
        try:
            header = read_header(shard_path)
        except InputError as exc:
            report.record_input_error(exc)
            continue
        report.base_shards_read += 1
        report.metadata_bytes += header.bytes_read
        for module in modules:
            key = module + ".weight"
            weight = header.tensors.get(key)
            if weight is None:
                code = "indexed_tensor_missing" if sharded else "target_missing"
                location = "indexed shard" if sharded else "base"
                report.findings.append(
                    Finding(
                        code,
                        f"{key!r} is absent from the {location} header",
                        module,
                        str(shard_path),
                    )
                )
                if sharded:
                    report.incomplete = True
            _check_pair(
                module, groups[module], _module_rank(module, rank, patterns), weight, report
            )
        del header


def check(base: Path, adapter: Path) -> Report:
    """Return a report for a local base path and a saved PEFT adapter directory."""
    report = Report()
    try:
        config_path = adapter / "adapter_config.json"
        config, size = read_json(config_path, MAX_CONFIG, "invalid_config")
        report.metadata_bytes += size
        rank = _rank(config, config_path)
        rank_pattern = _rank_pattern(config, config_path)
        name = config.get("base_model_name_or_path")
        report.declared_base = name if isinstance(name, str) and name else None
        adapter_path = adapter / "adapter_model.safetensors"
        header = read_header(adapter_path)
        report.metadata_bytes += header.bytes_read
        groups = _groups(header, adapter_path, report)
        report.modules = len(groups)
        _match(_select_base(base), groups, rank, report, rank_pattern)
    except InputError as exc:
        report.record_input_error(exc)
    return report
