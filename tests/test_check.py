import json
from pathlib import Path

import pytest

from lorafit._check import check
from lorafit._demo import Q, V, build_demo, write_weights


@pytest.fixture
def demo(tmp_path: Path) -> tuple[Path, Path, Path]:
    return build_demo(tmp_path)


def test_sharded_match_and_mismatch(demo: tuple[Path, Path, Path]) -> None:
    base, wrong, adapter = demo
    good = check(base, adapter)
    assert (good.exit_code, good.modules, good.matched, good.base_shards_read) == (0, 2, 2, 2)
    bad = check(wrong, adapter)
    assert (bad.exit_code, bad.matched) == (1, 1)
    (finding,) = bad.findings
    assert (finding.code, finding.module) == ("shape_mismatch", V)
    assert (finding.expected_shape, finding.actual_shape) == ((4, 16), (8, 16))


def _single_pair(adapter: Path, module: str, a: tuple[int, ...], b: tuple[int, ...]) -> None:
    file = adapter / "adapter_model.safetensors"
    file.unlink()
    write_weights(
        file,
        {
            f"base_model.model.{module}.lora_A.weight": a,
            f"base_model.model.{module}.lora_B.weight": b,
        },
        dtype="F16",
    )


def test_only_opens_shards_needed_for_targets(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    _single_pair(adapter, Q, (2, 16), (8, 2))
    (base / "model-00002-of-00002.safetensors").unlink()
    report = check(base, adapter)
    assert (report.exit_code, report.base_shards_read, report.base_shards) == (0, 1, 2)
    (base / "model-00001-of-00002.safetensors").unlink()
    report = check(base, adapter)
    assert report.exit_code == 2
    assert [item.code for item in report.findings] == ["file_missing"]


def test_single_file_and_no_suffix_matching(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    _single_pair(adapter, Q, (2, 16), (8, 2))
    file = base / "custom.safetensors"
    write_weights(file, {"other." + Q + ".weight": (8, 16)})
    report = check(file, adapter)
    assert report.exit_code == 1
    assert [item.code for item in report.findings] == ["target_missing"]
    file.unlink()
    write_weights(file, {Q + ".weight": (8, 16)}, dtype="BF16")
    assert check(file, adapter).exit_code == 0


def test_missing_partner(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    file = adapter / "adapter_model.safetensors"
    file.unlink()
    write_weights(file, {f"base_model.model.{Q}.lora_A.weight": (2, 16)})
    report = check(base, adapter)
    assert report.exit_code == 1
    assert [(item.code, item.detail) for item in report.findings] == [
        ("missing_pair", "missing lora_B.weight")
    ]


@pytest.mark.parametrize(("a", "b"), [((2, 16), (8, 3)), ((3, 16), (8, 3))])
def test_rank_mismatch(
    demo: tuple[Path, Path, Path], a: tuple[int, ...], b: tuple[int, ...]
) -> None:
    base, _, adapter = demo
    _single_pair(adapter, Q, a, b)
    report = check(base, adapter)
    assert report.exit_code == 1
    assert [item.code for item in report.findings] == ["rank_mismatch"]


@pytest.mark.parametrize(
    "change",
    [
        {"peft_type": "IA3"},
        {"rank_pattern": {"q_proj": 2}},
        {"use_dora": True},
        {"fan_in_fan_out": True},
        {"modules_to_save": ["lm_head"]},
    ],
)
def test_unsupported_config_is_not_a_mismatch(
    demo: tuple[Path, Path, Path], change: dict[str, object]
) -> None:
    base, _, adapter = demo
    config = {"peft_type": "LORA", "r": 2} | change
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    report = check(base, adapter)
    assert report.exit_code == 2
    assert [item.code for item in report.findings] == ["unsupported_config"]


def test_unsupported_tensors_prevent_pass(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    file = adapter / "adapter_model.safetensors"
    file.unlink()
    write_weights(
        file,
        {
            f"base_model.model.{Q}.lora_A.weight": (2, 16),
            f"base_model.model.{Q}.lora_B.weight": (8, 2),
            "base_model.model.lm_head.weight": (8, 16),
        },
    )
    report = check(base, adapter)
    assert (report.exit_code, report.matched) == (2, 1)
    assert [item.code for item in report.findings] == ["unsupported_tensor"]


def test_empty_adapter_prevents_pass(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    file = adapter / "adapter_model.safetensors"
    file.unlink()
    write_weights(file, {})
    report = check(base, adapter)
    assert report.exit_code == 2
    assert [item.code for item in report.findings] == ["empty_adapter"]


@pytest.mark.parametrize(
    ("dtype", "span", "code"),
    [("I8", 128, "unsupported_layout"), ("F32", 128, "invalid_matrix")],
)
def test_bad_base_matrix_is_not_reported_as_a_match(
    demo: tuple[Path, Path, Path], dtype: str, span: int, code: str
) -> None:
    base, _, adapter = demo
    _single_pair(adapter, Q, (2, 16), (8, 2))
    file = base / "custom.safetensors"
    header = json.dumps(
        {
            Q + ".weight": {
                "dtype": dtype,
                "shape": [8, 16],
                "data_offsets": [0, span],
            }
        }
    ).encode("utf-8")
    file.write_bytes(len(header).to_bytes(8, "little") + header + bytes(span))
    report = check(file, adapter)
    assert (report.exit_code, report.matched) == (2, 0)
    assert [item.code for item in report.findings] == [code]


@pytest.mark.parametrize(
    ("key", "value", "exit_code", "code"),
    [
        (Q, None, 1, "target_missing"),
        (Q, "model-00002-of-00002.safetensors", 2, "indexed_tensor_missing"),
    ],
)
def test_wrong_index_mapping(
    demo: tuple[Path, Path, Path], key: str, value: str | None, exit_code: int, code: str
) -> None:
    base, _, adapter = demo
    index = base / "model.safetensors.index.json"
    obj = json.loads(index.read_text(encoding="utf-8"))
    if value is None:
        del obj["weight_map"][key + ".weight"]
    else:
        obj["weight_map"][key + ".weight"] = value
    index.write_text(json.dumps(obj), encoding="utf-8")
    report = check(index, adapter)
    assert report.exit_code == exit_code
    assert [item.code for item in report.findings] == [code]


def test_ambiguous_base_needs_explicit_choice(demo: tuple[Path, Path, Path]) -> None:
    base, _, adapter = demo
    write_weights(base / "model.safetensors", {Q + ".weight": (8, 16), V + ".weight": (8, 16)})
    assert [item.code for item in check(base, adapter).findings] == ["ambiguous_base"]
    assert check(base / "model.safetensors", adapter).exit_code == 0


def test_base_not_found_suggests_variant_filenames(tmp_path: Path, demo: tuple[Path, Path, Path]) -> None:
    _, _, adapter = demo
    variant_dir = tmp_path / "variants"
    variant_dir.mkdir()
    for name in (
        "model.fp16.safetensors",
        "model.bf16.safetensors",
        "model.int8.safetensors",
        "model.zz.safetensors",
    ):
        write_weights(variant_dir / name, {Q + ".weight": (8, 16), V + ".weight": (8, 16)})
    report = check(variant_dir, adapter)
    assert report.exit_code == 2
    assert report.status == "incomplete"
    assert [item.code for item in report.findings] == ["base_not_found"]
    detail = report.findings[0].detail
    assert "model.bf16.safetensors" in detail
    assert "model.fp16.safetensors" in detail
    assert "model.int8.safetensors" in detail
    assert "model.zz.safetensors" not in detail
    assert "lorafit check" in detail
