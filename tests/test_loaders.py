import os

import pytest

from coldstart_lab.loaders import PyTorchBinLoader, SafetensorsLoader
from coldstart_lab.loaders.converters import to_pytorch_bin


def test_safetensors_loader_reads_all_tensors(synthetic_checkpoint):
    res = SafetensorsLoader(use_mmap=True).load(synthetic_checkpoint, device="cpu")
    assert res.tensor_count == 4
    assert res.bytes_read > 0
    assert res.total_ms > 0
    assert res.throughput_mib_s > 0
    # The three phases must all be present and sum to total.
    assert set(res.phases_ms) == {"open_and_read", "materialize_host", "to_device"}
    assert abs(sum(res.phases_ms.values()) - res.total_ms) < 1e-6


def test_safetensors_nommap_variant_names_itself():
    loader = SafetensorsLoader(use_mmap=False)
    assert loader.name == "safetensors-nommap"


def test_safetensors_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        SafetensorsLoader().load(str(tmp_path), device="cpu")


def test_converter_and_bin_loader_roundtrip(synthetic_checkpoint, tmp_path):
    bin_dir = str(tmp_path / "bin")
    bin_path = to_pytorch_bin(synthetic_checkpoint, bin_dir)
    assert os.path.isfile(bin_path)

    res = PyTorchBinLoader().load(bin_dir, device="cpu")
    assert res.tensor_count == 4
    assert res.loader == "pytorch_bin"
    assert res.total_ms > 0


def test_bin_loader_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        PyTorchBinLoader().load(str(tmp_path), device="cpu")


def test_loaders_agree_on_tensor_values(synthetic_checkpoint, tmp_path):
    """safetensors and bin must hold identical weights (converter correctness)."""
    import torch
    from safetensors import safe_open

    bin_dir = str(tmp_path / "bin")
    to_pytorch_bin(synthetic_checkpoint, bin_dir)
    bin_sd = torch.load(os.path.join(bin_dir, "pytorch_model.bin"), weights_only=True)

    with safe_open(
        os.path.join(synthetic_checkpoint, "model.safetensors"), framework="pt", device="cpu"
    ) as h:
        for key in h.keys():
            assert torch.equal(h.get_tensor(key), bin_sd[key])
