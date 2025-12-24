import pytest
import torch


def test_pytorch_version():
    version = torch.__version__
    print(f"{version}")
    assert version


def test_cuda_available():
    available = torch.cuda.is_available()
    print(f"{available}")
    assert available


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_cuda_device():
    device_name = torch.cuda.get_device_name(0)
    print(f"{device_name}")
    assert device_name


def test_cuda_version():
    cuda_version = torch.version.cuda
    print(f"{cuda_version}")
    assert cuda_version
