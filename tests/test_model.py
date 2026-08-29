import torch
from src.model import get_model
from src.dataset import get_transforms


def test_model_architecture():
    model = get_model("resnet18", num_classes=10)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_transforms():
    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)
    assert train_tf is not None
    assert val_tf is not None