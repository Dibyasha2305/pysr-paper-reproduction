import torch
from torch_playground import SimpleNet


def test_simple_net_output_shape():
    """
    Test that SimpleNet returns correct output shape.
    """
    model = SimpleNet()
    X = torch.randn(4, 3)
    y = model(X)

    assert y.shape == (4, 1)
