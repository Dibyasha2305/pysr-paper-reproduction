import torch
from models.torch_symbolic_net import SymbolicNet, train_symbolic_model



def test_symbolic_net_learns_linear_equation():
    """
    Test that SymbolicNet can learn y = 3x + 0.5
    """
    model = SymbolicNet()

    X = torch.randn(200, 1)
    y = 3.0 * X + 0.5

    train_symbolic_model(model, X, y, epochs=800)

    a, b = model.get_equation()

    assert abs(a - 3.0) < 0.05
    assert abs(b - 0.5) < 0.05
