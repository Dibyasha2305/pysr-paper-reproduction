import numpy as np
import deepchem as dc

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor



def test_dc_torch_symbolic_regressor():
    """
    Test DeepChem Torch symbolic regressor.
    """
    X = np.random.randn(200, 1)
    y = 3.0 * X + 0.5

    dataset = dc.data.NumpyDataset(X, y)

    model = DCTorchSymbolicRegressor(learning_rate=0.01, batch_size=200)
    model.fit(dataset, nb_epoch=4000)





    a, b = model.get_equation()

    assert abs(a - 3.0) < 1.0
    assert abs(b - 0.5) < 1.0


