import sys

sys.path.insert(0, './')
from src.utils.utils import count_local_params, count_parameters

from src.models import MLPClassificationModel


def test_count_local_params():
    """
    Test, if parameters are counted properly
    Returns
    -------

    """
    model = MLPClassificationModel(shared_embedding_size=3)

    layer_1 = count_local_params(model=model, n_personal_layers=1) == 18
    layer_2 = count_local_params(model=model, n_personal_layers=2) == 48
    layer_3 = count_local_params(model=model, n_personal_layers=3) == count_parameters(
        model)  # all layers for this particular model

    assert layer_1 and layer_2 and layer_3
