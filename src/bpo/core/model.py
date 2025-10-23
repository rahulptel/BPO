from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def initialize_model(train_x, train_obj):
    train_x_flat = train_x.reshape(-1, train_x.shape[-1])
    train_obj_flat = train_obj.reshape(-1, train_obj.shape[-1])

    models = []
    for i in range(train_obj_flat.shape[-1]):
        train_y = train_obj_flat[:, i].unsqueeze(-1)
        models.append(
            SingleTaskGP(train_x_flat, train_y, outcome_transform=Standardize(m=1))
        )

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model
