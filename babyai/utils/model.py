import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        if not torch.cuda.is_available():
            model = torch.load(path, map_location='cpu')
            if model.use_corrector:
                model.corrector.device = "cpu"
        else:
            model = torch.load(path)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)
