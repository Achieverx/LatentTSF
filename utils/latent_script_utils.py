from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def official_loader(args, flag, shuffle=False):
    data_set, _ = data_provider(args, flag)
    return DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def _strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", ""): value for key, value in state_dict.items()}
    return state_dict


def load_autoencoder(args, device, freeze=True, unfreeze_encoder=False):
    autoencoder = get_autoencoder(args).float().to(device)
    state_dict = torch.load(args.autoencoder_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "autoencoder_state_dict" in state_dict:
        state_dict = state_dict["autoencoder_state_dict"]

    autoencoder.load_state_dict(_strip_module_prefix(state_dict))

    if freeze:
        freeze_module(autoencoder)
        if unfreeze_encoder:
            encoder = autoencoder.module.encoder if hasattr(autoencoder, "module") else autoencoder.encoder
            for param in encoder.parameters():
                param.requires_grad = True

    return autoencoder


def load_forecaster_from_checkpoint(
    args,
    device,
    checkpoint_path,
    forecaster_cls,
    freeze=True,
    return_checkpoint=False,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    base_args = SimpleNamespace(**vars(args))
    for key, value in ckpt_args.items():
        setattr(base_args, key, value)

    model = forecaster_cls(base_args).to(device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]))

    if freeze:
        freeze_module(model)

    if return_checkpoint:
        return model, checkpoint
    return model


def merge_checkpoint_args(args, checkpoint_args, keys):
    overwritten = {}
    for key in keys:
        if key not in checkpoint_args:
            continue
        old_value = getattr(args, key, None)
        new_value = checkpoint_args[key]
        if old_value != new_value:
            overwritten[key] = {"old": old_value, "new": new_value}
        setattr(args, key, new_value)
    return overwritten
