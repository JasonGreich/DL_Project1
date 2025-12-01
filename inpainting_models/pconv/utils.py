import torch


def load_ckpt(ckpt_path, model, optimizer=None, for_predict=False, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Some checkpoints: {'model': state_dict}, others: just state_dict
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    # IGNORE missing buffers like mask_kernel_buffer
    load_info = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)

    if for_predict:
        model.eval()
        return model

    step = checkpoint.get('step', 0) if isinstance(checkpoint, dict) else 0
    if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, step