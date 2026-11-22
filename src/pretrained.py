"""Load and run the pretrained CloudSEN12 models (Hugging Face: isp-uv-es/cloudsen12_models).

Follows the official cloudsen12_models package (v1.0.2):
- UNet with a MobileNetV2 encoder (segmentation_models_pytorch), 4 output classes
  0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow (same as the human labels)
- input: L1C top-of-atmosphere reflectance = stored value / 10,000
- the 509x509 image is reflect-padded to 512 (divisible by 32) and the output cropped back
"""
import numpy as np
import torch

REPO = "isp-uv-es/cloudsen12_models"
L1C_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

MODELS = {
    # 13-band UNetMobV2 trained on CloudSEN12 (two published weight files)
    "cloudsen12":   {"file": "cloudsen12.pt",   "type": "weights", "bands": L1C_13},
    "UNetMobV2_V1": {"file": "UNetMobV2_V1.pt", "type": "weights", "bands": L1C_13},
    # 4-band model (NIR, red, green, blue) trained on CloudSEN12: relevant to band reduction
    "dtacs4bands":  {"file": "dtacs4bands.pt",  "type": "jit",     "bands": ["B8", "B4", "B3", "B2"]},
}


def load_model(name, device, folder="/tmp/weights"):
    """Download (once) and load a pretrained model in eval mode."""
    from huggingface_hub import hf_hub_download
    spec = MODELS[name]
    path = hf_hub_download(REPO, spec["file"], local_dir=folder)
    if spec["type"] == "jit":
        model = torch.jit.load(path, map_location=device)
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None,
                         in_channels=len(spec["bands"]), classes=4)
        weights = torch.load(path, map_location=device, weights_only=True)
        if "state_dict" in weights:   # saved from a wrapper whose sub-module is called "model"
            weights = {k.removeprefix("model."): v for k, v in weights["state_dict"].items()}
        model.load_state_dict(weights)   # strict: fails loudly if keys don't match
    return model.eval().to(device)


def predict_batch(model, x, device):
    """x: float32 array (B, C, 509, 509) of reflectance (stored value / 10,000).

    Returns uint8 array (B, 509, 509) with classes 0-3.
    """
    x = np.pad(x, ((0, 0), (0, 0), (1, 2), (1, 2)), mode="reflect")   # 509 -> 512
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
    pred = logits.argmax(dim=1)[:, 1:510, 1:510]
    return pred.to(torch.uint8).cpu().numpy()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, n_bands, size, device):
    """FLOPs for one image of size x size (1 multiply-add = 2 FLOPs). NaN if not measurable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, n_bands, size, size, device=device)
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        return counter.get_total_flops()
    except Exception as e:   # e.g. TorchScript models
        print(f"  FLOPs not measured ({type(e).__name__})")
        return float("nan")
