import sys
import torch
from typing import Literal

from nataili.model_manager import ModelManager
from nataili.modules import blip_decoder
if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

pkg = importlib_resources.files("nataili")

class BLIPModelManager(ModelManager):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        model_name: str = "BLIP",
        precision: str = "fp32",
        device: Literal["cpu", "cuda"] = "cuda",
        gpu_id: int = 0,
        blip_image_eval_size: int = 512,
    ):
        vit = "base" if model_name == "BLIP" else "large"
        model_path = self.get_model_files(model_name)[0]["path"]
        device = torch.device(f"cuda:{gpu_id}") if device == "cuda" else torch.device("cpu")
        model = blip_decoder(
            pretrained=model_path,
            med_config=pkg / "model_manager" / "med_config.json",
            image_size=blip_image_eval_size,
            vit=vit,
        )
        model = model.eval()
        model = (model if precision == "fp32" else model.half()).to(device)
        self.loaded_models[model_name] = {"model": model, "device": device}
        return True
