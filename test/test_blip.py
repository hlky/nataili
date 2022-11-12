import PIL
import sys
if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

pkg = importlib_resources.files("nataili")

from nataili import logger, BLIPModelManager, Caption

def test_blip():
    model_manager = BLIPModelManager()
    model_manager.init()
    image = PIL.Image.open(pkg / "image.png").convert("RGB")
    model = "BLIP"
    if model not in model_manager.available_models:
        logger.error(f"Model {model} not available")
        logger.info(f"Dowloading model {model}")
        model_manager.download_model(model)
    success = model_manager(model)
    assert success
    caption = Caption(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"])
    expected_result = "a woman with a red hat on her head"
    caption_result = caption(image, sample=False)
    logger.info(f"Caption result: {caption_result}")
    logger.info(f"Expected result: {expected_result}")
    assert caption_result == expected_result
    logger.info(f"All tests passed")

test_blip()