import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Transformers uses .isin for an op, which is not supported on MPS

from surya.detection import DetectionPredictor, InlineDetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from surya.texify import TexifyPredictor

from marker.builders.layout import LayoutBuilder

def create_model_dict(device=None, dtype=None) -> dict:
    return {
        "layout_model": LayoutPredictor(device=LayoutBuilder.LayoutPredictor_device, dtype=dtype),
        "texify_model": TexifyPredictor(device=LayoutBuilder.TexifyPredictor_device, dtype=dtype),
        "recognition_model": RecognitionPredictor(device=LayoutBuilder.RecognitionPredictor_device, dtype=dtype),
        "table_rec_model": TableRecPredictor(device=LayoutBuilder.TableRecPredictor_device, dtype=dtype),
        "detection_model": DetectionPredictor(device=LayoutBuilder.DetectionPredictor_device, dtype=dtype),
        "inline_detection_model": InlineDetectionPredictor(device=LayoutBuilder.InlineDetectionPredictor_device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=LayoutBuilder.OCRErrorPredictor_device, dtype=dtype)
    }