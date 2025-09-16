import torch
from PIL import Image
import typer

from ..utils.bbox import xyxy_to_xywh
from ..utils.visualization import show_image_with_boxes
from .architecture import load_model_and_processor

def run_inference(
    model_dir: str,
    image_path: str,
    threshold: float = 0.5,
    device: str = "cpu"
):
    """Run inference on a single image"""
    model, processor = load_model_and_processor(model_dir)
    model = model.to(device)
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device,dtype=torch.bfloat16)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    
    # Convert to xywh for visualization
    boxes = [xyxy_to_xywh(box.tolist()) for box in results["boxes"]]
    
    typer.echo(f"Found {len(boxes)} objects")
    show_image_with_boxes(image, boxes)
    
    return results
