from datasets import DatasetDict
from pathlib import Path
import typer

from ..utils.bbox import xyxy_to_xywh
from ..config import PROCESSED_DIR

def convert_bbox(example):
    """Convert bounding boxes and add metadata"""
    example["boxes"] = [xyxy_to_xywh(box) for box in example["bboxes"]]
    example["labels"] = [example["category"]] * len(example["bboxes"])
    example["area"] = [w * h for _, _, w, h in example["boxes"]]
    example["iscrowd"] = [0] * len(example["bboxes"])
    return example

def process_dataset(
    dataset_dir: Path = PROCESSED_DIR,
    output_dir: Path = PROCESSED_DIR / "processed"
):
    """Process dataset with bounding box conversion"""
    typer.echo(f"Loading dataset from {dataset_dir}")
    dataset = DatasetDict.load_from_disk(dataset_dir)
    
    processed = dataset.map(
        convert_bbox,
        remove_columns=["bboxes"],
        desc="Converting bounding boxes"
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(output_dir)
    typer.echo(f"Processed dataset saved to {output_dir}")

if __name__ == "__main__":
    typer.run(process_dataset)
