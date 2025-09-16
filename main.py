import typer
from pathlib import Path
from datasets import DatasetDict

# Import from src package
from src.data import loader, processor
from src.training import trainer
from src.models import inference
from src.utils.visualization import visualize_sample
from src.config import PROCESSED_DIR

app = typer.Typer(no_args_is_help=True)

@app.command()
def preprocess(
    csv_path: Path = loader.TRAIN_CSV,
    image_dir: Path = loader.IMAGE_DIR,
    output_dir: Path = PROCESSED_DIR
):
    """Preprocess raw data into training-ready dataset"""
    loader.save_dataset(
        loader.create_dataset(loader.load_raw_data()),
        output_dir
    )

@app.command()
def process(
    dataset_dir: Path = PROCESSED_DIR,
    output_dir: Path = PROCESSED_DIR / "processed"
):
    """Process dataset with bounding box conversion"""
    processor.process_dataset(dataset_dir, output_dir)

@app.command()
def train(
    dataset_dir: Path = PROCESSED_DIR / "processed",
    output_dir: str = "models/detection_model",
    epochs: int = 50,
    batch_size: int = 8
):
    """Train object detection model"""
    dataset = DatasetDict.load_from_disk(dataset_dir)
    config = trainer.TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size
    )
    trainer.train_model(dataset, config)

@app.command()
def infer(
    model_dir: Path,
    image_path: Path,
    threshold: float = 0.5
):
    """Run inference on a single image"""
    if not model_dir.exists():
        raise typer.BadParameter(f"Model directory not found: {model_dir}")
    
    inference.run_inference(str(model_dir), str(image_path), threshold)

@app.command()
def visualize(
    dataset_dir: Path = PROCESSED_DIR / "processed",
    split: str = "train",
    index: int = 0
):
    """Visualize a dataset sample"""
    dataset = DatasetDict.load_from_disk(dataset_dir)
    visualize_sample(dataset, split, index)

if __name__ == "__main__":
    app()
