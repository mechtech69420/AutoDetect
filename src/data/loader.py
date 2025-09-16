import pandas as pd
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence, DatasetDict
from datasets.features import Image as HFImage
import typer

from src.config import TRAIN_CSV, IMAGE_DIR, PROCESSED_DIR

def load_raw_data() -> pd.DataFrame:
    """Load and preprocess raw CSV data"""
    df = pd.read_csv(TRAIN_CSV)
    return (
        df.groupby("image")
        .apply(lambda x: x[["xmin", "ymin", "xmax", "ymax"]].values.tolist(), include_groups=False)
        .rename("bboxes")
        .reset_index()
    )

def create_dataset(df: pd.DataFrame) -> DatasetDict:
    """Convert DataFrame to Hugging Face Dataset"""
    df["image"] = df["image"].apply(lambda x: str(IMAGE_DIR / x))
    df["category"] = 1
    df["image_id"] = range(len(df))
    
    features = Features({
        "image_id": Value("int32"),
        "image": HFImage(),
        "bboxes": Sequence(Sequence(Value("float32"), length=4)),
        "category": Value("float32")
    })
    
    return Dataset.from_pandas(df, features=features).train_test_split(train_size=0.9)

def save_dataset(dataset: DatasetDict, output_dir: Path = PROCESSED_DIR):
    """Save processed dataset to disk"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    typer.echo(f"Dataset saved to {output_dir}")

def main():
    """CLI command for data preprocessing"""
    typer.echo("Loading raw data...")
    df = load_raw_data()
    
    typer.echo("Creating dataset...")
    dataset = create_dataset(df)
    
    save_dataset(dataset)
    typer.echo(f"Processed {len(dataset['train'])} training samples")

if __name__ == "__main__":
    typer.run(main)
