from transformers import TrainingArguments, Trainer
from dataclasses import dataclass
import typer

from ..models.architecture import load_model_and_processor
from ..config import BATCH_SIZE, LEARNING_RATE, EPOCHS

@dataclass
class TrainingConfig:
    output_dir: str = "models/detection_model"
    learning_rate: float = LEARNING_RATE
    per_device_train_batch_size: int = BATCH_SIZE
    num_train_epochs: int = EPOCHS
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    fp16: bool = True

def collate_fn(batch, processor):
    """Custom collate function for object detection"""
    images = [item["image"] for item in batch]
    annotations = []
    
    for sample in batch:
        ann = {
            "image_id": sample["image_id"],
            "annotations": [
                {
                    "image_id": sample["image_id"],
                    "bbox": box,
                    "category_id": sample["labels"][i],
                    "area": sample["area"][i]
                } for i, box in enumerate(sample["boxes"])
            ]
        }
        annotations.append(ann)
    
    return processor(
        images=images,
        annotations=annotations,
        return_tensors="pt",
        padding=True
    )

def train_model(
    dataset,
    config: TrainingConfig = TrainingConfig(),
    model_name: str = "facebook/detr-resnet-50"
):
    """Train object detection model"""
    model, processor = load_model_and_processor(model_name)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        fp16=config.fp16,
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=lambda batch: collate_fn(batch, processor),
    )
    
    typer.echo("Starting training...")
    trainer.train()
    typer.echo(f"Training completed. Best model saved at {trainer.state.best_model_checkpoint}")
    
    return trainer
