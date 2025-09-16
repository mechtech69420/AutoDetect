from transformers import DetrForObjectDetection, DetrImageProcessor
from ..config import MODEL_NAME, ID2LABEL, LABEL2ID, NUM_LABELS, DTYPE

def load_model_and_processor(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
    id2label: dict = ID2LABEL,
    label2id: dict = LABEL2ID,
    dtype: str= DTYPE
):
    """Load model and processor with custom configuration"""
    processor = DetrImageProcessor.from_pretrained(model_name)
    
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        num_labels=num_labels,
        dtype=dtype
    )
    
    return model, processor
