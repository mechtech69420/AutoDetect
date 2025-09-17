import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from src.models.architecture import load_model_and_processor

from transformers import AutoImageProcessor, AutoModelForObjectDetection

### 2. Setup preprocessing and helper functions ###

# Path to the fine-tuned car detection model
# Replace this with your model path (local or Hugging Face Hub)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir= "./model"

# Load processor and model
model, image_processor = load_model_and_processor(model_dir)
model = model.to(device)

# Override id2label with single class
id2label = {1: "car"}
label2id = {"car": 1}

# Define color for car boxes
color_dict = {"car": "red"}

### 3. Prediction function ###
def predict_on_image(image, conf_threshold):
    model.eval()
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt").to(device,dtype=torch.bfloat16)
        outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )[0]

    # Convert results to CPU
    for key, value in results.items():
        results[key] = value.cpu()

    # Draw boxes
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    detected_labels = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x, y, x2, y2 = box.tolist()

        draw.rectangle((x, y, x2, y2), width=3)
        text = f"({round(score.item(), 2)})"
        draw.text((x, y), text, fill="white", font=font)

    del draw

    # Text summary
    if not detected_labels:
        return image, f"No cars detected at threshold {conf_threshold}."
    return image, f"Detected {len(detected_labels)} car(s)."

### 4. Gradio Demo ###
description = """
Upload an image, and the model will detect cars.
"""

demo = gr.Interface(
    fn=predict_on_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="Confidence Threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Detections"),
        gr.Text(label="Summary"),
    ],
    title="🚗 Car Object Detection Demo",
    description=description,
    examples=[
        ["dataset/vid_4_12320.jpg", 0.8],
        ["dataset/vid_4_10050.png", 0.8],
        ["dataset/vid_4_12120.jpg", 0.8],
    ],
    cache_examples=True,
)

if __name__ == "__main__":
    demo.launch(server_port=8080)
