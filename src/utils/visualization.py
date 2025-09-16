import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import typer

def show_image_with_boxes(image: Image.Image, boxes: list):
    """Display image with bounding boxes"""
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for bbox in boxes:
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()

def visualize_sample(dataset, split: str = "train", index: int = 0):
    """Visualize a sample from the dataset"""
    example = dataset[split][index]
    show_image_with_boxes(example["image"], example["boxes"])
    typer.echo(f"Visualized sample {index} from {split} set")
