# Car Detection Project
![test](images/output.png)
|name|ID|
|---|----|
|Hisham Abdallah (Parllel) | 202174078|
|Ahmed al-khamery|202070033|
|Abdalazeez Ahmed| 201870170|
|Ahmed Al-agbergy| 202170127|
|Assell Al-kamal| 202170099|
|Wadea Al-qupaty| 202170066|


# Proejct tree
```
project/
│
├── app.py                  # Entry point for launching the app (e.g., API/GUI/Gradio/Streamlit)
├── main.py                 # Main script for running training or inference
│
├── src/
│   ├── __init__.py         # Marks src as a package
│   ├── config.py           # Centralized configuration (hyperparams, paths, settings)
│   │
│   ├── data/
│   │   ├── loader.py       # Data loading functions/classes
│   │   └── processor.py    # Preprocessing, augmentation, and transformations
│   │
│   ├── models/
│   │   ├── architecture.py # Model definitions and architectures
│   │   └── inference.py    # Inference logic, postprocessing, evaluation
│   │
│   ├── training/
│   │   └── trainer.py      # Training loop, validation, checkpointing
│   │
│   └── utils/
│       ├── bbox.py         # Bounding box utilities (IoU, NMS, conversions)
│       └── visualization.py# Visualization helpers for predictions and data
```


## Features
- Data preprocessing for bounding box detection
- Object detection using state-of-the-art DETR model
- Training with configurable hyperparameters
- Inference on new images with visualization
- Bounding box visualization tools
- User-friendly CLI interface using Typer
- Model saving and loading with full configuration
- gradio GUI
# Model training 
![test](images/loss.png)

# Model Quantization
```bash
-rw-r--r-- 1 root root  63M Sep 16 21:11 model_bfloat16.zip
-rw-r--r-- 1 root root  74M Sep 16 20:51 model_float16.zip
-rw-r--r-- 1 root root 411M Sep 16 20:56 model.zip
```
Command-Line Interface Documentation
=====================================

This project uses Typer to provide a command-line interface (CLI) for dataset
preparation, model training, inference, and visualization.

Run the app with:
    python app.py [COMMAND] [OPTIONS]


Available Commands
------------------

1. preprocess
~~~~~~~~~~~~~
Preprocess raw data into a training-ready dataset.

Usage:
    python app.py preprocess [OPTIONS]

Options:
    --csv-path PATH      Path to the raw dataset CSV file. (default: src.data.loader.TRAIN_CSV)
    --image-dir PATH     Path to the raw image directory. (default: src.data.loader.IMAGE_DIR)
    --output-dir PATH    Directory to save the preprocessed dataset. (default: src.config.PROCESSED_DIR)

Description:
    - Loads raw annotations & images.
    - Converts them into a Hugging Face DatasetDict.
    - Saves the dataset to disk.


2. process
~~~~~~~~~~
Process a dataset with bounding box conversion.

Usage:
    python app.py process [OPTIONS]

Options:
    --dataset-dir PATH   Path to the dataset directory. (default: src.config.PROCESSED_DIR)
    --output-dir PATH    Directory to save the processed dataset. (default: PROCESSED_DIR/processed)

Description:
    - Converts bounding boxes into the correct format.
    - Applies necessary preprocessing for training.


3. train
~~~~~~~~
Train an object detection model.

Usage:
    python app.py train [OPTIONS]

Options:
    --dataset-dir PATH   Path to the processed dataset. (default: PROCESSED_DIR/processed)
    --output-dir PATH    Directory to save trained models. (default: models/detection_model)
    --epochs INTEGER     Number of training epochs. (default: 50)
    --batch-size INTEGER Batch size per device. (default: 8)

Description:
    - Loads dataset from disk using Hugging Face DatasetDict.
    - Creates a TrainingConfig.
    - Runs training via trainer.train_model.


4. infer
~~~~~~~~
Run inference on a single image.

Usage:
    python app.py infer [OPTIONS] MODEL_DIR IMAGE_PATH

Arguments:
    model-dir PATH   Path to the trained model directory. (required)
    image-path PATH  Path to the input image. (required)

Options:
    --threshold FLOAT Detection confidence threshold. (default: 0.5)

Description:
    - Loads trained model from disk.
    - Runs inference on the given image.
    - Applies postprocessing with thresholding.


5. visualize
~~~~~~~~~~~~
Visualize a dataset sample.

Usage:
    python app.py visualize [OPTIONS]

Options:
    --dataset-dir PATH   Path to the dataset. (default: PROCESSED_DIR/processed)
    --split [train|test|validation]  Dataset split to visualize. (default: train)
    --index INTEGER      Index of the sample to visualize. (default: 0)

Description:
    - Loads dataset from disk.
    - Visualizes bounding boxes and labels on an image sample.


Example Workflows
-----------------

Preprocess + Process Data:
    python app.py preprocess --csv-path data/train.csv --image-dir data/images
    python app.py process

Train a Model:
    python app.py train --epochs 100 --batch-size 16

Run Inference:
    python app.py infer models/detection_model/checkpoint-100 image.jpg --threshold 0.7

Visualize a Sample:
    python app.py visualize --split validation --index 5

