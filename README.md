# Food Category Classification

A small Python project that trains a CNN based on MobileNetV2 to recognize food images from a 5-class subset of the Food-101 dataset and provides a local Streamlit app for image prediction.

## What this app does

1. Builds a reduced dataset from Food-101 using 5 food categories.
2. Trains a convolutional neural network on that dataset.
3. Evaluates the model with test-set metrics and a confusion matrix.
4. Runs a local Streamlit GUI where you upload an image and get a predicted food category.

## How the app works

- `split_train_test_folder.py` prepares the dataset by copying images into `food-5/train/` and `food-5/test/`.
- `train.py` loads the images, trains a MobileNetV2-based CNN model, evaluates it, and saves the best model and metadata into `outputs/`.
- `app.py` loads the saved model and class labels, then serves a simple web interface for uploading an image and showing a prediction.
- `utils.py` contains shared helper functions used by training and the app.

## Quick startup flow

1. Activate the project environment.
2. Prepare the dataset.
3. Train the model.
4. Start the Streamlit app.

## Files you need

- `app.py`: Streamlit prediction app
- `train.py`: model training and evaluation pipeline
- `split_train_test_folder.py`: dataset creation script
- `utils.py`: helper utilities
- `requirements.txt`: dependencies
- `README.md`: project documentation

## Typical commands

Activate the environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Prepare the dataset:

```powershell
python split_train_test_folder.py
```

Train the model:

```powershell
python train.py
```

Run the app:

```powershell
streamlit run app.py
```

## Notes

- `outputs/` stores the trained model, class names, metrics, and any generated evaluation files.
- If the model is already trained, you can skip `train.py` and run only `app.py`.
- Retrain only when the data, classes, or model code change.


To use different Food-101 classes:

```bash
python split_train_test_folder.py --classes sushi ramen pizza steak tacos
python train.py
```

The class names must match Food-101 folder names.

## Reducing Dataset Size

If training is too slow, recreate `food-5` with fewer images:

```bash
python split_train_test_folder.py --train-limit 50 --test-limit 20
python train.py --epochs 3 --image-size 96
```

This is faster but usually less accurate.

## Notes

- `food-101/` is only needed to create or recreate `food-5/`.
- `train.py` uses `food-5/train` and `food-5/test`.
- `app.py` uses `outputs/final_model.keras` and `outputs/class_names.json`.
- If the GUI says model files are missing, run `python train.py` first.
