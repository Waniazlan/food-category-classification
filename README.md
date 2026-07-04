# 5-Class Food Image Classification CNN

This project is a Python image classification system using TensorFlow/Keras. It
trains a Convolutional Neural Network (CNN) to classify food images into 5
classes and includes a simple local web GUI for prediction.

Default classes:

- `apple_pie`
- `cheesecake`
- `chicken_curry`
- `fried_rice`
- `pizza`

The project includes:

- Dataset preparation for training and testing
- CNN model training
- Test-set evaluation
- Confusion matrix
- Precision, recall, and F1-score
- Streamlit GUI for image prediction

## Project Files

```text
food101-classification/
  app.py                      # GUI app for prediction
  train.py                    # Train and evaluate the CNN model
  split_train_test_folder.py  # Create the 5-class dataset
  utils.py                    # Helper functions
  requirements.txt            # Python dependencies
  README.md                   # Instructions
```

Generated folders after running the project:

```text
food-101/     # Original downloaded Food-101 dataset
food-5/       # Smaller 5-class dataset used by this project
outputs/      # Trained model, confusion matrix, metrics, plots
.venv/        # Local Python virtual environment
```

These generated folders are not included in GitHub because they are large or
machine-specific.

## Fresh Clone Setup

Use these steps after cloning the repository.

### 1. Go to the Project Folder

```bash
cd food101-classification
```

### 2. Create a Python Virtual Environment

Use Python 3.10, 3.11, or 3.12. TensorFlow may not work with the newest Python
version.

On this machine, Python 3.12 works:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If your computer has Python 3.11 instead:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download Food-101 Dataset

```bash
curl -L -o food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz
```

After extraction, you should have:

```text
food-101/
  images/
  meta/
```

### 5. Create the 5-Class Dataset

```bash
python split_train_test_folder.py
```

This creates:

```text
food-5/
  train/
  test/
```

By default, the project uses:

```text
300 training images per class
100 testing images per class
5 classes
```

So the dataset size is:

```text
Training images: 1500
Testing images: 500
Total images: 2000
```

This is more than the project minimum requirement of 50 images per class.

### 6. Train and Evaluate the Model

```bash
python train.py
```

This trains the CNN and saves results to `outputs/`.

For a faster test run:

```bash
python train.py --epochs 3 --image-size 128
```

For the fastest demo run:

```bash
python train.py --epochs 1 --image-size 96
```

Lower epochs and smaller image size make training faster, but the accuracy may
be lower.

### 7. Run the GUI

After training finishes:

```bash
streamlit run app.py
```

The app will open in your browser. Upload a food image and click `Predict`.

## Normal Daily Usage

After setup and training are already done, you do not need to train every time.

Run only:

```bash
cd food101-classification
source .venv/bin/activate
streamlit run app.py
```

Run `python train.py` again only if:

- You changed the dataset
- You changed the selected classes
- You changed the model code
- You deleted the `outputs/` folder
- You want to improve accuracy by retraining

## Output Files

After training, check:

```text
outputs/
  best_model.keras              # Best saved model during training
  final_model.keras             # Final model used by the GUI
  class_names.json              # Class labels used by the GUI
  training_log.csv              # Training accuracy/loss per epoch
  training_history.png          # Accuracy/loss graph
  confusion_matrix.png          # Visual confusion matrix
  confusion_matrix.csv          # Confusion matrix values
  classification_metrics.csv    # Precision, recall, F1-score, accuracy
```

## Changing the Classes

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
