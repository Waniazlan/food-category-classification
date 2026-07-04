# 5-Class Food Image Classification CNN

This project trains a convolutional neural network with TensorFlow/Keras on a smaller
5-class subset of the Food-101 image dataset.

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
- Training history plot
- Confusion matrix
- Per-class precision, recall, and F1-score metrics
- Local web GUI for image prediction

## 1. Create Python Environment

TensorFlow does not usually support the newest Python versions immediately. Use
Python 3.10, 3.11, or 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download Dataset

Download and extract Food-101:

```bash
curl -L -o food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz
```

After extraction, the folder should look like:

```text
food-101/
  images/
  meta/
```

## 3. Prepare 5-Class Train/Test Dataset

```bash
python split_train_test_folder.py
```

This creates:

```text
food-5/
  train/
    apple_pie/
    cheesecake/
    chicken_curry/
    fried_rice/
    pizza/
  test/
    apple_pie/
    cheesecake/
    chicken_curry/
    fried_rice/
    pizza/
```

By default, it copies 300 training images and 100 test images per class. To use
more images:

```bash
python split_train_test_folder.py --train-limit 750 --test-limit 250
```

To choose different Food-101 classes:

```bash
python split_train_test_folder.py --classes sushi ramen pizza steak tacos
```

## 4. Train and Evaluate

```bash
python train.py
```

Useful faster test run:

```bash
python train.py --epochs 3 --image-size 128
```

The training script automatically evaluates on the test dataset and writes results
to `outputs/`.

## 5. Run the GUI

After training finishes, start the local prediction app:

```bash
streamlit run app.py
```

The app opens in your browser. Upload a food image and click `Predict`.

## Outputs

After training, check:

```text
outputs/
  best_model.keras
  final_model.keras
  training_log.csv
  training_history.png
  confusion_matrix.png
  confusion_matrix.csv
  classification_metrics.csv
  class_names.json
```

`confusion_matrix.png` is the visual confusion matrix. `classification_metrics.csv`
contains per-class precision, recall, F1-score, support, and overall accuracy.
