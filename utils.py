from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir, image_size, batch_size, seed):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train/test folders inside {data_dir}. "
            "Run split_train_test_folder.py first."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
        shuffle=True,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, test_ds, class_names


def collect_predictions(model, dataset):
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        predicted_labels.extend(np.argmax(predictions, axis=1))

    return np.array(true_labels), np.array(predicted_labels)


def plot_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
