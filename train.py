import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import (
    collect_predictions,
    load_datasets,
    plot_confusion_matrix,
    plot_training_history,
)


def build_cnn(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_classification_metrics(class_names, true_labels, predicted_labels, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, class_name in enumerate(class_names):
        true_positive = np.sum((true_labels == index) & (predicted_labels == index))
        false_positive = np.sum((true_labels != index) & (predicted_labels == index))
        false_negative = np.sum((true_labels == index) & (predicted_labels != index))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        support = np.sum(true_labels == index)

        rows.append(
            {
                "class": class_name,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "support": int(support),
            }
        )

    accuracy = np.mean(true_labels == predicted_labels)
    rows.append(
        {
            "class": "overall_accuracy",
            "precision": "",
            "recall": "",
            "f1_score": round(float(accuracy), 4),
            "support": int(len(true_labels)),
        }
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a 5-class food CNN.")
    parser.add_argument("--data-dir", default="food-5", help="Dataset folder with train/test subfolders.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.image_size, args.image_size)
    train_ds, test_ds, class_names = load_datasets(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = build_cnn(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(class_names),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    true_labels, predicted_labels = collect_predictions(model, test_ds)
    confusion_matrix = tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=len(class_names),
    ).numpy()

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    model.save(output_dir / "final_model.keras")
    plot_training_history(history, output_dir / "training_history.png")
    plot_confusion_matrix(confusion_matrix, class_names, output_dir / "confusion_matrix.png")
    np.savetxt(output_dir / "confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")
    save_classification_metrics(
        class_names,
        true_labels,
        predicted_labels,
        output_dir / "classification_metrics.csv",
    )

    print(f"Saved model and evaluation files in: {output_dir}")


if __name__ == "__main__":
    main()
