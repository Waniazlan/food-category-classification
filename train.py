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


def build_mobilenetv2(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.18),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="food_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model, base_model


def compile_for_fine_tuning(model, base_model, fine_tune_at, learning_rate):
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


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

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a 5-class food MobileNetV2 classifier.")
    parser.add_argument("--data-dir", default="food-5", help="Dataset folder with train/test subfolders.")
    parser.add_argument("--epochs", type=int, default=18, help="Frozen-base training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=12, help="Fine-tuning epochs after the head is trained.")
    parser.add_argument("--fine-tune-at", type=int, default=100, help="MobileNetV2 layer index to start fine-tuning.")
    parser.add_argument("--fine-tune-lr", type=float, default=0.00001)
    parser.add_argument("--target-recall", type=float, default=0.75, help="Minimum desired recall for each class.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
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

    model, base_model = build_mobilenetv2(
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
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=0.000001,
        ),
    ]

    head_history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    fine_tune_history = None
    if args.fine_tune_epochs > 0:
        compile_for_fine_tuning(
            model,
            base_model,
            fine_tune_at=args.fine_tune_at,
            learning_rate=args.fine_tune_lr,
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(head_history.history["loss"]),
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
    if fine_tune_history:
        plot_training_history(merge_histories(head_history, fine_tune_history), output_dir / "training_history.png")
    else:
        plot_training_history(head_history, output_dir / "training_history.png")
    plot_confusion_matrix(confusion_matrix, class_names, output_dir / "confusion_matrix.png")
    np.savetxt(output_dir / "confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")
    metrics_rows = save_classification_metrics(
        class_names,
        true_labels,
        predicted_labels,
        output_dir / "classification_metrics.csv",
    )

    below_target = [
        row for row in metrics_rows
        if row["class"] != "overall_accuracy" and row["recall"] < args.target_recall
    ]
    if below_target:
        print(f"Classes below {args.target_recall:.0%} recall:")
        for row in below_target:
            print(f"- {row['class']}: {row['recall']:.2%}")
    else:
        print(f"All classes reached at least {args.target_recall:.0%} recall.")

    print(f"Saved model and evaluation files in: {output_dir}")


if __name__ == "__main__":
    main()
