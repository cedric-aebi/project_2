import os.path
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# f6da7034a9bd66ab7f34094d760a01c00e9f49e155523b0a2a23c6c8024a9ddc
if __name__ == "__main__":
    if not os.path.exists("model.tflite"):
        model: Sequential = joblib.load("fine_tuned_model.joblib")

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save the model.
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Function to make predictions with TFLite model
    def predict_tflite(x):
        # Ensure we have a single sample with the right shape
        x = np.expand_dims(x, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        return output

    def evaluate_tflite_model(x_test, y_test):
        # Initialize counters for confusion matrix
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        total = len(x_test)

        # Convert y_test to numpy array if it's not already
        y_test = np.array(y_test)

        # Process one sample at a time
        for i in range(total):
            # Get single sample
            single_sample = x_test[i]

            # Get prediction
            prediction = predict_tflite(single_sample)
            predicted_class = (prediction > 0.5).astype(int).flatten()[0]

            # Get true class
            true_class = y_test[i]

            # Update confusion matrix counters
            if predicted_class == 1 and true_class == 1:
                true_positives += 1
            elif predicted_class == 0 and true_class == 0:
                true_negatives += 1
            elif predicted_class == 1 and true_class == 0:
                false_positives += 1
            elif predicted_class == 0 and true_class == 1:
                false_negatives += 1

            # Print progress every 1000 samples
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total} samples")

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total

        # Handle division by zero
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "tp": true_positives,
                "tn": true_negatives,
                "fp": false_positives,
                "fn": false_negatives,
            },
        }

    df = pd.read_pickle(Path(__file__).parent.parent.parent.parent / "datasets" / "nurse" / "paper" / "5C.pkl")
    x = df.drop(columns=["Label", "Participant"])
    y = df["Label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    # scale
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Then update your main code to print all metrics:
    metrics = evaluate_tflite_model(x_test, y_test)
    print(f"TFLite Model Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"TFLite Model Precision: {metrics['precision'] * 100:.2f}%")
    print(f"TFLite Model Recall: {metrics['recall'] * 100:.2f}%")
    print(f"TFLite Model F1 Score: {metrics['f1'] * 100:.2f}%")
    print(
        f"Confusion Matrix: TP={metrics['confusion_matrix']['tp']}, TN={metrics['confusion_matrix']['tn']}, "
        f"FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}"
    )

    # Load the original Keras model
    original_model = joblib.load("fine_tuned_model.joblib")

    # Function to make predictions with original model
    def predict_keras(x):
        # Ensure we're working with a batch
        x = np.asarray(x).astype(np.float32)
        if len(x.shape) == 1:  # If it's a single sample
            x = np.expand_dims(x, axis=0)
        return original_model.predict(x)

    def evaluate_keras_model(x_test, y_test):
        # Get predictions for all samples at once for efficiency
        predictions = predict_keras(x_test)
        predicted_classes = (predictions > 0.5).astype(int).flatten()

        y_test = np.array(y_test)

        # Calculate confusion matrix elements
        true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
        true_negatives = np.sum((predicted_classes == 0) & (y_test == 0))
        false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
        false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(y_test)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "tp": int(true_positives),
                "tn": int(true_negatives),
                "fp": int(false_positives),
                "fn": int(false_negatives),
            },
        }

    # Evaluate the original Keras model
    print("\nEvaluating original Keras model...")
    keras_metrics = evaluate_keras_model(x_test, y_test)

    # Print Keras model metrics
    print(f"Keras Model Accuracy: {keras_metrics['accuracy'] * 100:.2f}%")
    print(f"Keras Model Precision: {keras_metrics['precision'] * 100:.2f}%")
    print(f"Keras Model Recall: {keras_metrics['recall'] * 100:.2f}%")
    print(f"Keras Model F1 Score: {keras_metrics['f1'] * 100:.2f}%")
    print(
        f"Confusion Matrix: TP={keras_metrics['confusion_matrix']['tp']}, TN={keras_metrics['confusion_matrix']['tn']}, "
        f"FP={keras_metrics['confusion_matrix']['fp']}, FN={keras_metrics['confusion_matrix']['fn']}"
    )

    # Compare the models
    print("\nComparison (TFLite vs Original):")
    print(f"Accuracy diff: {(metrics['accuracy'] - keras_metrics['accuracy']) * 100:.4f}%")
    print(f"Precision diff: {(metrics['precision'] - keras_metrics['precision']) * 100:.4f}%")
    print(f"Recall diff: {(metrics['recall'] - keras_metrics['recall']) * 100:.4f}%")
    print(f"F1 diff: {(metrics['f1'] - keras_metrics['f1']) * 100:.4f}%")

    # Compare model sizes
    original_model_path = "fine_tuned_model.joblib"
    tflite_model_path = "model.tflite"

    original_size = os.path.getsize(original_model_path) / (1024 * 1024)  # Size in MB
    tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # Size in MB

    print("\nModel Size Comparison:")
    print(f"Original Keras model: {original_size:.2f} MB")
    print(f"TFLite model: {tflite_size:.2f} MB")
    print(f"Size reduction: {(1 - tflite_size/original_size) * 100:.2f}%")

    # Measure inference time
    print("\nInference Time Comparison:")

    # For TFLite model - measure average inference time over multiple samples
    tflite_times = []
    num_samples = min(100, len(x_test))  # Use up to 100 samples
    warmup_rounds = 5

    # Warmup
    for _ in range(warmup_rounds):
        _ = predict_tflite(x_test[0])

    # Actual timing
    for i in range(num_samples):
        start_time = time.time()
        _ = predict_tflite(x_test[i])
        tflite_times.append(time.time() - start_time)

    avg_tflite_time = (sum(tflite_times) / len(tflite_times)) * 1000  # Convert to ms

    # For original Keras model - measure average inference time
    keras_times = []

    # Warmup
    for _ in range(warmup_rounds):
        _ = predict_keras(x_test[0])

    # Actual timing
    for i in range(num_samples):
        start_time = time.time()
        _ = predict_keras(x_test[i])
        keras_times.append(time.time() - start_time)

    avg_keras_time = (sum(keras_times) / len(keras_times)) * 1000  # Convert to ms

    print(f"Original Keras model average inference time: {avg_keras_time:.2f} ms")
    print(f"TFLite model average inference time: {avg_tflite_time:.2f} ms")
    print(f"Speed improvement: {(1 - avg_tflite_time/avg_keras_time) * 100:.2f}%")

    # Optional: Also measure batch inference for Keras (which is typically more efficient)
    if len(x_test) > 100:
        batch_size = 100
    else:
        batch_size = len(x_test)

    start_time = time.time()
    _ = original_model.predict(x_test[:batch_size])
    batch_time = time.time() - start_time
    avg_batch_time = (batch_time / batch_size) * 1000  # ms per sample in batch

    print(f"Keras batch inference time (per sample): {avg_batch_time:.2f} ms")
    print(f"TFLite vs Keras batch speedup: {(1 - avg_tflite_time/avg_batch_time) * 100:.2f}%")
