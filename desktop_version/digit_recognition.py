"""
Handwritten Digit Recognition - Desktop Version (Tkinter)
Draw a digit (0-9) with the mouse and click Recognize.
"""

import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_model.h5")
CANVAS_SIZE = 280
BRUSH_SIZE = 18


def train_and_save_model(path):
    print("Model not found. Training a new model on MNIST (this may take a few minutes)...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {score[1] * 100:.2f}%")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")
    return model


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return train_and_save_model(MODEL_PATH)


class DigitRecognitionApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda event: None)

        self.clear_button = tk.Button(root, text="Clear", width=12, command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, padx=10, pady=5)

        self.recognize_button = tk.Button(
            root, text="Recognize", width=12, command=self.recognize_digit
        )
        self.recognize_button.grid(row=1, column=1, padx=10, pady=5)

        self.result_label = tk.Label(root, text="Predicted Digit: -", font=("Helvetica", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        self.confidence_label = tk.Label(root, text="Confidence: -", font=("Helvetica", 12))
        self.confidence_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Predicted Digit: -")
        self.confidence_label.config(text="Confidence: -")

    def recognize_digit(self):
        img = self.image.resize((28, 28))
        img_array = np.array(img).astype("float32")
        img_array = 255 - img_array  # invert: white background -> black background
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = self.model.predict(img_array, verbose=0)[0]
        digit = int(np.argmax(predictions))
        confidence = float(predictions[digit]) * 100

        self.result_label.config(text=f"Predicted Digit: {digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")


def main():
    try:
        model = load_or_train_model()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load or train model:\n{e}")
        return

    root = tk.Tk()
    DigitRecognitionApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
