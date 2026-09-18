"""
Handwritten Digit Recognition - Desktop Version (Tkinter)
Draw a digit (0-9) with the mouse and click Recognize.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_model import DigitCNN  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_model.pt")
CANVAS_SIZE = 280
BRUSH_SIZE = 18


def load_or_train_model():
    model = DigitCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        print("Model not found. Training a new model on MNIST (this may take a few minutes)...")
        import train_model

        train_model.main()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


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

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(img_tensor)[0]
            predictions = torch.softmax(logits, dim=0).numpy()

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
