# Desktop Version - Digit Recognition

Tkinter desktop app for handwritten digit recognition.

- `digit_recognition.py`: main app. Draws on a canvas, predicts with a CNN trained on MNIST.
- Model is loaded from `../models/mnist_model.pt` (PyTorch); trained automatically on first run if missing.
- `install_requirements.bat` / `run_digit_recognition.bat`: Windows helper scripts.
