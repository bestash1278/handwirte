"""Shared MNIST model and drawing preprocessing. Created 2026-09-21 KST."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'mnist.pt'
torch.set_num_threads(4)

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 10))

    def forward(self, x):
        return self.layers(x)

def preprocess(image):
    """Convert black ink on white paper to a centered MNIST image."""
    ink = ImageOps.invert(image.convert('L'))
    mask = ink.point(lambda p: 255 if p > 32 else 0)
    box = mask.getbbox()
    if box is None or np.count_nonzero(np.asarray(mask)) < 8:
        raise ValueError('Draw a digit first.')
    ink = ink.crop(box)
    scale = 20 / max(ink.size)
    ink = ink.resize(tuple(max(1, round(s * scale)) for s in ink.size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(ink, ((28 - ink.width) // 2, (28 - ink.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    yy, xx = np.indices(pixels.shape)
    mass = pixels.sum()
    dx = round(13.5 - float((pixels * xx).sum() / mass))
    dy = round(13.5 - float((pixels * yy).sum() / mass))
    shifted = Image.new('L', (28, 28))
    shifted.paste(canvas, (dx, dy))
    return np.asarray(shifted, dtype=np.float32) / 255

class Recognizer:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError('Model missing. Run: python train_model.py')
        self.model = DigitNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        self.model.eval()

    def predict(self, image):
        pixels = preprocess(image)
        with torch.inference_mode():
            scores = self.model(torch.from_numpy(pixels.copy())[None, None]).softmax(1)[0].tolist()
        digit = int(np.argmax(scores))
        return {'digit': digit, 'confidence': scores[digit], 'probabilities': scores,
                'pixels': (pixels * 255).astype('uint8').ravel().tolist()}
