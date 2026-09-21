<!-- Created 2026-09-21 KST -->
# Handwritten Digit Studio
Recognize one handwritten digit (0–9) using MNIST and a shared PyTorch CNN.

## Commands
- Install: `python -m pip install -r requirements.txt`
- Train: `python train_model.py --epochs 5`
- Desktop: `python desktop_version/digit_recognition.py`
- Web: `python web_version/server.py`
- Test: `python -m unittest discover -s tests -v`

## Architecture
`recognition.py` owns the network, normalization and inference. `train_model.py`
downloads checksum-verified MNIST and saves weights plus measured test metrics.
Both interfaces use `models/mnist.pt`. Never use test images for training.

## Conventions
Use English code/comments and creation-date comments in new source files.
Keep UI input black on white; model input is white on black, 28 × 28.
Do not send drawings to external services. Model scores are not calibrated certainty.
