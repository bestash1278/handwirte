"""Download MNIST, train on 60,000 images, evaluate on 10,000. Created 2026-09-21 KST."""
import argparse
import gzip
import hashlib
import json
import urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from recognition import DigitNet, ROOT, MODEL_PATH

FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}

def load_data(split):
    arrays = []
    for kind in ('images-idx3', 'labels-idx1'):
        name = f'{split}-{kind}-ubyte.gz'
        path = ROOT / 'data' / name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            print(f'Downloading {name}...', flush=True)
            with urllib.request.urlopen('https://storage.googleapis.com/cvdf-datasets/mnist/' + name, timeout=60) as response:
                content = response.read()
            if hashlib.md5(content).hexdigest() != FILES[name]:
                raise ValueError(f'Invalid download: {name}')
            path.write_bytes(content)
        if hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f'Corrupt dataset: {path}. Remove it and retry.')
        raw = gzip.decompress(path.read_bytes())
        arrays.append(np.frombuffer(raw, dtype=np.uint8, offset=16 if kind.startswith('images') else 8).copy())
    return TensorDataset(torch.from_numpy(arrays[0].reshape(-1, 1, 28, 28)).float() / 255,
                         torch.from_numpy(arrays[1]).long())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.manual_seed(42)
    train = DataLoader(load_data('train'), batch_size=128, shuffle=True)
    test = DataLoader(load_data('t10k'), batch_size=256)
    model = DigitNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, labels in train:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        print(f'Epoch {epoch + 1}/{args.epochs}: loss={total_loss / len(train.dataset):.4f}', flush=True)
    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in test:
            correct += (model(images).argmax(1) == labels).sum().item()
    metrics = {'test_accuracy': correct / len(test.dataset), 'test_correct': correct,
               'test_samples': len(test.dataset), 'train_samples': len(train.dataset),
               'epochs': args.epochs, 'seed': 42, 'torch_version': torch.__version__}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    temporary = MODEL_PATH.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(MODEL_PATH)
    (MODEL_PATH.parent / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
