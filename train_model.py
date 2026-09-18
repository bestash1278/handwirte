import os
import struct
import gzip
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("models", exist_ok=True)

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = "mnist_data"


def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in MNIST_URLS.items():
        path = os.path.join(DATA_DIR, f"{name}.gz")
        if not os.path.exists(path):
            print(f"{name} 다운로드 중...")
            urllib.request.urlretrieve(url, path)


def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def main():
    print("MNIST 데이터셋 다운로드 및 로드 중...")
    download_mnist()

    x_train = load_images(os.path.join(DATA_DIR, "train_images.gz")).astype("float32") / 255.0
    y_train = load_labels(os.path.join(DATA_DIR, "train_labels.gz")).astype("int64")
    x_test = load_images(os.path.join(DATA_DIR, "test_images.gz")).astype("float32") / 255.0
    y_test = load_labels(os.path.join(DATA_DIR, "test_labels.gz")).astype("int64")

    print(f"훈련 데이터: {x_train.shape}")
    print(f"테스트 데이터: {x_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    x_train_t = torch.from_numpy(x_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t), batch_size=128, shuffle=True
    )

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print("\n모델 학습 중...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / total:.4f} - "
            f"accuracy: {correct / total:.4f}"
        )

    print("\n모델 평가 중...")
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_t)
        test_accuracy = (outputs.argmax(1) == y_test_t).float().mean().item()
    print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), "models/mnist_model.pt")
    print("\n모델이 models/mnist_model.pt에 저장되었습니다.")


if __name__ == "__main__":
    main()
