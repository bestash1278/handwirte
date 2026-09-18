import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 모델 저장 경로
os.makedirs('models', exist_ok=True)

print("MNIST 데이터셋 로드 중...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"훈련 데이터: {x_train.shape}")
print(f"테스트 데이터: {x_test.shape}")

# 모델 정의
print("\n모델 구축 중...")
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# 모델 학습
print("\n모델 학습 중...")
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1
)

# 모델 평가
print("\n모델 평가 중...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"테스트 정확도: {score[1]*100:.2f}%")

# 모델 저장
model.save('models/mnist_model.h5')
print("\n모델이 models/mnist_model.h5에 저장되었습니다.")
