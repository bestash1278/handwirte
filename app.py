import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="손글자 숫자 인식", layout="centered")
st.title("✍️ 손글자 숫자 인식")
st.markdown("마우스로 숫자를 그려보세요 (0-9)")


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


@st.cache_resource
def load_model():
    if not os.path.exists("models/mnist_model.pt"):
        st.error("❌ 모델 파일을 찾을 수 없습니다. 먼저 train_model.py를 실행하세요.")
        st.stop()
    model = DigitCNN()
    model.load_state_dict(torch.load("models/mnist_model.pt", map_location="cpu"))
    model.eval()
    return model


model = load_model()

# Canvas 설정
col1, col2 = st.columns(2)

with col1:
    st.subheader("Canvas")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=4,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.subheader("결과")
    result_placeholder = st.empty()


def predict_digit(image_data):
    if image_data is None:
        return None

    img = np.array(image_data.getdata()).reshape(300, 300)

    # 흑백 반전 (모델은 검은 배경에 흰 숫자를 기대)
    img = 255 - img

    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized.astype("float32") / 255.0

    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)[0]
        predictions = torch.softmax(logits, dim=0).numpy()

    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit]) * 100

    return predicted_digit, confidence, predictions


# Canvas 결과 처리
if canvas_result.image_data is not None:
    result = predict_digit(Image.fromarray((canvas_result.image_data).astype("uint8")).convert("L"))

    if result:
        digit, confidence, all_probs = result

        with result_placeholder.container():
            st.metric("예측 숫자", digit, f"{confidence:.1f}% 확신도")

            st.bar_chart({str(i): float(p) for i, p in enumerate(all_probs)})

            st.write("**각 숫자별 확률:**")
            for i, prob in enumerate(all_probs):
                st.write(f"{i}: {prob*100:.2f}%")
else:
    with result_placeholder.container():
        st.info("🎨 캔버스에 숫자를 그려주세요!")

# 초기화 버튼
if st.button("초기화", key="reset"):
    st.rerun()

st.markdown("---")
st.markdown("""
### 사용방법
1. 왼쪽 캔버스에 마우스로 숫자를 그리세요
2. 오른쪽에 인식 결과가 실시간으로 표시됩니다
3. 초기화 버튼으로 캔버스를 비울 수 있습니다

### 팁
- 숫자를 크게 그릴수록 정확도가 높습니다
- 캔버스 중앙에 그리는 것이 좋습니다
""")
