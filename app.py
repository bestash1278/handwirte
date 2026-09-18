import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="손글자 숫자 인식", layout="centered")
st.title("✍️ 손글자 숫자 인식")
st.markdown("마우스로 숫자를 그려보세요 (0-9)")

# 모델 로드
@st.cache_resource
def load_model():
    if not os.path.exists('models/mnist_model.h5'):
        st.error("❌ 모델 파일을 찾을 수 없습니다. 먼저 train_model.py를 실행하세요.")
        st.stop()
    return keras.models.load_model('models/mnist_model.h5')

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

# 예측 함수
def predict_digit(image_data):
    if image_data is None:
        return None

    # PIL 이미지를 numpy 배열로 변환
    img = np.array(image_data.getdata()).reshape(300, 300)

    # 흑백 반전 (모델은 검은 배경에 흰 숫자를 기대)
    img = 255 - img

    # 28x28로 리사이즈
    img_resized = cv2.resize(img, (28, 28))

    # 0-1 정규화
    img_normalized = img_resized.astype('float32') / 255.0

    # 배치 차원 추가
    img_batch = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)

    # 예측
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit] * 100

    return predicted_digit, confidence, predictions

# Canvas 결과 처리
if canvas_result.image_data is not None:
    result = predict_digit(Image.fromarray((canvas_result.image_data).astype('uint8')))

    if result:
        digit, confidence, all_probs = result

        with result_placeholder.container():
            st.metric("예측 숫자", digit, f"{confidence:.1f}% 확신도")

            # 확률 차트
            st.bar_chart({
                "숫자": list(range(10)),
                "확률": [f"{p*100:.1f}%" for p in all_probs]
            })

            # 상세 확률 표시
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
