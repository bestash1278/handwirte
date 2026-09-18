# 손글자 숫자 인식 프로그램 ✍️

마우스로 그린 손글자 숫자를 AI가 인식하는 웹 애플리케이션입니다.

## 기능

- 🎨 **마우스로 그리기**: 웹 캔버스에서 손글자로 숫자 작성
- 🤖 **AI 인식**: TensorFlow 기반 MNIST 모델로 실시간 인식
- 📊 **신뢰도 표시**: 각 숫자별 확률을 시각화

## 기술 스택

- **프론트엔드**: Streamlit + Drawable Canvas
- **백엔드**: Python + TensorFlow/Keras
- **모델**: CNN (Convolutional Neural Network)
- **데이터셋**: MNIST (70,000개의 손글자 숫자 이미지)

## 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/[your-username]/handwriting-recognition.git
cd handwriting-recognition
```

### 2. 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 모델 학습 (처음 한 번만)
```bash
python train_model.py
```
- 약 5-10분 소요
- MNIST 데이터셋을 다운로드하고 학습합니다
- 학습된 모델은 `models/mnist_model.h5`에 저장됩니다

### 4. 웹 앱 실행
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 앱을 사용할 수 있습니다.

## 사용 방법

1. 웹 페이지의 왼쪽 **캔버스**에 마우스로 숫자(0-9)를 그립니다
2. 오른쪽에 **예측 결과**와 **신뢰도**가 실시간으로 표시됩니다
3. 각 숫자별 확률 분포를 확인할 수 있습니다
4. **초기화** 버튼으로 캔버스를 비우고 다시 그릴 수 있습니다

## 모델 정보

- **정확도**: ~99% (테스트 데이터셋 기준)
- **구조**: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense
- **입력**: 28×28 그레이스케일 이미지
- **출력**: 0-9 숫자 분류

## 프로젝트 구조

```
handwriting-recognition/
├── app.py                # Streamlit 웹 애플리케이션
├── train_model.py        # 모델 학습 스크립트
├── requirements.txt      # Python 라이브러리 의존성
├── .gitignore           # Git 제외 파일
├── README.md            # 이 파일
└── models/
    └── mnist_model.h5   # 학습된 모델 (자동 생성)
```

## 팁

- 숫자를 크게 그릴수록 인식 정확도가 높습니다
- 캔버스 중앙에 그리는 것이 좋습니다
- 느릿한 동작으로 명확하게 그려보세요

## 라이선스

MIT License

## 작성자

Claude AI with Streamlit
