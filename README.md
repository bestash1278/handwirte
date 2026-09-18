# 손글자 숫자 인식 프로그램 ✍️

마우스로 그린 손글자 숫자를 AI가 인식하는 프로그램입니다. 웹(Streamlit) 버전과 데스크톱(Tkinter) 버전 두 가지를 제공합니다.

## 웹 버전 (기본)

## 기능

- 🎨 **마우스로 그리기**: 웹 캔버스에서 손글자로 숫자 작성
- 🤖 **AI 인식**: PyTorch 기반 MNIST 모델로 실시간 인식
- 📊 **신뢰도 표시**: 각 숫자별 확률을 시각화

## 기술 스택

- **프론트엔드**: Streamlit + Drawable Canvas
- **백엔드**: Python + PyTorch
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
- 약 5-10분 소요 (이미 `models/mnist_model.pt`가 저장소에 포함되어 있어 생략 가능)
- MNIST 데이터셋을 다운로드하고 학습합니다
- 학습된 모델은 `models/mnist_model.pt`에 저장됩니다

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

## 팁

- 숫자를 크게 그릴수록 인식 정확도가 높습니다
- 캔버스 중앙에 그리는 것이 좋습니다
- 느릿한 동작으로 명확하게 그려보세요

## 데스크톱 버전 (Tkinter)

인터넷 없이 실행되는 윈도우 데스크톱 앱입니다. `desktop_version/` 폴더에 있습니다.

### 설치 및 실행

```bash
cd desktop_version
pip install -r requirements.txt
python digit_recognition.py
```

또는 윈도우에서 `install_requirements.bat`을 먼저 실행해 라이브러리를 설치하고, `run_digit_recognition.bat`으로 앱을 실행할 수 있습니다.

- 학습된 모델(`models/mnist_model.pt`)이 없으면 첫 실행 시 자동으로 MNIST 데이터로 학습합니다 (몇 분 소요)
- 캔버스에 마우스로 숫자를 그린 뒤 **Recognize** 버튼을 누르면 예측 숫자와 신뢰도가 표시됩니다
- **Clear** 버튼으로 캔버스를 지울 수 있습니다

## 프로젝트 구조

```
handwriting-recognition/
├── app.py                       # Streamlit 웹 애플리케이션
├── train_model.py               # 모델 학습 스크립트
├── requirements.txt             # Python 라이브러리 의존성 (웹 버전)
├── .gitignore                   # Git 제외 파일
├── README.md                    # 이 파일
├── models/
│   └── mnist_model.pt          # 학습된 모델 (저장소에 포함, 웹/데스크톱 공용)
└── desktop_version/
    ├── digit_recognition.py    # Tkinter 데스크톱 앱
    ├── requirements.txt        # Python 라이브러리 의존성 (데스크톱 버전)
    ├── install_requirements.bat
    └── run_digit_recognition.bat
```

## 라이선스

MIT License

## 작성자

Claude AI
