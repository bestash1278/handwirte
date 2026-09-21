<!-- Created 2026-09-21 KST -->
# 손글씨 숫자 인식 프로그램

제공된 PDF의 55–71쪽과 84–93쪽 제작 실습을 구현했습니다.
MNIST로 학습한 CNN이 **0–9 숫자 한 개**를 인식합니다. 한글·영문·문장 인식은 지원하지 않습니다.
PDF에 맞춰 코드, 주석, 화면 문구는 영어로 작성했습니다.

## 실행

Windows 탐색기에서 다음 파일을 더블 클릭하세요.

- **`run_desktop.bat`**: 데스크톱 그림판 실행
- **`run_web.bat`**: 웹 서버 실행 후 브라우저에서 <http://127.0.0.1:8000> 접속

Python과 pip, Tcl/Tk가 필요합니다. 현재 환경의 Python 3.14에서 구현했습니다.
필요한 라이브러리가 없으면 배치 파일이 설치합니다. 처음 설치할 때는 인터넷이 필요합니다.
학습된 `models/mnist.pt`가 포함되어 있으므로 이후 실행 시 재학습하지 않습니다.
모델 파일이 없으면 MNIST를 내려받아 자동 학습합니다. 실패하면 터미널의 오류를 확인하세요.
배치 파일은 Python 실행을 돕는 파일이며 Python이 필요 없는 단독 EXE는 아닙니다.

1. 흰 그림판에 숫자 하나를 크게 그립니다.
2. 기본적으로 획을 마친 뒤 자동 인식합니다. `Recognize` 버튼으로 직접 인식할 수도 있습니다.
3. 예측 숫자와 신뢰도를 확인합니다. 웹 버전에서는 숫자별 점수와 28×28 입력 영상도 표시합니다.
4. `Clear`로 지운 다음 새 숫자를 그립니다.

데스크톱은 창을 닫고, 웹 서버는 터미널에서 `Ctrl+C`로 종료합니다.
웹 버전은 마우스·터치·펜 입력을 지원합니다. 그림은 로컬 컴퓨터에서 처리하며 저장하지 않습니다.

## 직접 실행 및 재학습

프로젝트 폴더의 PowerShell에서:

```powershell
python -m pip install -r requirements.txt
python desktop_version/digit_recognition.py
python web_version/server.py
# If port 8000 is occupied:
python web_version/server.py --port 8001
# Retrain the shared model:
python train_model.py --epochs 5
# Run model and HTTP integration tests:
python -m unittest discover -s tests -v
```

## 모델과 평가

- 학습: MNIST 학습 세트 60,000장, 5 epochs, seed 42
- 평가: 학습에 사용하지 않은 MNIST 테스트 세트 10,000장
- 측정 정확도: **99.04% (9,904 / 10,000)**, `models/metrics.json`에 기록
- 전처리: 흑백 변환 → 색상 반전 → 여백 제거 → 최대 20픽셀로 비율 유지 축소 → 28×28 중심 정렬
- 두 화면이 동일한 모델과 전처리를 사용합니다.

MNIST 테스트 정확도는 실제 사용자가 그린 모든 글씨에 대한 보장이 아닙니다.
신뢰도는 모델의 softmax 점수이며 정답 확률로 보정한 값이 아닙니다.
숫자 여러 개, 글자, 그림에도 숫자 결과가 나올 수 있으므로 숫자 하나만 입력하세요.

## 파일 구조

```text
recognition.py             공통 모델·전처리·추론
train_model.py             MNIST 다운로드·학습·평가
models/                   학습 가중치와 평가 결과
desktop_version/          Tkinter 프로그램과 CLAUDE.md
web_version/              로컬 HTTP 서버·웹 화면·CLAUDE.md
tests/                    전처리·실제 MNIST·HTTP 테스트
run_desktop.bat            데스크톱 실행
run_web.bat                웹 서버 실행
setup.bat                 의존성 및 모델 확인
requirements.txt          검증한 라이브러리 버전
CLAUDE.md                 프로젝트 구조·개발 규칙
```

MNIST 출처: [CVDF MNIST](https://github.com/cvdfoundation/mnist).
모델 로딩 참고: [PyTorch 모델 저장·로딩](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
