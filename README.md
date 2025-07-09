python 3.10.13 (pyenv 사용)

---

## 프로젝트 환경설정 방법

### 1. Python 버전 설치
이 프로젝트는 Python 3.10.13 버전을 사용합니다. [pyenv](https://github.com/pyenv/pyenv)를 통해 Python 버전을 관리합니다.

```bash
# pyenv가 설치되어 있지 않다면 설치
brew install pyenv

# 프로젝트에 맞는 Python 버전 설치
pyenv install 3.10.13

# 프로젝트 디렉토리에서 해당 버전 사용
pyenv local 3.10.13
```

### 2. 가상환경 생성 및 활성화
가상환경을 생성하여 패키지 의존성을 분리합니다.

```bash
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치
`requirements.txt` 파일이 있다면 아래 명령어로 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
필요한 환경 변수(.env 파일 등)가 있다면 예시 파일을 참고하여 설정합니다.

### 5. 실행 방법
프로젝트의 실행 방법(예: main.py 실행 등)을 명시합니다.

```bash
python main.py
```
