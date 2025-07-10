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

#### 5-1. 블로그 url 크롤링
```bash
cd crawl
python get_blog_urls.py
```
#### 5-2. 블로그 url로 컨텐츠 크롤링
```bash
python get_blog_contents.py
```

#### 5-3. 성분 분석 정보 가져오기

```json
{
    "SCHEMA": "[YOUR SUPABASE SCHEMA]",
    "TABLE": "YOUR SUPABASE TABLE",
    "ITEM_NAME": "YOUR ITEM NAME",
    "BASE_URL": "YOUR SERVER BASE URL"
}
```
```bash
python get_toothpaste.py
```

#### 5-4. 감정분석 

```json 
{
    "file_path": [YOUR DATA FILE PATH],
    "distance_threshold": 0.5,
    "lang": "ko",
    "top_k": 10,
    "device": "cpu",
    "profile": true,
    "purpose": [YOUR PURPOSE],
    "debug": true,
    "sample_size": 500,
    "save": true,
    "benchmark": false,
    "openai_model": "gpt-4o-mini",
    "base_save_path": [YOURSAVE BASE PATH]
    "save_name": [YOUR SAVE DIRECTORY NAME]
}
```

```bash
python run.py
```