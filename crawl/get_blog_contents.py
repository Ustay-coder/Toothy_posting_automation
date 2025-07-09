# 로컬에서는 GPU 문제로 인해 parameter 다운로드 시간이 오래 걸리고 추론 시간도 느릴 것으로 예상됨
# 따라서 Colab에서 실행하거나 GPU 서버를 운용해서 실행해야 함. 

import pandas as pd
from transformers import pipeline
from konlpy.tag import Okt
from collections import Counter
import re
import os, json
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import re
import traceback
# 1. 데이터 불러오기
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
keyword = config.get("keyword", "아조나치약")


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def get_blog_contents(naver_urls):
    print(f"[DEBUG] 크롬 드라이버 및 옵션 설정 시작. URL 개수: {len(naver_urls)}")
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # 창을 띄우지 않고 실행
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    contents = []
    for idx, url in enumerate(naver_urls):
        try:
            print(f"[DEBUG] ({idx+1}/{len(naver_urls)}) 크롤링 중: {url}")
            driver.get(url)
            time.sleep(2)
            # 네이버 블로그는 iframe(mainFrame) 안에 본문이 있음
            iframe = driver.find_element(By.ID, "mainFrame")
            driver.switch_to.frame(iframe)
            source = driver.page_source
            html = BeautifulSoup(source, "html.parser")
            # 최신 에디터(2024년 기준) 본문 div
            content = html.select("div.se-main-container")
            if not content:
                # 구 에디터 대응
                content = html.select("div#postViewArea")
            content = ''.join(str(content))
            content = clean_html(content)
            content = content.replace('\n', '').replace('\u200b', '')
            contents.append(content)
        except Exception as e:
            print(f"[ERROR] 크롤링 실패: {url} / {e}")
            traceback.print_exc()
            contents.append('error')
    driver.quit()
    print(f"[DEBUG] 크롤링 완료. 총 {len(contents)}개 수집.")
    return contents

if __name__ == "__main__":
    print("[DEBUG] blog_content_crawler.py 시작")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        read_dir = os.path.join(script_dir, "reviews", "url", keyword)
        df = pd.read_csv(os.path.join(read_dir, "blog_urls.csv"), encoding='utf-8-sig')
        print(f"[DEBUG] blog_urls.csv 파일 로드 성공. {len(df)}개 URL")
    except Exception as e:
        print(f"[ERROR] blog_urls.csv 파일을 읽는 중 오류 발생: {e}")
        traceback.print_exc()
        exit(1)
    naver_urls = df['url'].tolist()
    titles = df['title'].tolist()
    postdates = df['date'].tolist()
    print(f"[DEBUG] {len(naver_urls)}개의 블로그 본문을 크롤링합니다.")
    contents = get_blog_contents(naver_urls)
    print("[DEBUG] DataFrame 생성 및 blog_full.csv 저장 시도")
    try:
        result_df = pd.DataFrame({'title': titles, 'content': contents, 'date': postdates, 'url': naver_urls})
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "reviews", "content", keyword)
        os.makedirs(save_dir, exist_ok=True)
        result_df.to_csv(os.path.join(save_dir, 'blog_full.csv'), index=False, encoding='utf-8-sig')
        print("[DEBUG] blog_full.csv 파일로 저장 완료!")
    except Exception as e:
        print(f"[ERROR] blog_full.csv 저장 중 오류 발생: {e}")
        traceback.print_exc() 