import requests
import urllib.parse
import re
import time
import os
import pandas as pd
import json

# 네이버 API 정보 입력
client_id = 'gViXcpkbNs6rFh4AinDb'
client_secret = 'tXrChF5Wcx'

# 이 파일은 네이버 블로그 검색 API를 활용하여, 키워드로 검색한 블로그의 url과 요약(description)을 CSV 파일로 저장하는 코드입니다.
# 네이버 블로그 url을 복사해오는 용도로 사용할 수 있습니다.

def get_blog_urls(keyword, display=10, end=1):
    naver_urls = []
    postdates = []
    titles = []
    encText = urllib.parse.quote(keyword)
    for start in range(end):
        url = f"https://openapi.naver.com/v1/search/blog?query={encText}&start={start*display+1}&display={display}"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()['items']
            for row in data:
                if 'blog.naver' in row['link']:
                    naver_urls.append(row['link'])
                    postdates.append(row['postdate'])
                    title = re.sub('<.*?>', '', row['title'])
                    titles.append(title)
            time.sleep(1)
        else:
            print("Error Code:", response.status_code)
    return naver_urls, titles, postdates

if __name__ == "__main__":
    # config.json에서 설정값 읽기
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    keyword = config.get("keyword", "아조나치약")
    display = config.get("display", 10)
    end = config.get("end", 1)

    naver_urls, titles, postdates = get_blog_urls(keyword, display=display, end=end)
    print(f"총 {len(naver_urls)}개의 블로그 URL을 수집했습니다.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "reviews", "url", keyword)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "blog_urls.csv")
    df = pd.DataFrame({'title': titles, 'date': postdates, 'url': naver_urls})
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"{save_path} 파일로 저장 완료!")