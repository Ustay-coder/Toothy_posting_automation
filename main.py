import os
import json
import pandas as pd

# 1. 블로그 URL 수집
from crawl.get_blog_urls import get_blog_urls
# 2. 블로그 본문 크롤링
from crawl.get_blog_contents import get_blog_contents
# 3. 감정 분석
from sentiment_analysis.anlysis import ReviewAnalyzer

def run_all_pipeline():
    # 통합 config.json 읽기
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    crawl_cfg = config["crawl"]
    senti_cfg = config["sentiment_analysis"]

    # 1. 블로그 URL 수집
    naver_urls, titles, postdates = get_blog_urls(
        crawl_cfg["keyword"], display=crawl_cfg["display"], end=crawl_cfg["end"]
    )
    url_save_dir = os.path.join("crawl", crawl_cfg["saveUrlBasePath"], crawl_cfg["keyword"])
    os.makedirs(url_save_dir, exist_ok=True)
    url_save_path = os.path.join(url_save_dir, "blog_urls.csv")
    pd.DataFrame({'title': titles, 'date': postdates, 'url': naver_urls}).to_csv(url_save_path, index=False, encoding='utf-8-sig')
    print(f"[1/3] 블로그 URL 저장 완료: {url_save_path}")

    # 2. 블로그 본문 크롤링
    contents = get_blog_contents(naver_urls)
    content_save_dir = os.path.join("crawl", crawl_cfg["saveContentBasePath"], crawl_cfg["keyword"])
    os.makedirs(content_save_dir, exist_ok=True)
    content_save_path = os.path.join(content_save_dir, "blog_full.csv")
    pd.DataFrame({'title': titles, 'content': contents, 'date': postdates, 'url': naver_urls}).to_csv(content_save_path, index=False, encoding='utf-8-sig')
    print(f"[2/3] 블로그 본문 저장 완료: {content_save_path}")

    # 3. 감정 분석
    senti_cfg["file_path"] = content_save_path  # 최신 데이터로 경로 갱신
    analyzer = ReviewAnalyzer(
        senti_cfg["file_path"],
        distance_threshold=senti_cfg["distance_threshold"],
        lang=senti_cfg["lang"],
        top_k=senti_cfg["top_k"],
        device=senti_cfg["device"],
        profile=senti_cfg["profile"],
        purpose=senti_cfg["purpose"],
        base_save_path=senti_cfg["base_save_path"],
        save_name=senti_cfg["save_name"],
    )
    pos, neg = analyzer.analyze(
        purpose=senti_cfg["purpose"],
        debug=senti_cfg["debug"],
        sample_size=senti_cfg["sample_size"],
    )
    analyzer.print_summary(pos, neg)
    if senti_cfg["save"]:
        analyzer.save_summary(pos, neg)
        analyzer.save_summary_image(pos, neg)
    print("[3/3] 감정 분석 완료")

if __name__ == "__main__":
    run_all_pipeline()