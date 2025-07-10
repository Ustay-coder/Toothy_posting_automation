import json
from anlysis import ReviewAnalyzer
from .classifier import SentenceOpinionClustering

def bench(device: str):
    # 테스트용 작은 샘플 (원한다면 ReviewAnalyzer._load_sentences 로딩)
    base = [
        "치약 향이 상쾌해요", "향이 정말 좋아요", "입안이 시원하고 기분이 좋아요",
        "가격이 너무 비싸요", "좀 비싼 편인 것 같아요", "효과가 별로 없네요"
    ]
    sentences = base * 2000           # 12,000 문장

    clusterer = SentenceOpinionClustering(
        distance_threshold=0.5,
        device=device,
        profile=True,                  # ← 프로파일링 켜기
    )
    # 군집만 돌리면 임베딩 성능이 그대로 드러남
    _ = clusterer._cluster_sentences(sentences)


# ────────────────────────────────────────────────────
# 5) 사용 예시
if __name__ == "__main__":
    config = json.load(open("config.json"))
    if config["benchmark"]:
        bench(config["device"]) 
    else:
        analyzer = ReviewAnalyzer(
            config["file_path"],
            distance_threshold=config["distance_threshold"],
            lang=config["lang"],
            top_k=config["top_k"],
            device=config["device"],
        profile=config["profile"],
        purpose=config["purpose"],
        base_save_path=config["base_save_path"],
        save_name=config["save_name"],
    )
    pos, neg = analyzer.analyze(
        purpose=config["purpose"],
        debug=config["debug"],          # False 로 바꾸면 전체 문장 대상
        sample_size=config["sample_size"],
    )
    analyzer.print_summary(pos, neg)
    if config["save"]:
        analyzer.save_summary(pos, neg)
        analyzer.save_summary_image(pos, neg)