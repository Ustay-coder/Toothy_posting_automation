# toothpaste_review_analyzer.py
from pathlib import Path
from typing import List, Tuple, Dict, Literal
import pandas as pd
import openai
from classifier import SentenceOpinionClustering
import numpy as np
import os
from dotenv import load_dotenv
from utils import clean_html, split_korean_sentences

load_dotenv()
# ────────────────────────────────────────────────────
# 4) 메인 분석 클래스
class ReviewAnalyzer:
    """
    blog_full.csv → 긍정/부정 TOP-k 클러스터 추출
    """

    def __init__(
        self,
        csv_path: str | Path = "blog_full.csv",
        *,
        encoding: str | None = "utf-8-sig",
        distance_threshold: float = 0.4,
        lang: Literal["ko", "en"] = "ko",
        top_k: int = 5,
        openai_model: str = "gpt-4o-mini",
        device: str = "cpu",
        profile: bool = False,
        purpose: str = "나는 이 리뷰를 통해 소비자들이 해당 제품을 사용했을 때 어떠한 경험을 했는지를 파악하고 싶다."
    ):
        self.csv_path = Path(csv_path)
        self.profile = profile
        self.device = device
        self.encoding = encoding
        self.lang = lang
        self.top_k = top_k
        self.purpose = purpose
        self.clusterer = SentenceOpinionClustering(
            distance_threshold=distance_threshold,
            device=device,
            profile=True,
            purpose=self.purpose
        )
        self.openai_model = openai_model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # ────────── 데이터 로드 & 문장 추출 ──────────
    def _load_sentences(self, *, min_tokens: int = 2) -> Tuple[pd.DataFrame, List[str]]:
        df = pd.read_csv(
            self.csv_path,
            usecols=["title", "content", "date", "url"],
            encoding=self.encoding,
            dtype={"title": str, "content": str, "url": str},
            parse_dates=["date"],
            infer_datetime_format=True,
        )
        for col in ["title", "content"]:
            df[col] = df[col].apply(clean_html)

        df["sentences"] = df["content"].apply(split_korean_sentences)
        all_sents = [s for lst in df["sentences"] for s in lst if len(s.split()) >= min_tokens]
        return df, all_sents

    # ────────── GPT 감정 분류 util ──────────
    @staticmethod
    def _normalize(lbl: str, lang: str) -> str:
        lbl = lbl.strip().lower()
        ko = {"긍정": "긍정", "중립": "중립", "부정": "부정",
              "positive": "긍정", "neutral": "중립", "negative": "부정"}
        en = {"positive": "positive", "neutral": "neutral", "negative": "negative",
              "긍정": "positive", "중립": "neutral", "부정": "negative"}
        return (ko if lang == "ko" else en).get(lbl, "중립" if lang == "ko" else "neutral")

    # ────────── GPT 감정 분류 util (업데이트) ──────────
    def _classify(self, phrase: str, purpose: str) -> str:
        """
        • purpose 문맥에 비추어 phrase 가 긍정/중립/부정인지 판정
        • purpose 와 무관하거나 치약 사용 경험이 명확치 않으면 중립
        • self.lang == 'ko' → 한글 라벨, 'en' → 영어 라벨
        """
        # ① system 프롬프트: 라벨 언어 고정 + 규칙 명시
        if self.lang == "en":
          sys_msg = (
            "You are a sentiment classifier focused on toothpaste reviews.\n"
            "Goal: Reflect the user's purpose, then respond with **exactly** "
            "'positive', 'neutral' or 'negative' (lowercase only).\n"
            "- If the sentence does not clearly describe the consumer's "
            "experience *after using* the toothpaste or is irrelevant to the purpose, "
            "return 'neutral'.\n"
            "- No other words, no punctuation."
          )
        else:  # 'ko'
          sys_msg = (
            "당신은 치약 리뷰에 대한 감정 분류기입니다.\n"
            "아래 목적을 반영한 뒤 반드시 '긍정', '중립', '부정' "
            "중 하나만 정확히 출력하세요 (다른 문장·기호 금지).\n"
            "- 문장이 소비자의 사용 후 경험을 명확히 말하지 않거나 목적과 관련 없으면 '중립'으로 분류합니다."
        )

        # ② user 프롬프트: 목적 + 문장
        user_prompt = (
            f"[Purpose]\n{purpose}\n\n"
            f"[Sentence]\n\"{phrase}\""
        )

        resp = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return self._normalize(resp.choices[0].message.content, self.lang)


    # ────────── 전체 파이프라인 ──────────
    def analyze(
        self,
        purpose: str,
        *,
        debug: bool = True,
        sample_size: int = 10,
        min_tokens: int = 2,
    ) -> Tuple[List[Tuple[int, str, int]], List[Tuple[int, str, int]]]:
        """
        반환: (긍정 TOP-k 리스트, 부정 TOP-k 리스트)
        각 원소: (cluster_id, 대표 구, 클러스터 문장 수)
        """
        _, sentences_all = self._load_sentences(min_tokens=min_tokens)
        sentences = sentences_all[:sample_size] if debug else sentences_all

        # clusters = self.clusterer.cluster(sentences)
        # phrases = {cid: self.clusterer.representative_phrase(sents)
        #            for cid, sents in clusters.items()}
        clusters, phrases = self.clusterer.summarize(sentences)

        bucket_pos, bucket_neg = [], []
        for cid, phrase in phrases.items():
            lbl = self._classify(phrase, purpose)
            if lbl in ("positive", "긍정"):
                bucket_pos.append((cid, phrase, len(clusters[cid])))
            elif lbl in ("negative", "부정"):
                bucket_neg.append((cid, phrase, len(clusters[cid])))
            # 중립은 버림

        bucket_pos.sort(key=lambda x: x[2], reverse=True)
        bucket_neg.sort(key=lambda x: x[2], reverse=True)

        return bucket_pos[:self.top_k], bucket_neg[:self.top_k]

    # ────────── 결과 콘솔 출력 ──────────
    def print_summary(self, pos: List[Tuple[int, str, int]], neg: List[Tuple[int, str, int]]) -> None:
        label_pos, label_neg = (("positive", "negative") if self.lang == "en" else ("긍정", "부정"))
        print(f"\n🟢 {label_pos.upper()} TOP-{len(pos)}")
        for cid, phr, sz in pos:
            print(f"[{cid}] ({sz}문장) {phr}")
        print(f"\n🔴 {label_neg.upper()} TOP-{len(neg)}")
        for cid, phr, sz in neg:
            print(f"[{cid}] ({sz}문장) {phr}")

    def save_summary(self, pos: List[Tuple[int, str, int]], neg: List[Tuple[int, str, int]]) -> None:
        label_pos, label_neg = (("positive", "negative") if self.lang == "en" else ("긍정", "부정"))
        with open(f"{self.csv_path.stem}_summary.txt", "w") as f:
            f.write(f"🟢 {label_pos.upper()} TOP-{len(pos)}\n")
            for cid, phr, sz in pos:
                f.write(f"[{cid}] ({sz}문장) {phr}\n")
            f.write(f"\n🔴 {label_neg.upper()} TOP-{len(neg)}\n")
            for cid, phr, sz in neg:
                f.write(f"[{cid}] ({sz}문장) {phr}\n")


        