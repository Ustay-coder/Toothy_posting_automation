# toothpaste_review_analyzer.py
from pathlib import Path
from typing import List, Tuple, Dict, Literal
import os, re, html, openai, pandas as pd
from collections import defaultdict
import time, gc, psutil, torch
import openai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from .utils import clean_html, split_korean_sentences


# ────────────────────────────────────────────────────
# 3) SBERT + AgglomerativeClustering
class SentenceOpinionClustering:
    """
    ① SBERT 임베딩으로 문장 군집화
    ② 각 군집에서 대표 감정 구 추출(GPT-4)
    ③ 결과(dict)·출력(print) 지원
    """

    def __init__(
        self,
        *,
        sbert_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        distance_threshold: float = 0.4,
        openai_model: str = "gpt-4o-mini",
        openai_temperature: float = 0.3,
        device: str = "cpu",
        profile: bool = False,
        purpose: str = "나는 이 리뷰를 통해 소비자들이 해당 제품을 사용했을 때 어떠한 경험을 했는지를 파악하고 싶다."
    ):
        self.device = torch.device(device)
        self.embedder = SentenceTransformer(sbert_model, device=self.device)
        self.distance_threshold = distance_threshold
        self.openai_model = openai_model
        self.openai_temperature = openai_temperature
        self.profile = profile
        self.purpose = purpose


        # 환경변수에 OPENAI_API_KEY 가 설정돼 있어야 함
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

     # ----------------------------------------------------------------
     # gpu / cpu 메모리 사용량 비교 분석을 위한 snapshot
    def _snapshot(self):
        """(RSS, GPU alloc, GPU reserved) MB"""
        rss = psutil.Process().memory_info().rss / 1024**2
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserv = torch.cuda.memory_reserved() / 1024**2
            return rss, alloc, reserv
        return rss, None, None
    # ────────────────────────────────────────────────────────────
    # 1) 문장 군집화
    def _cluster_sentences(self, sentences: List[str]) -> Dict[int, List[str]]:

        if self.profile:
            rss0, g0a, g0r = self._snapshot()
            t0 = time.perf_counter()

        embeddings = self.embedder.encode(sentences, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
        if self.profile:
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t1 = time.perf_counter()
            rss1, g1a, g1r = self._snapshot()
            print(
                f"[{self.device}]  embed {len(sentences):,} 문장  |  "
                f"time {t1 - t0:.2f}s  |  RAM {rss1 - rss0:+.1f} MB  "
                + (
                    f"|  VRAM {g1a - g0a:+.1f}/{g1r - g0r:+.1f} MB"
                    if self.device.type == "cuda" else ""
                )
            )
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        ).fit(embeddings)

        clusters: Dict[int, List[str]] = defaultdict(list)
        for sent, label in zip(sentences, clustering.labels_):
            clusters[label].append(sent)
        return dict(clusters)

    # ────────────────────────────────────────────────────────────
    # 2) GPT-4 로 대표 감정 구 추출
    def _extract_phrase(
        self,
        cluster_sentences: List[str],
    ) -> str:
        """
        주어진 클러스터에서 purpose 를 가장 잘 드러내는 감정 구절을 1줄로 추출
        """
        representative = cluster_sentences[0]           # 필요하면 중심 문장으로 교체

        prompt = (
            f"[Purpose]\n{self.purpose}\n\n"
            "다음 문장에서 위 목적을 가장 잘 보여주는 감정을 담은 **핵심 구절 하나**만 추출해줘.\n"
            f"\"{representative}\"\n"
            "한 문장 또는 아주 짧은 구로만."
        )

        res = openai.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.openai_temperature,
        )
        return res.choices[0].message.content.strip()

    # ────────────────────────────────────────────────────────────
    # 3) 공개 API
    def summarize(self, sentences: List[str]) -> Dict[int, str]:
        """
        입력 문장 → {클러스터 ID: 대표 감정 구} 딕셔너리 반환
        """
        clusters = self._cluster_sentences(sentences)
        summary = {cid: self._extract_phrase(sents) for cid, sents in clusters.items()}
        return clusters, summary

    def print_summary(self, sentences: List[str], *, sort_by_size: bool = False) -> None:
        """
        summarize() 결과를 보기 좋게 콘솔에 출력
        """
        clusters = self._cluster_sentences(sentences)
        items = clusters.items()
        if sort_by_size:
            items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

        for cid, sents in items:
            phrase = self._extract_phrase(sents)
            joined = ", ".join(sents)
            print(f"🧾 Cluster {cid} ({len(sents)}개)\n  • 대표 구: {phrase}\n  • 문장들: {joined}\n")