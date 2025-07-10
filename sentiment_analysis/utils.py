import re
from typing import List
import html

# 1) 문장 분리용 kss (없으면 정규식 fallback)
try:
    from kss import split_sentences as kss_split
    _HAS_KSS = True
except ImportError:
    _HAS_KSS = False
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s*')

def split_korean_sentences(text: str) -> List[str]:
    if not text:
        return []
    if _HAS_KSS:
        return [s.strip() for s in kss_split(text) if s.strip()]
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


# ────────────────────────────────────────────────────
# 2) 간단한 HTML 정제 util
_TAG_RE, _BR_RE, _SP_RE = re.compile(r"<[^>]+>"), re.compile(r"(?:<br\s*/?>)+", re.I), re.compile(r"[ \t]+")
def clean_html(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    txt = html.unescape(raw)
    txt = _BR_RE.sub("\n", txt)
    txt = _TAG_RE.sub("", txt)
    txt = _SP_RE.sub(" ", txt).strip()
    return txt
