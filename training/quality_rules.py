#!/usr/bin/env python3
import re
from collections import Counter


POSITIVE_PATTERNS = {
    "empathy": ["我能理解", "我能感受到", "听起来", "辛苦了", "不容易", "很难受", "委屈你了"],
    "support": ["如果你愿意", "我们可以一起", "慢慢来", "先照顾好自己", "可以试试", "我在这里"],
    "exploration": ["愿意多说一点", "发生了什么", "最难受的是什么", "什么时候开始的", "你怎么看"],
    "validation": ["你的感受很重要", "有这样的感受很正常", "能被理解", "不是你的错"],
}

NEGATIVE_PATTERNS = [
    "别想太多",
    "你应该",
    "你必须",
    "你需要",
    "赶紧",
    "这没什么",
    "想开点",
    "活该",
    "矫情",
    "后悔",
    "只有坚持",
    "喜欢就去追",
    "阳光点",
    "做点有意义的事吧",
]

MEDICAL_PATTERNS = [
    "确诊",
    "开药",
    "药物治疗",
    "处方",
]

JUNK_PATTERNS = [
    r"^assistant(\s|$)",
    r"^recovered(\s|$)",
    r"^oplayer(\s|$)",
    r"^ypse(\s|$)",
    r"^<tool_call>(\s|$)",
    r"^[.。!！?？]+$",
]

STYLE_DRIFT_PATTERNS = [
    r"<tool_response>",
    r"<tool_call>",
    r"^(?:[-./]?[A-Za-z_][A-Za-z0-9_./-]{2,20})\s*$",
    r"^(?:[-./]?[A-Za-z_][A-Za-z0-9_./-]{2,20})\n+",
    r"^(?:咨询师|来访者|支持者|用户|助理)[:：]",
    r"^(?:妹妹|姐妹|亲亲|宝宝|宝贝|哥哥|同学|亲爱的)[，,]",
    r"^(?:hi|hello)[~！!，,\s]",
    r"(?:责任和使命|随时向我汇报进展)",
    r"(?:🌟|✨|❤️|❤|🌱|🌈|💗|💖|🫶|🌸)",
    r"(?:压力山大|不是一个人在战斗)",
    r"(?:希望我的回答对你有帮助|随时来找我聊天|如果以后还有需要|祝你好运|祝福你)",
    r"(?:我是.*助理)",
]

IMPERATIVE_PATTERNS = [
    "你应该",
    "你必须",
    "你需要",
    "一定要",
    "赶紧",
    "立刻",
    "马上",
    "别再",
    "喜欢就去追",
    "阳光点",
    "做点有意义的事吧",
]

ADVICE_STRUCTURE_PATTERNS = [
    r"(?:首先|其次|另外|最后)",
    r"(?:以下是一些建议|可以从以下几个方面)",
    r"(?:\n\s*1\.|\n\s*2\.|\n\s*3\.)",
]


def looks_chinese(text: str, threshold: float = 0.2) -> bool:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese / max(len(text), 1) >= threshold


def has_think_leak(text: str) -> bool:
    lower = text.lower()
    return "<think>" in lower or "</think>" in lower or "思考：" in text


def has_style_drift(text: str) -> bool:
    text = (text or "").strip()
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in STYLE_DRIFT_PATTERNS)


def advice_heavy_score(text: str) -> float:
    text = (text or "").strip()
    score = 0
    if any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in ADVICE_STRUCTURE_PATTERNS):
        score += 1
    if text.count("你可以") >= 2:
        score += 1
    if text.count("可以尝试") >= 2:
        score += 1
    if text.count("建议") >= 2:
        score += 1
    return float(score > 0)


def passes_keep_gate(text: str) -> bool:
    breakdown = quality_breakdown(text)
    if breakdown["usable"] < 1.0:
        return False
    if breakdown["style_drift"] > 0:
        return False
    if breakdown["negative"] > 0:
        return False
    if breakdown["medicalized"] > 0:
        return False
    if breakdown["advice_heavy"] > 0:
        return False
    if breakdown["imperative_penalty"] > 0:
        return False
    return True


def is_junk_response(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 12:
        return True
    if not looks_chinese(text):
        return True
    if has_style_drift(text):
        return True
    return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in JUNK_PATTERNS)


def contains_negative_behavior(text: str) -> bool:
    return any(pattern in text for pattern in NEGATIVE_PATTERNS)


def contains_medicalized_language(text: str) -> bool:
    return any(pattern in text for pattern in MEDICAL_PATTERNS)


def repetition_penalty(text: str) -> float:
    tokens = [token for token in re.split(r"[\s，。！？；：,!.?;:]+", text) if token]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated) / max(len(tokens), 1)


def imperative_rate(text: str) -> float:
    hits = sum(marker in text for marker in IMPERATIVE_PATTERNS)
    return hits / max(len(IMPERATIVE_PATTERNS), 1)


def too_short(text: str, threshold: int = 20) -> bool:
    return len((text or "").strip()) < threshold


def quality_breakdown(text: str) -> dict[str, float]:
    text = (text or "").strip()
    scores = {
        "usable": 0.0 if is_junk_response(text) or has_think_leak(text) else 1.0,
        "style_drift": float(has_style_drift(text)),
        "empathy": sum(pattern in text for pattern in POSITIVE_PATTERNS["empathy"]),
        "supportiveness": sum(pattern in text for pattern in POSITIVE_PATTERNS["support"]),
        "exploration": sum(pattern in text for pattern in POSITIVE_PATTERNS["exploration"]),
        "validation": sum(pattern in text for pattern in POSITIVE_PATTERNS["validation"]),
        "negative": float(contains_negative_behavior(text)),
        "medicalized": float(contains_medicalized_language(text)),
        "advice_heavy": advice_heavy_score(text),
        "repetition_penalty": repetition_penalty(text),
        "imperative_penalty": imperative_rate(text),
        "too_short": float(too_short(text)),
        "think_leak": float(has_think_leak(text)),
    }
    total = (
        1.5 * scores["empathy"]
        + 1.25 * scores["supportiveness"]
        + 1.0 * scores["exploration"]
        + 0.75 * scores["validation"]
        - 2.0 * scores["negative"]
        - 1.5 * scores["medicalized"]
        - 1.5 * scores["advice_heavy"]
        - 2.0 * scores["too_short"]
        - 2.5 * scores["think_leak"]
        - 2.5 * (1.0 - scores["usable"])
        - 1.5 * scores["style_drift"]
        - 2.0 * scores["imperative_penalty"]
        - 1.5 * scores["repetition_penalty"]
    )
    scores["total"] = total
    return scores


def overall_score(text: str) -> float:
    return quality_breakdown(text)["total"]
