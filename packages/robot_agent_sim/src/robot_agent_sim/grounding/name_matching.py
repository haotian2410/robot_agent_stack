"""Conservative semantic name matching; raw substrings are never exact."""

from __future__ import annotations

import re


def normalize_name(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def canonical_name(value: str) -> str:
    return re.sub(r"_[0-9]+$", "", normalize_name(value))


def exact_name_match(query: str, label: str) -> bool:
    query = normalize_name(query)
    label = normalize_name(label)
    if not query or not label:
        return False
    if query == label or canonical_name(query) == canonical_name(label):
        return True
    # English multi-word aliases may differ only by separators/order-safe
    # token boundaries; never accept apple↔pineapple substring matches.
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    label_tokens = set(re.findall(r"[a-z0-9]+", label))
    return bool(query_tokens and query_tokens == label_tokens)
