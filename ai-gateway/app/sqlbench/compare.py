"""Comparing the result of a model's query with the result of the gold query. Column names and aliases
do not matter, the number of columns, the rows and the values do. Rows are compared as a multiset unless
the question asks for an order. Numbers are compared to two decimals."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    match: bool
    reason: str | None  # column_count, row_count or values


def _normalize(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_rows(rows) -> list:
    return [tuple(_normalize(v) for v in row) for row in rows]


def compare_results(gold_rows, predicted_rows, ordered: bool = False) -> Comparison:
    gold, predicted = normalize_rows(gold_rows), normalize_rows(predicted_rows)
    if gold and predicted and len(gold[0]) != len(predicted[0]):
        return Comparison(False, "column_count")
    if len(gold) != len(predicted):
        return Comparison(False, "row_count")
    same = gold == predicted if ordered else Counter(gold) == Counter(predicted)
    return Comparison(same, None if same else "values")