import re


_SELECTION_TOKENS = {
    "abs",
    "and",
    "False",
    "not",
    "or",
    "True",
}


def selection_columns(expr: str):
    if expr is None:
        return []
    tokens = re.findall(r"\b[A-Za-z_]\w*\b", str(expr))
    return list(dict.fromkeys(token for token in tokens if token not in _SELECTION_TOKENS))


def normalize_selection_expr(expr: str) -> str:
    normalized = expr.replace("&&", " and ").replace("||", " or ")
    normalized = re.sub(r"(?<![=!<>])!(?!=)", " not ", normalized)
    return " ".join(normalized.split())


def apply_selection(df, expr: str, label: str):
    if expr is None:
        return df
    expr = str(expr).strip()
    if not expr:
        return df
    normalized = normalize_selection_expr(expr)
    try:
        return df.query(normalized, engine="python").copy()
    except Exception as exc:
        raise ValueError(f"Invalid {label} expression: {expr}") from exc
