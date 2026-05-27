import re


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
