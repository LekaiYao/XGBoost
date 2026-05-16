import re

CHANNELS = ("X", "Bu", "Bd", "Bs")
_CHANNEL_SET = set(CHANNELS)


def split_channel_tag(tag: str):
    if not isinstance(tag, str) or not tag:
        raise ValueError("Empty tag is not allowed")
    parts = tag.split("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid tag '{tag}'. Expected '<channel>_<body>' where channel in {CHANNELS}."
        )
    channel, body = parts
    if channel not in _CHANNEL_SET:
        raise ValueError(
            f"Invalid channel prefix in tag '{tag}'. Expected one of {CHANNELS}."
        )
    if not body:
        raise ValueError(
            f"Invalid tag '{tag}'. Missing body after channel prefix."
        )
    return channel, body


def infer_sample_from_body(body_tag: str):
    if body_tag.startswith("pp"):
        return "pp"
    return "pbpb"


def parse_optuna_trials_from_group_body(body_tag: str):
    matches = re.findall(r"(?:^|_)(?:\d+o|o)(\d+)(?:_|$)", body_tag)
    if not matches:
        raise ValueError(
            f"Cannot infer optuna_n_trials from group tag body '{body_tag}'. "
            "Expected token like '_o200' or '_4o200'."
        )
    return int(matches[-1])
