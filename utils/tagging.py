import re

CHANNELS = ("X", "Psi2S", "Bu", "Bd", "Bs")
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
    _, objective_idx, n_trials, _ = parse_optuna_spec_from_group_body(body_tag)
    _ = objective_idx
    return n_trials


def parse_optuna_spec_from_group_body(body_tag: str):
    """
    Parse Optuna token in group tag body, supporting two modes.

    Explicit mode suffix:
      ..._{objective_idx}o{n_trials}_v{space_version}
    Example:
      pb24_v2_fid1_8v1_1o200_v1

    Legacy batch mode suffix:
      ..._{objective_idx}o{n_trials}
    Example:
      pb24_v2_fid1_8v1_1o200
    """
    m_explicit = re.search(r"(?:^|_)(\d+)o(\d+)_v(\d+)$", body_tag)
    if m_explicit:
        objective_idx, n_trials, space_version = m_explicit.groups()
        return "explicit", int(objective_idx), int(n_trials), f"v{int(space_version)}"

    m_legacy = re.search(r"(?:^|_)(\d+)o(\d+)$", body_tag)
    if m_legacy:
        objective_idx, n_trials = m_legacy.groups()
        return "legacy", int(objective_idx), int(n_trials), None

    raise ValueError(
        f"Cannot infer Optuna spec from group tag body '{body_tag}'. "
        "Expected suffix '_{n}o{N}_v{k}' (explicit mode) or '_{n}o{N}' (legacy batch mode)."
    )


def parse_optuna_spec_from_train_tag(tag: str):
    """
    Parse Optuna spec from a DAGMan train_tag.

    train_tag is expected as:
      <group_tag>_v<run_version>
    where group_tag ends with strict optuna suffix:
      ..._{n}o{N}_v{k}
    """
    _, body = split_channel_tag(tag)
    m = re.search(r"(?:^|_)(\d+)o(\d+)_v(\d+)(?:_v\d+)?$", body)
    if not m:
        raise ValueError(
            f"Cannot infer Optuna spec from train tag '{tag}'. "
            "Expected ..._{n}o{N}_v{k}_v<runVersion>."
        )
    objective_idx, n_trials, space_version = m.groups()
    return int(objective_idx), int(n_trials), f"v{int(space_version)}"
