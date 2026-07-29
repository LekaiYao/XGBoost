def resolve_apply_mc_input(default_mc_input, model_configs):
    if len(model_configs) != 1:
        return default_mc_input
    config = model_configs[0]
    weight_branch = config.get("signal_weight_branch")
    if not weight_branch:
        return default_mc_input
    signal_path = config.get("signal_path")
    if not signal_path:
        raise ValueError(
            f"Weighted model with branch '{weight_branch}' is missing signal_path"
        )
    return signal_path
