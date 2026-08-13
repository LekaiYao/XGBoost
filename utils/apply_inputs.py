def resolve_apply_mc_input(default_mc_input, model_configs):
    if len(model_configs) != 1:
        return default_mc_input
    config = model_configs[0]
    signal_input_override = config.get("signal_input_override")
    if signal_input_override is None:
        # Backward compatibility for weighted models trained before the
        # explicit override flag was added.
        signal_input_override = bool(config.get("signal_weight_branch"))
    if not signal_input_override:
        return default_mc_input
    signal_path = config.get("signal_path")
    if not signal_path:
        raise ValueError(
            "Model with signal_input_override=true is missing signal_path"
        )
    return signal_path
