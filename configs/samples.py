from __future__ import annotations

from copy import deepcopy

from utils.tagging import infer_sample_from_body, split_channel_tag


def _spec(path: str, tree: str) -> dict:
    return {"path": path, "tree": tree}


def _draw_from_apply_spec(tree: str) -> dict:
    # Draw consumes apply output DATA_with_score.root in selected/<tag>/.
    return _spec("__APPLY_OUTPUT__/DATA_with_score.root", tree)


def to_root_spec(entry: dict) -> str:
    return f"{entry['path']}:{entry['tree']}"


def split_root_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid ROOT spec (expected <path>:<tree>): {spec}")
    path, tree = spec.rsplit(":", 1)
    return path, tree


def _pbpb_channel_cfg(
    signal23,
    train_background23,
    apply_background23,
    signal24,
    train_background24,
    apply_background24,
    draw_tree,
    selection_profiles,
    fiducial_profiles,
    mass_windows,
    draw_plot,
):
    return {
        "default_dataset_year": "2024",
        "default_selection_profile": "pb24v1",
        "default_fid_profile": "fid",
        "datasets": {
            "2023": {
                "train": {
                    "signal": signal23,
                    "background": train_background23,
                },
                "apply": {
                    "mc": [signal23],
                    "data": [apply_background23],
                },
                "draw": {
                    "data": _draw_from_apply_spec(draw_tree),
                },
            },
            "2024": {
                "train": {
                    "signal": signal24,
                    "background": train_background24,
                },
                "apply": {
                    "mc": [signal24],
                    "data": [apply_background24],
                },
                "draw": {
                    "data": _draw_from_apply_spec(draw_tree),
                },
            },
        },
        "selection_profiles": selection_profiles,
        "fiducial_profiles": fiducial_profiles,
        "mass_windows": mass_windows,
        "draw_plot": draw_plot,
    }


def _pp_channel_cfg(datasets_by_year, draw_tree, selection_profiles, fiducial_profiles, mass_windows, draw_plot):
    cfg = deepcopy(datasets_by_year)
    for year_cfg in cfg.values():
        year_cfg["draw"] = {"data": _draw_from_apply_spec(draw_tree)}
    return {
        "default_dataset_year": "2024",
        "default_selection_profile": "pp24v2",
        "default_fid_profile": "fid",
        "datasets": cfg,
        "selection_profiles": selection_profiles,
        "fiducial_profiles": fiducial_profiles,
        "mass_windows": mass_windows,
        "draw_plot": draw_plot,
    }


SAMPLES = {
    "pbpb": {
        "channels": {
            "X": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_X3872.root", "ntmix_X3872"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root", "ntmix"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root", "ntmix"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_X3872.root", "ntmix_X3872"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA_SMALL.root", "ntmix"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root", "ntmix"),
                "ntmix",
                {
                    "pb24v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.6 and 15 < Bpt < 50",
                        "train_cut": {"by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": None, "bqvalue_max": None},
                    },
                    "pb24v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "train_cut": {"by_max": 1.2, "bpt_min": 10.0, "bpt_max": None, "centbin_min": 20.0, "bqvalue_max": None},
                    },
                },
                {
                    "fid": {"bqvalue_max": 0.13, "by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": 0.0, "centbin_max": 90.0},
                    "fid3": {"bqvalue_max": 0.2, "by_max": 1.2, "bpt_min": 10.0, "bpt_max": 50.0, "centbin_min": 20.0, "centbin_max": None},
                },
                {"signal": None, "sidebands": [(3.75, 3.83), (3.91, 4.00)]},
                {"mass_range": [3.62, 4.0], "bin_width": 0.01, "reference_masses": [3.686, 3.872]},
            ),
            "Bu": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKp_PbPb24_MC.root", "ntKp"),#no 2023 pbpb MC for Bu, using 2024 instead
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntKp_PbPb23_DATA.root", "ntKp"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntKp_PbPb23_DATA.root", "ntKp"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKp_PbPb24_MC.root", "ntKp"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKp_PbPb24_DATA.root", "ntKp"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKp_PbPb24_DATA.root", "ntKp"),
                "ntKp",
                {
                    "pb24v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",#no By limit, pT>10 or 5
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                        "train_cut": {"by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": None, "bqvalue_max": None},
                    },
                    "pb24v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "train_cut": {"by_max": 1.2, "bpt_min": 10.0, "bpt_max": None, "centbin_min": 20.0, "bqvalue_max": None},
                    },
                },
                {
                    "fid": {"bqvalue_max": None, "by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0},
                    "fid3": {"bqvalue_max": None, "by_max": 1.2, "bpt_min": 10.0, "bpt_max": 50.0, "centbin_min": 20.0, "centbin_max": None},
                },
                {"signal": None, "sidebands": [(5.0, 5.2), (5.36, 5.56)]},
                {"mass_range": [5.0, 5.6], "bin_width": 0.01, "reference_masses": []},
            ),
            "Bd": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKstar_PbPb24_MC.root", "ntKstar"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntKstar_PbPb23_DATA.root", "ntKstar"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntKstar_PbPb23_DATA.root", "ntKstar"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKstar_PbPb24_MC.root", "ntKstar"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKstar_PbPb24_DATA.root", "ntKstar"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntKstar_PbPb24_DATA.root", "ntKstar"),
                "ntKstar",
                {
                    "pb24v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                        "train_cut": {"by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": None, "bqvalue_max": None},
                    },
                    "pb24v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "train_cut": {"by_max": 1.2, "bpt_min": 10.0, "bpt_max": None, "centbin_min": 20.0, "bqvalue_max": None},
                    },
                },
                {
                    "fid": {"bqvalue_max": None, "by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0},
                    "fid3": {"bqvalue_max": None, "by_max": 1.2, "bpt_min": 10.0, "bpt_max": 50.0, "centbin_min": 20.0, "centbin_max": None},
                },
                {"signal": None, "sidebands": [(5.0, 5.2), (5.36, 5.56)]},
                {"mass_range": [5.0, 5.6], "bin_width": 0.01, "reference_masses": []},
            ),
            "Bs": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntphi_PbPb24_MC.root", "ntphi"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntphi_PbPb23_DATA.root", "ntphi"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb23/flat_ntphi_PbPb23_DATA.root", "ntphi"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntphi_PbPb24_MC.root", "ntphi"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntphi_PbPb24_DATA.root", "ntphi"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/Bmesons/PbPb24/flat_ntphi_PbPb24_DATA.root", "ntphi"),
                "ntphi",
                {
                    "pb24v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.6 and 15 < Bpt < 50",
                        "train_cut": {"by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": None, "bqvalue_max": None},
                    },
                    "pb24v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "train_cut": {"by_max": 1.2, "bpt_min": 10.0, "bpt_max": None, "centbin_min": 20.0, "bqvalue_max": None},
                    },
                },
                {
                    "fid": {"bqvalue_max": None, "by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0},
                    "fid3": {"bqvalue_max": None, "by_max": 1.2, "bpt_min": 10.0, "bpt_max": 50.0, "centbin_min": 20.0, "centbin_max": None},
                },
                {"signal": None, "sidebands": [(5.1, 5.29), (5.45, 5.64)]},
                {"mass_range": [5.0, 5.7], "bin_width": 0.01, "reference_masses": []},
            ),
        }
    },
    "pp": {
        "channels": {
            "X": _pp_channel_cfg(
                {
                    "2024": {
                        "train": {
                            "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root", "ntmix_X3872"),
                            "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                        },
                        "apply": {
                            "mc": [
                                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S_nonPrompt.root", "ntmix_PSI2S"),
                                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S.root", "ntmix_PSI2S"),
                                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872_nonPrompt.root", "ntmix_X3872"),
                                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root", "ntmix_X3872"),
                            ],
                            "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix")],
                        },
                    },
                },
                "ntmix",
                {
                    "pp24v1": {
                        "signal_selection": None,
                        "background_selection": "((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                        "train_cut": {"by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "bqvalue_max": None},
                    },
                    "pp24v2": {
                        "signal_selection": None,
                        "background_selection": "((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                        "train_cut": {"by_max": 2.4, "bpt_min": 5.0, "bpt_max": 50.0, "centbin_min": None, "bqvalue_max": None},
                    }
                },
                {
                    "fid": {"bqvalue_max": 0.2, "by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "centbin_max": None},
                    "fid2": {"bqvalue_max": 0.2, "by_max": 2.4, "bpt_min": 5.0, "bpt_max": 50.0, "centbin_min": None, "centbin_max": None},
                },
                {"signal": None, "sidebands": [(3.75, 3.80), (3.95, 4.00)]},
                {"mass_range": [3.62, 4.0], "bin_width": 0.01, "reference_masses": [3.686, 3.872]},
            ),
            "Bu": _pp_channel_cfg(
                {
                    "2024": {
                        "train": {
                            "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BU.root", "ntmix_BU"),
                            "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                        },
                        "apply": {
                            "mc": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BU.root", "ntmix_BU")],
                            "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix")],
                        },
                    },
                },
                "ntmix_BU",
                {
                    "pp24v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                        "train_cut": {"by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "bqvalue_max": None},
                    }
                },
                {"fid": {"bqvalue_max": None, "by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "centbin_max": None}},
                {"signal": None, "sidebands": [(3.75, 3.80), (3.95, 4.00)]},
                {"mass_range": [4.9, 5.7], "bin_width": 0.01, "reference_masses": []},
            ),
            "Bd": _pp_channel_cfg(
                {
                    "2024": {
                        "train": {
                            "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BD.root", "ntmix_BD"),
                            "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                        },
                        "apply": {
                            "mc": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BD.root", "ntmix_BD")],
                            "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix")],
                        },
                    },
                },
                "ntmix_BD",
                {
                    "pp24v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                        "train_cut": {"by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "bqvalue_max": None},
                    }
                },
                {"fid": {"bqvalue_max": None, "by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "centbin_max": None}},
                {"signal": None, "sidebands": [(3.75, 3.80), (3.95, 4.00)]},
                {"mass_range": [5.0, 5.7], "bin_width": 0.01, "reference_masses": []},
            ),
            "Bs": _pp_channel_cfg(
                {
                    "2024": {
                        "train": {
                            "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BS.root", "ntmix_BS"),
                            "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                        },
                        "apply": {
                            "mc": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BS.root", "ntmix_BS")],
                            "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix")],
                        },
                    },
                },
                "ntmix_BS",
                {
                    "pp24v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                        "train_cut": {"by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "bqvalue_max": None},
                    }
                },
                {"fid": {"bqvalue_max": None, "by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "centbin_max": None}},
                {"signal": None, "sidebands": [(3.75, 3.80), (3.95, 4.00)]},
                {"mass_range": [5.0, 5.7], "bin_width": 0.01, "reference_masses": []},
            ),
        }
    },
}


def _channel_cfg(sample: str, channel: str) -> dict:
    cfg = SAMPLES.get(sample, {})
    channels = cfg.get("channels", {})
    if channel not in channels:
        raise ValueError(f"Unsupported channel '{channel}' for sample '{sample}'.")
    return channels[channel]


def infer_channel_from_tag(tag: str) -> str:
    channel, _ = split_channel_tag(tag)
    return channel


def infer_sample_from_tag(tag: str) -> str:
    _, body = split_channel_tag(tag)
    return infer_sample_from_body(body)


def infer_dataset_year(tag: str, sample: str) -> str:
    _, body = split_channel_tag(tag)
    if body.startswith("pb23"):
        return "2023"
    if body.startswith("pb24") or body.startswith("pp24"):
        return "2024"
    channel = infer_channel_from_tag(tag)
    return _channel_cfg(sample, channel)["default_dataset_year"]


def infer_selection_profile(tag: str, sample: str) -> str:
    _, body = split_channel_tag(tag)
    channel = infer_channel_from_tag(tag)
    cfg = _channel_cfg(sample, channel)
    if sample == "pp" and body.startswith("pp24v1_"):
        return "pp24v1"
    if sample == "pp" and body.startswith("pp24v2_"):
        return "pp24v2"
    if sample == "pbpb" and body.startswith("pb24v1_"):
        return "pb24v1"
    if sample == "pbpb" and body.startswith("pb24v2_"):
        return "pb24v2"
    return cfg["default_selection_profile"]


def infer_fid_profile(tag: str, sample: str) -> str:
    _, body = split_channel_tag(tag)
    channel = infer_channel_from_tag(tag)
    cfg = _channel_cfg(sample, channel)
    if sample == "pp" and body.startswith("pp24v2_"):
        return "fid2"
    if sample == "pbpb" and body.startswith("pb24v1_"):
        return "fid"
    if sample == "pbpb" and (body.startswith("pb23v6_") or body.startswith("pb24v2_")):
        return "fid3"
    return cfg["default_fid_profile"]


def resolve_training_config(sample: str, channel: str, dataset_year: str, selection_profile: str) -> dict:
    cfg = _channel_cfg(sample, channel)
    ds = cfg["datasets"][dataset_year]["train"]
    sel = cfg["selection_profiles"][selection_profile]
    return {
        "dataset_year": dataset_year,
        "signal": deepcopy(ds["signal"]),
        "background": deepcopy(ds["background"]),
        "selection_profile": selection_profile,
        "signal_selection": sel["signal_selection"],
        "background_selection": sel["background_selection"],
        "train_cut": deepcopy(sel["train_cut"]),
        "mass_windows": deepcopy(cfg["mass_windows"]),
        "dataset_source": f"{sample}_{channel}_{dataset_year}",
        "channel": channel,
    }


def resolve_apply_config(sample: str, channel: str, dataset_year: str) -> dict:
    cfg = _channel_cfg(sample, channel)
    ds = cfg["datasets"][dataset_year]["apply"]
    return {
        "dataset_year": dataset_year,
        "mc": deepcopy(ds["mc"]),
        "data": deepcopy(ds["data"]),
        "dataset_source": f"{sample}_{channel}_{dataset_year}",
        "channel": channel,
    }


def resolve_draw_config(sample: str, channel: str, dataset_year: str) -> dict:
    cfg = _channel_cfg(sample, channel)
    ds = cfg["datasets"][dataset_year]["draw"]
    return {
        "dataset_year": dataset_year,
        "data": deepcopy(ds["data"]),
        "plot": deepcopy(cfg["draw_plot"]),
        "dataset_source": f"{sample}_{channel}_{dataset_year}",
        "channel": channel,
    }


def resolve_fiducial_config(sample: str, channel: str, fid_profile: str) -> dict:
    cfg = _channel_cfg(sample, channel)
    return deepcopy(cfg["fiducial_profiles"][fid_profile])
