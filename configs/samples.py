from __future__ import annotations

from copy import deepcopy

from utils.tagging import infer_sample_from_body, split_channel_tag


def _spec(path: str, tree: str) -> dict:
    return {"path": path, "tree": tree}


def to_root_spec(entry: dict) -> str:
    return f"{entry['path']}:{entry['tree']}"


def split_root_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid ROOT spec (expected <path>:<tree>): {spec}")
    path, tree = spec.rsplit(":", 1)
    return path, tree


def _pbpb_channel_cfg(signal23, signal24):
    return {
        "default_dataset_year": "2024",
        "default_selection_profile": "pb24v2",
        "default_fid_profile": "fid3",
        "datasets": {
            "2023": {
                "train": {
                    "signal": signal23,
                    "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root", "ntmix"),
                },
                "apply": {
                    "mc": [signal23],
                    "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root", "ntmix")],
                },
                "draw": {
                    "data": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root", "ntmix"),
                },
            },
            "2024": {
                "train": {
                    "signal": signal24,
                    "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA_SMALL.root", "ntmix"),
                },
                "apply": {
                    "mc": [signal24],
                    "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root", "ntmix")],
                },
                "draw": {
                    "data": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root", "ntmix"),
                },
            },
        },
        "selection_profiles": {
            "legacy": {
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
        "fiducial_profiles": {
            "fid": {"bqvalue_max": 0.13, "by_max": 1.6, "bpt_min": 15.0, "bpt_max": 50.0, "centbin_min": 0.0, "centbin_max": 90.0},
            "fid3": {"bqvalue_max": 0.2, "by_max": 1.2, "bpt_min": 10.0, "bpt_max": 50.0, "centbin_min": 20.0, "centbin_max": None},
        },
        "mass_windows": {
            "signal": None,
            "sidebands": [(3.75, 3.83), (3.91, 4.00)],
        },
    }


def _pp_channel_cfg(signal_train, signal_apply_list):
    return {
        "default_dataset_year": "2024",
        "default_selection_profile": "pp24v2",
        "default_fid_profile": "fid",
        "datasets": {
            "2024": {
                "train": {
                    "signal": signal_train,
                    "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                },
                "apply": {
                    "mc": signal_apply_list,
                    "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix")],
                },
                "draw": {
                    "data": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                },
            }
        },
        "selection_profiles": {
            "pp24v2": {
                "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                "train_cut": {"by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "bqvalue_max": None},
            }
        },
        "fiducial_profiles": {
            "fid": {"bqvalue_max": None, "by_max": None, "bpt_min": None, "bpt_max": None, "centbin_min": None, "centbin_max": None}
        },
        "mass_windows": {
            "signal": None,
            "sidebands": [(3.75, 3.80), (3.95, 4.00)],
        },
    }


SAMPLES = {
    "pbpb": {
        "channels": {
            "X": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_X3872.root", "ntmix_X3872"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_X3872.root", "ntmix_X3872"),
            ),
            "Bu": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_BU.root", "ntmix_BU"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_BU.root", "ntmix_BU"),
            ),
            "Bd": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_BD.root", "ntmix_BD"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_BD.root", "ntmix_BD"),
            ),
            "Bs": _pbpb_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_BS.root", "ntmix_BS"),
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_BS.root", "ntmix_BS"),
            ),
        }
    },
    "pp": {
        "channels": {
            "X": _pp_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root", "ntmix_X3872"),
                [
                    _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S_nonPrompt.root", "ntmix_PSI2S"),
                    _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S.root", "ntmix_PSI2S"),
                    _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872_nonPrompt.root", "ntmix_X3872"),
                    _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root", "ntmix_X3872"),
                ],
            ),
            "Bu": _pp_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BU.root", "ntmix_BU"),
                [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BU.root", "ntmix_BU")],
            ),
            "Bd": _pp_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BD.root", "ntmix_BD"),
                [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BD.root", "ntmix_BD")],
            ),
            "Bs": _pp_channel_cfg(
                _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BS.root", "ntmix_BS"),
                [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_BS.root", "ntmix_BS")],
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
    if sample == "pbpb" and body.startswith("pb24v2_"):
        return "pb24v2"
    return cfg["default_selection_profile"]


def infer_fid_profile(tag: str, sample: str) -> str:
    _, body = split_channel_tag(tag)
    channel = infer_channel_from_tag(tag)
    cfg = _channel_cfg(sample, channel)
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


def resolve_fiducial_config(sample: str, channel: str, fid_profile: str) -> dict:
    cfg = _channel_cfg(sample, channel)
    return deepcopy(cfg["fiducial_profiles"][fid_profile])
