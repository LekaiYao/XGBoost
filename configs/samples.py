from __future__ import annotations

from copy import deepcopy
import re

from utils.tagging import infer_sample_from_body, split_channel_tag

SINGLE_DAG_BODY_RE = re.compile(r"^(pp24|pb23|pb24)_v(\d+)_fid(\d+)_([0-9]+v[0-9]*)_xgb_v(\d+)$")


def _parse_single_dag_body_or_raise(tag: str):
    _, body = split_channel_tag(tag)
    m = SINGLE_DAG_BODY_RE.fullmatch(body)
    if not m:
        raise ValueError(
            f"Invalid single DAG tag '{tag}'. "
            "Expected format: {channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}, "
            "for example: X_pb24_v2_fid1_8v_xgb_v1"
        )
    dataset_token, selection_idx, fid_idx, varset_token, model_version = m.groups()
    return body, dataset_token, int(selection_idx), int(fid_idx), varset_token, int(model_version)


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
    draw_plot,
):
    return {
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
        "draw_plot": draw_plot,
    }


def _pp_channel_cfg(datasets_by_year, draw_tree, selection_profiles, fiducial_profiles, draw_plot):
    cfg = deepcopy(datasets_by_year)
    for year_cfg in cfg.values():
        year_cfg["draw"] = {"data": _draw_from_apply_spec(draw_tree)}
    return {
        "datasets": cfg,
        "selection_profiles": selection_profiles,
        "fiducial_profiles": fiducial_profiles,
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
                    "pb24_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb24_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                    "pb23_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb23_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                },
                {
                    "pb24_fid1": "BQvalue < 0.13 and abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0 and CentBin > 0.0 and CentBin < 90.0",
                    "pb24_fid2": "BQvalue < 0.2 and abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                    "pb23_fid1": "BQvalue < 0.13 and abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0 and CentBin > 0.0 and CentBin < 90.0",
                    "pb23_fid2": "BQvalue < 0.2 and abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                },
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
                    "pb24_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",#no By limit, pT>10 or 5
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb24_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                    "pb23_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb23_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                },
                {
                    "pb24_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb24_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                    "pb23_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb23_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                },
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
                    "pb24_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb24_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                    "pb23_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb23_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                },
                {
                    "pb24_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb24_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                    "pb23_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb23_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                },
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
                    "pb24_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb24_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                    "pb23_v1": {
                        "signal_selection": "abs(By) < 1.6 and 15 < Bpt < 50",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.6 and 15 < Bpt < 50",
                    },
                    "pb23_v2": {
                        "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                        "background_selection": "((5.1 < Bmass < 5.29) or (5.45 < Bmass < 5.64)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                    },
                },
                {
                    "pb24_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb24_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                    "pb23_fid1": "abs(By) < 1.6 and Bpt > 15.0 and Bpt < 50.0",
                    "pb23_fid2": "abs(By) < 1.2 and Bpt > 10.0 and Bpt < 50.0 and CentBin > 20.0",
                },
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
                    "pp24_v1": {
                        "signal_selection": None,
                        "background_selection": "((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                    },
                    "pp24_v2": {
                        "signal_selection": "BQvalue < 0.2",
                        "background_selection": "((BQvalue < 0.2) and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80)))",
                    }
                },
                {
                    "pp24_fid1": "BQvalue < 0.2",
                    "pp24_fid2": "BQvalue < 0.2 and abs(By) < 2.4 and Bpt > 5.0 and Bpt < 50.0",
                },
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
                    "pp24_v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                    }
                },
                {"pp24_fid1": None},
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
                    "pp24_v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                    }
                },
                {"pp24_fid1": None},
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
                    "pp24_v2": {
                        "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                        "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                    }
                },
                {"pp24_fid1": None},
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
    _, dataset_token, _, _, _, _ = _parse_single_dag_body_or_raise(tag)
    if dataset_token.startswith("pb23"):
        return "2023"
    if dataset_token.startswith("pb24") or dataset_token.startswith("pp24"):
        return "2024"
    raise ValueError(f"Unsupported dataset token '{dataset_token}' in tag '{tag}'.")


def infer_selection_profile(tag: str, sample: str) -> str:
    channel = infer_channel_from_tag(tag)
    cfg = _channel_cfg(sample, channel)
    _, dataset_token, selection_idx, _, _, _ = _parse_single_dag_body_or_raise(tag)
    profile = f"{dataset_token}_v{selection_idx}"
    if profile in cfg["selection_profiles"]:
        return profile
    raise ValueError(
        f"Selection profile '{profile}' not configured for sample='{sample}', channel='{channel}'."
    )


def infer_fid_profile(tag: str, sample: str) -> str:
    channel = infer_channel_from_tag(tag)
    cfg = _channel_cfg(sample, channel)
    _, dataset_token, _, fid_idx, _, _ = _parse_single_dag_body_or_raise(tag)
    profile = f"{dataset_token}_fid{fid_idx}"
    if profile in cfg["fiducial_profiles"]:
        return profile
    raise ValueError(
        f"Fiducial profile '{profile}' not configured for sample='{sample}', channel='{channel}'."
    )


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
    raw = deepcopy(cfg["fiducial_profiles"][fid_profile])
    if isinstance(raw, str):
        return {"expression": raw}
    if isinstance(raw, dict) and "expression" in raw:
        return {"expression": raw["expression"]}
    raise ValueError(f"Unsupported fiducial profile format: {type(raw)}")
