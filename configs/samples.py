from __future__ import annotations

from copy import deepcopy


def _spec(path: str, tree: str) -> dict:
    return {"path": path, "tree": tree}


def to_root_spec(entry: dict) -> str:
    return f"{entry['path']}:{entry['tree']}"


def split_root_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid ROOT spec (expected <path>:<tree>): {spec}")
    path, tree = spec.rsplit(":", 1)
    return path, tree


SAMPLES = {
    "pbpb": {
        "default_dataset_year": "2024",
        "default_selection_profile": "pb24v2",
        "default_fid_profile": "fid3",
        "datasets": {
            "2023": {
                "train": {
                    "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_X3872.root", "ntmix_X3872"),
                    "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root", "ntmix"),
                },
                "apply": {
                    "mc": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC_X3872.root", "ntmix_X3872")],
                    "data": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root", "ntmix")],
                },
                "draw": {
                    "data": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root", "ntmix"),
                },
            },
            "2024": {
                "train": {
                    "signal": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_X3872.root", "ntmix_X3872"),
                    "background": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA_SMALL.root", "ntmix"),
                },
                "apply": {
                    "mc": [_spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC_X3872.root", "ntmix_X3872")],
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
                "train_cut": {
                    "by_max": 1.6,
                    "bpt_min": 15.0,
                    "bpt_max": 50.0,
                    "centbin_min": None,
                    "bqvalue_max": None,
                },
            },
            "pb24v2": {
                "signal_selection": "abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                "background_selection": "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.2 and Bpt > 10 and CentBin > 20",
                "train_cut": {
                    "by_max": 1.2,
                    "bpt_min": 10.0,
                    "bpt_max": None,
                    "centbin_min": 20.0,
                    "bqvalue_max": None,
                },
            },
        },
        "fiducial_profiles": {
            "fid": {
                "bqvalue_max": 0.13,
                "by_max": 1.6,
                "bpt_min": 15.0,
                "bpt_max": 50.0,
                "centbin_min": 0.0,
                "centbin_max": 90.0,
            },
            "fid3": {
                "bqvalue_max": 0.2,
                "by_max": 1.2,
                "bpt_min": 10.0,
                "bpt_max": 50.0,
                "centbin_min": 20.0,
                "centbin_max": None,
            },
        },
    },
    "pp": {
        "default_dataset_year": "2024",
        "default_selection_profile": "pp24v2",
        "default_fid_profile": "fid",
        "datasets": {
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
                "draw": {
                    "data": _spec("/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root", "ntmix"),
                },
            }
        },
        "selection_profiles": {
            "pp24v2": {
                "signal_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5",
                "background_selection": "Bchi2Prob>0.02 and Btrk1dR<0.5 and ((Bmass > 3.95 and Bmass < 4.0) or (Bmass > 3.75 and Bmass < 3.80))",
                "train_cut": {
                    "by_max": None,
                    "bpt_min": None,
                    "bpt_max": None,
                    "centbin_min": None,
                    "bqvalue_max": None,
                },
            }
        },
        "fiducial_profiles": {
            "fid": {
                "bqvalue_max": None,
                "by_max": None,
                "bpt_min": None,
                "bpt_max": None,
                "centbin_min": None,
                "centbin_max": None,
            }
        },
    },
}


def infer_sample_from_tag(train_tag: str) -> str:
    if train_tag.startswith("pp"):
        return "pp"
    return "pbpb"


def infer_dataset_year(train_tag: str, sample: str) -> str:
    if train_tag.startswith("pb23"):
        return "2023"
    if train_tag.startswith("pb24") or train_tag.startswith("pp24"):
        return "2024"
    return SAMPLES[sample]["default_dataset_year"]


def infer_selection_profile(train_tag: str, sample: str) -> str:
    if sample == "pbpb" and train_tag.startswith("pb24v2_"):
        return "pb24v2"
    return SAMPLES[sample]["default_selection_profile"]


def infer_fid_profile(train_tag: str, sample: str) -> str:
    if sample == "pbpb" and (train_tag.startswith("pb23v6_") or train_tag.startswith("pb24v2_")):
        return "fid3"
    return SAMPLES[sample]["default_fid_profile"]


def resolve_training_config(sample: str, dataset_year: str, selection_profile: str) -> dict:
    cfg = SAMPLES[sample]
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
        "dataset_source": f"{sample}_{dataset_year}",
    }


def resolve_apply_config(sample: str, dataset_year: str) -> dict:
    cfg = SAMPLES[sample]
    ds = cfg["datasets"][dataset_year]["apply"]
    return {
        "dataset_year": dataset_year,
        "mc": deepcopy(ds["mc"]),
        "data": deepcopy(ds["data"]),
        "dataset_source": f"{sample}_{dataset_year}",
    }


def resolve_fiducial_config(sample: str, fid_profile: str) -> dict:
    return deepcopy(SAMPLES[sample]["fiducial_profiles"][fid_profile])
