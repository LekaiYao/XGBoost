import re

from utils.tagging import split_channel_tag

PBPB_VARSETS_X = {
    "5v6": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk2Pt", "BtrkPtimb"],
    "10v9": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Btrk2dR", "Bujeta", "Btktketa"],
    "11v5": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btrk1Eta", "Btrk2Eta", "Btktketa"],
    "12v4": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btrk1Eta", "Btrk2Eta", "Btktketa", "Btktkpt"],
    "12v5": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btrk1Eta", "Btrk2Eta", "Btktketa", "Bcos_dtheta"],
    "12v6": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btrk1Eta", "Btrk2Eta", "Btktketa", "Bchi2Prob"],
    "8v10": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta"],
    "8v11": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Btktketa"],
    "8v12": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Btrk1Eta"],
    "8v13": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Btrk2Eta"],
    "10v8": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btrk1Eta", "Btktketa"],
    "9v9": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Bujeta", "Btktketa"],
    "12v3": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Bujeta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR", "Btrk1Eta", "Btktketa", "Btktkpt", "Bcos_dtheta"],
    "7v9": ["Bmu1pt", "Bmu1eta", "Bmu2pt", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1dR"],
    "8v8": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR"],
    "8v9": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1Pt"],
    "9v8": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btktkmass"],
    "7v7": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "BtrkPtimb"],
    "7v8": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt"],
    "4v3": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "BtrkPtimb"],
    "5v4": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "BtrkPtimb", "Bchi2Prob"],
    "5v5": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "BtrkPtimb", "Balpha"],
    "9v7": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Btrk1Pt"],
    "8v7": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "BtrkPtimb"],
    "9v5": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Bchi2Prob"],
    "9v6": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "BtrkPtimb", "Balpha"],
    "10v7": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Btrk2dR"],
    "11v4": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Btrk2dR", "Bchi2Prob"],
    "5v1": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR"],
    "6v1": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb"],
    "7v6": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bmu1pt", "Bmu2pt", "Btrk2Pt", "Btrk1dR"],
    "10v3": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Btktkpt"],
    "10v4": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Bchi2Prob"],
    "10v5": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Bcos_dtheta"],
    "10v6": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Balpha"],
    "13v2": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "BtrkPtimb", "Btktkpt", "Bchi2Prob", "Bcos_dtheta"],
    "4v1": ["Btrk1Pt", "Btrk2Pt", "Btrk1dR", "BtrkPtimb"],
    "9v4": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk2Pt", "Btrk1dR", "BtrkPtimb"],
    "11v2": ["Balpha", "Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "BtrkPtimb"],
    "10v2": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "BtrkPtimb"],
    "12v2": ["Balpha", "Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb"],
    "11v3": ["Bmu1pt", "Bmu1phi", "Bmu1eta", "Bmu2pt", "Bmu2phi", "Bmu2eta", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb"],
    "8v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt"],
    "9v2": ["Btrk1dR", "Btrk2dR", "Btktkpt", "Btrk1Pt", "Btrk2Pt", "Bchi2Prob", "Bcos_dtheta", "BtrkPtimb", "BtktkvProb"],
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
    "5v3": ["Btrk1dR", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
    "6v4": ["Btrk2dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "6v5": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
    "7v1": ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Bmu2y", "Bmu1y", "Btrk1Phi","Btrk2Phi"],
    "7v2": ["Btrk1dR", "Btrk2dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "BtrkPtimb"],
    "7v3": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y"],
    "7v4": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu2pt"],
    "7v5": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu2y"],
    "8v1": ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "Bmu2y", "Bmu1y", "Btrk1Phi","Btrk2Phi"],
    "8v3": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu2pt"],
    "8v4": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu2y"],
    "8v5": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu1pt"],
    "8v6": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu2y", "Bmu2pt"],
    "9v1": ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "Bmu2y", "Bmu1y", "Bmu1pt", "Btrk1Phi","Btrk2Phi"],
    "9v3": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu2y", "Bmu1pt"],
    "10v1": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu2y", "Bmu1pt", "Bmu2pt"],
    "11v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt", "BujvProb", "Btrk2Eta", "Btrk2Phi", "Bmu1y"],
    "12v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb"],
    "13v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt"],
    "14v1": ["BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "14v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz"],
    "16v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz", "Btrk2Eta", "Btrk1Eta"],
    "17v1": ["PVz", "BujvProb", "Btrk2PtErr", "Btrk1PtErr", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "BtktkvProb"],
    "17v2": ["PVz", "BujvProb", "Btrk2PtErr", "Btrk1PtErr", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "18v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz", "Btrk2Eta", "Btrk1Eta", "Btrk1PtErr", "Btrk2PtErr"],
}

PBPB_VARSETS_PSI2S = {
    "5v1": ["Btrk1dR", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
    "6v1": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
    "7v1": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y"],
    "7v2": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu2pt"],
    "8v1": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt", "Bmu1y", "Bmu2pt"],
}

PBPB_VARSETS_BU = {
    "4v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D"],
    "5v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob"],
    "12v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D",  "Bchi2Prob", "PVz", "BujvProb", "Bmu2y", "Bmu1y", "Btrk1Eta", "Bmu1pt", "Bmu2pt"],
    "10v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btrk1Pt", "Btrk1dR", "Bchi2Prob", "Bmu2pt", "Bmu1pt", "PVz", "Bmu2y", "BujvProb"],
}

PBPB_VARSETS_BD = {
    "9v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt","BtrkPtimb", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "BtktkvProb"],
    "17v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "PVz", "BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "BtktkvProb"],
    "14v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "Bmu1pt", "Bmu2pt", "Btktkpt", "Btrk1Pt", "Btrk1dR", "Btrk2Pt", "BujvProb", "PVz", "Btrk2dR", "Bmu2y", "Btrk1Eta"],
}

PBPB_VARSETS_BS = {
    "6v1": ["Bchi2Prob", "Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "BtrkPtimb"],
    "7v1": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "Bchi2Prob"],
    "8v1": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "Bchi2Prob", "BtktkvProb"],
    "17v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "PVz", "BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "BtktkvProb"],
    "14v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "Bmu2pt", "Btktkpt", "Bmu1pt", "Btrk1dR", "Btrk2Pt", "Btrk1Pt", "Btrk2dR", "PVz", "Btrk2Eta", "Btrk1Eta", "BtktkvProb"],
}

PP_VARSETS_X = {
    "4v1": ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "5v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "5v2": ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb"],
    "7v1": ["Bmu1pt", "Bmu2pt", "Btrk1dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Btktkpt"],
    "8v1": ["Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Btktkpt"],
    "8v2": ["Bchi2Prob", "Btrk1dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Btktkpt", "BtrkPtimb", "BtktkvProb"],
    "11v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt", "BujvProb", "Btrk2Eta", "Btrk2Phi", "Bmu1y"],
    "12v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb"],
    "13v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt"],
    "14v1": ["BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "14v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz"],
    "16v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz", "Btrk2Eta", "Btrk1Eta"],
    "17v1": ["PVz", "BujvProb", "Btrk2PtErr", "Btrk1PtErr", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "BtktkvProb"],
    "17v2": ["PVz", "BujvProb", "Btrk2PtErr", "Btrk1PtErr", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk1Pt", "BtktkvProb"],
    "18v1": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Bmu2y", "Bmu1y", "Bmu1pt", "Bmu2pt", "BujvProb", "Btktkpt", "PVz", "Btrk2Eta", "Btrk1Eta", "Btrk1PtErr", "Btrk2PtErr"],
}

PP_VARSETS_PSI2S = {
    "6v1": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
}

PP_REWEIGHT_VARSETS_X = {
    "R3": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob"],
    "R4_noCos": ["Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt"],
    "R5": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt"],
    "R5v2": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk1Pt", "Btrk1dR"],
    "R5v3": ["Btrk1dR", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
    "R6": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt", "Btrk1dR"],
    "R6v2": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk1Pt", "Btrk1dR", "Btrk2dR"],
    "R8": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt"],
}

PP_REWEIGHT_VARSETS_PSI2S = {
    "R5": ["Bcos_dtheta", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt", "Btrk1dR"],
    "R6": ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt", "Btrk1dR"],
    "R8": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt"],
}

PP_VARSETS_BU = {
    "4v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D"],
    "5v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob"],
    "12v1": ["Btrk1dR", "Btrk1Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D",  "Bchi2Prob", "PVz", "BujvProb", "Bmu2y", "Bmu1y", "Btrk1Eta", "Bmu1pt", "Bmu2pt"],
    "10v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btrk1Pt", "Btrk1dR", "Bchi2Prob", "Bmu2pt", "Bmu1pt", "PVz", "Btrk1Eta", "BujvProb"],
}

PP_VARSETS_BD = {
    "9v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt","BtrkPtimb", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "BtktkvProb"],
    "17v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "PVz", "BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "BtktkvProb"],
    "14v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "Bchi2Prob", "Btktkpt", "Bmu2pt", "Bmu1pt", "Bmu1y", "Btrk1Eta", "Btrk2Eta", "BujvProb"],
}

PP_VARSETS_BS = {
    "7v1": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "Bchi2Prob"],
    "17v1": ["Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt", "Bcos_dtheta", "Bnorm_svpvDistance_2D", "Bchi2Prob", "PVz", "BujvProb", "Btktkpt", "Bmu2y", "Bmu1y", "Btrk2Eta", "Btrk1Eta", "Bmu1pt", "Bmu2pt", "BtktkvProb"],
    "14v1": ["Bcos_dtheta", "Bnorm_svpvDistance_2D", "Btktkpt", "Btrk1Pt", "Btrk2Pt", "Bchi2Prob", "Bmu2pt", "Bmu1pt", "Btrk2dR", "Btrk1dR", "PVz", "Bmu2y", "BtktkvProb", "BujvProb"],
}

VARSETS = {
    "pbpb": {
        "X": PBPB_VARSETS_X,
        "Psi2S": PBPB_VARSETS_PSI2S,
        "Bu": PBPB_VARSETS_BU,
        "Bd": PBPB_VARSETS_BD,
        "Bs": PBPB_VARSETS_BS,
    },
    "pp": {
        "X": PP_VARSETS_X,
        "Psi2S": PP_VARSETS_PSI2S,
        "Bu": PP_VARSETS_BU,
        "Bd": PP_VARSETS_BD,
        "Bs": PP_VARSETS_BS,
    },
}

DEFAULT_SAMPLE = "pbpb"
SUPPORTED_SAMPLES = tuple(VARSETS.keys())
SUPPORTED_VARSETS = tuple(sorted({k for ch in VARSETS[DEFAULT_SAMPLE].values() for k in ch.keys()}))
VARSET_COLUMNS = VARSETS[DEFAULT_SAMPLE]["X"]

REWEIGHT_VARSETS = {
    "pp": {
        "X": PP_REWEIGHT_VARSETS_X,
        "Psi2S": PP_REWEIGHT_VARSETS_PSI2S,
    },
}


def infer_sample_from_tag(tag):
    _, body = split_channel_tag(tag)
    if body.startswith("pp"):
        return "pp"
    return "pbpb"


def infer_channel_from_tag(tag):
    channel, _ = split_channel_tag(tag)
    return channel


def get_varset_columns(sample, varset, channel):
    if sample not in VARSETS:
        raise ValueError(f"Unsupported sample '{sample}'. Expected one of {SUPPORTED_SAMPLES}.")
    by_channel = VARSETS[sample]
    if channel not in by_channel:
        raise ValueError(f"Unsupported channel '{channel}' for sample '{sample}'. Expected one of {tuple(by_channel.keys())}.")
    if varset not in by_channel[channel]:
        raise ValueError(f"Unsupported varset '{varset}' for sample '{sample}' channel '{channel}'.")
    return list(by_channel[channel][varset])


def infer_varset_from_tag(tag, sample=None):
    sample_key = sample or infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    candidates = tuple(VARSETS.get(sample_key, {}).get(channel, {}).keys())
    for key in sorted(candidates, key=len, reverse=True):
        if not re.fullmatch(r"\d+v\d+", key):
            continue
        if f"_{key}_" in tag:
            return key
    return None


def get_reweight_varset_columns(sample, varset, channel):
    by_sample = REWEIGHT_VARSETS.get(sample)
    if by_sample is None:
        raise ValueError(
            f"Unsupported reweight sample '{sample}'. Expected one of {tuple(REWEIGHT_VARSETS.keys())}."
        )
    by_channel = by_sample.get(channel)
    if by_channel is None:
        raise ValueError(
            f"Unsupported reweight channel '{channel}' for sample '{sample}'. "
            f"Expected one of {tuple(by_sample.keys())}."
        )
    if varset not in by_channel:
        raise ValueError(
            f"Unsupported reweight varset '{varset}' for sample '{sample}' channel '{channel}'. "
            f"Expected one of {tuple(by_channel.keys())}."
        )
    return list(by_channel[varset])
