import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import uproot
from configs import samples
from utils import paths
from utils.model_backend import map_bdtg_score, BDTGModel
from utils.streaming_apply import score_frame

class BDTGTest(unittest.TestCase):
    def test_mapping_and_tag(self):
        np.testing.assert_array_equal(map_bdtg_score([-1,0,1]),[0,.5,1])
        for bad in [[np.nan],[np.inf],[1.01],[-1.01]]:
            with self.assertRaises(ValueError):map_bdtg_score(bad)
        for kind in ["xgb","bdtg"]:
            tag="X_pb18_v1_fid1_3v3_rw0_"+kind+"_v1"
            self.assertEqual(samples.infer_dataset_year(tag,"pbpb"),"2018")
            self.assertEqual(samples.infer_fid_profile(tag,"pbpb"),"pb18_fid1")
            self.assertTrue(paths.model_path(tag).endswith("bdtg.weights.xml" if kind=="bdtg" else "xgb_model.pkl"))
        with self.assertRaises(ValueError):samples.infer_direct_xgb_version(tag)

    def test_bdtg_dag_dependencies(self):
        from dag.make_bdtg_workflow import make_dag
        with tempfile.TemporaryDirectory() as d:
            tag="X_pb18_v1_fid1_3v3_rw0_bdtg_v1"
            text=make_dag(d,tag,Path(d)/"manifest.json").read_text()
            for edge in ["TRAIN CHILD APPLY","APPLY CHILD EXTRA","EXTRA CHILD FULL","FULL CHILD DRAW","DRAW CHILD SHAP","SHAP CHILD FIT_INTERFACE","FIT_INTERFACE CHILD VERIFY"]:
                self.assertIn("PARENT "+edge,text)
            post=make_dag(d,tag,Path(d)/"manifest.json",True).read_text()
            self.assertNotIn("JOB TRAIN",post)
            self.assertNotIn("JOB APPLY",post)
            self.assertIn("JOB DRAW",post)
            self.assertNotIn("PUNZI",text)

    @unittest.skipUnless(shutil.which("root"),"ROOT required")
    def test_native_seed_and_serialization(self):
        from workflows.bdtg_train import native_train
        from configs.direct_bdtg_settings import resolve_bdtg_params
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.RandomState(19)
            frame=pd.DataFrame(rng.normal(size=(160,3)),columns=["a","b","c"])
            labels=np.arange(160)<80;frame.loc[labels,"a"]+=1
            weights=np.linspace(.4,1.4,160);params=resolve_bdtg_params(1);params["NTrees"]=3
            a,b=native_train(frame,labels,weights,root/"m1",root/"t1",params)
            c,e=native_train(frame,labels,weights,root/"m2",root/"t2",params)
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,e)
            self.assertEqual(len(a),120);self.assertEqual(len(b),40)
            m=BDTGModel(root/"m1/bdtg.weights.xml",frame.columns)
            self.assertEqual(len(m.predict_raw(frame)),160)
            self.assertFalse(set(a)&set(b))

    @unittest.skipUnless(shutil.which("root"),"ROOT required")
    def test_full_root_dual_scores_and_legacy_score(self):
        macro=Path("utils/full_apply_root.C").resolve()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/"source.root"
            with uproot.recreate(source) as f:
                f.mktree("RECO",{"id":np.arange(5,dtype=np.int32),"Score":np.arange(5,dtype=np.float32)+10,"weight":np.ones(5)})
                f.mktree("GEN",{"id":np.arange(9,dtype=np.int32)})
            scores=np.array([0,.25,.5,1,-1],dtype=np.float32);raw=np.array([-1,-.5,0,1,np.nan],dtype=np.float32)
            scores.tofile(d/"score.bin");raw.tofile(d/"raw.bin")
            for data in [False,True]:
                dest=d/("data.root" if data else "mc.root")
                if not data:shutil.copy2(source,dest)
                args=[json.dumps(str(dest)),json.dumps("RECO"),json.dumps(str(d/"score.bin")),str(data).lower(),json.dumps(str(source)),json.dumps(str(d/"raw.bin"))]
                subprocess.run(["root","-l","-b","-q",str(macro)+"("+",".join(args)+")"],check=True,capture_output=True)
                with uproot.open(dest) as f:
                    arr=f["RECO"].arrays(library="np");n=4 if data else 5
                    np.testing.assert_array_equal(arr["id"],np.arange(n))
                    np.testing.assert_array_equal(arr["Score_source"],np.arange(n)+10)
                    np.testing.assert_array_equal(arr["Score"],scores[:n])
                    np.testing.assert_allclose(arr["Score_raw"],raw[:n],equal_nan=True)
                    self.assertEqual(f["GEN"].num_entries,9)
            # Optional argument omitted: legacy XGBoost writer remains compatible.
            dest=d/"xgb.root";shutil.copy2(source,dest)
            args=[json.dumps(str(dest)),json.dumps("RECO"),json.dumps(str(d/"score.bin")),"false",json.dumps(str(source))]
            subprocess.run(["root","-l","-b","-q",str(macro)+"("+",".join(args)+")"],check=True,capture_output=True)
            with uproot.open(dest) as f:
                self.assertNotIn("Score_raw",f["RECO"].keys())
                self.assertIn("Score_source",f["RECO"].keys())

if __name__=="__main__":unittest.main()
