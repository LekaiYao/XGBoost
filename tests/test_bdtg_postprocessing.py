import unittest
import numpy as np
from workflows.punzi_scan import validated_split

class NativeSplitTests(unittest.TestCase):
    def setUp(self):
        self.s=np.array([2,4,8,9]);self.b=np.array([11,13,15,16])
        self.f=dict(signal_entry=self.s,background_entry=self.b,
                    train_index=np.array([3,0,1,7,4,5]),test_index=np.array([2,6]))
        self.meta=dict(notes=dict(n_sig_train=3,n_bkg_train=3,n_sig_test=1,n_bkg_test=1))
    def test_native_membership_is_preserved(self):
        result=validated_split('bdtg',{},self.meta,self.s,self.b,self.f)
        np.testing.assert_array_equal(result['train_index'],self.f['train_index'])
    def test_missing_duplicate_source_and_count_rejected(self):
        for frozen in [None,dict(self.f,train_index=np.array([3,0,1,7,4,4])),
                       dict(self.f,signal_entry=self.s+1)]:
            with self.assertRaises(ValueError):validated_split('bdtg',{},self.meta,self.s,self.b,frozen)
        with self.assertRaises(ValueError):
            validated_split('bdtg',{},dict(notes=dict(n_sig_train=2)),self.s,self.b,self.f)
    def test_xgb_keeps_seed_split(self):
        from sklearn.model_selection import train_test_split
        tr,te=train_test_split(np.arange(8),train_size=.75,random_state=42)
        f=dict(self.f,train_index=tr,test_index=te)
        result=validated_split('xgb',dict(train_fraction=.75,split_random_state=42),dict(notes={}),self.s,self.b,f)
        np.testing.assert_array_equal(result['train_index'],tr)
        with self.assertRaises(ValueError):
            validated_split('xgb',dict(train_fraction=.75,split_random_state=42),dict(notes={}),self.s,self.b,self.f)

class ShapBoundaryTests(unittest.TestCase):
    def test_exact_cut_and_adjacent_float32_for_both_directions(self):
        import tempfile
        from pathlib import Path
        import shap
        from utils.tmva_shap import forest_from_xml
        for cut in [1., .1, .100000003]:
            for ctype in [0,1]:
                xml=f'<MethodSetup Method="BDT::BDTG"><Options><Option name="BoostType">Grad</Option><Option name="VarTransform">None</Option></Options><Variables><Variable Expression="x"/></Variables><Weights><BinaryTree><Node NCoef="0" IVar="0" Cut="{cut}" cType="{ctype}"><Node pos="l" NCoef="0" IVar="-1" res="-0.3"/><Node pos="r" NCoef="0" IVar="-1" res="0.3"/></Node></BinaryTree></Weights></MethodSetup>'
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'tree.xml';p.write_text(xml)
                    r=np.float32(cut);x=np.array([[np.nextafter(r,np.float32(-np.inf))],[r],[np.nextafter(r,np.float32(np.inf))]],dtype=np.float32)
                    e=shap.TreeExplainer(forest_from_xml(p,['x']),data=x,feature_perturbation='interventional')
                    expected=np.where(x[:,0].astype(float)>=cut,.6,-.6)*(1 if ctype else -1)
                    np.testing.assert_allclose(e.model.predict(x),expected,atol=1e-12)
                    np.testing.assert_allclose(e.expected_value+e.shap_values(x).sum(axis=1),expected,atol=1e-7)

if __name__=='__main__':unittest.main()
