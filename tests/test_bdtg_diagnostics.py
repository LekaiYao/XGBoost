import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from configs.direct_bdtg_settings import resolve_bdtg_params
from utils import paths
from utils.model_backend import BDTGModel
from utils.bdtg_diagnostics import export_training_diagnostics,validate_training_diagnostics
from workflows.bdtg_train import native_train

class DiagnosticsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('root'),'ROOT required')
    def test_native_prefix_losses_and_required_artifacts(self):
        tag='X_pb18_v1_fid1_3v3_rw0_bdtg_v1'
        with tempfile.TemporaryDirectory() as d,patch.object(paths,'MODELS_DIR',str(Path(d)/'models')),patch.object(paths,'TRAINING_DIR',str(Path(d)/'training')):
            rng=np.random.RandomState(19);x=pd.DataFrame(rng.normal(size=(240,3)),columns=['a','b','c']);labels=np.arange(240)<120;x.loc[labels,'a']+=1.2
            physical=np.where(labels,np.linspace(.03,.08,len(x)),1.);fit=physical*np.where(labels,20.,1.)
            params=resolve_bdtg_params(1);params['NTrees']=3
            md=Path(paths.model_dir(tag));td=Path(paths.training_dir(tag));tr,te=native_train(x,labels,fit,md,td,params)
            model=BDTGModel(md/'bdtg.weights.xml',x.columns);score=model.predict_proba(x)[:,1]
            np.savez_compressed(td/'split_identity.npz',train_index=tr,test_index=te)
            (md/'model_config.json').write_text(json.dumps(dict(model_params=params)))
            result=export_training_diagnostics(tag,x,labels,physical,tr,te,score,list(x.columns));self.assertEqual(result['status'],'PASS')
            self.assertEqual(validate_training_diagnostics(tag)['n_estimators'],3)
            history=json.loads((td/'training_history.json').read_text())
            root=ET.parse(md/'bdtg.weights.xml').getroot()
            for k in range(1,4):
                limited=copy.deepcopy(root);forest=limited.find('Weights');forest.set('NTrees',str(k))
                for tree in list(forest)[k:]:forest.remove(tree)
                for option in limited.find('Options'):
                    if option.get('name')=='NTrees':option.text=str(k)
                path=Path(d)/f'prefix{k}.xml';ET.ElementTree(limited).write(path)
                probability=BDTGModel(path,x.columns).predict_proba(x)[:,1]
                for name,ids in [('train',tr),('test',te)]:
                    loss=np.average(-np.where(labels[ids],np.log(probability[ids]),np.log1p(-probability[ids])),weights=physical[ids])
                    self.assertAlmostEqual(loss,history[name+'_logloss'][k-1],places=11)
            (td/'ks.json').unlink()
            with self.assertRaises(FileNotFoundError):validate_training_diagnostics(tag)

if __name__=='__main__':unittest.main()
