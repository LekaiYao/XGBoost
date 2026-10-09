import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils.punzi import denominator, scan, optimal, sidebands_from_selection
from dag.make_feature_selection_workflow import make_dag
from configs.feature_selection import FEATURE_SELECTION_EXPERIMENTS

class PunziTest(unittest.TestCase):
    def test_delivered_cases(self):
        cases=json.loads(Path('tests/fixtures/punzi_method_examples.json').read_text())
        from fractions import Fraction
        for case in cases['cases']:
            i,e=case['input'],case['expected']
            eps=float(i['mc_pass_sumw'])/float(i['mc_pre_score_sumw'])
            b=(i['left_entries']+i['right_entries'])*float(Fraction(i['width_ratio']))/float(i['inclusion_probability'])
            d=float(denominator(b))
            np.testing.assert_allclose([eps,b,d,eps/d],[float(e[k]) for k in ['epsilon','B','D','FOM']],rtol=1e-12,atol=1e-15)
        for scope,rows in cases['optimal_fixture_scopes'].items():
            self.assertEqual(optimal(rows)['score'],cases['optimal_fixture_scopes_expected'][scope])
        self.assertEqual(optimal(cases['optimal_fixture'])['exact_ties'],[.5,.51])

    def test_strict_edges_and_probability(self):
        bands=[(3.8,3.85),(3.89,3.94)]
        rows=scan(np.array([.5,.6]),np.array([1.,3.]),np.array([3.8,3.82,3.9,3.94]),np.array([1.,.5,.6,1.]),bands,.25)
        r=rows[45]
        self.assertEqual(r['score'],.5)
        self.assertEqual((r['left_entries'],r['right_entries']),(0,1))
        self.assertAlmostEqual(r['epsilon'],.75)
        self.assertAlmostEqual(r['B'],2.4)
        self.assertEqual(len(rows),91);self.assertEqual(rows[-1]['score'],.95)
        for weights in [[],[0],[-1],[np.nan]]:
            with self.assertRaises(ValueError):scan(np.zeros(len(weights)),weights,[],[],bands)
        self.assertTrue(optimal(scan([0],[1],[],[],bands))['uninformative'])

    def test_union_not_mean(self):
        bands=[(3.8,3.85),(3.89,3.94)]
        total=scan([.8,.2],[1,3],[3.82,3.91],[.8,.2],bands)[45]
        train=scan([.8],[1],[3.82],[.8],bands,.75)[45]
        test=scan([.2],[3],[3.91],[.2],bands,.25)[45]
        self.assertEqual(total['mc_pass_sumw'],train['mc_pass_sumw']+test['mc_pass_sumw'])
        self.assertAlmostEqual(total['B'],.75*train['B']+.25*test['B'])
        self.assertNotAlmostEqual(total['FOM'],(train['FOM']+test['FOM'])/2)

    def test_mass_metadata(self):
        self.assertEqual(sidebands_from_selection('((Bmass > 3.80 and Bmass < 3.85) or (Bmass > 3.89 and Bmass < 3.94)) and Bpt > 15'),[(3.8,3.85),(3.89,3.94)])
        with self.assertRaises(ValueError):sidebands_from_selection('Bmass >= 3.8 and Bmass < 3.9')

    def test_add1_all_seven_default_shap(self):
        exp='fs_2v1_add1_X_pb23_v1_fid1_2v1_rw0_xgb_v1'
        from configs.samples import SAMPLES
        from utils.varsets import VARSETS
        config = dict(baseline='X_pb23_v1_fid1_2v1_rw0_xgb_v1', operation='add1',
                      candidates=[dict(train_tag=f'X_pb23_v1_fid1_3v{i}_rw0_xgb_v1',
                                       added_variable=f'feature_{i}') for i in range(1, 7)])
        # Synthetic registrations keep the workflow test independent of archived production tags.
        with patch.dict(FEATURE_SELECTION_EXPERIMENTS, {exp: config}, clear=True), \
             patch.dict(VARSETS['pbpb']['X'], {'2v1': ['a', 'b'], **{f'3v{i}': ['a', 'b', f'feature_{i}'] for i in range(1, 7)}}, clear=True), \
             patch.dict(SAMPLES['pbpb']['channels']['X']['selection_profiles'],
                        {'pb23_v1': dict(signal_selection='True', background_selection='True')}, clear=True), \
             tempfile.TemporaryDirectory() as d, patch('dag.make_feature_selection_workflow.Path.is_file',return_value=False),patch('dag.make_feature_selection_workflow.Path.exists',return_value=False):
            # Path.exists patch can interfere with mkdir only for already-created out dir: exist_ok handles it.
            path=make_dag(exp,d)
            text=path.read_text()
        self.assertEqual(text.count('submit_train_job.sub'),7)
        self.assertEqual(text.count('submit_shap_job.sub'),7)
        self.assertEqual(text.count('submit_draw_job.sub'),7)
        self.assertIn('PARENT DRAW_6 CHILD SHAP_6',text)

if __name__=='__main__':unittest.main()
