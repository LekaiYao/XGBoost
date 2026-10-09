import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dag import submit_bdtg_workflow as workflow
from utils.training_identity import sha256

TAG = 'X_pb18_v1_fid1_3v3_rw0_bdtg_v1'


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        for name, content in {
            'configs/example.json': '{}', 'utils/example.py': 'x = 1\n',
            'dag/submit_templates/submit_bdtg_job.sub': 'executable = run_bdtg_job.sh\nqueue 1\n',
            'wrappers/run_bdtg_job.sh': '#!/bin/bash\nexit 0\n',
        }.items():
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        self.source = self.repo / 'reference.root'
        self.source.write_bytes(b'source')
        stat = self.source.stat()
        self.resolved = dict(train_tag=TAG, reference_tag=TAG.replace('bdtg', 'xgb'),
                            input_revisions=[dict(path=str(self.source), tree='RECO',
                                size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(self.source))],
                            protected_sha256={str(self.source): sha256(self.source)})
        def capture(args, cwd=None):
            if args[0] == 'root-config': return '6.36.00'
            if args[-1] == 'freeze': return 'numpy==2.0.0'
            if args[:2] == ['git', 'ls-files']:
                return '\n'.join(str(p.relative_to(self.repo)) for p in self.repo.rglob('*') if p.is_file())
            if args[0] == 'condor_q': return ''
            if args[0] == 'hostname': return 'lxplus-test'
            if args[0] == 'condor_config_val': return 'schedd-test' if args[1] == 'SCHEDD_HOST' else 'pool-test'
            return 'test'
        self.addCleanup(patch.stopall)
        patch.object(workflow, 'capture', side_effect=capture).start()

    def prepare(self):
        return workflow.prepare(self.repo, self.resolved, postprocess_only=True)

    def test_prepare_is_offline_and_contains_complete_snapshot(self):
        with patch.object(workflow.subprocess, 'run') as run:
            bundle = self.prepare()
            run.assert_not_called()
        receipt = workflow.validate_bundle(bundle)
        self.assertEqual(receipt['state'], 'prepared_not_submitted')
        self.assertEqual(receipt['stages'], ['draw', 'shap', 'fit_interface', 'verify'])
        self.assertTrue((bundle / 'source_snapshot/utils/example.py').is_file())
        self.assertNotEqual(bundle, self.prepare())

    def test_full_dag_and_existing_target_rejection(self):
        bundle = workflow.prepare(self.repo, self.resolved)
        receipt = json.loads((bundle / 'preparation.json').read_text())
        self.assertEqual(Path(receipt['dag']).read_text().count('JOB '), 8)
        target = self.repo / 'existing'
        target.mkdir()
        (target / 'model').write_text('retained')
        with patch.object(workflow.paths, 'model_dir', return_value=str(target)):
            with self.assertRaises(FileExistsError): workflow.validate_bundle(bundle)

    def test_reject_source_and_bundle_mutation(self):
        bundle = self.prepare()
        source = self.repo / 'utils/example.py'
        source.write_text('x = 2\n')
        with self.assertRaisesRegex(ValueError, 'Changed since'): workflow.validate_bundle(bundle)
        source.write_text('x = 1\n')
        (bundle / 'run_bdtg_job.sh').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'bundle changed'): workflow.validate_bundle(bundle)

    def test_receipt_and_duplicate_guard_without_real_submission(self):
        bundle = self.prepare()
        def run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, '1 job(s) submitted to cluster 12345.\n', '')
        with patch.object(workflow.subprocess, 'run', side_effect=run) as runner:
            workflow.submit(bundle, self.repo / 'mock_afs')
            self.assertEqual(runner.call_count, 2)
            with self.assertRaises(FileExistsError): workflow.submit(bundle, self.repo / 'mock_afs')
            self.assertEqual(runner.call_count, 2)
        receipt = json.loads((bundle / 'submission.json').read_text())
        self.assertEqual(receipt['dag_cluster'], 12345)
        self.assertEqual(receipt['state'], 'submitted')
        self.assertIn('condor_q', receipt['query_command'])

    def test_failure_is_recorded_and_not_retried(self):
        bundle = self.prepare()
        results = [subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 1, '', 'failed')]
        with patch.object(workflow.subprocess, 'run', side_effect=results):
            with self.assertRaises(subprocess.CalledProcessError): workflow.submit(bundle, self.repo / 'mock_afs')
        receipt = json.loads((bundle / 'submission.json').read_text())
        self.assertEqual(receipt['state'], 'submit_failed')
        with self.assertRaises(FileExistsError): workflow.submit(bundle, self.repo / 'mock_afs')


if __name__ == '__main__':
    unittest.main()
