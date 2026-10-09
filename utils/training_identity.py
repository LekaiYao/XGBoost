"""Freeze source revisions and the actual candidate split for later diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from configs.samples import split_root_spec
from utils.paths import training_dir


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def source_revision(spec, checksum=True):
    path, tree = split_root_spec(spec)
    st = Path(path).stat()
    result = dict(path=path, tree=tree, size=st.st_size, mtime_ns=st.st_mtime_ns)
    if checksum: result['sha256'] = sha256(path)
    check_revision(result)
    return result


def check_revision(revision):
    st = Path(revision['path']).stat()
    if (st.st_size, st.st_mtime_ns) != (revision['size'], revision['mtime_ns']):
        raise ValueError('Input source changed: '+revision['path'])


def save_identity(tag, signal_ids, background_ids, train_ids, test_ids, revisions):
    for revision in revisions: check_revision(revision)
    out = Path(training_dir(tag)); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'split_identity.npz', signal_entry=np.asarray(signal_ids, dtype=np.int64),
                        background_entry=np.asarray(background_ids, dtype=np.int64),
                        train_index=np.asarray(train_ids, dtype=np.int64), test_index=np.asarray(test_ids, dtype=np.int64))
    (out/'split_identity.json').write_text(json.dumps(dict(sources=revisions,
        policy='actual train_test_split indices; source tree entry identities',
        npz_sha256=sha256(out/'split_identity.npz')), indent=2)+'\n')
