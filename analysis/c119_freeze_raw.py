"""Freeze the raw counts of every device job the paper uses, so analysis never needs the vendor API.

    python c119_freeze_raw.py

Read-only retrieval of finished jobs (no device time). Writes data/raw/<job>.json with the job id,
backend, creation time and per-pub counts, and data/raw/MANIFEST.json with sha256 of each file.
The token is read at runtime from outside the repository and is never written anywhere.
c114 (the sealed replication) is already frozen in data/sealed/ and is only listed in the manifest.
"""
import hashlib
import json
import os

import c99_run_device as R

HERE = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(HERE, 'data', 'raw')
JOBS = {'dam6td02fm4c73f2gl1g': ('QTM161', 'c101 engineered-noise sweep (Supplementary)'),
        'daoa3j8pqrnc7399miv0': ('DQC methods', 'c108 held-out experiment; c110 paired regression (Fig. 3)'),
        'daobh65r85ps73ff233g': ('DQC methods', 'c112 target-resolved calibration (Fig. 3b)')}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


if __name__ == '__main__':
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = json.load(open(R.KEYPATH, encoding='utf-8'))['apikey']
    os.makedirs(RAWDIR, exist_ok=True)
    manifest = {}
    for jid, (inst, use) in JOBS.items():
        path = os.path.join(RAWDIR, f'{jid}.json')
        if not os.path.exists(path):
            svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=inst)
            job = svc.job(jid)
            res = job.result()
            counts = [res[i].data.c.get_counts() for i in range(len(res))]
            json.dump(dict(job=jid, backend=job.backend().name, created=str(job.creation_date),
                           n_pubs=len(counts), counts=counts), open(path, 'w'))
        manifest[jid] = dict(file=f'raw/{jid}.json', sha256=sha(path), use=use,
                             bytes=os.path.getsize(path))
        print(f'{jid}  {manifest[jid]["bytes"]/1e6:7.2f} MB  {manifest[jid]["sha256"][:16]}  {use}')
    sp = os.path.join(HERE, 'data', 'sealed', 'c114_raw.json')
    raw114 = json.load(open(sp))
    manifest[raw114['job']] = dict(file='sealed/c114_raw.json', sha256=sha(sp),
                                   use='c114 sealed replication (Fig. 4, 5)', bytes=os.path.getsize(sp))
    json.dump(manifest, open(os.path.join(RAWDIR, 'MANIFEST.json'), 'w'), indent=1)
    print('manifest written')
