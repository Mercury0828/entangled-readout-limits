"""Re-run the engineered-noise re-analysis (c106) on the FROZEN counts of job dam6td02fm4c73f2gl1g.

    python c123_rerun_c106_frozen.py > data/selfcheck/c106_frozen.txt

Executes c106's own analysis code unchanged, replacing only the vendor-API retrieval by the frozen file, so the
Supplementary engineered-noise table is regenerated from data in the repository.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'c106_recalibration.py'), encoding='utf-8').read()
head, main = src.split("if __name__ == '__main__':", 1)
old = """    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=INSTANCE)
    job = svc.job(JOB_C101)
    print(f'job {JOB_C101}: {job.status()}')
    res = job.result()
    _c, meta = T.build()
    prod, bell, mode = collect(lambda i: res[i].data.c.get_counts(), meta)"""
new = """    frozen = json.load(open(os.path.join(HERE, 'data', 'raw', JOB_C101 + '.json')))['counts']
    print(f'job {JOB_C101}: frozen counts, {len(frozen)} pubs')
    _c, meta = T.build()
    assert len(frozen) == len(_c), 'frozen counts do not match the rebuilt schedule'
    prod, bell, mode = collect(lambda i: frozen[i], meta)"""
assert main.count(old) == 1, 'c106 retrieval block changed; update this wrapper'
g = {'__name__': 'c106_frozen', '__file__': os.path.join(HERE, 'c106_recalibration.py')}
sys.path.insert(0, HERE)
exec(compile(head, 'c106_recalibration.py', 'exec'), g)
g['HERE'] = HERE
g['os'] = os
exec(compile('if True:\n' + main.replace(old, new), 'c106_recalibration.py:main', 'exec'), g)
