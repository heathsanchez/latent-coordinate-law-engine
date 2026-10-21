"""V13: source binding is a distinct cost from retaining semantic expression."""
import argparse, json, sys, itertools, time
from pathlib import Path
from research.prospective_native_v12.search import run as cold_search, probe
from research.prospective_native_v10.run import scan, score

def main(source,out):
    root=Path(source).resolve(); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    records=scan(root)[:400]
    checks=[tuple(x) for n in range(7) for x in itertools.product(range(3),repeat=n)]
    # Source A recipe is deliberately not misrepresented as a resolved B symbol.
    retained={'semantic_contract':'finite adjacent pairs, ordered, list',
              'parameter_role':'window_size=2','result_conversion':'list',
              'source_a_symbol':'more_itertools.recipes.sliding_window',
              'source_b_symbol':None}
    cold=cold_search(root,out/'cold',budget=len(records))
    ordered=sorted(records,key=lambda c:(-score(c),c['file'],c['name']))
    start=time.perf_counter();attempted=0;winner=None
    for cand in ordered:
        attempted+=1
        result=probe(root,cand,checks)
        if result['status']=='PASS':
            winner={'name':cand['name'],'file':cand['file'],'result':result};break
    elapsed=time.perf_counter()-start
    outcome={'retained':retained,'direct_transfer':'UNKNOWN_NO_SOURCE_B_BINDING',
             'binding_reconstruction_attempts':attempted,'binding_reconstruction_seconds':elapsed,
             'qualified_after_binding':winner,'cold_baseline':cold['results']['baseline'],
             'scope_cases':len(checks),'source':'exposed boltons calibration',
             'scientific_status':'NO_ZERO_SEARCH_TRANSFER_CLAIM'}
    (out/'transport.json').write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.source,a.out)
