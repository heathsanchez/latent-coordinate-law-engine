"""Frozen source-C test: unchanged V21 scanner/adapter and matched semantic baseline.

Mode `discover` only uses six predeclared acquisition cases. Mode `verify`
imports the separate 1,093-case oracle in a new process. The source pin is
an identity condition, not evidence of unrestricted correctness.
"""
import argparse
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from research.prospective_native_v10.run import scan, score
from research.prospective_native_v11.adapter import adaptations
from research.semantic_v21.discover import fingerprints

SOURCE_SHA='451af60dec590a6010e2babdbf391ea8f815122f'
REPO='pytoolz/toolz'
CASES=[[],[0],[1,2],[0,1,2],[2,2,1,0],[0,2,1,2]]
TERMS=(('adjacent',4),('consecutive',4),('pair',3),('sliding',3),
       ('window',2),('tuple',1),('iterable',1),('list',1))


def check_source(root):
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=SOURCE_SHA:
        raise ValueError('SOURCE_C_COMMIT_MISMATCH')
    subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD'],check=True)


def baseline_score(record):
    doc=record['doc'].lower()
    return sum(value for word,value in TERMS if word in doc)


def candidate_universe(root):
    return sorted(scan(root)[:400],key=lambda c:(-score(c),c['file'],c['name']))[:160]


def evaluate(root,records):
    sys.path.insert(0,str(root))
    matches=[];attempts=0;first_hit_attempt=None
    began=time.perf_counter()
    for c in records:
        module='.'.join(Path(c['file']).with_suffix('').parts)
        try:
            function=getattr(importlib.import_module(module),c['name'])
            variants=adaptations(function)
        except Exception:
            continue
        for kwargs,materialize in variants:
            attempts+=1
            try:
                ok=True
                for xs in CASES:
                    result=function(list(xs),**kwargs)
                    if materialize:
                        # Correct adjacent-pairs result length is max(0,len(xs)-1).
                        result=list(itertools.islice(result,max(1,len(xs))))
                    expected=[(xs[i],xs[i+1]) for i in range(len(xs)-1)]
                    if type(result)!=list or result!=expected:
                        ok=False;break
                if ok:
                    if first_hit_attempt is None:first_hit_attempt=attempts
                    matches.append({'file':c['file'],'name':c['name'],
                                    'kwargs':kwargs,'materialize':materialize})
            except Exception:
                continue
    return {'candidates':matches,'attempts':attempts,
            'first_hit_attempt':first_hit_attempt,
            'elapsed_seconds':time.perf_counter()-began}


def discovery(root,out):
    check_source(root)
    universe=candidate_universe(root)
    original,original_attempts,original_scanned=fingerprints(root)
    ordinary=evaluate(root,universe)
    comparator=evaluate(root,sorted(universe,key=lambda c:(-baseline_score(c),c['file'],c['name'])))
    if ordinary['candidates']!=original or ordinary['attempts']!=original_attempts:
        raise AssertionError('SOURCE_C_REIMPLEMENTATION_DISAGREEMENT')
    result={'source_repository':REPO,'source_sha':SOURCE_SHA,
            'candidate_limit':160,'scanned_functions':original_scanned,
            'acquisition_probes':CASES,'source_generic':ordinary,
            'equal_information_semantic_baseline':comparator,
            'status':'CANDIDATES_PENDING_INDEPENDENT_VERIFICATION'}
    out.mkdir(exist_ok=True)
    (out/'source_c_discovery.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'source':SOURCE_SHA,'scanned':original_scanned,
          'source_generic_matches':len(ordinary['candidates']),
          'baseline_matches':len(comparator['candidates']),
          'source_generic_attempts':ordinary['attempts'],
          'baseline_attempts':comparator['attempts']},indent=2))


def verification(root,out):
    check_source(root)
    # The 1,093-case verifier is deliberately imported only in this process.
    from research.semantic_v21.verify import check
    acquired=json.loads((out/'source_c_discovery.json').read_text())
    if acquired.get('source_sha')!=SOURCE_SHA:
        raise ValueError('ACQUISITION_SOURCE_MISMATCH')
    results={}
    for key in ('source_generic','equal_information_semantic_baseline'):
        checks=[{'candidate':c,'result':check(root,c)}
                for c in acquired[key]['candidates']]
        accepted=[record for record in checks if record['result'].get('pass') and
                  record['result'].get('checked')==1093]
        results[key]={'checked_candidates':len(checks),'qualified':accepted,
                      'first_hit_attempt':acquired[key]['first_hit_attempt'],
                      'attempts':acquired[key]['attempts']}
    status='WARRANTED_BOUNDED_SOURCE_C' if results['source_generic']['qualified'] else \
           'UNKNOWN_BUDGET_OR_BINDING'
    result={'source_repository':REPO,'source_sha':SOURCE_SHA,
            'status':status,'protected_oracle_cases':1093,
            'results':results,'unique_advantage':False,
            'reason':'MATCHED_SEMANTIC_SEARCH_CAN_USE_SAME_INFORMATION',
            'not_proved':['arbitrary input equivalence','unbounded portability',
                          'new language invention','owner intent or permission']}
    (out/'source_c_evidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':status,'qualified':{
      k:len(v['qualified']) for k,v in results.items()},
      'protected_oracle_cases':1093,'unique_advantage':False},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('discover','verify'))
    p.add_argument('--root',type=Path,default=Path('/tmp/toolz-v21-c'))
    p.add_argument('--out',type=Path,default=Path('v21_out'))
    args=p.parse_args()
    if args.mode=='discover':discovery(args.root,args.out)
    else:verification(args.root,args.out)


if __name__=='__main__':main()
