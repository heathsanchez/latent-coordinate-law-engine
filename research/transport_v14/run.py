"""V14 honest compiled fallback versus source binding with equal-info baseline."""
import argparse,itertools,json,time,sys
from pathlib import Path
from research.prospective_native_v12.search import run as cold
from research.prospective_native_v10.run import ref

def fallback(seq):
    return list(zip(seq,seq[1:]))
def check(fn,cases):
    return all(fn(list(x))==ref(list(x)) for x in cases)
def run(source,out):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    cases=[tuple(a) for n in range(7) for a in itertools.product(range(3),repeat=n)]
    start=time.perf_counter()
    retained_verified=check(fallback,cases)
    verified_s=time.perf_counter()-start
    start=time.perf_counter()
    cold_result=cold(source,p/"cold",budget=160)
    cold_s=time.perf_counter()-start
    # Equal-information conventional baseline is permitted exactly the same
    # fallback; compare identical implementation and do not subtract old acquisition.
    result={"cases":len(cases),"retained_compiled_fallback":retained_verified,
            "same_info_baseline_fallback":check(fallback,cases),
            "retained_verification_seconds":verified_s,"cold_total_seconds":cold_s,
            "cold_source_binding":cold_result["results"]["baseline"],
            "new_source_binding_obtained_by_retained":False,
            "source_b_reused_without_binding":True,
            "source_b_native_implementation_invoked_by_retained":False,
            "status":"BOUNDED_FALLBACK_EQUIVALENCE_NO_COMPARATIVE_ADVANTAGE",
            "limitations":"hand-authored fallback; source B bypassed; full source-independent development not established"}
    (p/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--source",required=True);a.add_argument("--out",required=True)
    args=a.parse_args();run(args.source,args.out)
