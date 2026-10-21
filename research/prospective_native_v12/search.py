"""V12 completeness-preserving bounded adapter search on exposed calibration source.

Ranking orders attempts, never prunes eligible functions. A declared cap bounds
attempts; if the cap is exhausted without a witness return UNKNOWN, not IMPOSSIBLE.
"""
import argparse, importlib, itertools, json, sys, time
from pathlib import Path
from research.prospective_native_v10.run import scan, score, equal_info_keyword_baseline, ref
from research.prospective_native_v11.adapter import adaptations

def probe(root,c,cases):
    module=".".join(Path(c["file"]).with_suffix("").parts)
    try:
        f=getattr(importlib.import_module(module),c["name"])
        variants=adaptations(f)
    except Exception as exc:
        return {"status":"UNRESOLVED","error":type(exc).__name__}
    if not variants:return {"status":"NO_ADMISSIBLE_BINDING"}
    verdicts=[]
    for kwargs,materialize in variants:
        correct=0
        try:
            for values in cases:
                result=f(list(values),**kwargs)
                if materialize:result=list(result)
                if type(result) is not list or result!=ref(list(values)):break
                correct+=1
            status="PASS" if correct==len(cases) else "REFUTED"
        except Exception as exc:
            status="UNRESOLVED_"+type(exc).__name__
        verdicts.append(dict(kwargs=kwargs,materialize=materialize,correct=correct,status=status))
        if status=="PASS":break
    return {"status":"PASS" if any(v["status"]=="PASS" for v in verdicts) else "NOT_QUALIFIED","variants":verdicts}

def run(root,out,budget=160):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    records=scan(root)[:400]
    candidate=sorted(records,key=lambda c:(-score(c),c["file"],c["name"]))
    base=equal_info_keyword_baseline(records)
    # The baseline shares the same total universe, but uses a different order.
    remaining={(c["file"],c["name"]) for c in base}
    base += [c for c in sorted(records,key=lambda x:(x["file"],x["name"])) if (c["file"],c["name"]) not in remaining]
    cases=[tuple(v) for n in range(7) for v in itertools.product(range(3),repeat=n)]
    results={}
    for name,order in (("candidate",candidate),("baseline",base)):
        begin=time.perf_counter();attempts=[]
        for c in order[:budget]:
            result=probe(root,c,cases)
            attempts.append({"name":c["name"],"file":c["file"],**result})
            if result["status"]=="PASS":break
        winner=next((x for x in attempts if x["status"]=="PASS"),None)
        results[name]={"attempts":len(attempts),"winner":winner,"seconds":time.perf_counter()-begin,
                       "status":"QUALIFIED_FINITE_BEHAVIOR" if winner else ("UNKNOWN_BUDGET" if len(order)>budget else "NO_WITNESS")}
    result={"source":"more-itertools exposed calibration","universe":len(records),"budget":budget,
            "cases":len(cases),"results":results,"scientific_status":"EXPOSED_CALIBRATION_NOT_PROSPECTIVE"}
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True);parser.add_argument("--out",required=True);parser.add_argument("--budget",type=int,default=160)
    args=parser.parse_args();run(args.root,args.out,args.budget)
