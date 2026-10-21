"""V11 source-general adapter search for known finite sequence obligation.

This *calibration* runner adapts callables with a common iterable-first
signature by searching explicit argument bindings and list materialization.
It never assumes target callable names; all candidates are ranked from AST.
Never claim source-native novelty from this exposed-source calibration.
"""
import ast, importlib, inspect, itertools, json, sys, time
from pathlib import Path
from research.prospective_native_v10.run import scan, score, equal_info_keyword_baseline, ref

def adaptations(function):
    try: signature=inspect.signature(function)
    except (TypeError,ValueError):return []
    params=list(signature.parameters.values())
    if not params or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return []
    rest=params[1:]
    knobs=[]
    for p in rest:
        if p.name in ("n","size","window_size","length","count"):
            knobs.append((p.name,2))
        elif p.default is inspect.Parameter.empty:
            return []
    # Materialization is explicit and budgeted, never tacit.
    return [(dict(knobs),False),(dict(knobs),True)]

def execute_candidate(root,c,cases,attempt_budget=2):
    mod=".".join(Path(c["file"]).with_suffix("").parts)
    try:
        f=getattr(importlib.import_module(mod),c["name"])
        variants=adaptations(f)[:attempt_budget]
    except (ImportError,AttributeError,TypeError,ValueError) as exc:
        return [{"verdict":"UNRESOLVED","reason":type(exc).__name__}]
    reports=[]
    for kw,materialize in variants:
        correct=0;start=time.perf_counter()
        try:
            for values in cases:
                out=f(list(values),**kw)
                if materialize:out=list(out)
                if type(out) is not list or out!=ref(list(values)):break
                correct+=1
            result="PASS" if correct==len(cases) else "REFUTED"
        except Exception as exc:
            result="UNRESOLVED";reason=type(exc).__name__
        else:reason=None
        reports.append({"kwargs":kw,"materialize":materialize,"correct":correct,
                        "cases":len(cases),"verdict":result,"reason":reason,
                        "seconds":time.perf_counter()-start})
    return reports

def calibrate(root,out):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    records=scan(root)[:400]
    ranked=sorted(records,key=lambda x:(-score(x),x["file"],x["name"]))
    baseline=equal_info_keyword_baseline(records)
    cases=[tuple(t) for n in range(7) for t in itertools.product(range(3),repeat=n)]
    def evaluate(candidates):
        return [{"name":c["name"],"file":c["file"],"score":score(c),
                 "variants":execute_candidate(root,c,cases)} for c in candidates[:5]]
    candidate=evaluate(ranked)
    baseline_result=evaluate(baseline)
    result={"source_root":str(root),"functions_scanned":len(records),
            "cases":len(cases),"candidate":candidate,"baseline":baseline_result,
            "qualified_candidate":[x for x in candidate if any(v["verdict"]=="PASS" for v in x["variants"])],
            "qualified_baseline":[x for x in baseline_result if any(v["verdict"]=="PASS" for v in x["variants"])],
            "status":"EXPOSED_CALIBRATION_NOT_PROSPECTIVE"}
    (out/"calibration.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();calibrate(a.root,a.out)
