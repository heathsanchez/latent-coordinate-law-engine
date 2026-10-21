"""V16 cumulative reuse: identical authority and correctness gates for each arm.

Descriptive Python timing on synthetic tasks. Not a proof of net advantage.
"""
import itertools,json,time,hashlib
from pathlib import Path

def spec(seq):return list(zip(seq,seq[1:]))
def implementation(seq):return list(zip(seq,seq[1:]))
def verify_full(fn):
    return all(fn(list(xs))==spec(list(xs)) for n in range(7) for xs in itertools.product(range(3),repeat=n))
def verify_boundary(seq,authority,expected):
    return authority==expected and isinstance(seq,list) and len(seq)<=12
def tasks():
    return [list(x) for n in range(1,9) for x in itertools.islice(itertools.product(range(3),repeat=n),0,8)]
def run(out="cumulative_v16_out"):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    inputs=tasks()
    modes=("cold","conventional_cache","retained_full","retained_boundary")
    events=[]
    for mode in modes:
        stamp="source-v1";authorized=True;cached=False;proof=False
        for i,x in enumerate(inputs):
            if i==25: stamp="source-v2";cached=False;proof=False
            if i==40: authorized=False;cached=False;proof=False
            if i==44: authorized=True;stamp="source-v3"
            start=time.perf_counter()
            full_checks=0;boundary_checks=0;acquisitions=0
            if mode=="cold":
                full_checks=1;acquisitions=1
                valid=verify_full(implementation)
            elif mode=="conventional_cache":
                if not cached:
                    full_checks=1;acquisitions=1
                    cached=verify_full(implementation)
                valid=cached
            elif mode=="retained_full":
                full_checks=1
                valid=verify_full(implementation)
            else:
                if not proof:
                    full_checks=1
                    proof=verify_full(implementation)
                valid=proof
            boundary_checks=1
            eligible=valid and verify_boundary(x,"trusted" if authorized else "withdrawn","trusted")
            result=implementation(x) if eligible else None
            elapsed=time.perf_counter()-start
            events.append(dict(mode=mode,index=i,source=stamp,authorized=authorized,
                full_checks=full_checks,boundary_checks=boundary_checks,acquisitions=acquisitions,
                eligible=eligible,correct=(result==spec(x) if eligible else result is None),
                seconds=elapsed))
    summary={}
    for mode in modes:
        rs=[r for r in events if r["mode"]==mode]
        summary[mode]=dict(tasks=len(rs),full_checks=sum(r["full_checks"] for r in rs),
            acquisitions=sum(r["acquisitions"] for r in rs),
            boundary_checks=sum(r["boundary_checks"] for r in rs),
            elapsed_seconds=sum(r["seconds"] for r in rs),
            permitted=sum(r["eligible"] for r in rs),
            correct=sum(r["correct"] for r in rs))
    report=dict(summary=summary,limitations="generated tasks; shared hand-authored implementation; source-version simulation; no separately qualified source correspondence",status="CALIBRATION_NOT_PROSPECTIVE_NO_ADVANTAGE_CLAIM")
    (p/"events.json").write_text(json.dumps(events,indent=2))
    (p/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":run()
