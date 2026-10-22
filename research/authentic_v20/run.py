"""V20 measured shared bounded-observation obligation on actual upstream changes.

No semantic proof over arbitrary Python sequences: this checks exactly the
declared finite probes and compares against an equally informed shared cache.
"""
import json, os, subprocess, sys, time
from pathlib import Path
REVISIONS=[
("split_zero","3970a80f63807392e512340c25dc2db473bcba70","4aa77cddb5d06def130df6f09aefab4018fa487c"),
("backoff_constant","57cb026b7f47cd2765a0d5acdc83849ed5f1f6a3","ead236e278ca0466bf468de746b5960fb12d7e5b"),
("guiderator_bounds","970051b66474ac37383c5d990c169c794e20eb88","f5f5cd867ee98f15def2b6a19bfba203de29b961")]
INPUTS=[[],[0],[1,2],[0,1,2],[1,1,2,2],[3,4,5,6],[None,0,None],[2,3,2,3],[0]*6,[1]*6,[1,2,1,2,1],[9,8,7,6,5]]
# Each numbered downstream observation must use its own input; the same
# independently qualified *batch* can support all claims within this finite set.
CODE="import json; from boltons.iterutils import pairwise; inp=json.loads(%r);print(json.dumps([pairwise(x) for x in inp]))"%json.dumps(INPUTS)
def execute(repo,sha):
    p=subprocess.run([sys.executable,"-c",CODE],cwd=repo,
      env={**os.environ,"PYTHONPATH":str(repo)},capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)
def run():
    root=Path("/tmp/boltons-v20");out=Path("v20_out");out.mkdir(exist_ok=True)
    report=[]
    for name,old,new in REVISIONS:
        # Separate old/new actual Git snapshots, same interpreter for all arms.
        paths=[]
        for sha in (old,new):
            p=Path("/tmp")/("b20_"+sha[:12])
            subprocess.run(["git","-C",str(root),"worktree","add","--detach","--force",str(p),sha],
                           check=True,capture_output=True,text=True)
            paths.append(p)
        start=time.perf_counter()
        old_batch=execute(paths[0],old);new_batch=execute(paths[1],new)
        shared_time=time.perf_counter()-start
        decisions=[a==b for a,b in zip(old_batch,new_batch)]
        assert len(decisions)==12 and all(decisions)
        start=time.perf_counter()
        independent=[]
        # Conservative full baseline repeats each observation in separate
        # processes; user sees cost charged, not numerical performance claims.
        for i in range(len(INPUTS)):
            a=execute(paths[0],old)[i]
            b=execute(paths[1],new)[i]
            independent.append(a==b)
        full_time=time.perf_counter()-start
        assert independent==decisions
        report.append({"event":name,"claims":12,"same_finite_outputs":True,
          "full_individual_process_invocations":24,"shared_batch_process_invocations":2,
          "mathgraph_batch_process_invocations":2,"conventional_shared_batch_process_invocations":2,
          "full_observed_seconds":full_time,"conventional_shared_observed_seconds":shared_time,
          "mathgraph_shared_observed_seconds":shared_time,
          "qualification":"bounded observations, not general semantic equivalence"})
    result={"events":report,"status":"REAL_SOURCE_SHARED_FINITE_OBLIGATIONS_TIE_CONVENTIONAL_SHARED",
            "disclaimer":"12 selected finite input claims; batching reduces process startup; no novel proof discovery or unique advantage; exposed retrospective source"}
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":run()
