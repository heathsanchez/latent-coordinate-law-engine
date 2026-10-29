"""V21 automatically discover cross-repository finite semantic correspondence."""
import importlib,itertools,json,sys,time
from pathlib import Path
from research.prospective_native_v10.run import scan,score
from research.prospective_native_v11.adapter import adaptations

def oracle(xs):return [(xs[i],xs[i+1]) for i in range(len(xs)-1)]
def function(root,c):
    mod=".".join(Path(c["file"]).with_suffix("").parts)
    return getattr(importlib.import_module(mod),c["name"])
def fingerprints(root,limit=160):
    sys.path.insert(0,str(root))
    records=scan(Path(root))[:400]
    records=sorted(records,key=lambda x:(-score(x),x["file"],x["name"]))[:limit]
    acquisition=[[],[0],[1,2],[0,1,2],[2,2,1,0],[0,2,1,2]]
    candidates=[];attempts=0
    for c in records:
        try:
            f=function(root,c);variants=adaptations(f)
        except Exception:continue
        for kwargs,materialize in variants:
            attempts+=1
            try:
                ok=True
                for xs in acquisition:
                    y=f(list(xs),**kwargs)
                    if materialize:y=list(itertools.islice(y,max(1,len(xs))))
                    if type(y)!=list or y!=oracle(xs):ok=False;break
                if ok:candidates.append(dict(file=c["file"],name=c["name"],kwargs=kwargs,materialize=materialize))
            except Exception:pass
    return candidates,attempts,len(records)
def run():
    out=Path("v21_out");out.mkdir(exist_ok=True)
    ra=Path("/tmp/more-v21");rb=Path("/tmp/boltons-v21")
    start=time.perf_counter()
    a,aa,na=fingerprints(ra);b,ab,nb=fingerprints(rb)
    # Independently check in separate process; never trust the acquisition screen.
    result={"source_a_candidates":a,"source_b_candidates":b,"candidate_attempts":aa+ab,
      "scanned_functions":na+nb,"discovery_elapsed_seconds":time.perf_counter()-start,
      "baseline":"equal-information semantic fingerprinting over same universe",
      "discovery_status":"CANDIDATES_PENDING_INDEPENDENT_CHECK"}
    (out/"discovery.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":run()
