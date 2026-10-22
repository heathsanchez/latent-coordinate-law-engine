"""Independent behavioral verifier. Exhaustive bounded inputs, not theorem proof."""
import importlib,itertools,json,sys
from pathlib import Path
def expected(xs):return list(zip(xs,xs[1:]))
def check(root,c):
    sys.path.insert(0,str(root))
    try:
        module=".".join(Path(c["file"]).with_suffix("").parts)
        fn=getattr(importlib.import_module(module),c["name"])
        total=0
        for n in range(7):
            for t in itertools.product((0,1,2),repeat=n):
                xs=list(t);result=fn(xs,**c["kwargs"])
                if c["materialize"]:result=list(result)
                if type(result)!=list or result!=expected(xs):return {"pass":False,"checked":total}
                total+=1
        return {"pass":True,"checked":total}
    except Exception as e:return {"pass":False,"error":str(e)}
def run():
    p=Path("v21_out");d=json.loads((p/"discovery.json").read_text())
    evidence={}
    for key,root in (("a",Path("/tmp/more-v21")),("b",Path("/tmp/boltons-v21"))):
        options=d["source_"+key+"_candidates"]
        checks=[{"candidate":c,"result":check(root,c)} for c in options]
        qualified=[x for x in checks if x["result"]["pass"]]
        evidence[key]={"checked_candidates":len(checks),"qualified":qualified}
    assert evidence["a"]["qualified"] and evidence["b"]["qualified"],"No cross-source verified correspondence"
    result={"evidence":evidence,"cross_source_obligation":"same finite adjacent-pair behavior",
       "conventional_semantic_baseline":"can find identical matches with equal evidence and verifier",
       "unique_advantage":False,"source_dependent_ast_only":"does not directly connect unrelated implementations",
       "status":"BOUNDED_CROSS_IMPLEMENTATION_EQUIVALENCE_NOT_GENERAL_PROOF"}
    (p/"evidence.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"qualified_a":len(evidence["a"]["qualified"]),"qualified_b":len(evidence["b"]["qualified"]),"status":result["status"]},indent=2))
if __name__=="__main__":run()
