"""V10 frozen generic native-contract selector; no source-specific code."""
import argparse, ast, importlib, json, os, re, subprocess, sys, time
from pathlib import Path
from itertools import product

OBLIGATION = "materialized list of contiguous adjacent pairs, two elements each, preserve order, no terminal fill"
def ref(values): return list(zip(values,values[1:]))
def scan(root):
    result=[]
    for file in sorted(root.rglob("*.py")):
        rel=file.relative_to(root)
        if any(x in ("tests","test","vendor","docs","build",".tox") for x in rel.parts):continue
        try: tree=ast.parse(file.read_text("utf8"))
        except (SyntaxError,UnicodeError):continue
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                doc=ast.get_docstring(node) or ""
                args=[a.arg for a in node.args.args]
                returns=ast.unparse(node)[:900]
                result.append(dict(name=node.name,file=str(rel),args=args,doc=doc,body=returns))
    return result

def score(c):
    doc=c["doc"].lower()
    name=c["name"].lower()
    terms={"adjacent":4,"consecutive":4,"pair":3,"sliding":3,"window":2,"tuple":1,"iterable":1,"list":1}
    v=sum(w for term,w in terms.items() if term in doc)
    # Distinguish list-return contract and lazy iterator as relevant to protected output type.
    if "returns a list" in doc or "return a list" in doc:v+=4
    if "yield" in c["body"] and "return list(" not in c["body"]:v-=2
    if "iter" in name:v-=1
    return v
def equal_info_keyword_baseline(records):
    # Equal docs/signatures but no return-body analysis.
    def b(c):
        d=c["doc"].lower()
        return sum(w for token,w in [("adjacent",4),("consecutive",4),("pair",3),("sliding",3),("window",2),("tuple",1),("iterable",1),("list",1)] if token in d)
    return sorted(records,key=lambda c:(-b(c),c["file"],c["name"]))[:5]

def execute(root,c,tests):
    parts=Path(c["file"]).with_suffix("").parts
    mod=".".join(parts)
    # Only require positional input and a defaulted or explicit 'size' when present.
    try:
        module=importlib.import_module(mod)
        f=getattr(module,c["name"])
        if not callable(f):return "UNRESOLVED",None
        args=c["args"]
        kw={}
        if "size" in args:kw["size"]=2
        for values in tests:
            observed=f(list(values),**kw)
            if not isinstance(observed,list) or observed!=ref(list(values)):
                return "REFUTED",repr((values,observed))[:250]
        return "PASS",None
    except Exception as e:
        return "UNRESOLVED",type(e).__name__+":"+str(e)[:150]

def run(root,out):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    records=scan(root)[:400]
    ranking=sorted(records,key=lambda c:(-score(c),c["file"],c["name"]))
    tselect=time.monotonic()-start
    tests=[tuple(x) for n in range(7) for x in product(range(3),repeat=n)]
    verified=[]
    for c in ranking[:5]:
        start=time.monotonic()
        verdict,note=execute(root,c,tests)
        verified.append(dict(name=c["name"],file=c["file"],score=score(c),verdict=verdict,note=note,seconds=time.monotonic()-start))
    baseline=[dict(name=c["name"],file=c["file"]) for c in equal_info_keyword_baseline(records)]
    upstream=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True)
    report=dict(source_commit=upstream.stdout.strip(),source_function_count=len(records),
                top_candidates=[dict(name=c["name"],file=c["file"],score=score(c)) for c in ranking[:5]],
                evaluated=verified,keyword_baseline=baseline,
                selection_seconds=tselect,holdout_count=len(tests),
                provisional_verified=[r for r in verified if r["verdict"]=="PASS"],
                native_tests_status="NOT_RUN",
                claim="PROVISIONAL_ONLY_UNTIL_UPSTREAM_TEST_SUITE_RUNS")
    (out/"selection.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    run(a.root,a.out)
