"""V18: evaluate real boltons parent/head revisions, preserving cost/accounting honesty."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
REVISIONS=[
("split_zero","3970a80f63807392e512340c25dc2db473bcba70","4aa77cddb5d06def130df6f09aefab4018fa487c","split_iter"),
("backoff_constant","57cb026b7f47cd2765a0d5acdc83849ed5f1f6a3","ead236e278ca0466bf468de746b5960fb12d7e5b","backoff_iter"),
("guiderator_bounds","970051b66474ac37383c5d990c169c794e20eb88","f5f5cd867ee98f15def2b6a19bfba203de29b961","GUIDerator")
]
def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True)
def body(src,symbol):
    tree=ast.parse(src)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name==symbol:
            return ast.dump(node,include_attributes=False)
    raise ValueError("missing symbol "+symbol)
def probe(repo,sha,case):
    # Use actual upstream checkout and interpreter. For matching author tests,
    # subprocess invocation also isolates module import caches between revisions.
    checkout=repo/("wt_"+sha[:12])
    checkout.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","-C",str(repo),"worktree","add","--detach","--force",str(checkout),sha],
       check=True,capture_output=True,text=True)
    script={
      "pairwise":"from boltons.iterutils import pairwise;print(repr(pairwise([1,2,3,4])))",
      "split_zero":"from boltons.iterutils import split;print(repr(split(iter([1,None,2]),maxsplit=0)))",
      "backoff_constant":"from boltons.iterutils import backoff;\ntry: print(repr(backoff(5,5,factor=1.0)))\nexcept Exception as e: print(type(e).__name__+':'+str(e))",
      "guiderator_bounds":"from boltons.iterutils import GUIDerator;\ntry: GUIDerator(size=19);print('accepted')\nexcept Exception as e: print(type(e).__name__+':'+str(e))"
    }
    outputs={}
    for name in ["pairwise",case]:
        x=subprocess.run([sys.executable,"-c",script[name]],cwd=checkout,env={**os.environ,"PYTHONPATH":str(checkout)},capture_output=True,text=True,timeout=30)
        if x.returncode:raise RuntimeError("%s %s %s"%(sha,name,x.stderr))
        outputs[name]=x.stdout.strip()
    return outputs
def run():
    repo=Path("/tmp/boltons-v18").resolve()
    out=Path("v18_out");out.mkdir(exist_ok=True)
    results=[]
    for name,old,new,changed in REVISIONS:
        a=time.perf_counter()
        src_old=git(repo,"show",old+":boltons/iterutils.py")
        src_new=git(repo,"show",new+":boltons/iterutils.py")
        # AST identity is only a candidate for preservation, never enough for
        # source semantic equivalence without stable dependency assumptions.
        unchanged_pairwise=(body(src_old,"pairwise")==body(src_new,"pairwise"))
        changed_ast=(body(src_old,changed)!=body(src_new,changed))
        b=time.perf_counter()
        old_outputs=probe(repo,old,name);new_outputs=probe(repo,new,name)
        c=time.perf_counter()
        rec={"event":name,"old":old,"new":new,"changed_symbol":changed,
             "pairwise_ast_identical":unchanged_pairwise,"edited_symbol_ast_changed":changed_ast,
             "pairwise_native_equal":old_outputs["pairwise"]==new_outputs["pairwise"],
             "edited_native_equal":old_outputs[name]==new_outputs[name],
             "old_observations":old_outputs,"new_observations":new_outputs,
             "ast_check_seconds":b-a,"native_probe_seconds":c-b,
             "conventional_function_ast_cache_candidate":unchanged_pairwise,
             "mathgraph_preservation_candidate":unchanged_pairwise,
             "qualifier":"function AST identity is not complete transitive semantic proof"}
        results.append(rec)
    assert all(x["pairwise_ast_identical"] and x["pairwise_native_equal"] for x in results)
    assert all(x["edited_symbol_ast_changed"] for x in results)
    assert any(not x["edited_native_equal"] for x in results)
    report={"experiments":results,"status":"REAL_COMMITS_BOUNDED_CALIBRATION_NO_UNIQUE_ADVANTAGE",
            "limitations":"source commits previously inspected; AST equivalence conditional on dependencies; short behavior probes; no full semantics or strong incremental verifier speed-up established"}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":run()
