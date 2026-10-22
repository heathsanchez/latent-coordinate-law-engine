"""V19 conservative module dependency-closed AST warrant candidates.

An unchanged function AST is insufficient when referenced module-level bindings
change. Dynamic calls, imports, attributes and closure references remain UNKNOWN.
This is a syntactic guard, not a Lean proof of semantic preservation.
"""
import ast, json
from pathlib import Path

def module_bindings(tree):
    bindings={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            bindings[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t,ast.Name):bindings[t.id]=ast.dump(node,include_attributes=False)
    return bindings

def closure_status(old,new,symbol):
    a=ast.parse(old);b=ast.parse(new)
    before=module_bindings(a);after=module_bindings(b)
    if symbol not in before or symbol not in after:return {"status":"UNKNOWN_MISSING"}
    queue=[symbol];seen=set();changes=[];external=[]
    while queue:
        name=queue.pop()
        if name in seen:continue
        seen.add(name)
        x=before.get(name);y=after.get(name)
        if x!=y:
            changes.append(name);continue
        # Locate refs in the declaration, deliberately overapproximate.
        for tree in (a,b):
            decl=next((n for n in tree.body if getattr(n,"name",None)==name),None)
            if decl is None:continue
            locals_set=set()\n            if isinstance(decl,(ast.FunctionDef,ast.AsyncFunctionDef)):\n                locals_set.update(a.arg for a in decl.args.posonlyargs+decl.args.args+decl.args.kwonlyargs)\n                if decl.args.vararg:locals_set.add(decl.args.vararg.arg)\n                if decl.args.kwarg:locals_set.add(decl.args.kwarg.arg)\n            for n in ast.walk(decl):
                if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load):
                    if n.id in before or n.id in after:queue.append(n.id)
                    elif n.id not in locals_set and n.id not in ("range","len","list","zip","enumerate","tuple","iter","next","min","max"):
                        external.append(n.id)
    if changes:return {"status":"WITHHOLD_CHANGED_DEPENDENCY","dependencies":sorted(seen),"changed":sorted(set(changes))}
    if external:return {"status":"UNKNOWN_EXTERNAL_OR_LOCAL_REFERENCES","dependencies":sorted(seen),"external":sorted(set(external))}
    return {"status":"CONDITIONAL_AST_CLOSURE_UNCHANGED","dependencies":sorted(seen),
            "qualification":"requires unchanged runtime, imports and ambient state; not proof of semantics"}

def fixture():
    old="OFFSET=1\ndef outcome(x):\n    return x+OFFSET\n"
    new="OFFSET=2\ndef outcome(x):\n    return x+OFFSET\n"
    first=closure_status(old,new,"outcome")
    assert first["status"]=="WITHHOLD_CHANGED_DEPENDENCY",first
    namespace1={};namespace2={}
    exec(old,namespace1);exec(new,namespace2)
    assert namespace1["outcome"](3)!=namespace2["outcome"](3)
    same=closure_status(old,old,"outcome")
    assert same["status"]=="CONDITIONAL_AST_CLOSURE_UNCHANGED",same
    # Regression negative: untouched function body but changed dependency.
    return {"unchanged_function_changed_global":first,"stable_source":same,
            "behavioral_counterexample":[namespace1["outcome"](3),namespace2["outcome"](3)]}

if __name__=="__main__":
    result=fixture();p=Path("v19_out");p.mkdir(exist_ok=True)
    p.joinpath("dependency_closure.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
