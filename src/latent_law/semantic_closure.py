"""Bounded typed arithmetic closure with semantic deduplication.

Search is restricted to supplied integer constants, observed variables,
unary negation, addition, subtraction, multiplication. No XOR/modulo primitive.
Semantic keys are exact integer output vectors on the acquisition domain;
the independent evaluator checks candidates on the separately declared scope.
"""
from dataclasses import dataclass
from itertools import product
@dataclass(frozen=True)
class Expr:
    src: str
    values: tuple
    size: int
    def __str__(self): return self.src

def domain():
    return tuple((a,q,c) for a,q,c in product((0,1),repeat=3))
def objective(row):return row[0]^row[1]
def enumerate_closure(rows, max_size=7):
    rows=tuple(rows)
    levels={}
    seen={}
    for name,values in [('a',tuple(x[0] for x in rows)),('q',tuple(x[1] for x in rows)),
                        ('c',tuple(x[2] for x in rows)),
                        ('0',(0,)*len(rows)),('1',(1,)*len(rows)),('2',(2,)*len(rows))]:
        seen.setdefault(values,Expr(name,values,1))
    levels[1]=list(seen.values())
    tested=0
    for size in range(2,max_size+1):
        candidates=[]
        for left_size in range(1,size-1):
            right_size=size-1-left_size
            for l in levels.get(left_size,[]):
                for r in levels.get(right_size,[]):
                    for symbol,op in [('+',lambda a,b:a+b),('-',lambda a,b:a-b),('*',lambda a,b:a*b)]:
                        if symbol in ('+','*') and l.src>r.src:continue
                        vals=tuple(op(a,b) for a,b in zip(l.values,r.values))
                        tested+=1
                        if vals in seen or max(map(abs,vals),default=0)>16:continue
                        candidates.append(Expr('('+l.src+symbol+r.src+')',vals,size))
        # Within-level signatures and previously seen semantics are collapsed.
        fresh={}
        for e in candidates:
            if e.values not in fresh or e.src<fresh[e.values].src:fresh[e.values]=e
        levels[size]=list(fresh.values())
        seen.update(fresh)
    return seen,tested

def propose(rows,labels,max_size=7):
    seen,tested=enumerate_closure(rows,max_size)
    goal=tuple(labels)
    winner=seen.get(goal)
    return winner,{"unique_semantics":len(seen),"syntactic_candidates":tested,
                   "max_size":max_size,"grammar":"integer {0,1,2}, a,q,c, +,-,*"}
def eval_expr(expr,row):
    # Independent structural evaluator rather than replay from stored training vectors.
    import ast
    env={'a':row[0],'q':row[1],'c':row[2]}
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Constant) and type(n.value)==int:return n.value
        if isinstance(n,ast.Name) and n.id in env:return env[n.id]
        if isinstance(n,ast.BinOp) and type(n.op) in (ast.Add,ast.Sub,ast.Mult):
            a,b=visit(n.left),visit(n.right)
            return a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b
        raise ValueError("outside frozen grammar")
    return visit(ast.parse(expr.src,mode='eval'))
def certify(expr,rows,goal):
    return expr is not None and all(eval_expr(expr,r)==goal(r) for r in rows)
def run():
    rows=domain()
    # Closure synthesis uses only acquisition rows, while verifier independently checks full domain.
    acquisition=tuple(r for r in rows if r[2]==0)
    expr,counts=propose(acquisition,[objective(r) for r in acquisition])
    accepted=certify(expr,rows,objective)
    return {"expression":str(expr) if expr else None,"accepted":accepted,
            "verified_worlds":len(rows),"acquisition_worlds":len(acquisition),**counts,
            "boundary":"finite Boolean input domain; supplied arithmetic grammar"}
if __name__=="__main__":
    import json
    print(json.dumps(run(),indent=2))
