"""Untrusted coordinate proposals behind an exhaustive, independent verifier.

Uses the existing latent_law.coordinates synthesis implementation. Proposal
selection observes only acquisition examples. Checker sees a disjoint exhaustive
finite domain, never exposes labels to the constructor during proposal.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from latent_law.coordinates import synthesize_coordinates

def obs(world):
    a,b,c=world
    return {"a":a,"q":a ^ b,"c":c}
def worlds():
    return [(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)]
def label(world):return world[1]
@dataclass(frozen=True)
class Proposal:
    column: str
    threshold: float
    positive: bool
    grammar: str = "latent_law.coordinates.synthesize_coordinates"
def propose(rows):
    """Only acquisition inputs; no verifier internals or held-out targets."""
    data=pd.DataFrame([r[0] for r in rows])
    made=synthesize_coordinates(data,targets=[],max_base_features=3,max_new_features=80,
                                second_order=False,include_pca=False)
    best=None
    for col in made.columns:
        v=made[col].to_numpy(float)
        if not np.isfinite(v).all():continue
        values=sorted(set(v))
        cuts=[(x+y)/2 for x,y in zip(values,values[1:])]
        for threshold in cuts:
            for positive in (True,False):
                score=sum(((x>threshold)==positive)==int(y) for x,y in zip(v,(r[1] for r in rows)))
                candidate=(score,col,threshold,positive)
                if best is None or score>best[0]:best=candidate
    return None if best is None else Proposal(best[1],float(best[2]),bool(best[3]))
def _replay_feature(column, observed):
    """Replay supported syntactic coordinate families without batch refitting."""
    import math
    if column in observed:
        return float(observed[column])
    unary = {
        "coord_abs__": abs,
        "coord_square__": lambda x: x*x,
        "coord_sqrt__": lambda x: math.sqrt(max(0.0,x)),
        "coord_log__": lambda x: math.log1p(max(0.0,x)),
    }
    for prefix,fn in unary.items():
        if column.startswith(prefix):
            key=column[len(prefix):]
            if key in observed:return fn(float(observed[key]))
    operations = (
        ("coord_diff__", "__minus__", lambda a,b:a-b),
        ("coord_sum__", "__plus__", lambda a,b:a+b),
        ("coord_mul__", "__times__", lambda a,b:a*b),
        ("coord_ratio__", "__over__", lambda a,b:a/b if b else 0.0),
    )
    for prefix,separator,fn in operations:
        if column.startswith(prefix):
            parts=column[len(prefix):].split(separator)
            if len(parts)==2 and all(p in observed for p in parts):
                return fn(float(observed[parts[0]]),float(observed[parts[1]]))
    raise ValueError("unsupported coordinate replay: "+column)

def apply(proposal, observed):
    value=_replay_feature(proposal.column,observed)
    return int((value>proposal.threshold)==proposal.positive)

def apply_batch(proposal, observations):
    try:
        return [apply(proposal,observation) for observation in observations]
    except (ValueError,KeyError,TypeError,ZeroDivisionError,OverflowError):
        return None
def verify(proposal, cases):
    if proposal is None:return False
    predictions=apply_batch(proposal,[obs(w) for w in cases])
    return predictions is not None and predictions==[label(w) for w in cases]
def run():
    domain=worlds()
    # Acquisition is intentionally only 4 worlds; verification sees all 8.
    acquisition=[w for w in domain if w[2]==0]
    proposal=propose([(obs(w),label(w)) for w in acquisition])
    accepted=verify(proposal,domain)
    return {"proposal":str(proposal),"admitted":accepted,"acquisition":len(acquisition),
      "verifier_worlds":len(domain),"boundary":"fixed algebraic proposal grammar; finite exhaustive verifier"}
if __name__=="__main__":
    import json
    print(json.dumps(run(),indent=2))
