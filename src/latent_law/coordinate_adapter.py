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
def apply(proposal, observed):
    df=synthesize_coordinates(pd.DataFrame([observed]),targets=[],
         max_base_features=3,max_new_features=80,second_order=False,include_pca=False)
    # One-row generation drops constant features; use domain-fixed batch evaluator instead.
    raise RuntimeError("Use apply_batch with stable feature schema")
def apply_batch(proposal, observations):
    data=pd.DataFrame(observations)
    made=synthesize_coordinates(data,targets=[],max_base_features=3,max_new_features=80,
                                second_order=False,include_pca=False)
    if proposal.column not in made:return None
    return [int(((x>proposal.threshold)==proposal.positive)) for x in made[proposal.column]]
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
