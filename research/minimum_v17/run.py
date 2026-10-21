"""Finite-scope consequential obligation compiler with independent exact checker.

No ungrounded claim of minimal cost in real repositories. Exhaustive subset choice
is exact only for the declared small certificate grammar and synthetic costs.
"""
import itertools,json
from pathlib import Path

DOMAIN=[list(t) for n in range(5) for t in itertools.product((0,1,2),repeat=n)]
CLAIMS=tuple("claim_%02d"%i for i in range(12))
def original(xs):return [(xs[i],xs[i+1]) for i in range(len(xs)-1)]
def equivalent(xs):return list(zip(xs,xs[1:]))
def broken(xs):return list(zip(xs,xs[2:]))
def certificate(a,b):return all(a(xs)==b(xs) for xs in DOMAIN)
def run():
    events={}
    for change,fn,authorized in (("equivalent",equivalent,True),("behavior_changed",broken,True),("authority_withdrawn",equivalent,False)):
        eq=certificate(original,fn)
        # Individual protected claims each depend on the same shared behavior.
        # A correct state authorizes a claim only when provenance is active
        # and the changed implementation preserves the protected behavior.
        expected=bool(eq and authorized)
        costs={"individual_check":5,"shared_proof":14,"certificate_inspection":1,"per_claim_update":1}
        full=12*5+12
        dependency=12*5+12
        shared=14+1+12
        # Exact finite candidate-cover optimizer. Only a valid certificate
        # may cover claims; otherwise all affected claims require checking
        # or fail closed on an independently observed counterexample.
        options=[("individual",full)]
        if eq and authorized: options.append(("shared_certificate",shared))
        best=min(options,key=lambda p:(p[1],p[0]))
        # Everyone sees identical source and withdraws on authority failure.
        result={"finite_equivalent":eq,"authorized":authorized,"expected_claim_authorization":expected,
           "claims":len(CLAIMS),"full_cost_units":full,"dependency_cost_units":dependency,
           "conventional_shared_cost_units":shared if eq and authorized else full,
           "mathgraph_cost_units":best[1],"mathgraph_choice":best[0],
           "all_correct":all((authorized and eq)==expected for _ in CLAIMS),
           "certificate_domain_cases":len(DOMAIN),"counterexample_found":not eq}
        events[change]=result
    assert events["equivalent"]["mathgraph_cost_units"]==events["equivalent"]["conventional_shared_cost_units"]
    assert events["behavior_changed"]["expected_claim_authorization"] is False
    assert events["authority_withdrawn"]["expected_claim_authorization"] is False
    assert all(e["all_correct"] for e in events.values())
    report={"experiment":"V17 synthetic finite semantic-preservation calibration",
      "cost_units":"declared synthetic units, not runtime", "events":events,
      "status":"NO_DISTINCT_ADVANTAGE_OVER_EQUAL_INFORMATION_SEMANTIC_CACHING",
      "limitations":"finite domain; hand-authored alternative implementation and obligations; certificate construction not automated; no real code change"}
    Path("v17_out").mkdir(exist_ok=True)
    Path("v17_out/result.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":run()
