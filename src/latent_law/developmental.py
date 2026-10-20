"""Finite consequence-driven capability admission (bounded reference implementation).

The constructor receives labelled acquisition cases and a predeclared XOR grammar.
No claim of autonomous invention outside this grammar. The independent checker
certifies every admitted rule on the complete eight-world acquisition universe.
"""
from dataclasses import dataclass
from itertools import product, combinations

WORLDS = tuple(dict(a=a,b=b,c=c) for a,b,c in product((0,1), repeat=3))
OBSERVATIONS = ("a","c","q")
def observe(w): return dict(a=w["a"],c=w["c"],q=w["a"] ^ w["b"])
def target_action(w): return w["b"]
def old_rules():
    return tuple((col,flip) for col in OBSERVATIONS for flip in (False,True))
def rule_eval(rule, obs):
    left,right=rule
    return (obs[left] ^ obs[right]) if right is not None else obs[left]
def constructor(acquisition):
    """Enumerate supplied grammar, with no hidden-world access."""
    for left,right in combinations(OBSERVATIONS,2):
        rule=(left,right)
        if all(rule_eval(rule,o)==answer for o,answer in acquisition):
            return rule
    return None
def certify(rule, worlds=WORLDS):
    return all(rule_eval(rule,observe(w))==target_action(w) for w in worlds)
def old_closure_obstruction(worlds=WORLDS):
    obs=[observe(w) for w in worlds]
    labels=[target_action(w) for w in worlds]
    # The installed old grammar permits only individual observations and negations.
    return not any(all((o[col] ^ int(flip))==y for o,y in zip(obs,labels))
                       for col,flip in old_rules())
@dataclass(frozen=True)
class Capability:
    rule: tuple
    scope: str
    parents: tuple
    witnessed: bool
class Ledger:
    def __init__(self): self.records={}
    def admit(self,key,rule,scope,parents=()):
        if not certify(rule): return False
        if not all(p in self.records for p in parents): return False
        self.records[key]=Capability(rule,scope,tuple(parents),True)
        return True
    def revoke(self,key):
        to_remove={key}
        changed=True
        while changed:
            changed=False
            for k,v in self.records.items():
                if k not in to_remove and any(p in to_remove for p in v.parents):
                    to_remove.add(k);changed=True
        for k in to_remove:self.records.pop(k,None)
        return to_remove
    def act(self,key,observed,scope):
        cap=self.records.get(key)
        if cap is None or cap.scope != scope:return None
        return rule_eval(cap.rule,observed)
def experiment():
    assert old_closure_obstruction(), "do not expand when old closure suffices"
    training=tuple((observe(w),target_action(w)) for w in WORLDS)
    proposal=constructor(training)
    ledger=Ledger()
    admitted=proposal is not None and ledger.admit("derived",proposal,"heldout")
    # Distinct presentation and order, with no access to the hidden target at invocation.
    transfer=tuple(reversed(WORLDS))
    predicted=[ledger.act("derived",observe(w),"heldout") for w in transfer]
    correct=sum(p==target_action(w) for p,w in zip(predicted,transfer))
    removed=ledger.revoke("derived")
    ablated=[ledger.act("derived",observe(w),"heldout") for w in transfer]
    return dict(old_language_obstructed=True,proposed=str(proposal),
                independently_certified=admitted,correct=correct,total=len(transfer),
                ablation_denies_all=all(p is None for p in ablated),revoked=sorted(removed),
                claim_boundary="finite exhaustive worlds; supplied XOR constructor grammar")
if __name__=="__main__":
    import json
    print(json.dumps(experiment(),indent=2))
