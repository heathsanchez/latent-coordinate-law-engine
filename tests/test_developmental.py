from latent_law.developmental import (
 WORLDS,observe,target_action,constructor,certify,old_closure_obstruction,Ledger,experiment
)
def test_old_closure_proof_is_exhaustive():
    assert len(WORLDS)==8 and old_closure_obstruction()
def test_proposed_candidate_checked_independently():
    rows=[(observe(w),target_action(w)) for w in WORLDS]
    rule=constructor(rows)
    assert rule is not None and certify(rule)
    assert not certify(("a","c"))
def test_scope_and_dependency_revocation():
    rows=[(observe(w),target_action(w)) for w in WORLDS]
    ledger=Ledger()
    assert ledger.admit("base",constructor(rows),"heldout")
    assert ledger.admit("child",constructor(rows),"heldout",parents=("base",))
    assert ledger.act("base",observe(WORLDS[0]),"other") is None
    assert ledger.revoke("base")=={"base","child"}
    assert not ledger.records
def test_finite_transfer_and_ablation():
    r=experiment()
    assert r["independently_certified"]
    assert (r["correct"],r["total"])==(8,8)
    assert r["ablation_denies_all"]
