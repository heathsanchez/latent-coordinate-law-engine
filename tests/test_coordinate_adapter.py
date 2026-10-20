from latent_law.coordinate_adapter import worlds,obs,label,propose,verify,run
def test_proposed_rule_needs_independent_verification():
    w=worlds()
    candidate=propose([(obs(x),label(x)) for x in w if x[2]==0])
    assert candidate is not None
    assert isinstance(verify(candidate,w),bool)
def test_wrong_target_is_not_automatically_accepted():
    w=worlds()
    candidate=propose([(obs(x),label(x)) for x in w if x[2]==0])
    if verify(candidate,w):
        from latent_law.coordinate_adapter import Proposal
        wrong=Proposal(candidate.column,candidate.threshold,not candidate.positive)
        assert not verify(wrong,w)
def test_report_declares_boundary():
    assert "fixed algebraic" in run()["boundary"]

def test_independent_checker_accepts_replayable_conjunction():
    from latent_law.coordinate_adapter import bounded_admission_demo
    report=bounded_admission_demo()
    assert report["conjunction"]["authorized"], report
    assert report["conjunction"]["correct"]==8

def test_xor_claim_only_when_independently_correct():
    from latent_law.coordinate_adapter import bounded_admission_demo
    report=bounded_admission_demo()
    if report["xor"]["authorized"]:
        assert report["xor"]["correct"]==8

def test_qualified_admission_scope_ablation_and_refusal():
    from latent_law.coordinate_adapter import CoordinateAuthority,Proposal,apply
    domain=worlds()
    objective=lambda w:w[0] & (w[0]^w[1])
    acquisition=[w for w in domain if w[2]==0]
    proposal=propose([(obs(w),objective(w)) for w in acquisition])
    authority=CoordinateAuthority(domain)
    assert authority.admit("conjunction",proposal,"source-A",objective)
    assert all(authority.invoke("conjunction",obs(w),"source-A")==objective(w) for w in domain)
    assert authority.invoke("conjunction",obs(domain[0]),"source-B") is None
    wrong=Proposal(proposal.column,proposal.threshold,not proposal.positive)
    assert not authority.admit("wrong",wrong,"source-A",objective)
    assert authority.revoke("conjunction")
    assert authority.invoke("conjunction",obs(domain[0]),"source-A") is None
