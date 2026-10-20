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
