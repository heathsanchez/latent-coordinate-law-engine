from latent_law.semantic_closure import domain,objective,propose,certify,run,Expr,enumerate_closure

def test_xor_recovered_from_arithmetic_without_xor_primitive():
    rows=domain()
    acquire=[r for r in rows if r[2]==0]
    expr,counts=propose(acquire,[objective(r) for r in acquire])
    assert expr is not None
    assert certify(expr,rows,objective)
    assert counts['syntactic_candidates']>=counts['unique_semantics']

def test_nonrepresentable_depth_one_must_not_pass():
    rows=domain()
    expr,_=propose(rows,[objective(r) for r in rows],max_size=1)
    assert expr is None

def test_wrong_expression_rejected_by_independent_semantics():
    assert not certify(Expr("a",(0,)*8,1),domain(),objective)

def test_result_reports_full_scope():
    result=run()
    assert result['accepted'] and result['verified_worlds']==8
