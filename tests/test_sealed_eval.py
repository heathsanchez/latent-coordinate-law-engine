import numpy as np
import pandas as pd
import pytest
from latent_law.sealed_eval import candidates, sealed_evaluate

def test_target_is_excluded():
    df=pd.DataFrame({'a':[0,0,1,1]*25,'b':[0,1,0,1]*25,'y':[0,1,1,0]*25})
    with pytest.raises(ValueError):
        sealed_evaluate(df,'y',['a','y'])

def test_candidates_are_label_free_and_deterministic():
    rng=np.random.default_rng(4)
    df=pd.DataFrame({'a':rng.normal(size=200),'b':rng.normal(size=200)})
    assert candidates(df,['a','b']).equals(candidates(df.iloc[::-1].iloc[::-1],['a','b']))

def test_xor_pair_recovery():
    rng=np.random.default_rng(7)
    a=rng.integers(0,2,4000); b=rng.integers(0,2,4000)
    df=pd.DataFrame({'a':a,'xor':a^b,'y':b})
    result=sealed_evaluate(df,'y',['a','xor'])
    assert result['test_accuracy']>.98, result

def test_random_target_negative_control():
    rng=np.random.default_rng(17)
    df=pd.DataFrame({'a':rng.normal(size=1000),'b':rng.normal(size=1000),'y':rng.integers(0,2,1000)})
    result=sealed_evaluate(df,'y',['a','b'])
    assert result['test_accuracy']<.65, result
