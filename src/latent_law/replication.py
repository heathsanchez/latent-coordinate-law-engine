"""Independent-seed coordinate replication; does not modify original A++ scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from latent_law.sealed_eval import candidates
from latent_law.challenge import generate_feynman_like,generate_gas_law_like,generate_ca_entropy_probe,generate_phase_probe,_feature_columns,_target

def one(probe, seed):
    frame=probe.data.drop(columns=[probe.hidden_coordinate])
    target=_target(frame,probe.targets).reset_index(drop=True)
    cols=[c for c in _feature_columns(frame,probe.targets) if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    x=frame[cols].reset_index(drop=True)
    ix,jx=train_test_split(np.arange(len(x)),test_size=.25,stratify=target,random_state=seed)
    tr,va=train_test_split(ix,test_size=.3,stratify=target.iloc[ix],random_state=seed+101)
    expanded=candidates(x,cols).replace([np.inf,-np.inf],np.nan).fillna(0)
    names=list(expanded.columns)
    from itertools import combinations
    # Precommitted exhaustive single and pair search, no test-label use.
    groups=[(c,) for c in names]+list(combinations(names,2))
    def tree(features,train,depth=3):
        m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=2,random_state=seed)
        m.fit(expanded.iloc[train][list(features)],target.iloc[train])
        return m
    scores=[]
    for g in groups:
        model=tree(g,tr)
        scores.append((accuracy_score(target.iloc[va],model.predict(expanded.iloc[va][list(g)])),g))
    best=max(scores,key=lambda item:(item[0],-len(item[1]),item[1]))[1]
    selected=tree(best,ix)
    raw=tree(cols,ix)
    forest=RandomForestClassifier(n_estimators=120,min_samples_leaf=2,random_state=seed,n_jobs=1)
    forest.fit(x.iloc[ix],target.iloc[ix])
    def acc(pred):return float(accuracy_score(target.iloc[jx],pred))
    return dict(domain=probe.name,seed=seed,n_train=len(ix),n_test=len(jx),chosen=";".join(best),
                coordinate=acc(selected.predict(expanded.iloc[jx][list(best)])),
                raw_tree=acc(raw.predict(expanded.iloc[jx][cols])),
                raw_forest=acc(forest.predict(x.iloc[jx])),
                candidate_count=len(groups))

def run(out="replication_out", seeds=(41,42,43,44,45)):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        probes=[generate_feynman_like(seed=seed+10),generate_gas_law_like(seed=seed+20),
                generate_ca_entropy_probe(seed=seed+30),generate_phase_probe(seed=seed+40)]
        for probe in probes: rows.append(one(probe,seed))
    df=pd.DataFrame(rows)
    df.to_csv(p/"results.csv",index=False)
    report={"n":len(df),"seeds":list(seeds),"domains":sorted(df.domain.unique()),
            "mean_coordinate":float(df.coordinate.mean()),
            "mean_raw_tree":float(df.raw_tree.mean()),
            "mean_raw_forest":float(df.raw_forest.mean()),
            "wins_raw_tree":int((df.coordinate>df.raw_tree).sum()),
            "wins_raw_forest":int((df.coordinate>df.raw_forest).sum()),
            "mean_delta_tree":float((df.coordinate-df.raw_tree).mean()),
            "mean_delta_forest":float((df.coordinate-df.raw_forest).mean()),
            "status":"BOUNDED_GENERATED_PROBES_NOT_NATURAL_TRANSFER"}
    (p/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    run()
