# V21 — Cross-implementation semantic obligation discovery
Frozen protocol, before CI execution. Both upstream projects are previously exposed; retrospective calibration.
Source A: more-itertools pinned 4aeb4338ae1407d8da9723e86da99efeb446f719.
Source B: boltons pinned f5f5cd867ee98f15def2b6a19bfba203de29b961.
Goal: automatically enumerate eligible top-level functions and eligible size/list-materialization adapters, find any A/B pair satisfying the same protected adjacent-pairs consequence across 1093 finite ternary inputs (length 0..6). Never seed matching symbol names. Independent checker in separate script re-executes both discovered functions and compares against separate mathematical oracle. Negative control deliberately tests wrong window size if possible.
Source-native integration baseline: ordinary fingerprint-driven semantic search over exactly the same universe and checker, plus AST-only dependency method. The strong baseline can discover the same relation; do NOT label semantic search a unique MathGraph benefit.
Stop on bounded search budget; no-found = UNKNOWN_BUDGET. Counts and measured time descriptive. No universal Python equivalence or source-authorship novelty claim.
