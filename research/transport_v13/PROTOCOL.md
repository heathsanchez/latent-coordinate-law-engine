# V13 capability transport protocol (pre-source inspection)
2026-10-21. Frozen before fetching target code.
Source A: more-itertools at 4aeb4338ae1407d8da9723e86da99efeb446f719 (exposed calibration).
Source B: boltons at a source commit recorded by the runner (previously exposed in V7–V9; NOT unopened).
Protected behavior: materialized list of adjacent ordered pairs, no terminal sentinel, on finite sequences. 
Retained object is a semantic contract plus known source A invocation recipe (sliding_window with size 2 and list materialization), NOT a source-B import path.
Transfer attempt: apply frozen recipe to B by a generic source-scan search. If a B function name must be searched again, count that as reconstruction of binding, not zero-search transfer.
Search baseline: equal-information complete source scan with same bounded adapter grammar; charge all attempts and elapsed time.
No source name hints in target selection. Verifier uses 1,093 exact ternary sequences.
Negative control: a mismatched window width or sentinel must fail.
Qualified transfer requires source-B binding without semantic search and complete scope verification. Otherwise label PARTIAL/UNKNOWN.
