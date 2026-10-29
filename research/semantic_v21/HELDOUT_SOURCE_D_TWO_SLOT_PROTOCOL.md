# V21 two-slot binding: frozen prospective source-D protocol

Status: CANDIDATE / preregistered protocol, **not** a result.

This branch extends the prior source-C trial *after* its independently completed
negative verdict (run 40837544539). It must not change or overwrite that verdict.

## Causal motivation, declared before D inspection

Source C `pytoolz/toolz@451af60dec590a6010e2babdbf391ea8f815122f`
contains `sliding_window(n, seq)`, whose `n=2` application represents
the original adjacent-pair obligation, but the legacy V11 adapter permits
only iterable-first signatures. The new hypothesis is that a single
signature-directed second-input-slot rule accounts for C's miss and
transfers without source-D-specific edits.

C is a **post-hoc mechanism/ablation diagnostic**, never a prospective success.

## Frozen source D (selected before inspecting source code)

`Suor/funcy@a876c7e1cf11e7c746e91fce4632ebfa93865428`, branch `master`
as selected 2026-10-29. No D implementation inspected before this protocol.

## Frozen task and budgets

- Contract: for each finite list, return a **list** of ordered contiguous
  adjacent pairs of the original elements, no final padding.
- Acquisition only: the same six V21 cases `[],[0],[1,2],[0,1,2],[2,2,1,0],[0,2,1,2]`.
- Independent verification, second process: all 1,093 ternary lists
  length 0..6, same reference behavior. No probe leakage.
- Universe: first 400 top-level public AST functions in path order,
  scored by frozen V10 ranking; evaluate the best 160. No name whitelist.
- New binding grammar, globally applied and held fixed:
  1. Legacy iterable-first: first required positional parameter receives
     the list; any known optional size knobs (`n`, `size`,
     `window_size`, `length`, `count`) are set to 2.
  2. Additional size-first: first required positional parameter is one of
     those exact size knobs, set to 2; second required positional parameter
     receives the list **only** if its name is one of
     `seq`, `sequence`, `iterable`, `xs`, `values`, `items`, `it`.
     Any other required parameters make this variant ineligible.
- Both non-materialized and materialized variants are admitted. Every
  materialization is `islice(output,max(1,len(xs)))` before casting to list.
- Independent equal-information comparator: same candidate universe,
  same binding grammar, same acquisition and verification, ordered by
  V10's frozen keyword-doc score (no code-body adjustment). Do **not**
  claim a MathGraph advantage when comparator ties.
- C post-hoc diagnosis must check old-adapter ablation and direct
  signature-correct `sliding_window(2,xs)`, plus distinguish from
  `partition(2,xs)` / wrong window size. Each positive variant must
  be independently verified on all protected 1,093 inputs.
- Output statuses: `WARRANTED_BOUNDED_SOURCE_D` only with a protected
  verified match; `UNKNOWN_BUDGET_OR_BINDING` for no candidate;
  `REJECTED_CANDIDATE` for a discovery hit refuted by verifier.
  No result proves unrestricted portability or autonomous constructor
  invention.

## Anti-leakage sequence

1. This protocol commits before source-D checkout or inspection.
2. RED test for two-slot adapter is run against absent implementation.
3. Implement generic rule, pass local unit tests and C post-hoc diagnostic.
4. Run D acquisition in CI, then second-process independent verification.
5. Publish exact run, artifact, statuses, negatives, and baselines to ROS.

Never silently revise the above after viewing D; if follow-up is needed,
register a new source E and separate new protocol.
