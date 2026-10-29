# V21 source C — prospectively frozen before source inspection

State: **CANDIDATE / protocol only**. This is a separate research trial, not
an extension of the successful source-A/B verification boundary.

- Source C: `pytoolz/toolz`, Git commit
  `451af60dec590a6010e2babdbf391ea8f815122f` (master head read from
  GitHub metadata on 2026-10-29; no source files inspected for this trial).
- Protected behavior: for each finite sequence `xs`, return the fully
  materialized ordered list of 2-tuples `(xs[i], xs[i+1])` for adjacent
  elements, no terminal fill.
- Frozen acquisition probes: `[]`, `[0]`, `[1,2]`, `[0,1,2]`, `[2,2,1,0]`,
  `[0,2,1,2]`. This is inherited unchanged from V21 A/B.
- Frozen independent verifier: every ternary input over `{0,1,2}` of length
  0–6 (1,093 inputs), using separate `verify.check`; exact `list` required.
- Frozen search: same V21 AST scan (first 400 top-level non-private functions
  in sorted source order, top 160 by existing `score`), same iterable-first
  `adaptations`, two materialization variants and bounded output of
  `max(1, len(xs))`. No C-specific function names, parameter adapters,
  ranking changes or post-view tuning. Same Python 3.11 execution.
- Fixed matched baseline: semantic fingerprint search on *exactly the same*
  source universe, acquisition probes, adapter grammar, 1,093-case verifier
  and bounded materialization, with candidate attempts and wall time recorded.
  It can tie the source-generic V21 search; no MathGraph novelty asserted by
  merely finding the same candidate.
- Failure outcome: `UNKNOWN_BUDGET_OR_BINDING`, not a refutation of source
  semantics. A binding outside the frozen grammar is not silently added.
- Success outcome: `WARRANTED_BOUNDED_SOURCE_C` only for the pinned finite
  inputs and source revision, with a verified nonempty candidate witness and
  independently checked baseline, if any. This does not prove general
  capability invention, unrestricted transport, or economic superiority.
- Traces must preserve empty and unary input cases and any divergent types,
  failures or unsupported signatures. After inspection, no edits to target,
  source commit, oracle, grammar or search cap on this protocol lineage.
