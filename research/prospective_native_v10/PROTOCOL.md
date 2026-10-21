# V10 frozen prospective source-native contract experiment
Frozen BEFORE external target source or tests are read.
Date: 2026-10-21. Parent code commit: 78a1a6b73bd38ce6b8da709fe06f7e9783fa868b.
Target repository: more-itertools/more-itertools (external independent maintainers), default HEAD resolved by GitHub Actions checkout; record exact SHA.
No target function name, file path or source excerpt may be supplied to candidate selector.
Protected task: Given finite sequence x, return a MATERIALIZED LIST of all contiguous adjacent 2-tuples in original order, no fill/sentinel. Empty or singleton -> [].
Candidate: Python AST scan of all external source .py files (excluding test/build/vendor paths), extract public function signature, first docstring and return-body information. Apply generic token-ranking and static return-materialization heuristic; propose best candidate and arguments if inferable.
Comparator: a conventional equal-documentation keyword/return-shape search applied to same AST records, no hidden target function name. Report ties.
Verifier: independent reference implementation list(zip(x,x[1:])); exhaustive ternary inputs length 0..6, type/order checks; run source library's own pytest suite independently. Candidate callable must be invoked from installed external library without manual adapter; only allowed signature adaptation is automatic src/iterable parameter and size=2 if explicitly exposed. If unable to resolve callable mechanically, UNKNOWN.
Budget: max 400 public functions scanned, top 5 candidate executions, no fitting on held-out cases. Upstream test execution capped at 10 min; do not treat failure to run as positive.
Holdout protection: no source tests or source contents inspected before this protocol commit. The evaluator's reference examples and exhaustive test grid are frozen here.
Admission: full target match, independent upstream tests passing, source commit SHA, no tie among materially distinct best candidates, no human mapping. Otherwise UNKNOWN. Any code change after inspecting source is exploratory V10.1, not prospective V10.
Negative controls: swapped, noncontiguous, chunked, sentinel-adding operations must fail exact output equality.
Cost: log candidate count and wall times for selector, native verification and baseline; no speed advantage unless actually measured against equal-info comparator.
