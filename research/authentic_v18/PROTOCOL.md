# V18 real-upstream-commit calibration, frozen before running
External source: mahmoud/boltons, three authentic commits (2026), old parent vs head.
- 4aa77cddb5d06def130df6f09aefab4018fa487c split_iter maxsplit=0
- ead236e278ca0466bf468de746b5960fb12d7e5b backoff_iter factor=1.0
- f5f5cd867ee98f15def2b6a19bfba203de29b961 GUIDerator diagnostic bounds
Source and affected files were inspected before writing this protocol. This is exposed retrospective calibration, NOT prospective.
Protected contracts: pairwise behavior (unrelated unchanged function) and the commit-named changed API's observed behavior.
A textual identity check of the *function AST* permits transport of the identical function body only under the restricted Python/source environment; it is not a proof of observational equivalence if globals/dependencies change. Report dependencies separately.
A full native behavior probe runs on each actual parent/head checkout, preserving exception class and tested output.
Four baselines: full rerun, file dependency invalidation, ordinary AST function-hash reuse, and MathGraph-style semantic certificate with same function-hash evidence. Both strong baselines must be permitted to reuse exact AST certificates.
Counts and wall time are descriptive; do not claim advantage over conventional hashing.
Negative controls: reported changed function and unrelated pairwise preservation.
CI must fail if sources unavailable, author tests fail, or declared output checks mismatch.
