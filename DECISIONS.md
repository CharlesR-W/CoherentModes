# Open design decisions (for the project owner)

Originally extracted from `THEORY.md` v1.3 and re-checked against v1.5 on 2026-11-07: still open, unchanged in content. Each is a choice the theory cannot make on its own; the consequence of each option is noted. **Provisional defaults** (agent-chosen, not owner-approved; use only to proceed and say so) follow the list.

1. **[A1]** is the load-bearing design decision. Does "the site is the atom, identity transport, attribution is a separate step" match the intent, including for recordings?
2. **Bistochastic normalization [M3]** makes sites uniformly weighted. This matches "frequency enters through the dataset". Agree, or is a non-uniform site weighting wanted (e.g. down-weighting repeated tokens)?
3. **Read vs carried context [P5]**: keep both as first-class, or make "carried" the spine and "read" an attribution tool?
4. **Noise temperature [P8]** needs stochastic resamples at fixed parameters, i.e. a few extra forward passes or minibatch steps per checkpoint. Acceptable as a requirement for the headline number?
5. **[A3]** for pretrained models without their corpus: substitute a held-out sample from the same distribution?
6. **Stitching sensitivity weight (Section 8)**: output-Fisher sensitivity rather than gradient pressure. Agree?

## Provisional defaults (not owner-approved)

1. Keep [A1] as stated: the site is the atom, identity transport, attribution a separate predictive step.
2. Uniform site weighting under bistochastic normalization.
3. Both context operators first-class; "read" used for attribution and the ladder, "carried" for the architecture-free characterization.
4. Stochastic resamples at fixed parameters required for the headline temperature; report the nearest-neighbour temperature where they cannot be produced.
5. For pretrained models without their corpus, a held-out sample from the same distribution, flagged.
6. Output-Fisher sensitivity (block-aggregated, rotation-invariant) as the stitching weight.
