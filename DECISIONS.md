# Open design decisions (for the project owner)

From `THEORY.md` v1.3. Each is a choice the theory cannot make on its own; the consequence of each option is noted.

1. **[A1]** is the load-bearing design decision. Does "the site is the atom, identity transport, attribution is a separate step" match the intent, including for recordings?
2. **Bistochastic normalization [M3]** makes sites uniformly weighted. This matches "frequency enters through the dataset". Agree, or is a non-uniform site weighting wanted (e.g. down-weighting repeated tokens)?
3. **Read vs carried context [P5]**: keep both as first-class, or make "carried" the spine and "read" an attribution tool?
4. **Noise temperature [P8]** needs stochastic resamples at fixed parameters, i.e. a few extra forward passes or minibatch steps per checkpoint. Acceptable as a requirement for the headline number?
5. **[A3]** for pretrained models without their corpus: substitute a held-out sample from the same distribution?
6. **Stitching sensitivity weight (Section 8)**: output-Fisher sensitivity rather than gradient pressure. Agree?
