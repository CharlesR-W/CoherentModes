# Planted residual MLP (PLAN §6b): what was measured

Exploratory, measured not asserted. Script `mlp_planted.py`, numbers in `data/mlp.json`.

**Setup.** A, B uniform on Z_8, y = (A+B) mod 8; x = E_A[A] + E_B[B] + Gaussian nuisance (input dimension 32, embedding norm about 1, nuisance norm about 0.5). Model: pre-LN residual MLP, width 256, 4 blocks, trained on y only (AdamW, 4000 steps x 512 fresh samples). Training took 94 s on this CPU (1083 s on a re-run under CPU contention, with identical results). Held-out y accuracy was 1.000.

Views are x, h0 (the embedding), h1..h4 (the residual stream) and the logits. The probe set is 64 (A,B) combinations x 30 = 1920 points, with a held-out set of the same size. Kernels: K-st(15) and K-vmf(kappa = 1/0.07, declared, not tuned); t in {1, 4, 16, 64}. Labels are used only to validate.

## Findings

1. **Hidden layers do not merge A and B in information.** Closed-form ridge decodability of A and B (probe to held-out) is 1.000 at every layer from x to h4. y is not linearly decodable at x or h0 (0.17, 0.19; chance is 0.125) and is 1.000 from h1 on.
   - Only the logits lose A and B linearly (0.21 / 0.24), and even there kNN recovers them at 0.86.
   - kNN decodes y at 1.000 even at x, because the input has 64 separable (A,B) clusters. So kNN decodability does not separate "computed" from "computable".

2. **Whether the geometry merges depends on the kernel.**
   - Under K-st the largest eigengap of P comes after 64 modes at every hidden layer, and r=64 SEBA recovers all 64 (A,B) states through h4 with H(AB | state) = 0.
   - Under K-vmf the gap moves to after 8 modes at h4 (only 8 eigenvalues > 0.9). The (A,B) contrast span's smoothness falls from 0.90 at x to 0.19 at h4 (t=1), and A-contrasts specifically to 0.09.
   - So at the declared cosine resolution, h4 looks like 8 y-clusters while A and B remain perfectly linearly decodable. That is a geometric merge at a resolution, not an information merge. It is the trained-network counterpart of unit test (iii).

3. **The largest smoothness change from the input is not task-specific.** From x to h1..h3, the top positive Delta modes put 98-100% of their energy in the 63-dim (A,B) contrast span, with eigenvalues up to 0.99 (K-st, t >= 4). But their principal cosines with the y-span are at most 0.12 for t <= 4.
   - The network tightens all 64 (A,B) clusters against the nuisance: K-st (A,B)-span smoothness at t=4 goes from 0.12 at x to 0.90 at h1.
   - A viewer listing "top became-smooth modes" for early blocks will show this, not y.

4. **y separates from A and B at h3.**
   - K-st at h3, t=64: y-span smoothness is 0.99 against A-span 0.24.
   - K-vmf at h3, t=1: 1.00 against 0.61; at h4: 1.00 against 0.09.
   - The y-span enters the top-7 positive modes of Delta(x -> l) at h3 for K-vmf at t >= 4 (cos 0.95) and for K-st only at t >= 16 (cos 1.0). Under K-st it enters at h4 from t = 4 (0.94) and at the logits from t = 1 (0.93).
   - The stopped-being-smooth modes at h4 and the logits contain the A and B spans (largest cosine 0.90 / 0.73 at K-vmf h4, t=1).
   - When y "appears" depends on the declared resolution, as planted in test (iii).

5. **Hard states and entropies.**
   - With r = 8, the SEBA states are exactly y from h3 on (H(y | state) = 0 for both kernels; K-vmf h2 gives 0.36 bits).
   - At x..h1 the r = 8 states are poorly supported: SEBA's distance from its input subspace is 0.41-0.54, and K-st coverage at h1 is 0.39. Entropies there should not be read as structure.
   - H(c_l | c_x) and H(c_x | c_l) are monotone non-decreasing in depth for both kernels and both r (r=8 K-st: 0.63, 1.00, 1.43, 2.68, 2.68, 2.68 and 0.59, 2.04, 2.50, 2.71, 2.71, 2.71 for h0..logits). With r = 64 they stay 0 until h3 and rise only at the logits (about 1 bit).
   - No same-layer split-half baseline was computed for the MLP, so the assignment-noise floor is unknown (it is likely near 0 for r = 64 given the purity).

6. P had no material negative spectrum here (min eigenvalue >= 0 to 3 decimals), so the R = P^2 choice did not matter for this model.

## Contrary to or beyond the plan's framing
- The plan asks "whether hidden layers merge (A,B)". The answer splits by notion: no in information, no under K-st geometry, yes under K-vmf geometry at h4.
- The dominant early Delta is generic cluster sharpening, so the Delta modes shown in a viewer need the span-energy or validation context beside them.
