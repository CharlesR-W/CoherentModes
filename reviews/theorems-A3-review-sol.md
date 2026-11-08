# Review of A3 (sample complexity and cost) by GPT-6.1 Sol, 2026-11-07

Saved by the orchestrator from the reviewer's report (the reviewer agent could not write files).

**Summary.** No substantive counterexample to the operator bounds, rank argument, approximation guarantees, or arithmetic costs. Two precision fixes; neither changes the conclusions.

## Findings

1. **[NIT] A.49 should explicitly require identically distributed resamples.** Replace "independent stochastic resamples" with conditionally i.i.d. resamples from one fixed resampling protocol, with $Y, Y'$ independent draws from the common conditional law; independence alone allows heterogeneous laws whose pair-average expectation need not equal the displayed $\theta$. Exponent correct: paired observations in [0,4], Hoeffding gives $2\exp(-\lfloor n/2\rfloor u^2/8)$.
2. **[NIT][CITE] A.47 should define its canonical functions and cite the non-leading-mode extension.** State the empirical coefficient $\hat a_p = (\hat C_{pp} + \rho_N I)^{-1/2}\hat u_p$, its evaluation with the centered feature map, and the limit $C_{pp}^{-1/2}u_p$; specify that the $L^2$ assertion concerns marginally centered functions. Fukumizu–Bach–Gretton Theorem 2 concerns the leading pair; for an arbitrary simple nonzero singular value also cite the remark following Theorem 2. Range condition matches the article's stronger assumption; no missing exponent.

## Checks (all pass)

- **A.40–A.45.** Continuous Sinkhorn proof valid. A.44's common-space lift is well-defined ($A_s u(z) = s_\mu(z)\langle\Phi(z), u\rangle$, $A_s^*A_s = C_P$, $A_sA_s^* = P_\mu$; empirical rows $s_N(z_j)\Phi(z_j)/\sqrt N$); constants check: $D \le 2\epsilon$, $t \le 1/2$, $E_N \le (12R+7)\epsilon$; conditional $(S_0^2 + 6L_0)e^\beta h$ bound follows; no universal $e^\beta$ multiplier advertised unconditionally. A.45 carries the $\sqrt N$ conversion and $1/\lambda$ penalty correctly.
- **A.46–A.48.** FBG Lemma 6 gives $O_P(N^{-1/2}\rho_N^{-3/2})$; Theorem 1 condition (9) is $N\rho_N^3 \to \infty$, draft correctly excludes $\rho_N = N^{-1/3}$. Source-condition exponents $1/(2\gamma+3)$, $\gamma/(2\gamma+3)$ correct. A.48 polar-decomposition argument valid; centered rank $\le k-1$.
- **A.50–A.54.** A.50 quantile-margin argument correct. A.52 verifies Tropp Theorem 6.1 ($L = c_\ell d_{\rm eff}$, variance $\le mL$, bound $2N\exp(-3m/(28L))$, factor 10 suffices). A.53 uses paired sine/cosine Gaussian Fourier features for the chordal Gaussian times $\sqrt{e^\beta}$; expectation is exactly the exponential-of-cosine kernel; Hoeffding constant, $N^2$ union bound, $e^{4\beta}$ positivity budget correct. A.51 dense Sinkhorn count $4N^2$ under its convention; cost table separates formation, iterations, Krylov sweeps, leverage acquisition, JVP work. A.54 variance and bounds correct.

Reran author's `/tmp/cm_a3_check.py`: normalized slope −0.517, CCA −0.5, CCA tail −0.503, probe variance measured/theory 0.998, all passed. No `\{` issues.

Sources checked: Fukumizu–Bach–Gretton JMLR 8 (2007) Lemma 6, Theorems 1–2 and remark; Tropp 2012 Theorem 6.1.

## Verdict: approve with fixes
1. Make A.49's common conditional resampling law and i.i.d. hypothesis explicit.
2. Define A.47's empirical and limiting canonical functions, their centering, and cite the post-Theorem-2 remark.
