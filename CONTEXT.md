# Owner's intent and taste

Written 2026-11-07 by the founding session's main agent (Claude Fable 5.1) from the owner's two long dictated messages in that session (T3 Code thread "i have been thinking lately", scratch dir `2026-11-07-i-have-been-thinking-lately-0bc4471f`, Claude session `acc5e881-ba9a-44f4-8eec-12deb21608f4`; the transcript is in the alignment-hive session record for that directory, not in this repo). It records *why* the project exists and what the owner considers principled: the motivations, trade-offs and interview answers, which are not derivable from the theory documents. Provenance of each statement: **(owner)** marks a close paraphrase of the owner's words; unmarked sentences are the agent's synthesis of what the owner said. Nothing here is a verbatim quotation.

## The founding idea

The owner started from two inspirations: HAVOK-style time-lagged (Hankel) analysis, and the picture of neurons as detectors of coherent sets (Pughe-Sanford et al. 2025). They wanted to apply diffusion geometry to the cloud of residual-stream activations of a transformer: build a Gram matrix on activations at layer L, a comparable "dynamic" version through to layer L', and identify coherent structure, with the hope of obtaining a *basis* (possibly overcomplete) rather than a segmentation. The question put to the agent was: what is the smart and principled way to use diffusion geometry on the activation cloud to obtain such a basis, and to make it useful for describing weights, training dynamics, and comparisons across layers, training and models?

## What "principled" means to the owner

- (owner) No positing of obviously incalculable quantities, and nothing that smuggles in the semantic meaning we want to *discover in the network's own ontology*, such as a "task distribution".
- (owner) Hell-or-high-water, everything ends in an SVD, because the SVD is the only thing they trust to be interpretable. The SVD and its associated metric are, in a spiritual sense, a connection to perturbation theory: they give a notion of when two things are really close.
- (owner) Do not train anything on top of a model if it can be helped. Closed-form fits (ridge regression) are tolerated as fits, and are labelled as fits. Trained *baselines* for comparison (an SAE, an actual stitch) are fine as long as they are outside the construction and labelled as baselines.
- (owner) The owner differs from the mainstream in this and knows it.
- The one substantive geometric assumption the owner is willing to make, knowing it matters, is the choice of dot product (cosine on normalized residual-stream activations); they regard the residual stream as "pretty L2-y for a variety of reasons". A spherical variant with a log-norm coordinate is acceptable if needed.
- No assumption of sparsity. If SAEs are right, the method should discover SAE-like concepts; if hierarchical SAEs are right, it should discover that hierarchy; but it must not assume either.
- Pro-tem operationalizations are fine where the natural quantity is intractable or does not exist, provided they are clearly flagged as such. (owner) "Definitely be clear about this."
- Everything must be computationally acceptable: fine if that restricts us to small models on normal dataset sizes, but there must be useful approximations, and the knobs (samples, bandwidth, truncation) should have a visible relationship to accuracy and to the level of coarse-graining. It should work on MNIST.

## What the owner wants the language to do

- Describe the activation cloud, and coarse-grain it so as to talk about things the network "acts on similarly", while noting that if activations were linear combinations of sparse concepts those would be treated roughly independently, so a strict partition into coherent sets would be wrong.
- Compare the description over training and over layers, and characterize the hierarchical refinement of the network's treatment of activations during training.
- "Pull back" the activation-space description to the weights, so that spectra decompose the weights' action, ideally per activation group, and so that gradients can be read as "pressure to refine this classification" or "interference from group Y". (owner) Most of the description should be about activation space; the pullback to weights should only *constrain* the update to the activation cloud.
- Avoid deciding ex ante whether small spectral components are noise or error-correction.
- Be comparable across models trained on the same data, ideally predicting model stitching.
- Relate to the Golowich–Liu–Shetty low-rank-logits / ISAN picture (they called the paper "lingen"); the construction should in some sense decompose how an LLM computes that.
- (owner) Ideally be usable, with reasonable changes and abstracting away measurement difficulties, for a brain; perhaps even a geneticist could use something like it.

## The owner's answers to the design interview

1. **Metric.** L2/cosine at intermediate layers is the preferred commitment. Fisher–Rao at the logit layer is fine. Fisher at intermediate layers is methodological, to be settled by experiment, not a theoretical obstacle.
2. **Context (KV cache).** Marginalizing over contexts is unacceptable as a description ("these simply contain too much information") though it is a lower bound. Fitting a model that learns the hidden state is unacceptable. ISAN-like context embeddings are also only a lower bound. The version the owner would most like is a Hankel-like construction on the layer-L stream treated as its own process, whose SVD gives an embedding φ^L(past), and a good way to say how φ^L(past) acts on the current h^L. Switching the *problem* to an RNN or Mamba with a true hidden state, to make the context Markovian, would dodge the question rather than face it; studying Mamba alongside the transformer (Stage 4) is wanted, since architecture-agnosticism is the point.
3. **Temperature.** Acceptable as a knob for coarse-graining and persistence; hesitant about treating functions of the temperature as primitives. Open to it if reasonable, useful and tractable. Candidate natural scales mentioned: noise, edge-of-stability or batch noise, mean nearest-neighbour distance, steradians per activation.
4. **Coarse-graining.** There should be a spine of the theory where spectral truncation is never required, and truncation can be applied at any step while still proceeding. Re-embedding from the diffusion graph back into activation space need not be a primitive, but something canonical like "replace with the centroid" would be welcome, with the caveat that it does not handle clusters along a manifold.
5. **Pullback to weights.** Unsure. Intuitions: use the natural bottleneck of a linear layer feeding a unit; coherent structures in tangent space feel right; weight magnitudes implicitly compare to zero-ablation, which is wrong; the activation–weight duality is suspect unless there is a very general version.
6. **Toy models to get right.** Toy models of superposition; the computation-in-superposition toy of Ferreira da Silva and Heimersheim (2026); grokking; HMM belief-state geometry (Shai et al.), where a true Hankel exists; then dSprites, MNIST, CIFAR. Toy models are "nearly never actual models of what happens in neural networks", so failing some is not fatal, but most should be recovered. CIFAR may fail; an Inception-scale failure would be unhappier; the dream is Pythia or Gemma. A failure is acceptable if the language lets us analyse it.
7. **Deliverable.** A theory document and an experiment plan in a project folder, developed with iterative Codex (GPT) critique, then a fresh-eyes review for readability; the owner may have an agent rewrite for clarity post facto.

## How the owner likes to work with agents

Operational preferences (compute on Modal, kindness to subagents, reuse over rewriting, cross-family critique) are in the owner's global `~/.claude/CLAUDE.md`, which is authoritative; only project-specific observations are kept here.

- Dictates; infer obvious mistranscriptions ("lingen" was Golowich–Liu–Shetty; "L4 paper" was the fourth-power loss, not an $L^4$ norm).
- Wants taste-level questions asked rather than guessed, and said it is fine to err on the side of asking; also said the agent may work alone for an hour or more between check-ins.
- For this project specifically asked for: iterative Codex (GPT) critique of the theory, a fresh-eyes review at the end, and an owner-facing report written by a separate agent with its own critique loop.
