# Where the LLM Ends: A Boundary, Its Sins, and the Two Axes of Agentic Intelligence

**Edward Y. Chang, Stanford University**
Position note, v1.0. First posted August 9, 2026. Elaborations to appear in *The Path to AGI*, Volume 3 (ACM Books, forthcoming).

## Summary

Public discourse uses the term "LLM" for two different objects: the trained weights, and the deployed system built around them. This note fixes the boundary with one operational test, derives the consequences for chain-of-thought and for fine-tuning, and states a two-axis map of how agentic intelligence is strengthened. The claims below condense positions developed across my AGI trilogy; I post them here to state them early and in one place.

## 1. The boundary: does a tool touch model parameters?

Define the **LLM** as the weights: a transformer pretrained on a large corpus by maximum-likelihood next-token prediction, plus all post-training (SFT, RLHF, the hundreds or thousands of fine-tuning refinements behind a modern release). Define the **LLM system** as the LLM plus every run-time layer above it: prompts, chain-of-thought orchestration, retrieval, tools.

The operational test is one question: does the tool touch model parameters? If yes, it is part of the LLM. If no, it lives above the line. Fine-tuning touches parameters, yet it changes no category: it is still gradient training on token prediction or preference, and its effect is to resculpt the pattern repository, moving which clusters are dense and near. An LLM with a thousand fine-tunings is still a System-1 pattern repository: sculpted, aligned, and semantics-free. When I say "LLMs are pattern repositories," I mean the weights. When a demo impresses, you are watching the system.

## 2. Chain-of-thought, correctly located

Chain-of-thought lives above the line, so it must be, and is, run-time, association-based construction. Three statements, all true: (i) the CoT layer orchestrates by first eliciting a typical workflow for researching and answering the question at hand; (ii) that workflow is produced by the same pattern completion as any other output; (iii) there is no pre-prepared template: nothing discrete is stored and fetched; the workflow is assembled on demand from learned regularities. CoT helps because it forces deeper consideration: more material is brought into context, aggregated, and weighed before concluding, rather than tracking the single maximum-likelihood continuation.

The verdict on its status: CoT is System-2's form executed by System-1's function. It supplies seriality and added compute, but not the supervisory function that defines System-2: it does not monitor its steps, cannot veto an erroneous one, and does not revise a failing procedure; errors are inherited confidently. CoT is half of System-2, namely procedure selection and execution without outcome evaluation and revision. Building the missing half outside the weights, monitoring, veto, revision, and regulation, is the program of my trilogy.

## 3. The sins of fine-tuning, and where values should live

Below-the-line modification carries four structural costs. **Interference**: parameters are shared, so improving task A moves mass that other tasks stood on, and the damage lands randomly; my base-arithmetic transfer experiments measure this directly. **Provenance**: reinforcement from human feedback optimizes toward someone's preferences; an annotator pool's judgment becomes physics for every downstream user. **Non-portability**: parameters are global while norms are local and contested; alignment tuned to one culture ships unchanged to all, and adapting re-invokes interference. **Entrenchment**: outcome reward cements right answers obtained through wrong reasons. Aggravating all four: weights carry no changelog; a fine-tune cannot be selectively inspected, attributed, or undone.

The corollary is architectural: adaptable commitments, including ethical constraints, belong above the line, expressed as declared, scoped, versioned, and auditable artifacts, values as data with a paper trail, not as physics baked into one prior for everyone.

## 4. Two axes of strengthening, and the substrate thesis

Agentic intelligence is strengthened along two orthogonal axes. The **horizontal axis** runs across steps, the long horizon: CoT chains evolved into agentic workflows; then, at a qualitative break, chains of generated steps became chains of admitted steps, with validation and state moved outside the generator (SagaLLM, ALAS for transactional planning and execution; Mnemosyne for governed memory with past, present, and future views; TRACE for typed reasoning records; temporal regret, horizon contracts, and world-model refresh for the temporal dimension). The **vertical axis** runs within one step, treating each workflow step as one atomic question strengthened by anchoring quality, structured critique and debate, causal audit, and epistemic regret. The Quadrivium composes both axes, and TRACE couples them: each step's vertical quality is written into the horizontal ledger, where cross-step learning can consume it.

This is more robust than the prevailing practice of pairing a vector database with hand-rolled retry logic, and the field has run this movie before. Flat files with per-application locking gave way to the DBMS, which promoted recovery, isolation, and audit into substrate guarantees. The same move is due for agents: admission, compensation, effective state, and durable obligations promoted from application hacks into substrate guarantees. Mnemosyne is to agent state what the DBMS was to flat files.

## Provenance and citation

These positions build on Volumes 1 and 2 of *The Path to AGI* (semantic anchoring and UCCT; multi-agent debate; causal audit; epistemic and temporal regret; SagaLLM and ALAS) and on the Volume 3 manuscript (Mnemosyne, TRACE, the Quadrivium, physical grounding). The condensed formulation above, in particular the parameter-boundary test, the half-of-System-2 verdict on CoT, the four sins of fine-tuning, and the two-axes map with the generated-to-admitted break, was crystallized in August 2026 during preparation of my nine-lecture AGI Trilogy tour and is posted here to declare it early.

To cite this note: E. Y. Chang, "Where the LLM Ends: A Boundary, Its Sins, and the Two Axes of Agentic Intelligence," position note, Stanford University, August 9, 2026. [URL of this page]

*Optional line, keep or cut:* Drafted with AI assistance in dialogue; the framing, claims, and responsibility are the author's.
