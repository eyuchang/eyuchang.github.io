# Enabling LLMs to Learn: ERM and RLER, and the End of What-Only Feedback

**Edward Y. Chang, Stanford University**
Position statement, v1.0. August 9, 2026.

Current pipelines teach models with outcome feedback: what was right, what was wrong. I take the position that what-only feedback cannot produce learning worth the name, because learning depends on knowing why, when, and how.

Why. Epistemic Regret Minimization (ERM) critiques the causal structure of a reasoning trace independently of the answer: unexamined confounders, correlation-intervention conflation, unchecked back-door paths. On 1,360 causal-trap scenarios across six frontier models, the models bifurcate. Compliant models flip on any credible push. Stubborn models, which are precisely the reasoning-enhanced ones, resist outcome-only correction at 25 to 31 percent recovery yet respond to causal critique at 78 to 91 percent, and residual rung collapse falls from 55 to 70 percent to 4 percent. The gain is not verbosity: against matched-richness but causally empty critique, the causal vocabulary is worth 5.9 points at p equal to 0.006. And when no answer key exists, fully label-free critique trades peak recovery for zero false flips, the deployment-safe operating point.

When. Temporal regret measures how long a miscalibrated causal model is tolerated. The guarantees are conditional and sharp: under observationally equivalent confounding without an intervention channel, miscalibration can persist linearly in the number of episodes even after training-time outcome regret reaches zero; with a persistent causal log and budgeted probes, temporal regret drops to order log E. Zero training loss and linear deployment regret can coexist. That sentence should be printed above every agent dashboard.

How. RLER (Reinforcement Learning from Epistemic Regret) closes the loop: accumulated interventional evidence becomes the reward signal, in the label-free regime where deployment actually lives. A separation theorem states that outcome-only reward cannot close the gap in principle, and controlled simulation confirms it with a thirty-eight-fold regret separation over outcome-only baselines.

What-only feedback rewards being right. Why, when, and how feedback produces being right for the right reasons, caught quickly, and improved durably. That is the difference between scoring and learning.

Full development: *The Path to AGI*, Vol. 2 (Chs. 6, 8, 9, 10).
