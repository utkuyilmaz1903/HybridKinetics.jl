# Scope and limitations

HybridKinetics is a research tool for a narrow problem: a small biochemical
network whose interaction graph and kinetics are known except for one
destruction term. This page collects every caveat in one place.

## Scope

- **Unknown destruction terms only, one per node.** Any number of
  destruction terms on distinct nodes may be unknown; two on the same node
  and an unknown production term are refused with an error. What several
  terms cost is measured and listed under "Several unknown terms" below.
  The reference recovery protocol behind the benchmarks
  (`run_recovery_suite`) is a single-term instrument and raises an error for
  zero or several unknown terms.
- **Known graph.** The graph-local library is built from the interaction
  graph you supply. Inferring the graph itself is out of scope.
- **Small networks.** The benchmarks cover two-, three-, and six-state
  networks. Larger known graphs use the same machinery, but there is no
  evidence in the repository for them.
- **Validation on synthetic data; two measured datasets.** The benchmarks and
  the tests use data generated from the compiled ground-truth mechanism; the
  CSV in `examples/data/` is synthetic, and the repressilator fixture with
  published dimensionless parameters is an ODE, not a measured series. The
  primary real-data example is the [laccase/ABTS case study](case-study-laccase.md):
  on nine published substrate-depletion curves, the two-state
  product-inhibition model yields an accepted rational rate, a saturating
  function of the substrate (the one-state model does not). The
  [p53–Mdm2 case study](case-study-p53.md) is a partially observed
  illustration: 40 MCF7 cells with p53 measured and Mdm2 an unobserved
  state, and no accepted rate. Both are downloaded by scripts with
  checksums (the data do not ship with the package) and run outside the
  tests and CI. They exercise the workflow on real measurements; neither
  has both states of a regulated destruction term measured.

## What the diagnostics do and do not establish

- **The identifiability diagnostic is local and practical.** It flags an
  edge as unidentifiable when the Fisher condition number exceeds `1e6` or
  when the trajectory sensitivities to the production rate and to the scale
  of the unknown term have a cosine of at least 0.95, on one trajectory at
  one parameter point. It is not a structural identifiability proof. A
  passing check does not certify the recovered mechanism, and a raised
  warning does not stop the workflow.
- **A good fit is not mechanism recovery.** A small trajectory residual shows
  that the hybrid model reproduces the data. It does not by itself show that
  the true mechanism was found.
- **Coefficients are not biological constants** unless the scale of
  production or destruction is fixed by outside information. With observed
  concentrations alone, the production rate and the scale of the unknown term
  trade off against each other; freezing the production rate or normalizing
  the sampled rate does not remove this.
- **The functional-identifiability diagnostic is a diagnostic.** Agreement
  between independently trained rate functions across restart seeds is
  evidence, not a proof of uniqueness, and it is not an acceptance criterion.
- **Held-out validation is reported, not enforced.** The reference protocol
  trains on seven of nine initial conditions and reports the residual and the
  destruction-rate error on the other two. Those numbers are evidence; they
  are never compared with a threshold. Two held-out initial conditions are
  not an out-of-distribution test.

## Discovery

- **The discovered form is a rational function, not a canonical Hill law.**
  The reference protocol recovers the true monomials of the Hill term (the
  acceptance criterion is recall of at least 0.99), but extra terms remain,
  a constant and a linear term. Combined support F1 is scored against a floor
  of 0.50; it is not an acceptance criterion. The package does not turn the
  neural term into a Hill expression with named parameters.
- **What the support F1 of 0.57 measures.** It is not a mislabelled truth and
  not an unattainable ceiling: a candidate carrying exactly the recorded true
  support reproduces the Hill law to 1e-12 and scores 1.0, which the test
  suite checks. The extra terms carry coefficients of the same order as the
  true ones — no sparsity threshold below 0.2 removes anything, and one large
  enough to remove them also removes true monomials — and they describe the
  gap between the trained network's rate and the true law rather than the
  mechanism: removing one costs three to seven times the regression residual
  while leaving the fit to the true rate unchanged or better, where removing a
  true term costs twenty times and makes it worse. Read the 0.57 as a measure
  of how far the learned rate sits from the true law in the library's basis.
  Information criteria, separate numerator and denominator thresholds, and
  derivative rows were all tested in 0.18 and none of them helps; the numbers
  are under "Why the extra terms survive" on the
  [Benchmarks](benchmarks.md#Why-the-extra-terms-survive) page.
- **Michaelis-Menten unknown terms** are checked on the neural-rate error and
  the residual only; canonical Michaelis-Menten support from the trained
  network is not claimed.
- **Noise.** Analytical recovery with finite-difference derivatives holds up
  to 2% rate noise and fails at 5%. The trained-model protocol is measured at
  0 and 2% observation noise.
- **Partial observation** is limited to discovery from subsampled
  destruction-rate values and the residual of the resulting hybrid model.
  Training on states that are never observed is not supported.

## Robustness

- The trained-model library comparison (graph-local versus global versus
  wrong-graph on one trained model) and the five-restart
  functional-identifiability diagnostic are implemented.
- A multi-seed robustness study of the full protocol is not implemented.
  `benchmark/recovery_seeds.jl --ude` runs the protocol on five seeds as a
  report; the continuous-integration check uses seeds 103 and 104.
- How much one run varies is measured (0.18 study, benchmarks page). Over ten
  random initialisations of the neural term on the same data, five seeds and
  fifty runs, the discovered support is the same every time, while the final
  training loss spans a factor of 6 to 90 within a seed and the learned-rate
  error runs from 0.022 to 0.084. One run's loss is one draw from that
  spread. The warm-up length matters less than the draw: the package's
  length, twice it and none at all differ by less than the spread between
  initialisations, although dropping the warm-up entirely is the only setting
  in which two seeds disagree about the support.
- The hybrid model built from the discovered rate resimulated in all 15 runs
  of the reference protocol at noise 0, 0.02 and 0.05, and in all 60 runs of
  the four-state fixture's default sample design at 0.05. The runs that do
  not resimulate, in the stored studies, belong to the constant sample design
  at 0.05 noise, which has not been the default since 0.12. Their candidates
  have a denominator identically 1, so the denominator-safety check cannot
  flag them and no candidate-only check was added: the quantity that does
  separate them, a rate that turns negative outside the sampled range, also
  flags 75 of 232 runs that resimulated without trouble.

## The optimisation gap

- A better solution than training reaches exists, and both of the workflow's
  headline weaknesses follow from not reaching it. On the reference two-state
  protocol, the same data with the network pre-trained to the true rate before
  training starts give the true support with no extra terms in 5 of 5 seeds,
  at a learned-rate error of 0.005 against 0.047 and about a thirteenth of the
  final training loss (0.19 study, benchmarks page). The support F1 of 0.571 is
  therefore a property of where training stops, not of the library or of the
  discovery step.
- That solution is in a different basin, not further down the same one. The
  training loss along the straight line between the two rises to 139 to 19,055
  times the higher endpoint. It is not a matter of the rate's scale either: the
  best single scalar on the learned rate recovers at most 12 per cent of the
  gap.
- What decides whether the extra terms appear is the shape of the learned
  rate's error, not its size. A rate several times less accurate gives the true
  support if its error has a different shape.
- Of the approaches measured, only a better starting point helps materially. A
  destruction rate estimated from the data alone and used to pre-train the
  network closes a median 88 per cent of the loss gap and gives the better
  support in 2 of 5 seeds. Choosing the lowest-loss of ten random
  initialisations improves the rate by 17 per cent and the held-out residual by
  41 per cent and changes the support in none. Smoothing the learned rate
  changes nothing: it is already smooth, so a roughness penalty has nothing to
  act on.
- `restarts` on the training configuration fits several initialisations and
  keeps the lowest final loss. It defaults to 1, which is the single fit the
  package has always done.

## Extensions and integrations

- The GPU extension only transfers arrays; there is no batched GPU training.
- SBML import reads species, reactions, and stoichiometry but does not parse
  kinetic laws; such reactions become unknown neural terms.
- The ModelingToolkit export represents neural terms as placeholder
  variables.
- The DataDrivenSparse backend cannot currently be loaded in the package
  environment because of a dependency version conflict.

## Several unknown terms

- One unknown destruction term per node. A second unknown term on the same
  node is refused, because two rates multiplying the same state are one rate
  to the data.
- No unknown production term. The model form is
  `du_i/dt = P_i(u) − D_i(u)·u_i`; an unknown `P_i` is not multiplied by
  `u_i`, needs a different network input and a different library, and on the
  same node it is entangled with the destruction scale that the diagnostic
  already flags. It is refused with an error naming this scope; it is out
  of scope at 0.17.
- What a second term costs is measured, not assumed (the two-term study on
  the [Benchmarks](benchmarks.md#Two-unknown-terms) page). Two terms on
  nodes that do not regulate each other's term separate cleanly: same
  support recovery as one term, a learned-rate error within 1.15 times the
  single-term run's, training 1.5 times longer. Two terms on adjacent nodes
  keep their supports but the downstream term's learned rate comes out
  15 to 30% low in every run (median 16%), two to six times the single-term
  error, and the noise-free held-out residual doubles. The cross-term
  diagnostic separates the two cases (0.37–0.43 against 0.95–0.96) and warns above
  `CROSS_TERM_COLLINEARITY_THRESHOLD` = 0.46, the value below which no run
  of the study showed a measurable cost; it is local to the fit,
  concerns the scales only, does not say which of the two terms drifted,
  and is not a structural result. The 0.18 study asked what that bias is:
  pre-training both terms to their true rates and then fitting with the
  normal settings leaves the biased term at under a fifth of its usual bias
  in 5 of 5 seeds, at about a tenth of the final training loss, so the data
  support a fit within 1 per cent of the truth on the median and 5 per cent
  at worst, and the joint search does not find it. That
  is a diagnosis and not a remedy — a user has no true rates — and the three
  corrections that were measured do not help: weighting each node's residual
  by its observed variance makes the bias larger, fitting one term first and
  freezing it moves the bias onto whichever term is fitted second in either
  ordering, and the residual surface over the two rates' scales shows their
  ratio about six times better determined than their common scale, so a
  constraint on the product would act where the data are already
  informative. The 15 to 30 per cent figure therefore stands as the measured
  cost, now known to be a property of the optimiser's path rather than a
  limit of the data (0.18 study, benchmarks page).
- The number of terms is not limited by the code. Two terms were measured
  in full; the three-term fixture ran at noise 0.0 only (its noisy cells
  were dropped at the study's budget). There, the adjacent pair came out
  1.4 and 2.2 times further from the truth than the single-term runs and
  the third term lost its support in two of five seeds, with cross-term
  values of 0.5 to 0.6, so three terms should be read as a measured but
  weaker case, not as two terms plus one. Beyond that the
  training time grows with the number of networks and nothing has been
  measured.

## Not in scope

Inferring the interaction graph, general reaction-network solving, several
unknown terms at once, experimental design, wet-lab decision support, and
integration with language models.

## Roadmap

The next items, in no particular order and with no dates: a multi-seed
robustness study of the reference protocol, a broader set of benchmark
networks, and registration in the General registry once the public API has
settled.
