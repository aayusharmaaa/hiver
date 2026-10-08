# Hiver SDE Intern Take-home - Decision Log

1. **Pick VirginTrains.** Ranked brands on volume, reconstructable multi-turn cases, repeated templates, intent diversity and visible public resolutions. Public evidence mattered more than raw volume.
2. **Rebuild support cases, not individual tweets.** Customer episodes preserve context and let retrieval learn from the actual resolution sequence.
3. **Use a candidate taxonomy.** The 10-intent taxonomy came from discovery/clustering work, but weak separation and overlapping intents meant it was treated as provisional rather than truth.
4. **Freeze the golden set.** Golden IDs were isolated from train/dev and reserve data so evaluation could not leak into development.
5. **Use 100 blind labels as the headline.** The 250-case set has mixed provenance: 100 independent human labels plus 150 human-confirmed assistant drafts. The blind 100 is the cleanest primary benchmark.
6. **Keep taxonomy review separate from tuning.** Gold labels were used diagnostically, not to retroactively redefine the benchmark taxonomy.
7. **Report both trivial and simple baselines.** Majority establishes the floor; TF-IDF + Logistic Regression tests how much of the task a conventional classifier can solve.
8. **Use hybrid retrieval with a soft intent bonus.** BM25 covers lexical matches, embeddings recover semantic matches, and a soft intent signal is more robust than hard filtering under intent errors.
9. **Put evidence before generation.** Historical resolutions constrain the LLM instead of being decorative citations added after generation.
10. **Keep risk policy deterministic.** Live-status, money, complaint and sensitive cases cannot become automatable merely because an LLM sounds confident.
11. **Make grounding fail closed.** Unsupported amounts, dates, links or policy claims block the draft; grounding can only downgrade AUTO_HANDLE to ESCALATE.
12. **Use always-escalate as the routing baseline.** It establishes the safety floor against which any automation policy should be judged.
13. **Track safe automation, not coverage alone.** Raw automation rate hides whether the system stayed within policy and evidence.
14. **Treat the LLM judge as a diagnostic.** n=14 and groundedness kappa 0.00 make it unsuitable as a safety authority.
15. **Avoid production-shaped plumbing.** No Hiver/Twitter API, auth, deployment, vector DB, fine-tuning or live train integration. The take-home is evaluated on reasoning, evaluation design and safe decision logic.
