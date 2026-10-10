---
description: Run a NeuroDrive study following the experiment protocol and write its report
argument-hint: <study id such as S6, or a question>
---
Run the study **$ARGUMENTS**.

1. Find it in `docs/EXPERIMENT_PROTOCOL.md` §5 (or, for a new question, propose a study ID).
2. Before running anything, write down the question, hypothesis, the single variable being
   changed, the controls, configs, seeds (≥ 5), budgets and exam version, and show this to the user.
   Wait for a go-ahead if the total run time is estimated above 30 minutes.
3. Run with `nd train` / `nd sweep` / `nd exam`. Record the run IDs.
4. Analyse with `experiment/stats.py`: median, IQR, and bootstrap confidence intervals for any
   A-vs-B claim. Keep loss, fitness and reward in separate, labelled panels; compare methods only
   on Driving Exam metrics.
5. Inspect at least one failure in the linked views and describe it (run, unit, track, step).
6. Write `docs/studies/<ID>.md` using `docs/WORKFLOW.md` §4. Save figures in
   `docs/studies/figures/`. Add a journal entry and update the checklist.
