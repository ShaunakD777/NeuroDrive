# NeuroDrive — Task Workflow and Templates

Every task in `ROADMAP.md` follows the same loop. The user starts a task by naming it (or with
`/do-task <id>`). That is the approval to implement **that task only**.

## 1. The loop

1. **Read.** The task entry in `ROADMAP.md`, its row in `CHECKLIST.md`, the relevant contracts in
   `ARCHITECTURE.md`, and any ADRs it touches.
2. **Brief.** Post the Brief (template below). If the task is marked ⚑ (decision needed), or you
   found a decision that is not covered, stop here and wait for the user.
3. **Implement.** Only the files named in the Brief. Mark the task ◐ in the checklist.
4. **Verify.** Run the task's verification. Fix and re-run if needed. After two failed fix attempts,
   stop and report the diagnosis.
5. **Report.** Post the Report (template below). Mark ☑ then ✅ once verification has passed.
6. **Record.** Append a journal entry, add new glossary terms, add an ADR if a decision was made,
   and commit `task X.Y: <summary>` on the phase branch (staging files by name; see §7).
7. **Stop.** Name the next task, but do not start it.

If the user says "brief only", do steps 1–2 and stop.

## 2. Brief template

```
## Task X.Y — <title>

**1. What we're building:** 1–3 sentences.
**2. Concept introduced:** the idea in plain language, with an everyday analogy if helpful.
**3. Why it's needed:** what later tasks depend on it.
**4. The maths:** each symbol defined in words, then a worked example with small round numbers.
   (Write "none, plumbing task" when that is the case.)
**5. Steps (pseudocode):** numbered plain-language steps for any algorithm.
**6. Design:** files to create or modify, each with its responsibility; key choices and the main
   alternative considered.
**7. Verification plan:** which tests or experiments, and what success looks like.
**8. What you'll observe:** what the user should see or be able to do afterwards.
```

## 3. Report template

```
## Task X.Y — Report

**1. What changed:** one line per file.
**2. How the code maps to the maths:** walk through the main function(s) in prose, step by step,
   linking each step to the formula. Re-run the Brief's worked example and show the code gives
   the same number. (No code blocks.)
**3. Verification:** the exact command(s) run, a summary of the real output, and what it proves.
**4. Try it yourself:** 1–3 small things the user can run or click, each with the expected result.
**5. Check your understanding:** 2–3 questions, answers given at the end of the report.
**6. Limitations / open issues:** anything not done or uncertain.
**7. Next task:** ID and one line. Not started.
```

## 4. Experiment report template (used by `/experiment`)

```
## Study <ID> — <question>

**Question and hypothesis:** written before running anything.
**What changed vs. what was held fixed:** one variable changes; list the controls.
**Setup:** configs, seeds, budgets, tracks, exam version, run IDs.
**Results:** table (median and IQR) plus chart files; native metrics shown separately per method.
**Failure analysis:** at least one inspected failure, with run, unit, track and step, and what the
   Network view showed.
**Conclusion:** only what the evidence supports. Say where it is uncertain.
**What this teaches about AI:** 2–4 sentences linking back to the central question.
```

Save study reports in `docs/studies/<ID>.md` and figures in `docs/studies/figures/`.

## 5. Journal entry template

```
### <date> — Task X.Y <title>
- Concept: ...
- Key equation(s) in words: ...
- What we observed: ...
- Surprise or difficulty: ...
- Open question: ...
```

## 6. Explanation style reminders
- Plain language first, then symbols. Define every symbol.
- Small numbers in worked examples (for instance weights 0.5 and −1, inputs 2 and 1).
- Describe code by file and function name. No code blocks in explanations.
- Connect each piece back to what the car does whenever possible.

## 7. Version control

**Per task (inside `/do-task`):**
1. Check you are on the current phase branch (`phase-<N>-<short-name>`), not `main`.
2. Run `git status`. Read the list of changed and new files. Anything unexpected (run outputs,
   databases, `.env`, large files, files outside the task's scope) is not staged; find out why it
   appeared.
3. Stage the task's files by name, then run `git diff --staged` and look for anything that resembles
   a secret, a personal path or a large binary.
4. Commit `task X.Y: <summary>`. The pre-commit hooks run automatically; if one fails, fix the
   cause and commit again (never `--no-verify`).
5. Do not push.

**Per phase (`/finish-phase <N>`):**
1. Every task of the phase is ✅ in the checklist, and the full test suite and
   `pre-commit run --all-files` pass.
2. Update the docs: checklist header row gets the tag name, journal gets a phase summary, README
   status line is updated.
3. Commit `docs: complete phase <N>`.
4. Switch to `main`, pull, merge the phase branch with a merge commit (no fast-forward), and create
   the annotated tag `phase-<N>` with a one-line summary of what the phase delivered.
5. Show the user what will be pushed (`main`, the phase branch, the tag) and push only after they
   confirm. Then check that CI is green on GitHub.
6. Create the next phase's branch from `main`, but do not start its first task.

**Going back to a version:** any completed phase can be inspected by checking out its tag
(`phase-<N>`) in a separate worktree or a temporary branch. Never move or delete a tag.

