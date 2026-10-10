---
description: Run the full task loop for exactly one roadmap task, then stop
argument-hint: <task-id, e.g. 1.4>
---
Carry out NeuroDrive task **$ARGUMENTS** using the loop in `docs/WORKFLOW.md` §1.

1. **Read** the task in `docs/ROADMAP.md`, `docs/CHECKLIST.md`, the relevant
   `docs/ARCHITECTURE.md` contracts and ADRs. Confirm the dependencies are ✅.
2. **Brief** (template `docs/WORKFLOW.md` §2). If the task is ⚑ or a new architectural decision is
   needed, stop after the Brief and ask.
3. **Implement** only the files named in the Brief. Mark the task ◐ in the checklist. Follow every
   rule in `CLAUDE.md` §3–4 (scope, determinism, seeded randomness, no ML frameworks, layering).
4. **Verify** by running the task's tests or experiment. Show the commands and summarise the real
   output. If it still fails after two fix attempts, stop and report the diagnosis.
5. **Report** (template `docs/WORKFLOW.md` §3) in plain language, with no code blocks: walk through
   how the code maps to the maths and re-run the worked example.
6. **Record:** checklist ☑ → ✅ with date and note; journal entry in `docs/LEARNING_JOURNAL.md`;
   new terms in `docs/GLOSSARY.md`; an ADR if a decision was made; update `CLAUDE.md` §6 if a
   command changed. Commit on the phase branch following `docs/WORKFLOW.md` §7 (check
   `git status`, stage files by name, review the staged diff for secrets, message
   `task $ARGUMENTS: <summary>`). Do not push.
7. **Stop.** Name the next task but do not start it.
