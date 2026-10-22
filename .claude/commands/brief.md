---
description: Produce the Brief for one roadmap task and save it to task-notes/, without changing project files
argument-hint: <task-id, e.g. 1.4>
---
Prepare the Brief for NeuroDrive task **$ARGUMENTS**, and save it locally.

1. Read the task in `docs/ROADMAP.md`, its row in `docs/CHECKLIST.md`, the contracts it touches in
   `docs/ARCHITECTURE.md` §6, and any related ADRs in `docs/DECISIONS.md`.
2. Check that its dependencies are ✅ in the checklist. If not, say which ones are missing and stop.
3. Write the Brief using the template in `docs/WORKFLOW.md` §2. Follow the explanation style in
   `CLAUDE.md` §1: plain language, every symbol defined, a worked example with small numbers, and
   pseudocode as numbered steps. No code blocks.
4. If the task is marked ⚑ or needs a decision that is not documented, present the options with
   trade-offs and a recommendation.

5. Save the Brief to `task-notes/phase-<N>/task-$ARGUMENTS-brief.md` and add or update its row in
   `task-notes/README.md` (format: `docs/WORKFLOW.md` §2a). The folder is ignored by git.

Do **not** create or edit any other files. Stop after the Brief.
