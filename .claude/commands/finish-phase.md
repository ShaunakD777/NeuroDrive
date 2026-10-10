---
description: Complete a phase - verify, merge into main, tag the version, push after confirmation
argument-hint: <phase number, e.g. 1>
---
Finish NeuroDrive phase **$ARGUMENTS** following `docs/WORKFLOW.md` §7 ("Per phase").

1. **Check completeness.** Every task of phase $ARGUMENTS is ✅ in `docs/CHECKLIST.md`, and the
   phase's "Phase done when" criteria in `docs/ROADMAP.md` are met. If anything is missing, list it
   and stop.
2. **Check health.** Run the full Python test suite, the web tests that exist so far, and
   `uv run pre-commit run --all-files`. All must pass.
3. **Check safety.** Run `git status` and review `git diff main...HEAD --stat`: no `runs/`,
   `data/`, databases, `.env` files, keys, large binaries or personal paths may be included.
4. **Update docs:** add the tag `phase-$ARGUMENTS` to the phase header row in the checklist, add a
   phase summary to `docs/LEARNING_JOURNAL.md` (what was built, what was learned, open questions),
   and update the README status line. Commit `docs: complete phase $ARGUMENTS`.
5. **Merge and tag:** switch to `main`, pull, merge the phase branch with a merge commit
   (no fast-forward), then create the annotated tag `phase-$ARGUMENTS` with a one-line summary.
6. **Confirm before pushing:** show the user exactly what will be pushed (`main`, the phase branch,
   the tag) and wait for a yes. Then push, and report the CI result on GitHub.
7. **Prepare the next phase:** create the next phase's branch from `main`
   (`phase-<next>-<short-name>`). Do not start its first task.

Never force-push, never use `--no-verify`, never move or delete an existing tag.
