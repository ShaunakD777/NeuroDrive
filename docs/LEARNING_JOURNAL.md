# NeuroDrive — Learning Journal

One entry per verified task (template in `WORKFLOW.md` §5). Newest at the bottom.
Claude Code drafts each entry; the user may edit or add notes in their own words.

---

### 2026-10-10 — Planning
- Concept: one environment, three feedback channels (teacher labels, episode summary, step reward).
- Key idea: loss, fitness and reward are different objectives; only the Driving Exam is compared
  across methods.
- Open question: which method will generalise best to held-out tracks, and why?

### 2026-10-10 — Task 0.1 Python project skeleton
- Concept: a package (importable folder), a virtual environment (private Python + libraries in
  `.venv/`), a lock file (exact versions of everything installed), a test runner, and an entry
  point that turns a function into the `nd` terminal command.
- Key equation(s) in words: none, plumbing task.
- What we observed: `uv sync` downloaded its own Python 3.12 even though the system has 3.11, and
  wrote `uv.lock`; pytest found and passed 1 test; `nd --help` lists the `version` subcommand.
- Surprise or difficulty: a Typer app with a single command collapses into that command, so a
  callback is needed to keep `nd` a group of subcommands. Ruff 0.17 also checks Python code
  blocks inside Markdown files.
- Open question: `pyproject.toml` says what we want, `uv.lock` says what we got — how strict
  should we be about upgrading the lock, given that results must stay reproducible?

### 2026-10-10 — Task 0.2 Seeded randomness module
- Concept: a pseudo-random generator follows a fixed recipe set by its seed; each sub-component
  gets its own named child stream so changing one never shifts another.
- Key equation(s) in words: child stream = f(master seed, family-tree position, name). It does
  not depend on how many numbers the parent has produced or on the order children are requested.
- What we observed: seed 42 gives 0.774, 0.439, 0.859 every time; seed 43 gives 0.652, 0.044,
  0.020; the child "init" of seed 42 gives 0.890, 0.832, 0.118 even after the parent has drawn
  1000 numbers. A planted `np.random.rand` in the source was caught by the scan test.
- Surprise or difficulty: asking twice for the same child name returns identical streams, by
  design; a component must keep its child, not re-ask. Also, one test I first wrote could not
  fail, so it was removed: a test must be able to fail to prove anything.
- Open question: when evolution runs many seeds in parallel processes, how will each worker
  get its generator without sharing state?

### 2026-10-11 — Task 0.3 Server skeleton
- Concept: REST is request and response; a WebSocket is a connection that stays open so the server
  can push messages (live driving needs this). A health endpoint answers "are you up, and which
  version?".
- Key equation(s) in words: none, plumbing task.
- What we observed: `nd serve` starts on port 8000; `/api/health` returns status ok and the
  version; a WebSocket client receives `hello` first, then gets back whatever it sends.
- Surprise or difficulty: the in-process test client never touches Uvicorn, so it would pass even
  if the real server could not accept WebSockets (plain Uvicorn needs a WebSocket library). We
  added `websockets` up front and confirmed with a real running server, not just the tests.
  Starlette also now deprecates `httpx` for its test client in favour of `httpx2`.
- Open question: when the web app on port 5173 starts calling REST on port 8000, should we allow
  that with CORS or hide the difference behind a Vite proxy?

### 2026-10-12 — Task 0.4 Web skeleton with three panels
- Concept: the UI is split into views that share one selection store (what the user picked); the
  TypeScript types are generated from the Python message models, so the two sides cannot drift.
- Key equation(s) in words: none, plumbing task.
- What we observed: three resizable panels and a badge that turns green only after the server's
  `hello` message arrives; stopping the Python server turns it red, restarting turns it green
  again within seconds (reconnect every 2 seconds).
- Surprise or difficulty: the first generated types were wrong in two ways (a field named
  `version` produced two clashing type names, and fields with defaults came out optional). Both
  were fixed on the Python side (a shared base model, and dropping per-field titles). Several
  tools had new major versions (TypeScript 7, Vite 8, react-resizable-panels 4), so I read their
  installed APIs instead of relying on memory. React's development mode opens each socket twice
  on purpose (it mounts, unmounts and remounts to expose cleanup bugs), which is why the server
  log shows two connections per page load.
- Open question: when the generated types are committed, what is the cheapest check that stops
  them going stale when someone edits `schemas.py`? (Task 0.6's CI is a natural place.)

### 2026-10-12 — Task 0.5 Docs and conventions check
- Concept: documentation is a contract that goes stale; test it the way you test code, by
  running every documented command and comparing every described file with the real tree.
- Key equation(s) in words: none, plumbing task.
- What we observed: every §6 command that exists today runs (`uv sync`, pytest, ruff, `nd serve`,
  `npm install`/`dev`/`test`/`e2e`/`gen:types`); the Vite proxy forwards `/api/health` to the
  server; `nd train`, `sweep`, `exam`, `db` correctly do not exist yet.
- Surprise or difficulty: (1) running `npm install` as documented modified a committed file,
  `web/package-lock.json`, because I had changed the web version after creating it in 0.4;
  fixed here. (2) `uv run pre-commit` fails on this machine: Windows Application Control blocks
  the `pre-commit.exe` launcher ("Permission denied"), although the Python package itself is
  fine (`python -m pre_commit --version` works). This affects task 0.6, which installs git hooks.
  (3) `.env.example` lists `ND_SERVER_PORT` and `ND_RUNS_DIR`, but nothing reads them yet.
  (4) README still says "Planning complete" and does not mention installing uv or Node.
- Open question: for 0.6, how should git hooks run on a machine that blocks the `.exe` launcher?
  Options are using `python -m pre_commit` in the documented commands, asking for the launcher
  to be allowed, or relying on CI for the checks.

### 2026-10-12 — Task 0.4b Views as pages, with a companion drawer and a layout switch
- Concept: client-side routing (the address hash names the page, and changing it never reloads
  the document) and focus + context (one large view plus a compact linked view in a drawer).
- Key equation(s) in words: none, plumbing task.
- What we observed: the app opens at `#/environment`; tabs and the back button switch pages;
  Split and All work and the choice survives a reload. Switching pages kept the same document
  (a marker set on the window survived) and opened no new WebSocket, which is the proof that the
  selection store and the live connection survive page changes.
- Surprise or difficulty: layout got its own store rather than joining the selection store,
  because "how the screen is arranged" and "what the user picked" change for different reasons.
  Saved settings are validated on load, so a stale or hand-edited value falls back to defaults
  instead of breaking the page.
- Open question: on a phone-width screen the open drawer is cramped; should it start closed there,
  or slide over the page instead of sitting beside it?

### 2026-10-12 — Task 0.6 Version control safety net
- Concept: a pre-commit hook is a gate git runs before recording a commit; secret scanning looks
  for key-shaped text; CI re-runs every check on a fresh machine for every push; pinning versions
  keeps checks stable over time.
- Key equation(s) in words: none, plumbing task.
- What we observed: a random GitHub-token-shaped string was caught by gitleaks (rule
  `github-pat`); a real `git commit` of a fake private key was refused by `detect-private-key` and
  gitleaks, and HEAD did not move. A full-history gitleaks scan of all 9 commits found no leaks.
  Every command CI runs passed locally.
- Surprise or difficulty: Smart App Control blocked the `pre-commit.exe` launcher but none of the
  hooks: the installed git hook calls Python directly, and even the gitleaks program that
  pre-commit built from source (downloading Go by itself) was allowed to run. Ruff runs through
  `uv run` so the hook and CI use one ruff version.
- Open question: should the Playwright smoke test also run in CI, given it needs a browser
  download and both servers?

### 2026-10-12 — Phase 0 complete (tag `phase-0`)
- What was built: a uv-managed Python package (`nd` CLI); seeded, name-keyed random streams with a
  guard against global randomness; a FastAPI server with a health check and a live WebSocket
  (api v1.1); a React + TypeScript viewer with three views as pages (companion drawer, Split and
  All layouts) and a live connection badge; TypeScript types generated from the Python models;
  commit hooks (gitleaks, key detection, large files, ruff) and CI (Python, web, type drift,
  full-history secret scan). 21 Python tests, 14 web unit tests, 6 browser tests.
- What was learned: reproducibility starts before any maths (lock files, seeds, pinned tools);
  tests only prove what they check (an in-process test client never touches the real server, and
  a test that cannot fail proves nothing); documentation drifts unless it is run like code;
  security tooling must be checked by planting a fake secret, not assumed.
- Decisions: ADR-015 (dev proxy), ADR-016 (api v1.1), ADR-017 (views as pages).
- Open questions: run the Playwright smoke test in CI? How strictly to upgrade lock files? Should
  the companion drawer start closed on narrow screens?

### 2026-10-13 — Task 1.1 Activations and the single neuron
- Concept: a neuron is a weighted sum plus a bias, squashed by an activation function; the
  activation alone fixes the range of possible outputs.
- Key equation(s) in words: z is each input times its weight, added up, plus the bias; the output
  is f of z. tanh gives −1..1, sigmoid 0..1, ReLU 0 upwards, linear anything.
- What we observed: inputs [2, 1], weights [0.5, −1], bias 0.2 give z = 0.2 and tanh 0.19738,
  sigmoid 0.54983, ReLU 0.2. Random sigmoid neurons with huge weights never produced a negative
  output, previewing the Milestone 1 sigmoid failure.
- Surprise or difficulty: the obvious sigmoid formula overflows for very negative sums, so it is
  computed from e^(−|z|) instead; a test at ±1000 with warnings turned into errors guards this.
- Open question: when do we need the derivatives (Phase 3), and will names alone stay enough once
  networks mix activations per layer?

### 2026-10-13 — Task 1.2 Track model
- Concept: a road is a centreline plus a width; arc length s is the distance along it, the tangent
  its direction, and the edges are the centreline pushed sideways by half the width along the
  left normal. Projecting a point gives s (how far along) and e (how far to the left).
- Key equation(s) in words: left edge = centreline point + (w/2) times the left normal; e is the
  sideways distance from the nearest centreline point, positive on the left.
- What we observed: a 50 m radius circular road has length 314.14 m (true 314.16 m); a point 3 m
  inside gives e = +3.00 and 3 m outside gives −3.00. train_loop_01 is a 731 m kidney-shaped loop
  whose tightest bend has a 22.5 m radius.
- Surprise or difficulty: my first corner points made a 15 m bend and the "wider than the bend"
  test I first wrote did not fire, because a circle whose half-width exceeds its radius just
  flips into a smaller circle without crossing itself. So the loader now also checks the bend
  radius against half the width, and a dumbbell-shaped road shows the edge-crossing check works.
- Open question: is a 22.5 m minimum radius gentle enough for a one-neuron driver at the speeds
  task 1.3 will allow?

### 2026-10-15 — Task 1.3 Car physics (batched)
- Concept: a kinematic bicycle car stepped in fixed slices of 1/30 s: speed first, then heading
  with the new speed, then position with the new heading. A grip limit caps how sharply the car
  can turn at speed.
- Key equation(s) in words: speed gains throttle acceleration and loses drag times speed; the
  heading turns by speed times curvature times dt, where curvature is the steering's wish
  (tangent of the wheel angle over the wheelbase) but never more than 9 divided by speed squared.
- What we observed: from 10 m/s with full throttle and full right steer one step gives speed
  10.2333 and heading change −0.029316 rad (the grip limit decides); at 4 m/s the steering does.
  Full throttle reaches 30 m/s in 4.7 s. A batch of three cars matched three single-car runs
  exactly, bit for bit.
- Surprise or difficulty: at 30 m/s even full lock turns only about 0.3 rad per second, a turn
  radius of 100 m, so the car cannot follow the train loop's bends at top speed; the throttle will
  matter a lot. The contract's starting parameters needed no tuning.
- Open question: will a fixed throttle (task 1.9) be slow enough to stay inside the grip limit on
  the 22.5 m bends?

### 2026-10-15 — Task 1.4 Raycast sensors
- Concept: a sensor reading is the distance along a ray to the first wall, divided by 50 m so it
  lies in 0..1; the ray-versus-segment maths solves for the distance t along the ray and the
  position u along the wall.
- Key equation(s) in words: t is cross(wall start minus car, wall vector) divided by
  cross(ray direction, wall vector); u is cross(wall start minus car, ray direction) over the
  same denominator; a hit needs t at least 0 and u between 0 and 1; a zero denominator means the
  ray is parallel to the wall.
- What we observed: a wall 10 m ahead gives centre 0.2, +-30 degrees 0.2309 (10 / cos 30 = 11.55 m)
  and +-60 degrees 0.4 (20 m). At the start of train_loop_01 the side rays read about 7 m
  (6 / sin 60 = 6.93 m) and the centre ray sees nothing (0.95). Batched and scalar versions
  agreed to 1e-9 on 60 track cases and 150 random-wall cases. Breaking the code on purpose (letting
  rays hit behind the car, or shrinking the allowed range along a wall) made the tests fail.
- Surprise or difficulty: a ray aimed exactly at the shared corner of two walls can slip between
  them through rounding, so u gets a 1e-9 slack; the corner test fails without it. 100 cars take
  about 21 ms per call, fine for live driving, a likely bottleneck for populations later (ADR-013).
- Open question: should rays be tested against only nearby wall segments rather than all 1462?

### 2026-10-15 — Task 1.5 Progress, crash and termination
- Concept: progress is the running total of how far the car's arc-length position changes each
  step, wrapped so crossing the start line is not a jump; the episode ends on a crash, finish,
  stall or time limit, and finished cars freeze while the others carry on.
- Key equation(s) in words: ds is the new arc length minus the old one, shifted by whole loop
  lengths into the range minus half to plus half a lap; P adds up ds; laps is floor(P / L); a crash
  is |e| greater than half the road width minus the car's half-width (5 m here).
- What we observed: a simple scripted driver (steer from heading error and lateral offset, hold
  10 m/s) laps train_loop_01 in 76 s with P = 731.02 m against a 730.74 m loop; driving
  backwards gives negative P; a stopped car stalls at exactly step 60 (2 s). With the wrap rule
  removed, three lap tests failed at once, so they do guard it.
- Surprise or difficulty: my first lap test timed out because 8 m/s for 60 s covers only 480 m,
  less than the 731 m loop; the default 60 s limit is tight for slow hand-tuned drivers, a point
  for task 1.10. A car exactly at a segment joint is equally near two segments, so windowed and
  global searches may pick different neighbours there; arc length and offset still agree.
- Open question: should reset accept a seeded random start offset (the Driving Exam uses fixed
  starts, but training may want variety), and which seed would it use?

### 2026-10-15 — Task 1.5 follow-up: 2 laps, 90 s, seeded random starts
- Concept: episode rules and start positions are choices, so they live in config and in a seeded
  generator passed in by the caller, never in hidden global state.
- What we observed: defaults are now 2 laps and a 90 s limit (matching the Driving Exam). `reset`
  can draw random start positions along the loop, sideways offsets and heading offsets from a
  seeded generator: the same seed gives the same starts, a different seed differs, and without
  randomness the generator is not used at all.
- Surprise or difficulty: the repo's guard test bans the text "np.random" everywhere except
  `core/rng.py`, even in type hints, so `core/rng.py` now exports a type alias `Rng` for hints.
- Open question: should training use random starts by default while exams keep fixed ones?

### 2026-10-15 — Task 1.6 Policies, episode runner and recording
- Concept: a policy is a function from observation to action; an episode is a policy run in a
  world; a recording stores the start and the actions, so the drive can be reproduced exactly.
- Key equation(s) in words: the observation is five ray readings, speed divided by top speed, and
  the heading error (car heading minus road heading, wrapped into minus pi to plus pi) divided
  by pi.
- What we observed: a tiny hand-made driver that sees only the observation (steer away from the
  nearer wall and towards the road direction) drives 369 m of train_loop_01 before clipping a
  wall; its 625-step recording is about 99 KB; saving, loading and replaying it gave identical
  arrays, and so did a four-car run from seeded random starts.
- Surprise or difficulty: a replay needs the start the world actually used after random draws, so
  `World` now keeps `start`; and a recording needs one more frame of state than of actions (T + 1
  against T), a classic off-by-one worth writing down.
- Open question: should the network view's per-step ForwardTrace live inside the same file or in
  a companion file once Phase 2 adds it?

### 2026-10-15 — Task 1.7 Live session over WebSocket
- Concept: a simulation loop that keeps running on a clock while commands arrive at any moment;
  the clock decides how many fixed physics steps are due, the speed factor scales that number, and
  a frame is sent once max(1, speed) steps are due.
- Key equation(s) in words: steps due per real second is 30 times the speed; frames per second is
  the smaller of 30 and 30 times the speed.
- What we observed: with a hand-driven clock, speeds 0.5, 1, 2 and 8 all advanced 30 times speed
  steps per second with at most 30 frames per second; against the real server, 2 seconds at
  speed 1 gave 60 frames and 60 steps, and speed 4 gave about 118 steps in a second. Planting two
  bugs (removing the frame limit, ignoring set_speed) made five tests fail.
- Surprise or difficulty: on Windows the timer rounds a 1/60 s sleep up to about 31 ms, which
  halved the frame rate to 16 per second until the loop's tick was shortened and frames were tied
  to due steps instead of to the gap since the last frame. Also, a constant-throttle car drives
  straight off the curving road within about 7 s, so speed tests must stay short.
- Open question: when 1.8 draws these frames, will 30 per second look smooth enough, or should
  the browser interpolate between frames?

### 2026-10-16 — Task 1.8 Environment view v1
- Concept: the browser only displays and sends commands. World metres become screen pixels with a
  scale and a flipped y axis; a camera either follows the car or fits the whole track; rays are
  drawn from the server's distances, never recomputed in the browser.
- Key equation(s) in words: screen x is half the canvas width plus (world x minus camera x) times
  pixels-per-metre; screen y is half the height minus (world y minus camera y) times pixels-per-metre.
- What we observed: with a centre at (10, 5) and 2 pixels per metre on a 200 by 100 canvas, world
  (11, 6) lands at pixel (102, 48). Holding the up arrow for 1.8 s at 1x took the car to 13 m/s
  with side rays reading 6 to 9 m; driving straight at 8x crashed it, the banner said "Crashed: the
  car left the road", the nearest ray read under 6 m, and Reset cleared it.
- Surprise or difficulty: after choosing a speed from the dropdown the menu kept keyboard focus,
  so the arrow keys changed the menu instead of driving; the menus now release focus. On a phone
  the companion drawer takes nearly half the width; the view scrolls, but the drawer should
  probably start closed on narrow screens (an old open question from task 0.4b).
- Open question: will the 30 frames per second look smooth enough, or should the browser
  interpolate between frames? Also, a manual lap of the track is still for the user to try.

### 2026-10-16 — Task 1.8 follow-up: scenario controls in the UI
- Concept: a feature the server supports but the UI cannot reach is invisible; the controls now
  cover everything `load_scenario` and `select_car` can do: driver, number of cars, random start
  with a seed, and the selected car.
- What we observed: 12 cars with seed 1 spread around the loop; the same seed gave the same start
  again, a different seed gave a different one; the constant driver moves with no keys held; a bad
  number (500 cars, an empty box) is never sent, the box shows the real value again.
- Surprise or difficulty: the check box and the numbers show what the *server* loaded, so a click
  changes them only after the server answers; a test that expected an instant tick failed until it
  waited. A first full browser-test run once failed 7 tests and could not be reproduced in three
  reruns (15 of 15 passed each time); worth watching.
- Open question: should the number of cars be capped lower in the UI than the server's 100?

### 2026-10-16 — Task 1.9 Neuron policy and Neuron view
- Concept: a neuron as a tiny controller. It reads two ray distances, scales each by a weight,
  adds a bias, and the activation turns the total into a steering command. The weights are the
  behaviour: change a number and the car drives differently.
- Key equation(s) in words: z is the left weight times the left ray, plus the right weight times
  the right ray, plus the bias; the steering is the activation of z; positive turns right. A wall
  close on the left means a small left reading, so a useful left weight is negative and the right
  one positive.
- What we observed: with weights −4 and +4 and bias 0.1, a close left wall (0.2 and 0.5) gives
  z = 1.3 and tanh 0.8617 (hard right); a close right wall gives z = −1.1 and tanh −0.8005 (hard
  left); the same inputs through sigmoid give 0.7858 and 0.2497, which both steer right. Weights
  of ±4 to ±24 kept the car on the road for the whole 90 s; all zeros crashed after 9.8 s and
  swapping the signs crashed too.
- Surprise or difficulty: the weights that work are large (about ±8) because a ray reading changes
  only about 0.05 for a metre of sideways drift. The generated TypeScript types also needed the
  schema exporter to strip field titles from nested models, otherwise they came out as clashing
  aliases.
- Open question: with a throttle of 0.15 the car settles near 12 m/s and covers about 1.3 laps in
  90 s, so a 2-lap finish needs a higher throttle; how fast can a hand-tuned neuron safely go?

### 2026-10-16 — Task 1.10 Milestone 1 experiments (study S1)
- Concept: hand-tuning as "the human is the learning algorithm". A good controller needs the right
  *sign* pattern first, a big enough size second, and an output whose range can express the answer.
- Key equation(s) in words: z is the left weight times the left ray, plus the right weight times the
  right ray, plus the bias; steer is the activation of z. A wall close on the left makes the left
  reading small; to turn right the left weight must be negative (a small value removes less from the
  sum than a big one) and the right weight positive, so the weights need opposite signs.
- What we observed: weights −12 and +12, bias 0, tanh, throttle 0.24 finish the exam's 2 laps in
  86.3 s from all 10 exam-like starts and all 10 seeded random starts; the lap was recorded and
  replayed identically. Flipping the signs crashes in 3.6 s (the car steers towards the nearer
  wall, which makes it nearer, positive feedback). 108 sigmoid settings never finished and never
  once steered below zero.
- Why sigmoid only ever steers one way: its output is always between 0 and 1, so the steering
  command is never negative, and negative means left. At the crash the sigmoid neuron's sum was
  −5.4, a clear "go left", and sigmoid turned that into +0.005, "straight". With bias 0 it is worse:
  sigmoid(0) = 0.5, so on a straight road the car is pulled half-right all the time. The weights
  cannot fix this; only the output range (tanh, from −1 to +1) can.
- Surprise or difficulty: the usable throttle window is tiny, 0.23 to 0.25 (about 18.4 to 20 m/s):
  slower misses the 90 s limit, faster cannot take the bends, and a neuron with a fixed throttle
  cannot brake. Weight size hardly matters once it is 4 to 8 or more. The study took 16 minutes to
  run, which is a cue to batch cars or speed up the sensors before the learning phases.
- Open question: would a sigmoid neuron do better on a clockwise (mirrored) track, and is a second
  neuron for the throttle enough to widen the throttle window?

### 2026-10-18 — Task 1.9 follow-up: the neuron keeps its settings
- What changed: Reset already kept the weights; a scenario change (more cars, a random start,
  another track, another driver) built a fresh all-zero neuron. Now a live session keeps one neuron
  for its whole life, and the browser sends the last settings back after a reconnect (ADR-020).
- What we observed: weights −12/+12, bias 0.1, throttle 0.24 and sigmoid survived Reset, 3 cars, a
  random start and a detour through the keyboard driver, in the server tests and in the browser.
- Surprise or difficulty: the browser test first failed because Playwright reused an `nd serve`
  that had been started before the change; a running server keeps running the old code until it
  is restarted.

### 2026-10-18 — Task 1.11 Half-size Monza, both directions
- Concept: a real circuit, traced point by point from a map, scaled to half of Monza's 5,793 m. The
  same road becomes two tracks: clockwise (Monza's racing direction, a training track) and
  counter-clockwise (held out, never trained or tuned on).
- Key equation(s) in words: the scale is half the real length divided by the traced outline's
  length in pixels (1.65 m per pixel); y is flipped because image rows count downwards; driving the
  other way is the same corner points in reverse order, starting from the same start line.
- What we observed: 2,900 m a lap in both directions; long straights, three chicanes, the two
  Lesmos and the Parabolica. The first trace had a 6.1 m bend at the T1 chicane, which left the
  inside edge almost a point, so the chicane was drawn slightly wider: tightest bend now 11.3 m.
- Surprise or difficulty: a track four times longer made the edge check compare 8 million pairs of
  segments (2.4 s and 280 MB per load); comparing only segments whose boxes overlap brought it to
  0.05 s. The track list then still built every track just to read its split, which delayed the
  web page's first frame by over 5 s; it now reads only that field.
- Open question: what time limit should a 2.9 km track have (task 1.12)?

### 2026-10-18 — Task 1.12 Study S1b: the hand-wired neuron on Monza, both directions
- Concept: transfer (a controller used on a track it was not tuned on); the same road driven both
  ways; a per-track time limit; a spatial grid that makes the sensors faster without changing a
  single reading.
- Key equation(s) in words: the time limit is 2 laps times the lap length, divided by 16 m/s,
  rounded up (2 × 2,900.3 / 16 = 362.5, so 363 s). The fastest speed through a bend of radius r
  is √(9 · r), since the tyres allow 9 m/s² sideways: a 17.5 m bend allows √157.5 ≈ 12.5 m/s. A
  fixed throttle u settles at 80 · u m/s, so 0.24 means about 19 m/s.
- What we observed: no fixed throttle finished 2 laps on Monza in either direction. Up to
  throttle 0.19 (clockwise) or 0.1825 (counter-clockwise) the car survives but runs out of time
  (it would need 392 s or 408 s); above that it crashes in a chicane. The S1 baseline crashes in
  the first tight bend every time. Flipped signs and zero weights still crash, so the sign logic
  transfers; only the throttle does not.
- Surprise or difficulty: at throttle 0.1925 the car survives lap 1 and crashes on lap 2 at the
  same chicane, because on lap 1 it was still accelerating (15.2 m/s) and on lap 2 it was at its
  steady 15.4 m/s: 0.2 m/s decided it. Also, throttle 0.13 fails the stall rule before it moves
  2 m. Sigmoid gained almost nothing from a clockwise track.
- Open question: a network that sets its own throttle (Phase 2) could brake for the chicanes; how
  will learning find that?

### 2026-10-18 — Phase 1 complete (tag `phase-1`)
- What was built: activations and a single neuron (`core`); the track model (smooth centreline,
  edges, projection) with `train_loop_01` and half-size Monza clockwise (train) and
  counter-clockwise (held out); batched kinematic-bicycle physics with a grip limit; five raycast
  sensors with a wall grid; progress, crash, stall, finish and a per-track time limit; policies
  (constant, keyboard, neuron), an episode runner and recordings that replay bit for bit; a live
  WebSocket session (api v1.3); the Environment view (canvas, HUD, controls, car selection) and
  the Neuron view (inputs, weights, sum, activation curve, steering dial, live sliders). Studies S1
  and S1b with figures and raw results. 234 Python tests, 58 web unit tests, 26 browser tests.
- What was learned: weights are behaviour, and a sign flip turns a driver into a wall-seeker; an
  output's range decides what behaviour is even possible (sigmoid cannot steer left); a working
  controller can sit in a very narrow window (S1: throttle 0.23 to 0.25), and that window does not
  transfer (S1b: on Monza no fixed throttle is both safe and fast enough); determinism makes every
  result reproducible and every bug repeatable; a speed-up is only trusted when a test proves it
  gives exactly the same numbers.
- Decisions: ADR-018 to ADR-019 (live protocol v1.2 and v1.3), ADR-020 (the neuron keeps its
  settings for the session), ADR-021 (detect-secrets locally, gitleaks in CI), ADR-022
  (per-track time limit).
- Open questions: how will a network that also sets the throttle learn to brake for chicanes?
  Should the Playwright tests run in CI? Phone layout (deferred by the user).

### 2026-10-19 — Task 2.1 Network spec and parameters
- Concept: a layer is a weight matrix (one row per neuron, one column per input) plus a bias
  vector; a network is a list of layers; all its numbers laid end to end are the genome.
- Key equation(s) in words: a layer with n_in inputs and n_out neurons owns n_out times n_in
  weights plus n_out biases. For [7,16,16,2]: 128 + 272 + 34 = 434. Flat order: each weight matrix
  row by row, then that layer's biases, layer after layer.
- What we observed: the hand example [2,2,1] flattens to 1 to 9 in order; flatten then unflatten
  and save then load both return the same numbers bit for bit; 266 tests pass.
- Surprise or difficulty: NumPy adds ".npz" to a file name you give it, so saving through an open
  file keeps checkpoint names exactly as chosen.
- Open question: how big must the initial random weights be so the 2 outputs do not saturate
  (task 2.4)?

### 2026-10-19 — Task 2.2 Forward pass with trace
- Concept: forward propagation, layer by layer, keeping every intermediate value in a trace.
- Key equation(s) in words: the weighted sums z are the weight matrix times the previous layer's
  outputs plus the bias; the outputs a are the activation applied to z; a_0 is the observation.
  For a batch the code computes A times W-transposed plus b, all rows at once.
- What we observed: the hand 2-2-1 example gives z1 = [-1.5, 1.5], a1 = [-0.9051, 0.9051] and
  output -0.8051; a batch of 20 matches one-at-a-time passes to 1e-9.
- Surprise or difficulty: rows of a batch are stored as rows, so the matrix is transposed; the
  maths in the docs is written for a single column vector.
- Open question: does a batch give bit-identical results to single passes? Here only to 1e-9.

### 2026-10-19 — Task 2.3 Population forward pass
- Concept: stacked weights. N networks of one shape keep each layer as one array with a leading
  network index, so one batched matrix product runs every network on its own car.
- Key equation(s) in words: for car i, the weighted sums are network i's weight matrix times car
  i's previous outputs plus network i's bias; then the activation. The same as one network, with
  an extra index.
- What we observed: 20 different random networks on 20 different observations match 20 separate
  single passes to 1e-9; a hand example where the two networks give 3 and 1.5 (network 0 alone
  would have given 8 for the second car).
- Surprise or difficulty: a matrix-vector product per car is written as a matrix times a column
  of height n_in, then the spare dimension is dropped.
- Open question: how fast is this for hundreds of cars (Phase 7 performance check)?

### 2026-10-19 — Task 2.4 Initialisation
- Concept: weights start random so that neurons differ (symmetry breaking), and the scale of the
  start matters: Xavier picks it from the layer sizes.
- Key equation(s) in words: Xavier draws each weight evenly from minus a to plus a, where a is
  the square root of 6 divided by (inputs plus neurons). The variance is then 2 divided by
  (inputs plus neurons). For a 16 to 16 layer: a = 0.4330, variance 0.0625.
- What we observed: the demo shows zero weights give outputs (0, 0) for every input, and equal
  weights of 0.1 give 16 identical hidden neurons (spread 0.0000) with steer equal to throttle;
  Xavier weights give spread 0.36 and 0.28 and different outputs. 200 random default networks
  almost never hit the +-1 walls.
- Surprise or difficulty: equal non-zero weights are as bad as zeros for symmetry, but harder to
  notice, because the network does react to its input.
- Open question: Phase 3 should show that the gradients stay identical too, so learning cannot
  separate symmetric neurons.

### 2026-10-19 — Task 2.5 Freeze the observation/action contract v1
- Concept: every method must see the same 7 inputs and act through the same 2 outputs, otherwise
  a difference in driving might come from the inputs and not from the learning algorithm.
- Key equation(s) in words: the five rays are distance divided by 50 m, far-left to far-right;
  speed is v divided by v_max; heading error is the wrapped angle between the car and the road
  divided by pi, so it lies in (-1, 1] and facing backwards is +1 from either side. Positive steer
  turns right; throttle +1 accelerates and -1 brakes; anything outside [-1, 1] is clipped.
- What we observed: nothing needed to change in the behaviour; the contract was already what the
  docs say. Three deliberately planted bugs (reversed ray order, heading error divided by 2 pi,
  steer sign flipped) were each caught by the new test.
- Surprise or difficulty: the observation range for the heading error excludes -1, because the
  angle wrap puts the seam at +pi; the test pins this so nobody expects a -1.
- Open question: o7 uses track knowledge (the road direction); is that a fair input once methods
  are compared in Phase 8?

### 2026-10-19 — Task 2.6 MLP and Population policies, multi-car scenarios
- Concept: one brain per car. A policy that wraps a network turns the 7 observations into the 2
  actions; a population policy gives car i network i, all in one batched calculation.
- Key equation(s) in words: the action of car i is the output of network i on observation i
  (the forward pass of task 2.2); the world only clips it to [-1, 1].
- What we observed: with seed 0, all 20 random brains choose different first actions, but 14 of
  them brake from the start and never move (a car at speed 0 that brakes stays at 0, so it stalls)
  and the other 6 crash within seconds. Seeds 1 and 2 are similar: random brains are very poor
  drivers, and the best one covered about 96 m of a 731 m lap.
- Surprise or difficulty: most random cars do nothing at all, not something wild. The stall comes
  from the throttle output being negative for about half of random networks. Also, a seed in the
  live scenario changes both the brains and the starts, so a test of the brains has to look at the
  networks directly. A planted bug (the seed ignored for the brains) slipped through until it did.
- Open question: will a fitness that rewards progress be enough to get evolution out of the
  "braking at zero speed" trap in Phase 7?
- Also done: the observation code duplicated in the episode runner and the live session is now one
  function (`observe`), and the runner no longer casts the rays twice per step; the action contract
  has named constants like the observation.

### 2026-10-19 — Task 2.7 Network view v1
- Concept: reading a network diagram. Each column is a layer, each line one entry of a weight
  matrix, each circle one value of the forward trace. Line colour is the weight's sign, thickness
  its size; a circle's fill is the neuron's activation right now.
- Key equation(s) in words: nothing new; the picture shows z (weights times the previous layer's
  values plus the bias) and a (the activation of z) from task 2.2. In the browser, hovering the
  steer output of car 5 read z = 0.5383 and tanh(z) = 0.4917, and tanh(0.5383) is 0.4917.
- What we observed: with 400 lines the diagram stays readable because weak weights are drawn faint
  and strong ones thick and on top; hovering one neuron dims everything unrelated, so its incoming
  and outgoing connections stand out. The input circles equal the rays divided by 50, and the
  numbers change as the car moves.
- Surprise or difficulty: the trace kept by the last `act` call belongs to the step before the
  frame being drawn (the world moves after the policy decides), so a frame needs a trace computed
  for its own observation, otherwise the input circles lag one step behind the rays. A test
  that compares them for exact equality caught the lag when I planted it.
- Open question: with 16 neurons per layer it is hard to say what any one hidden neuron means;
  will the contribution view (task 2.9) make a decision easier to read than the diagram alone?

### 2026-10-19 — Task 2.8 Selection linking (and a compact network for the drawer)
- Concept: one shared selection drives every view. The car, neuron and weight the user points at
  live in one store; the track, the full diagram and the drawer all read it and all change it.
- Key equation(s) in words: none (plumbing). The only rule is a mapping: input k of the
  observation, for k from 0 to 4, is ray k, so selecting it (or a weight leaving it) emphasises
  that ray on the track; speed and heading error have no ray.
- What we observed: clicking a car on the track switches the diagram to that car's weights; the
  same steer neuron selected on car 1 stays selected on car 2 and shows car 2's own values
  (z = 0.2097 on one car, for example); a cell clicked in the drawer is outlined in the full
  diagram and the other way round.
- Surprise or difficulty: two sources of truth fight unless one wins: the server decides which
  car's details it sends, so the store copies the server's choice from every frame, and only an
  explicit action (`selectCar`) sends the command, so following the server never echoes back.
  Also, after a click the mouse is still over the circle, so the readout has to say "Selected"
  whenever the thing shown is the selection, hovered or not.
- Design note: the drawer's compact network was unreadable as a shrunken 41-circle diagram, so
  it is now a summary for a narrow column: input bars, a strip of 16 coloured cells per hidden
  layer, and big output bars with words such as "right" and "accelerating".
- Open question: with the output neuron selected, which inputs matter most? That is task 2.9
  (contributions and sensitivity).

### 2026-10-19 — Task 2.9 Explain: contributions and sensitivity (finite differences)
- Concept: two ways to ask why a network output what it did. Contributions say who pushed this
  neuron and how hard, right now (a receipt for z). Sensitivity says how much the output would
  move if one input changed a little (a slope).
- Key equation(s) in words: contribution of input i to neuron j = weight from i to j times the
  value of i; the contributions plus the bias add up to z. Sensitivity of output m to input k =
  (output with input k nudged up by h, minus output with it nudged down by h) divided by 2h.
- What we observed: in the 2-2-1 example the hidden neuron's receipt is 0.5 and -2 (total -1.5),
  the output's is -1.810297 and +0.905148 plus a bias of 0.1 (z = -0.805148). The output's
  sensitivity is +0.361413 to input 0 and -0.361413 to input 1, matching the chain rule by hand
  to 1.8e-9. Over 200 random networks the receipts matched z - b to within 8.9e-16 (40 % of
  neurons bit for bit; the rest only differ because the forward pass adds in another order).
  When the hidden neurons are pushed into saturation (huge weights), every sensitivity falls
  below 1e-6: the inputs can no longer reach the output.
- Surprise or difficulty: contributions and sensitivity can disagree. A large contribution
  through a saturated neuron has almost no sensitivity, because a saturated tanh is flat: the
  input matters to the current value but a small change would not move it.
- Open question: Phase 3 computes the same slopes exactly with backpropagation; will the two
  agree to 1e-8 on every driving network, including ReLU ones with kinks?

### 2026-10-19 — Task 2.10 Output inspector v1 and what-if
- Concept: explaining one decision. The inspector shows the value in words, the receipt of
  contributions that made it, and the sensitivity to each input; the what-if panel lets you test
  the sensitivity's prediction against the network's real answer.
- Key equation(s) in words: predicted output = output now + sum over inputs of (sensitivity times
  the change in that input). It is a straight-line guess from the slope; the what-if value is the
  network's real answer for the changed inputs.
- What we observed: car 5 (seed-0 random brain, step 12) steers 0.492 ("right"); its biggest
  contributions come from hidden-2 neurons 9 and 7 (+0.189, +0.154). Moving the far-left ray from
  0.14 to 0.24 gave steer 0.496 against a predicted 0.497: the straight line is very good for a
  small nudge. Every one of its five ray sensitivities is positive (+0.05 to +0.38): a longer ray
  in any direction pushes it to steer right. A wall-avoiding driver needs opposite signs for the
  left and right rays (like S1's hand neuron), so this random brain cannot be a good driver.
- Surprise or difficulty: a question about a moving car is stale 30 times a second, so what-if is
  allowed only while paused, and both the sliders and the answer are dropped on the next step.
  Choosing where the computation lives (frame, command or REST) was a real design decision
  (ADR-025), because a live network has no address outside its session.
- Open question: in study S2a, do the circling cars show same-sign ray sensitivities and the
  crashing cars tiny ones (saturation)?

### 2026-10-19 — Task 2.10 follow-up: the what-if ghost car and choosing a car on the Network page
- Concept: a what-if is easier to judge when you see where it leads. The ghost car shows where
  the selected car would be in 2 s if it held the what-if action, next to a dashed path for
  holding its real action. Both come from the same car physics as the simulation.
- Key equation(s) in words: none new; the car model of task 1.3 is stepped 60 times with a fixed
  steer and throttle, once for each action.
- What we observed: car 5 at step 20 (about 2 m/s) with the +30 degree ray set to 0.9 and the
  heading error to -0.9: steer drops from 0.464 to 0.204 and throttle from 0.345 to 0.147, and
  the ghost ends 3.5 m from where the real action would take it. With a 1 s look-ahead the
  ghost was hidden under the slow car, so the look-ahead is 2 s.
- Surprise or difficulty: asking "what if nothing changes" did not return exactly the car's real
  action, for two reasons: the browser's copy of the observation is rounded to 4 decimals, and
  the real action came from the population's batched forward pass while the what-if used a
  single-network pass (different summation order, last-bit differences that 60 physics steps
  can push across a rounding boundary). Sending only the changed inputs and using one function
  for both fixed it: no change now gives bit-identical actions and paths.
- Open question: would a closed-loop ghost (the network deciding at every step) teach more
  than a held action once a network can actually drive?

### 2026-10-20 — Task 2.11: decision trail and looking back
- Concept: one frame shows one decision; a trail shows the sequence. Colouring the trail by the
  selected output turns "how did it steer all the way to the wall?" into a picture, and looking
  back at a step shows what the network saw and computed at that moment.
- Key equation(s) in words: none new. The weights never change while driving, so the trace at step
  t is just the forward pass on the observation stored for step t; recomputed by the same
  function, it equals the live one bit for bit.
- What we observed: seed 1, car 12 drives 311 steps (about 78 m, up to 12.8 m/s) almost straight
  (largest |steer| 0.38, mostly about 0.02 to 0.05) and leaves the road in a bend. At its last step
  the +60 degree ray reads 0.022 (wall 1.1 m away on the left), yet it steers -0.021, slightly
  towards the wall: its sensitivity to that ray is only -0.004, while to the -30 degree ray it is
  +1.25. This random brain listens to the wrong ray. Looking back costs about 0.2 ms per step;
  60 s of 20 cars take 3.8 MB on the server.
- Surprise or difficulty: random brains are slow, so their trails bunch up under the car at normal
  zoom; a click near the car could mean the car or a recent trail point, so the nearer of the two
  wins. On Windows `Trail.tsx` and `trail.ts` count as the same file name, so the helpers became
  `trailLayout.ts`.
- Open question: when trained networks drive whole laps, will 60 s of trail be enough to find the
  moment a run went wrong, or will Phase 5's replays take over that job?

### 2026-10-20 — Task 2.11 follow-up: an honest HUD while looking back
- Concept: every number on screen should describe the same moment. While looking back, the HUD
  now shows the earlier step's speed, status, progress, rays and the action chosen there.
- What we observed: a test that read the HUD at step 2 live and again while looking back at step 2
  got identical numbers, except steer and throttle: live, the HUD shows the last action *applied*
  (chosen one step earlier), while looking back it shows the action *chosen* at that step, which
  is the value the live HUD showed one step later.
- Surprise or difficulty: the first browser run failed on progress only, because Playwright reused
  an `nd serve` still running the previous API version (v1.8), which did not send progress yet.

### 2026-10-20 — Task 2.12: Study S2a, random brains
- Concept: an untrained network's behaviour is set entirely by its random weights; the explain
  tools (receipts, sensitivity, the decision trail) can read those accidental rules off it.
- Key equation(s) in words: the change in an output between two moments is about the sum, over
  the inputs, of sensitivity times the input's change (the what-if prediction, read backwards);
  the tightest circle at full lock is 2 x wheelbase / tan(max wheel angle) = 9.15 m across.
- What we observed: of 2,000 random brains, 72 % stall (956 of the 962 that never move brake at
  step 0: the first throttle is a coin flip, positive 52 % of the time), 28 % crash, 13 drive the
  full 92 s too slowly and one finishes (seed 10, car 11). Crashers' wiring signs match four coin
  flips; long drivers are the lucky well-wired ones. No brain circles: untrained tanh outputs stay
  moderate and a full-lock circle would not fit on the road anyway.
- Surprise or difficulty: the expected "circling car" never appeared in 4,000 tries, so the study
  explains a self-stopper instead. Many "stalled" cars were still speeding up when the 2 m in 2 s
  check ended them. And steering chatter turned out to come from rays grazing the tips of walls: a
  0.3 degree turn flips a reading between 47 m and 16 m.
- Open question: will learned networks learn to ignore grazing rays, or will the sensors need
  smoothing before learning works well?

### 2026-10-20 — Task 2.12 follow-up: random brains on Monza and in other shapes
- Concept: does a harder track or a different network shape change what random brains do?
- Key equation(s) in words: Xavier draws a weight with variance 2 / (n_in + n_out), so a first
  layer with many neurons gets small weights; its neurons' sums of the 7 inputs are then small, and
  the whole network's outputs are gentler (for 64 first-layer neurons the typical sum is about a
  third of a no-hidden-layer network's).
- What we observed: on Monza the shares of stalls and crashes are the same as on the loop, since
  random brains fail in their first few metres. Three of 6,000 app-shaped brains finished Monza,
  something no fixed-throttle neuron managed in S1b, by easing off the throttle when the centre ray
  shortened. The brains that last are largely the same on every track. Wider networks steer more
  gently, stall more and crash less; depth alone changes little; no shape circles.
- Surprise or difficulty: two quick claims in the first draft of the report ("every turn under 90
  degrees", "no brain ever near full lock") were wrong once all 36,000 brains were in; checking
  every claim against the results file caught them. The one brain that held full lock was too slow
  to get anywhere before the stall rule ended it.
- Open question: will a learning method find the "ease off when the road ahead shortens" rule on
  purpose, and will it transfer across tracks the way the lucky brains' rules did?

### 2026-10-22 — Phase 2 complete (tag `phase-2`)
- What was built: in `core/nn`, the network spec and flat genome ([7,16,16,2] = 434 numbers,
  `.npz` files), the forward pass with a full trace, the population forward pass (N networks in
  one batched step) and initialisation (zeros, uniform, Xavier); the observation/action contract
  v1 frozen (ADR-005); MLP and population policies and the "20 random networks" scenario; in
  `explain/`, contributions and finite-difference sensitivity; the live Network view (diagram,
  legend, hover readouts) and its compact drawer version; one shared selection across the views;
  the output inspector with what-if and a what-if ghost car; the decision trail with looking back
  at any step of the last 60 s, the HUD included; study S2a (36,000 random brains over 3 tracks and
  6 shapes) with view screenshots. Live API v1.4 to v1.9. 424 Python tests, 114 web unit tests,
  54 browser tests.
- What was learned: a network is matrices, and its forward pass is a few matrix products whose
  every intermediate value can be kept and shown; zero initialisation makes every hidden neuron
  identical, while Xavier keeps signals a sensible size and sets how gentle a random network is;
  a decision can be explained exactly (contributions add up to z) and locally (sensitivity, the
  straight-line what-if, read forwards or backwards); untrained networks are coin flips (72 %
  stall, 28 % crash, wiring signs like coin flips) and the rare survivors are lucky, well-wired
  brains whose luck can even carry them round Monza; a smooth network fed a jumpy input (a ray
  grazing a wall tip) makes jumpy decisions; Python stays the only source of numbers, so every
  value in the browser can be traced back and re-checked.
- Decisions: ADR-023 (random-networks driver, api v1.4), ADR-024 (network message and trace,
  v1.5), ADR-025 (selected.explain and what-if, v1.6; REST explain/forward deferred to Phase 5),
  ADR-026 (ghost car, overrides-only what-if, v1.7), ADR-027 (decision trail and looking back,
  v1.8), ADR-028 (the HUD shows the step looked at, v1.9).
- Fixed at the end: quick clicks on the Network page's ‹ › lost a click when the server had not
  answered yet; the selection store now remembers the pending request.
- Open questions: will learned networks learn to ignore grazing rays, or do the sensors need
  smoothing? Will a learning method find "ease off when the road ahead shortens" on purpose? Leaky
  ReLU and other activations (deferred by the user); Playwright in CI; the phone layout.

### 2026-10-22 — Task 3.1 Loss functions
- Concept: a loss turns a whole batch of predictions into one "how wrong" number; its derivative
  with respect to each prediction says which way and how hard to push that prediction. That array
  is where backpropagation will start.
- Key equation(s) in words: with residual r = prediction − target and M = number of entries,
  MSE = average of r², derivative 2r / M; Huber = average of ½r² (when |r| ≤ δ) or δ(|r| − ½δ)
  (when |r| > δ), derivative = r capped at ±δ, divided by M.
- What we observed: for ŷ = [2, 0, 3], y = [1, 0, 0], MSE is 3.3333 and Huber is 1.0. Under MSE
  the miss of 3 pushes 3 times harder than the miss of 1; under Huber both push equally. With a
  miss of 100 next to one of 0.2, MSE pushes the outlier 500 times harder, Huber only 5 times.
  Doubling the batch with copies leaves the loss unchanged and halves each entry's gradient
  (because of the 1/M).
- Surprise or difficulty: the first finite-difference test drew random misses and, for one seed,
  none fell below δ = 0.5, so Huber's squared piece was never tested; the test now spreads the
  misses evenly over −4…4. Also, Windows Application Control started blocking the `pytest.exe`
  launcher; `uv run python -m pytest` works.
- Open question: Huber's derivative never exceeds δ / M. Will that make imitation learning steadier
  when the expert's labels are noisy, or just slower when the network is far off?

### 2026-10-22 — Task 3.2 Activation derivatives
- Concept: an activation's derivative f′(z) is the gear ratio between a neuron's weighted sum and its
  output. Backprop multiplies by it at every neuron, so where it is near 0 (saturation) the
  learning signal is almost switched off.
- Key equation(s) in words: tanh′(z) = 1 − tanh(z)²; sigmoid′(z) = s(1 − s) with s = sigmoid(z);
  ReLU′ = 1 for positive z, 0 for negative (and 0 at the corner by convention); linear′ = 1.
- What we observed: at z = 0 the slopes are 1 (tanh), 0.25 (sigmoid), 0 (ReLU, by convention)
  and 1 (linear); at z = 2 tanh's has dropped to 0.0707 and sigmoid's to 0.105. tanh′(5) is under
  0.0002. Through 5 sigmoid layers the factor is at most 0.25⁵ ≈ 0.001 — a first hint of
  vanishing gradients (study S2b, task 3.9).
- Surprise or difficulty: the finite-difference estimate of ReLU's slope at exactly 0 is 0.5 (half
  of each side), which matches neither convention; the test skips points near the corner. A
  planted "relu′(0) = 1" bug was therefore caught only by the hand-value test.
- Open question: how often do ReLU neurons in the driving networks sit at z < 0 for every
  observation ("dead" neurons that can never learn)?

### 2026-10-22 — Task 3.3 Backprop for one neuron
- Concept: backpropagation is the chain rule walked backwards. The loss's push on the output (the
  upstream gradient g) times the activation's slope gives the error signal δ = ∂L/∂z; every other
  gradient of the neuron is δ times one more factor.
- Key equation(s) in words: δ = g · f′(z); weight gradient = δ times that weight's input; bias
  gradient = δ; input gradient = δ times that input's weight. For a batch, weight and bias
  gradients are summed over examples (the 1/M is already inside g).
- What we observed: for the tanh neuron w = [0.5, −1], b = 0.5 on x = [2, 1] with target 1, the
  output 0.462 is too low, δ = −0.846, and every weight and bias gradient is negative ("raise me").
  One step at learning rate 0.1 drops the loss from 0.289 to 0.055. A saturated neuron (z = 10)
  passes back almost nothing (gradients below 1e-7).
- Surprise or difficulty: the input gradient uses the weights and the weight gradient uses the
  inputs; swapping them is an easy mistake and broke 55 of 72 tests. Leaving out f′ still passed
  every linear-activation test, because there f′ = 1.
- Open question: the input gradient δ · w is the neuron's sensitivity to its inputs; will it
  match the finite-difference sensitivity of task 2.9 for whole networks (task 3.6)?

### 2026-10-24 — Task 3.4 Backprop for the full MLP
- Concept: backprop through a network is the one-neuron step of task 3.3 repeated layer by layer
  from the top down. What a layer pushes onto its inputs is exactly the upstream gradient of the
  layer below, and a neuron feeding several others receives the sum of their pushes.
- Key equation(s) in words: per layer, error signals Δ = G ⊙ f′(Z); weight gradient = Δ transposed
  times the layer's inputs (summed over the batch); bias gradient = column sums of Δ; push down
  G = Δ W. The last push is the gradient for the network's inputs.
- What we observed: on the 2-2-1 example (x = [1, 2], target 0) the gradient for the inputs,
  [−0.582, +0.582], equals the loss's push g = −1.610 times task 2.9's finite-difference
  sensitivity [0.361, −0.361]: the same quantity, now exact and from one backward pass. One step
  at learning rate 0.1 on all nine parameters drops the loss from 0.648 to 0.035. All 192
  finite-difference cases agree to 1e-5.
- Surprise or difficulty: two plausible bugs (forgetting f′, or passing G down instead of Δ) still
  pass about half the tests, every case where the layer is linear and f′ = 1. Mixed activations
  in the tests are what catch them.
- Open question: in the [7, 16, 16, 2] driving network with tanh everywhere, how much smaller is
  Δ in the first layer than in the last on real observations (task 3.9, vanishing gradients)?

### 2026-10-24 — Task 3.5 Gradient checker
- Concept: backprop is checked against finite differences, which use only the forward pass:
  nudge one parameter up and down by h, rerun, and divide the loss change by 2h. Agreement to
  many digits on hundreds of random networks is strong evidence that backprop is right.
- Key equation(s) in words: central difference n_k = (L(θ_k + h) − L(θ_k − h)) / 2h; relative
  error r = length of (b − n) divided by (length of b + length of n); pass when r < 1e-6.
- What we observed: all 240 sweep cases (every hidden × output activation, 5 architectures, 3
  seeds) pass, typically r ≈ 4e-11. The worst, 3.2e-7, is a saturated tanh output whose gradients
  are tiny (norm 4e-4): there the rounding noise of the estimate is large compared with the
  gradient. A smaller h made it worse (3.4e-6 at h = 1e-6, 3.2e-5 at 1e-7), the h trade-off in
  action. A 1 % error in one layer's gradient gives r ≈ 1e-3 and the checker names the layer.
- Surprise or difficulty: a planted "average the bias gradient instead of summing it" bug passed
  17 tests, all with a single example, where mean and sum are equal; batches are essential in
  gradient tests.
- Open question: the sweep uses unit-variance weights, harsher than Xavier; should the training
  code (3.7) run a quick gradient check on its first batch as a safety net?

### 2026-10-24 — Task 3.6 Sensitivity via backprop
- Concept: the sensitivity table is the network's Jacobian. Backprop gives the input gradient of
  whatever number its upstream gradient points at; pointing it at one output (G = 1 there, 0
  elsewhere) gives that output's row. Two copies of the observation with G = the identity give
  steer's and throttle's rows in one backward pass.
- Key equation(s) in words: row m of J = backprop's input gradient with G = e_m; for a hidden
  layer, the same on the network cut off after that layer; the loss's input gradient (task 3.4) is
  g times this table.
- What we observed: the exact and the measured tables agree to a median of 1.4e-12 and at worst
  3.9e-9 over 720 tables (every layer, every activation pair); the hand example's finite-difference
  values were off by about 1e-9, their truncation error. At a ReLU corner the two genuinely differ
  (0 vs 0.5). The Network view's numbers do not change at 4 decimals.
- Surprise or difficulty: the first agreement test refused ReLU networks whose weighted sums were
  exactly 0 (zero biases, all inputs from switched-off ReLUs), although nothing could cross a
  corner there; the right condition is "no nudge switches a ReLU on or off". And the exact version
  is not faster for 7 inputs (84 µs vs 50 µs): finite differences cost grows with the number of
  inputs, backprop's with the number of outputs.
- Open question: in the edge glow of task 3.8, should edges glow by the loss gradient (learning)
  or by this sensitivity (explaining), or should the view offer both?

### 2026-10-24 — Task 3.7 Toy tasks and plain gradient descent
- Concept: gradient descent walks downhill on the loss: every epoch, compute ∂L/∂θ by backprop
  and move every parameter a small step η against it. The learner owns that rule; the runner owns
  the loop and the record on disk (the run folder), so later methods plug into the same frame.
- Key equation(s) in words: θ ← θ − η · ∂L/∂θ, once per pass over the whole toy dataset.
- What we observed: the hand example (one linear neuron, x = [0, 1, 2], y = [1, 3, 5], η = 0.1)
  goes loss 11.67 → 3.21 → 0.90 exactly as computed by hand. The line task recovers slope 2 and
  intercept −0.5 to machine precision; XOR (2-4-1, tanh/sigmoid) goes from 0.263 to 0.0002 on every
  seed tried; sine reaches 0.0034. A too-large step (η = 50 on sine) makes the loss explode to
  1e301 in 50 epochs and the run is stopped as diverged, with the last finite network kept.
- Surprise or difficulty: on sine a *bigger* learning rate gave a *worse* result: η = 0.1 ended
  at 0.019 against 0.0034 for 0.05, and η = 0.2 diverged after 444 epochs; steps that are too
  large bounce across the valley instead of settling. In the prototype, XOR with a linear output diverged to NaN for
  every seed at η = 1 and 2 (4 of 10 seeds at η = 0.5), while a sigmoid output never did; likely
  because the sigmoid's slope (at most 0.25) shrinks every step through it.
- Open question: XOR is solved by every seed with 4 hidden neurons; with only 2 (the textbook
  minimum) how many seeds get stuck, and does the activation matter (study S2b, task 3.9)?

### 2026-10-25 — Task 3.7 follow-up: the driving network copies the hand neuron
- Concept: the same gradient descent that solved XOR, applied to our real driving network with
  a teacher's actions as targets (a preview of imitation learning, Phase 6). The only driver we
  have is the hand-wired neuron of S1, so it is the teacher (`scripts/copy_the_neuron.py`).
- Key equation(s) in words: minimise the mean squared gap between the network's (steer, throttle)
  and the teacher's on every observation the teacher saw; θ ← θ − η ∂L/∂θ with η = 0.5.
- What we observed: 51,743 examples from 20 teacher drives; loss 0.413 → 0.023 (epoch 10) →
  0.0092 (100) → 0.0083 (2,000), with held-out drives tracking it (0.0080), so no memorising.
  At epochs 1 and 10 the student stalls at the start (throttle not yet near 0.24, the S2a
  problem); from epoch 100 on it finishes all 20 drives, like the teacher, although its steering
  is off by 0.07–0.10 on average.
- Surprise or difficulty: the loss stops at about 0.008 and wobbles (0.0083 → 0.0089 → 0.0083),
  and the steering error even grows from 0.065 to 0.10 after epoch 500: a fixed step struggles
  with the teacher's steep steering (weight 12). Driving is forgiving: imperfect copies still lap.
- Open question: the student never leaves the teacher's paths here; what happens on a track where
  a small error takes it somewhere the teacher never went (distribution shift, Phase 6)?

### 2026-10-25 — Task 3.8 Toy lab in the Experiment view
- Concept: watching learning happen, three ways at once: the learning curve says how much the
  network improves, the prediction plot says what it has learned, and the gradient glow says
  where the next step will change it. Every number is computed in Python from the run folder
  (the same forward, backward and loss as training); the browser only draws.
- Key equation(s) in words: glow of a weight = |dL/dw| at that checkpoint divided by the largest
  |dL/dw| of the checkpoint; edge colour = the direction of the next step, −dL/dw; predictions =
  the checkpoint's network on a grid of inputs.
- What we observed: sine (1-16-1) goes from a flat line (loss 1.89) to a curve through the data
  (0.0097) in 300 epochs; at epoch 5 it is a single rising S, and the wave's two bends appear
  later. XOR's map turns from a nearly uniform grey-blue (every output near 0.5) to two blue
  diagonal bands around the 1-corners, with the grey decision boundary between; its learning curve
  is flat for about 50 epochs, then falls from about 0.25 to 0.0008. Late in training the largest
  gradient is tiny (6e-4 for XOR at epoch 1000), and for XOR the brightest edges sit next to the
  output.
- Surprise or difficulty: a flaky browser test exposed a real race (the previous run's slider
  stayed usable while a new run loaded). Mid-task, Windows Smart App Control began refusing two
  of Python's own asyncio DLLs in uv's downloaded Python, which stopped pytest and the server;
  python.org's signed Python 3.12.10 fixed it, Python 3.15 could not install PyYAML (no wheels).
- Open question: in study S2b (task 3.9), will sigmoid networks show dim glow in the first layer
  (vanishing gradients) while tanh and ReLU light it up?

### 2026-10-25 — Task 3.9 Study S2b: activations and vanishing gradients
- Concept: backprop multiplies the learning signal by each layer's activation slope, so the
  activation decides whether the signal reaches the first layers. Sigmoid's slope is at most 0.25;
  tanh's and ReLU's are 1 at best; ReLU's is 0 when the neuron is off.
- Key equation(s) in words: ρ = gradient size of the first weight layer divided by that of the
  output layer, at epoch 1; for sigmoid it shrinks by about 1 / 0.25 = 4 per extra layer.
- What we observed: 960 runs (Study S2b). ρ for sigmoid fell 4.7-fold (sine) and 4.4-fold (XOR)
  per extra layer, to 0.0008 at five layers; tanh, ReLU and linear stayed between 0.75 and 2.
  Deep sigmoid networks learned to predict the average (XOR loss 0.25) within 5 epochs and sat on
  that plateau for the rest of the 3,000; giving them their best learning rate did not help at
  depth 5. ReLU fitted sine 8–16 times better than tanh; on XOR the two were indistinguishable.
  ReLU's failures were all-or-nothing: in the stuck XOR run the whole last hidden layer died, the
  output became a constant 0.5 and every gradient was exactly zero. Linear hidden layers never
  beat the best straight line.
- Surprise or difficulty: saturation was not the cause of sigmoid's vanishing gradients (0 % of
  neurons saturated at the start); the 0.25 slope ceiling alone was enough. And a dead ReLU in a
  deeper layer can come back to life when the layers below it change, while one in the first
  layer cannot. In the toy lab the two failures look opposite: sigmoid's glow fades towards the
  inputs, a tanh run's glow is brightest at the inputs, and the dead ReLU network is dark.
- Open question: would He initialisation or a larger learning rate rescue deep sigmoid and ReLU
  networks, and do the [7, 16, 16, 2] driving networks of Phase 6 show any of these failures?

### 2026-10-26 — Phase 3 complete (tag `phase-3`)
- What was built: in `core`, MSE and Huber losses and the four activation derivatives; backprop
  for one neuron and for the full batched multilayer network (gradients for every weight, every
  bias and the inputs); a gradient checker that compares backprop with central differences;
  `explain` sensitivity switched to backprop, with finite differences kept as the test oracle;
  toy tasks (line, sine, XOR, two spirals), plain gradient descent, learner v0 and run folder v0;
  the toy lab in the Experiment view (loss curve, prediction plot, edge glow by gradient size)
  with six read-only run endpoints; study S2b (960 runs); a side experiment in which the driving
  network copies the hand-wired neuron by gradient descent. API v1.10. 1,261 Python tests, 129
  web unit tests, 57 browser tests.
- What was learned: the loss says how wrong the network is, and the derivative says which way to
  nudge each number; backprop is the chain rule applied layer by layer from the output back, so a
  gradient is a sensitivity that is multiplied by an activation slope at every layer; the gradient
  check is what makes such a long chain of signs and indices trustworthy (worst relative error
  3.2e-7 over 240 networks, median 4e-11); a batch gradient is the sum of the single-example
  gradients; the activation decides whether the learning signal reaches the first layers: sigmoid's
  slope ceiling of 0.25 shrinks the first layer's gradient about 4.5-fold per layer (deep sigmoid
  networks solve 0 of 10), tanh and ReLU keep it near 1, and ReLU can fail all at once when a whole
  layer dies; XOR is solved (loss below 0.01 for seeds 0–4).
- Decisions: ADR-029 (task notes kept as local files), ADR-030 (learner v0 and run folder v0),
  ADR-031 (one-task toy lab and the run endpoints, api v1.10).
- Fixed at the end: a race in the toy lab (the old run's slider shown while a new run loads),
  found by a flaky browser test.
- Environment: Windows Smart App Control blocked uv's Python and several launcher .exe files, so
  the virtual environment now uses python.org 3.12.10 and the server starts with
  `python -m neurodrive serve`.
- Open questions: would He initialisation or a larger learning rate rescue deep sigmoid and ReLU
  networks? Does the [7, 16, 16, 2] driving network show any of these failures in Phase 6? Plain
  gradient descent is slow on the plateau, so do momentum and Adam (Phase 4) change that? Whether
  the user can do backprop on a 2-2-1 network on paper is for the user to confirm; the hand 2-2-1
  example is in the task 3.4 notes.

### 2026-10-26 — Task 4.1 Minibatching and splits
- Concept: an epoch is one pass through the training data; a minibatch is a small group of
  examples used for one learning step; a validation set is held back so its loss can reveal
  memorising instead of learning.
- Key equation(s) in words: batches per epoch K = ceil(N / B); validation size = round(f x N).
- What we observed: 10 numbered examples with B = 4 are dealt as 4, 4, 2 in a new order each epoch
  (for example [4, 6, 2, 7] [3, 5, 9, 0] [8, 1], then [2, 9, 3, 6] [0, 4, 8, 7] [5, 1]); every
  number appears once per epoch; the same seed repeats the orders exactly; f = 0.3 gives 7 training
  and 3 validation examples.
- Surprise or difficulty: none; the care went into rejecting settings that silently misbehave
  (a fraction that rounds to zero validation examples, a batch size of zero).
- Open question: with a small batch the gradient is noisy; does that noise help or hurt on the
  sine task once optimizers arrive (4.2 onwards)?

### 2026-10-26 — Task 4.2 Optimizer contract and SGD
- Concept: an optimizer is the rule that turns a gradient into a parameter change; SGD moves each
  parameter against its gradient by a fixed fraction, the learning rate.
- Key equation(s) in words: new value = old value minus learning rate times gradient.
- What we observed: on the bowl (w - 3)^2 from w = 0 with learning rate 0.1 the walk is 0.6, 1.08,
  1.464, ... and the distance to 3 shrinks by a factor 0.8 each step (1 minus twice the rate); with
  rate 1.1 the factor is -1.2, so the walk overshoots and the error grows.
- Surprise or difficulty: none; the useful discovery is that for a bowl the stability limit is
  exactly rate 1 (factor of size 1), which gives a first rule of thumb for "too high".
- Open question: real losses are not bowls, so where is the too-high threshold for the sine
  network, and does Adam (4.5) move it?

### 2026-10-26 — Task 4.3 Momentum
- Concept: momentum keeps a velocity per parameter, a decaying sum of past gradients, and steps by
  the learning rate times that velocity, like a heavy ball rolling downhill.
- Key equation(s) in words: new velocity = beta times old velocity plus gradient; new parameter =
  old parameter minus learning rate times new velocity. With beta 0 it is plain SGD.
- What we observed: on the bowl (w - 3)^2 the walk is 0.6, 1.62, 2.814 (SGD: 0.6, 1.08), overshoots
  3 and settles; in a narrow valley (one steep and one shallow direction) 100 steps leave momentum
  0.0012 from the minimum and SGD 0.405.
- Surprise or difficulty: settling is slow (the wobble shrinks about 5 percent per step), so my own
  test needed 400 steps rather than 200 to get within 1e-6.
- Open question: when does the overshoot hurt more than the speed-up helps (large beta with a large
  learning rate), and does Adam's normalisation make the choice less delicate?

### 2026-10-31 — Task 4.4 RMSProp
- Concept: RMSProp gives every parameter its own step size by dividing its gradient by the root
  mean square of its recent gradients (an exponential moving average of g squared).
- Key equation(s) in words: s = rho times old s plus (1 - rho) times gradient squared; new
  parameter = old parameter minus learning rate times gradient over (square root of s plus eps).
- What we observed: gradients 10 and 0.1 both give a first step of 0.3162 (= 0.1 / sqrt(0.1));
  the bowl walk is 0.31623, 0.53318, 0.70823; multiplying the loss by 1000 leaves the 50-step walk
  unchanged to nine digits while SGD's steps grow 1000-fold; in the narrow valley RMSProp ends
  0.124 from the minimum (SGD 0.405) with both directions at 0.0876, moving at the same pace.
- Surprise or difficulty: two of my test expectations were wrong. eps is not negligible for a
  gradient of 1e-4 (it shortens the step by 3 parts in 10,000), and with a fixed learning rate
  RMSProp hovers about one step size from the minimum instead of settling, because its steps do not
  shrink as the gradient does.
- Open question: does Adam's bias correction remove the oversized first steps, and does its
  momentum part reduce the hovering?

### 2026-10-31 — Task 4.5 Adam
- Concept: Adam keeps a running average of the gradient (direction, like momentum) and of the
  squared gradient (size, like RMSProp), steps by their ratio, and corrects both for starting at 0.
- Key equation(s) in words: m = beta1 m + (1 - beta1) g; v = beta2 v + (1 - beta2) g squared;
  m-hat = m / (1 - beta1^t), v-hat = v / (1 - beta2^t); new parameter = old minus learning rate
  times m-hat over (square root of v-hat plus eps).
- What we observed: the bowl walk is 0.1, 0.19990, 0.29962, about one learning rate per step from
  the very first; without the correction step 1 would be 3.16 times too long. In the narrow valley
  after 100 steps: SGD 0.405, Momentum 0.0012, RMSProp 0.124, Adam (0.01) 0.317, Adam (0.05) 0.006.
- Surprise or difficulty: Adam with the same learning rate was slower than RMSProp in the valley,
  because its learning rate is almost literally its step length; comparing optimizers at one shared
  learning rate is unfair, so each needs its own tuned value.
- Open question: on the sine task, which optimizer wins when each gets its best learning rate, and
  how wide is each one's range of learning rates that work (Study S3)?

### 2026-10-31 — Task 4.6a Toy learner on the optimizers (split from 4.6, ADR-032)
- Concept: an epoch is now K = ceil(N / B) optimizer steps, one per minibatch, with any of the four
  optimizers; a held-back validation set gives a second loss that can reveal overfitting.
- Key equation(s) in words: steps per epoch = training examples divided by batch size, rounded up;
  each step uses the batch's gradient, an estimate of the full gradient.
- What we observed: the defaults reproduce Phase 3 networks bit for bit; on sine after 1000 epochs
  (from 1.888) SGD reaches 0.0056, Momentum 0.00081, RMSProp 0.0064 and Adam 0.00014, Adam starting
  slower but ending 40 times lower than SGD; with batch 16 and a quarter held back, validation loss
  tracks training loss (0.0031 against 0.0016 at epoch 300), so no overfitting here.
- Surprise or difficulty: keeping Phase 3 bit-identical needed the full batch not to be shuffled,
  because adding the same numbers in another order changes the last bits of a floating-point sum.
- Open question: what does an overfitting run look like on these tasks (a big network, few points),
  and how early does validation loss turn upward?

### 2026-10-31 — Task 4.6b Config system and `nd train` (ADR-032)
- Concept: a config file is the written recipe of a run; validation refuses a bad recipe before
  any work starts; the run folder keeps the resolved recipe, every default written out.
- Key equation(s) in words: none, plumbing task.
- What we observed: `nd train --config configs/toy_sine.yaml` took sine from 1.888 to 0.00341 in
  3000 epochs; changing two lines to Adam with learning rate 0.01 reached 0.0000042, about 800
  times lower; a file with `learning_rat` and `tahn` was refused with three named problems and no
  run folder. The browser tests now train their runs with `nd train` too.
- Surprise or difficulty: storing the nested form in run folders would have crashed the toy lab's
  summary line, which read the old flat fields; caught while checking what reads `config.yaml`,
  fixed with a display helper that reads both forms.
- Open question: which settings deserve a command-line override (today seed, tag, epochs), and
  which should always be changed in a copied file so the change is recorded?

### 2026-10-31 — Task 4.7a Statistics and `nd sweep` (split from 4.7, ADR-033)
- Concept: one run is one roll of the dice, so results are summarised over seeds: the median (not
  dragged by one wild seed), the IQR (spread of the middle half) and a bootstrap interval (how much
  the median moves when the seeds are re-drawn with replacement).
- Key equation(s) in words: percentile position = (n - 1) times p / 100, interpolating between
  neighbours; a diverged run counts as infinity.
- What we observed: SGD on sine, 300 epochs, seeds 0-4: median final loss 0.00996 [IQR 0.00974 to
  0.01222]; Adam (0.01): 0.00508 [0.00264 to 0.00861]; paired bootstrap of the difference 0.0049
  with 95 % interval [0.0011, 0.0110], which excludes 0, although Adam's seeds spread widely.
- Surprise or difficulty: with an odd number of seeds every bootstrap median is one of the data
  values, so the interval's ends are always actual results and do not change with the bootstrap
  seed; my test had wrongly expected them to.
- Open question: how many seeds does it take before the Adam interval is narrow enough to rank
  optimizers that are closer together than SGD and Adam?

### 2026-10-31 — Task 4.7b Compare-runs view (ADR-033)
- Concept: a curve band, Q1 to Q3 of the seeds at every epoch around the median line, shows whether
  a gap between two methods is bigger than the spread between seeds.
- Key equation(s) in words: per epoch, median and quartiles across runs (4.7a); a diverged run is
  infinity, which the server sends as null and the chart draws at the top edge.
- What we observed: SGD and Adam on sine (5 seeds, 300 epochs) overlap until about epoch 30, then
  Adam's median falls below SGD's while its band widens (one seed at 0.0012, one at 0.0094); the
  view's table shows the same medians and IQRs that `nd sweep` printed.
- Surprise or difficulty: a component called `Charts.tsx` next to `charts.ts` is the same file name
  to Windows, so TypeScript refused it; it became `CompareChart.tsx`. The phone-width screenshot
  showed a clipped tab switch that the tests could not see.
- Open question: Adam's band is much wider than SGD's at the end; is Adam less reliable across
  seeds, or just further along a curve that SGD has not reached yet (Study S3 can run longer)?
- Follow-up (same day, after the user's screenshot): the hover text smeared on a large screen
  because SVG text and its outline are stretched with the chart (about 1.9 times at 1917 px); the
  read-out is now an HTML box. Validation loss only exists for runs that hold examples back, and no
  shipped config did; `configs/toy_sine_val.yaml` does (final validation loss median 0.00677 over 5
  seeds at 1000 epochs).

### 2026-10-31 — Task 4.10 Environment view: mouse pan and zoom (added at the user's request)
- Concept: the camera (a world point at the canvas centre and a scale in pixels per metre); zooming
  around the cursor keeps the point under the mouse fixed; panning moves the centre against the drag.
- Key equation(s) in words: new centre = point under the cursor minus the cursor's offset from the
  canvas centre divided by the new scale; pan: centre minus drag divided by scale (y flipped).
- What we observed: the worked example gives centre (12.5, 0) at scale 4 with 25 m still under the
  cursor; in the browser the point under the cursor moved by less than a pixel; a click had to move
  from pointer press to pointer release so that a press can still turn into a drag.
- Surprise or difficulty: the browser tests caught a real bug (+ in free mode did not redraw because
  the mode did not change), and three of my own test mistakes (reading the camera before the first
  frame, and zooming the car off the canvas before clicking it).
- Open question: should the free view be remembered per page, and should the drawer and the full view
  share one camera?

### 2026-10-31 — Task 4.8 Study S3: optimizers and learning rates
- Concept: the learning rate decides stability (an edge above which training blows up); optimizers
  decide how each weight's step is sized; a fair comparison tunes each method on some seeds and
  reports on fresh ones; overfitting shows as validation loss turning up while training loss falls.
- Key equation(s) in words: gradient descent is stable along a direction of curvature lambda while
  learning rate times lambda is below 2; with momentum while it is below 2 times (1 + beta).
- What we observed: SGD and Momentum on sine blow up between learning rates 0.1 and 0.3; RMSProp and
  Adam never overflow, but RMSProp ends at a loss of 392 at rate 3; at their best rates on fresh seeds
  Adam ends about 580 times lower than SGD, Momentum is fastest to 0.01; a big network on 48 noisy
  points has its best validation loss near epoch 500 and ends 2.5 times worse, yet its best point beats
  the small network's best.
- Surprise or difficulty: Momentum's edge was not 10 times lower than SGD's, as I predicted, but the
  same on the grid and in theory about twice as high; the only unsolved run spiked in its last 20
  epochs; and the study exposed three bugs in the Compare-runs view (a crash on large runs, labels
  that ran ahead of the data, identical truncated names).
- Open question: would a learning-rate schedule (or keeping the best checkpoint) remove the late
  spikes near the edge, and is RMSProp's best rate below the grid?
