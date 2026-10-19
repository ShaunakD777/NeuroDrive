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
