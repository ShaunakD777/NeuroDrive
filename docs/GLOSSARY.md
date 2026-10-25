# NeuroDrive — Glossary (plain language)

Add a term the first time a task uses it. Keep definitions short and concrete.

| Term | Meaning |
|---|---|
| Neuron | Multiplies each input by a weight, adds them up, adds a bias, then passes the total through an activation function. |
| Weight | How strongly (and in which direction) one input pushes a neuron. |
| Bias | A constant added to the weighted sum; shifts when the neuron "switches on". |
| Activation function | The squashing or bending function applied to the weighted sum (tanh, sigmoid, ReLU). It makes networks nonlinear. |
| tanh | Activation with outputs from −1 to +1; zero in the middle. Good for steering. |
| Sigmoid | Activation with outputs from 0 to 1. Never negative. |
| ReLU | Activation that outputs 0 for negative inputs and the input itself otherwise. |
| Layer | A group of neurons that all read the same inputs. |
| Forward propagation | Computing the output by passing values layer by layer from inputs to outputs. |
| Weighted sum (z) / activation (a) | z is a neuron's weights times its inputs plus its bias, before squashing; a is the output after the activation function. |
| Look back (scrubbing) | Moving through the remembered steps of a drive (the trail slider, or a click on a trail point) to see what the car saw and what its network computed at that moment. The simulation does not move; Python recomputes that step's trace from the stored observation. |
| Forward trace | The record of every z and a in one forward pass, kept for display, explanation and backprop. |
| Shared selection | The one place the web app remembers what the user picked (car, neuron, weight, output). Every view reads and changes it, so a click in one view updates all the others. |
| Network diagram | A drawing of a network: a column of circles per layer, a line per weight. Line colour = weight sign, thickness = size of the weight, circle fill = the neuron's current activation. |
| Random brain | A network with randomly initialised weights and no training. Drives poorly, and differently from every other random brain. |
| Population (of networks) | N networks of the same shape whose weights are stacked into one array per layer, so all N run in one batched calculation, car i using network i. |
| Parameters | All weights and biases of a network. |
| Weight matrix | One layer's weights as a table: one row per neuron, one column per input. Layer sizes 16 → 16 give a 16 × 16 matrix. |
| Bias vector | One layer's biases as a list, one number per neuron. |
| Network spec | The shape of a network (layer sizes, activation names, version) with none of its numbers. |
| Flatten / unflatten | Laying all of a network's parameters out as one list (weights row by row, then biases, layer by layer), and the exact reverse. |
| Initialisation | How the parameters are set before learning starts. |
| Symmetry (of neurons) | Neurons in a layer that start with identical weights compute identical things, and learning treats them alike, so they stay identical. Random initialisation breaks it. |
| Xavier (Glorot) initialisation | Weights drawn evenly from [-a, a] with a = sqrt(6 / (n_in + n_out)), so the variance is 2 / (n_in + n_out) and the signal keeps a steady size from layer to layer. |
| Saturation | A tanh or sigmoid neuron whose weighted sum is so large that the output sits at its limit (-1 or +1) and barely reacts to its input. |
| Observation | The numbers the car's network receives each step (rays, speed, heading error). |
| Contract (observation / action) | The fixed agreement on what the 7 input numbers and 2 output numbers mean, in what order and range. Every method uses it, so differences in driving come from the learning, not the inputs. Changing it needs a new version. |
| Action | The numbers the network outputs (steer, throttle). |
| Policy | Any rule that turns observations into actions. A network is one kind of policy. |
| Episode | One drive from start until crash, time limit, stall or finish. |
| Loss | A number measuring how wrong predictions are; supervised learning makes it smaller. |
| MSE | Mean squared error: the average of (prediction − target)². |
| Residual | One prediction's miss with its sign: prediction − target. |
| Huber loss | Squares small misses (½ r²) but counts misses bigger than a threshold δ only in proportion to their size, so one wild miss cannot dominate. Its derivative is the miss capped at ±δ. |
| Outlier | A data point far from the rest, for example a badly wrong target label. |
| Derivative / gradient | How much an output changes when an input or parameter changes a tiny bit. |
| Activation derivative (f′(z)) | How much a neuron's output moves per unit change of its weighted sum z. Backprop multiplies by it at every neuron; tanh's is at most 1, sigmoid's at most 0.25, ReLU's is 0 or 1. |
| Chain rule | To find how A affects C through B, multiply (how A affects B) × (how B affects C). |
| Upstream gradient (g = ∂L/∂a) | The push arriving at a neuron's output from further along the chain: how the loss changes per unit of that output. For the last layer it comes from the loss's derivative. |
| Error signal (δ, delta) | ∂L/∂z, how the loss changes per unit of a neuron's weighted sum: the upstream gradient times the activation's slope, δ = g · f′(z). Every weight, bias and input gradient of that neuron is δ times one more factor. |
| Backpropagation | Applying the chain rule backwards through the network to get every parameter's gradient. |
| Element-wise product (⊙) | Multiplying two arrays of the same shape entry by entry (not a matrix product). Backprop uses it for Δ = G ⊙ f′(Z). |
| Fan-out | A neuron feeding several neurons in the next layer. In backprop it receives the sum of their pushes: G = Δ W. |
| Gradient check | Comparing backprop's gradients with ones estimated by nudging each parameter; catches bugs. |
| Relative error (gradient check) | ‖b − n‖ / (‖b‖ + ‖n‖): the size of the gap between backprop's gradient b and the finite-difference estimate n, compared with their sizes. 0 = identical, about 1 = unrelated; backprop passes below 1e-6. |
| Rounding error | The tiny error from storing numbers with about 16 significant digits. Subtracting two nearly equal losses magnifies it, so a finite-difference step h cannot be made arbitrarily small. |
| Gradient descent | Repeatedly moving parameters a small step opposite to the gradient to reduce the loss. |
| Learning rate | The size of each gradient-descent step. |
| Epoch | One pass through the whole training dataset. |
| Learning curve | The loss (or another score) plotted against training time, here epochs. Usually drawn on a log scale because the loss falls by factors of ten. |
| Log scale | An axis where each factor of ten takes the same distance (0.001 → 0.01 → 0.1), so both large early changes and small late ones stay visible. |
| Decision boundary | Where a classifier's output crosses halfway between the two labels (0.5 for labels 0 and 1): the line separating the inputs it calls one class from those it calls the other. The toy lab's XOR map shows it as the grey band. |
| Gradient glow | The toy lab's picture of where learning happens: each connection is drawn as bright as its loss gradient |dL/dw| relative to the largest one in that checkpoint. |
| Full-batch gradient descent | Gradient descent where every step uses the whole dataset at once (one step per epoch); minibatches (task 4.1) use a small part per step. |
| Divergence (training) | Learning breaking down: the steps overshoot so badly that the loss and the weights grow without bound, until they become infinite or NaN ("not a number"). Usually a too-large learning rate. |
| Learner / runner | A learner owns a learning method (its networks and update rule) and does one unit of learning on request; the runner owns the loop around it: how long to train, what to record, when to save and when to stop. |
| Run folder | The folder on disk that records one training run: its config, metadata, a metrics line per unit, checkpoints and extras. The master record of the run. |
| Minibatch | A small random subset of the data used for one update. |
| Momentum / RMSProp / Adam | Gradient-descent variants that smooth or rescale steps to learn faster and more stably. |
| Overfitting | Doing well on training data but badly on new data. |
| Imitation learning (behavioural cloning) | Supervised learning where the targets are an expert's actions. |
| Distribution shift | The learner ends up in situations the expert never showed it, so errors compound. |
| DAgger | Let the learner drive, ask the expert what it *should* have done there, add that to the data, retrain. |
| Fitness | One score for a whole drive, used by evolutionary methods. |
| Genome | All of a network's parameters laid out as one list of numbers. |
| Selection / crossover / mutation | Pick the better drivers / mix two parents' genomes / add small random changes. |
| Elitism | Copy the best few drivers unchanged into the next generation. |
| Evolution Strategies (ES) | Estimate which direction improves fitness by trying many random nudges, then step that way. |
| NEAT | Neuroevolution that evolves the network's structure as well as its weights. |
| Reward | A small score given after each step in reinforcement learning. |
| Return | The (discounted) sum of future rewards from a moment onward. |
| Discount factor (γ) | How much future rewards count compared with immediate ones. |
| Policy gradient | Make actions that led to higher return more likely. |
| Value function / critic | A network that predicts the expected return from a state. |
| Advantage | How much better an action turned out than the critic expected. |
| PPO | A policy-gradient method that limits how much the policy changes per update, for stability. |
| Seed | The starting number for a random generator; the same seed gives the same "random" numbers. |
| Driving Exam | The fixed test (tracks, starts, limits, metrics) that every network takes, however it was trained. |
| Held-out track | A track never used during training or tuning; it tests generalisation. |
| Transfer | Using a controller or model in a setting other than the one it was built or trained for (here: a neuron tuned on one loop, driven on Monza). |
| Spatial grid | A speed-up that sorts objects into square cells, so a query only looks at objects in nearby cells instead of all of them. The answer is unchanged; only the work is less. |
| Per-track time limit | The episode time limit set from the track's length: 2 laps at an average of 16 m/s, rounded up to a whole second (ADR-022). |
| Median / IQR | The middle value / the range covering the middle half of results. |
| Checkpoint | A saved snapshot of a network at a moment during training. |
| Package | A folder of Python files that can be imported by name (`import neurodrive`). |
| Virtual environment (venv) | A private copy of Python and libraries for one project (`.venv/`), so projects cannot break each other. |
| Dependency | A library our code needs; listed in `pyproject.toml`. |
| Lock file | `uv.lock`: the exact version of every installed library, so every machine installs the same thing. |
| Test runner | A tool (pytest) that finds functions named `test_…`, runs them and reports pass or fail. |
| Smoke test | A minimal test that only checks the project starts and imports; it proves wiring, not correctness. |
| Entry point | A line in `pyproject.toml` that turns a Python function into a terminal command (`nd`). |
| Pseudo-random generator | A machine that produces random-looking numbers by a fixed recipe set by its seed; same seed, same numbers. |
| Stream | The sequence of numbers one generator produces. |
| Child stream | A generator derived from a parent's seed plus a name ("init", "mutation"); independent of the parent's usage and of other children. |
| Seed sequence | NumPy's way of expanding one seed number into a well-mixed starting state for a generator. |
| Client / server | The client (the browser) asks for things; the server (our Python program) answers and holds the real data. |
| REST | Request and response: the client asks once, the server answers once, the conversation ends. |
| WebSocket | A connection that stays open so either side can send messages at any time; used for live driving. |
| Endpoint | One address on the server (such as `/api/health`) that does one job. |
| Health check | An endpoint whose only job is to say "I am running" (and which version). |
| Selection store | One shared object in the web app that remembers what the user picked (car, neuron, checkpoint, ...), so every view can react to a click in another. |
| Proxy (dev) | The web dev server passing requests for `/api` and `/ws` on to the Python server, so the browser sees only one address. |
| CORS | A browser rule that blocks a page from one address (port 5173) calling a server at another (port 8000) unless the server allows it. |
| Generated types | TypeScript type definitions produced automatically from the Python message models, so both sides agree on every message shape. |
| JSON Schema | A standard, machine-readable description of what a JSON message must look like (field names and types). |
| Smoke test (end-to-end) | A test that opens the real app in a real browser and checks the main things work together. |
| Documentation drift | Docs slowly stop matching the code because the code changed and the docs did not; fixed by re-running and re-reading them. |
| Launcher (.exe) | The small program a tool installs so you can type its name (such as `pre-commit`) in a terminal; it starts the real Python code. |
| Application Control | A Windows security policy that can stop unapproved programs (here, an unsigned launcher) from running. |
| Client-side routing | Changing which page is shown by changing the address (here the part after `#`) without reloading the document, so app state survives and back/forward work. |
| Hash (URL fragment) | The part of a web address after `#`, such as `#/network`; changing it never reloads the page. |
| Focus + context | A layout with one large main view and a small linked view beside it, so both stay visible. |
| Local storage | A small per-browser store a web page can save settings in; it can be unavailable, so the app must work without it. |
| Pre-commit hook | A check git runs just before recording a commit; if it fails, the commit is refused. |
| Secret scanning | Searching text for things shaped like passwords, tokens or keys before they are published. |
| Continuous integration (CI) | A fresh machine (here GitHub Actions) that re-runs the tests and checks on every push. |
| Pinning (versions) | Naming an exact tool version so a check behaves the same today and later. |
| Checksum | A fingerprint of a file; comparing it with the published one proves a download was not altered. |
| Linter / formatter | Tools (ruff) that flag likely mistakes and rewrite code into one consistent style. |
| Weighted sum (z) | A neuron's total before squashing: each input times its weight, all added up, plus the bias. |
| Linear activation | The "do nothing" activation: the output equals the weighted sum, with no limit on its range. |
| Output range | The set of values an activation can produce (tanh −1..1, sigmoid 0..1, ReLU 0..∞); weights cannot change it. |
| Centreline | The line down the middle of the road; the track is this line plus a width. |
| Arc length (s) | Distance travelled along the road from its starting point, like a kilometre marker. |
| Tangent | The unit-length direction the road points at a given place. |
| Normal | A unit-length direction at right angles to the tangent; the left normal points to the left of the direction of travel. |
| Offset curve | A curve made by pushing every point of another curve sideways by a fixed distance; the road edges are offsets of the centreline. |
| Lateral offset (e) | How far a point is to the side of the centreline: positive on the left, negative on the right. |
| Projection (onto the track) | Finding the nearest centreline point to a position, which gives its arc length s and lateral offset e. |
| Catmull-Rom spline | A smooth curve that passes exactly through a list of corner points. |
| Kinematic bicycle model | A simple car model: front wheel angle sets how fast the heading turns, with no tyre forces. |
| Time step (dt) | The fixed slice of time (1/30 s) the simulation advances by each update. |
| Semi-implicit Euler | An update order where speed changes first and the new speed is then used for heading and position, which is more stable than using old values everywhere. |
| Curvature (kappa) | How sharply a path bends: 1 divided by the turn radius, in 1/metres. |
| Grip limit | The cap on sideways acceleration (v squared times curvature); beyond it the car cannot turn tighter and runs wide. |
| Drag | A slowing effect proportional to speed; with no throttle, speed shrinks by a fixed fraction each step. |
| Raycast | Firing an imaginary line (a ray) from a point in a direction and finding the first thing it hits. |
| Ray reading | The distance a ray travelled before hitting a wall, divided by the 50 m range, so it lies between 0 (wall touching) and 1 (nothing in range). |
| Normalisation | Rescaling a number to a standard range (here 0 to 1) so every network input is on a similar scale. |
| Cross product (2-D) | For vectors p and q, p_x*q_y - p_y*q_x; its sign says which side of p the vector q lies on. |
| Property-based test (Hypothesis) | A test that checks a rule on many automatically generated inputs, including awkward ones, instead of a few hand-picked cases. |
| Reference implementation | A slow, obviously correct version of an algorithm, used only in tests to check the fast version. |
| World | The object that holds a track and N cars and advances them step by step, tracking progress and when each car's episode ends. |
| Episode | One attempt: from placing the cars at the start until each car crashes, finishes, stalls or times out. |
| Progress (P) | Total distance driven along the road since the start: forward adds, backward subtracts. |
| Wrap-around (of arc length) | Treating the loop as circular, so going from 725 m to 3 m on a 731 m loop counts as +9 m, not -722 m. |
| Windowed search | Looking for the nearest road point only among segments near where the car was a moment ago, which is faster and cannot jump to a far-away stretch of road. |
| Crash | The car's side reaching a road edge: sideways offset beyond half the road width minus the car's half-width. |
| Stall | Making less than 2 m of progress in the last 3 seconds (checked after the first 2 seconds). |
| Time limit / timeout | The maximum episode length; a car still running when it is reached is stopped with reason "timeout". |
| Observation | The 7 numbers a car sees each step: five ray readings, speed over top speed, and heading error over pi. |
| Heading error | The angle between where the car points and where the road points, wrapped to between minus and plus pi; positive means pointing left of the road. |
| Policy | A function from what a car sees (observation) to what it does (steer and throttle); a neural network is one kind of policy. |
| Action | The two numbers a policy outputs each step: steer and throttle, each between -1 and 1. |
| Episode recording | A saved episode: the start, every action, and the resulting states, enough to replay it exactly. |
| Replay | Re-running a recording's actions from its start; with no hidden randomness it reproduces the original trajectory exactly. |
| WebSocket | A connection that stays open so the server and browser can send each other messages at any moment, which live driving needs. |
| Live session | One browser connection's own simulation: a loaded track and driver, a clock, a speed and a selected car. |
| Frame (live) | One snapshot of every car sent to the browser: positions, speeds, actions and, for the selected car, rays and observation. |
| Speed factor | How much faster or slower than real time the simulation runs; it changes steps per second, never the size of a step. |
| Column arrays | Sending one list per quantity (all x, all y, ...) instead of one object per car; compact and fast to loop over. |
| API version | A number (now v1.2) recording the shape of the messages between server and browser; new messages bump the minor number. |
| Canvas | A drawing surface in the web page that code paints pixels onto; the track, cars and rays are drawn on one. |
| Camera (drawing) | The choice of which world point sits at the centre of the canvas and how many pixels one metre takes up (zoom). |
| HUD (heads-up display) | The numbers drawn over the picture: speed, steering and throttle, progress, ray distances and status. |
| Store (web) | One shared object in the web app that views read from and write to; the live store holds the latest frame from the server. |
| Neuron policy | A driver whose steering is one neuron: it reads the far-left and far-right rays, weights them, adds a bias, and passes the sum through an activation. |
| Contribution | One input's share of the weighted sum: its weight times its value. |
| Sensitivity | How much an output moves per unit change of one input, with the other inputs held still (the slope, or partial derivative). +0.4 means: if the input rose by 0.1, the output would rise by about 0.04. |
| Jacobian | The whole sensitivity table: one row per output, one column per input, entry [m, k] = ∂output_m / ∂input_k. Backprop finds row m by starting from an upstream gradient that is 1 for output m and 0 elsewhere. |
| Random brain | An untrained network whose weights were drawn at random (Xavier, biases 0). Its "driving rules" are accidents of the draw; Study S2a shows most of them stall or crash. |
| Straight-line breakdown | Explaining how much each input caused an output's change between two moments: input k's share is its sensitivity times its change, s_k x (change in input k), and the shares add up to the predicted change. Treats the network as a flat slope around that moment, so it is trustworthy only when the prediction comes close to the real change. The what-if prediction, read backwards. |
| Wiring signs | Whether the signs of steer's sensitivity to the side rays point the wall-avoiding way: negative for the left rays, positive for the right rays (0 to 4 of the four right). Random brains get them right like coin flips. |
| Grazing ray (chatter) | A ray nearly touching the tip of a wall: a tiny turn switches its reading between the far wall and the near one (e.g. 47 m and 16 m), so a network that listens to that ray makes its output jump from step to step. |
| Decision trail | The line drawn behind the selected car showing where it has been in the last 60 s; coloured by the selected output (blue = steer right or accelerate, orange = steer left or brake, thicker = stronger), so you can see the sequence of decisions, not just one. |
| Ghost car | A see-through copy of the selected car drawn where it would be in 2 s if it held the what-if action, next to a dashed path for holding its real action. Computed by the Python car physics. |
| What-if | Asking what a network would output if some of its inputs were different, without changing the simulation. Compared with the sensitivity's prediction, it shows how far the straight-line guess holds. |
| Central difference | Measuring a slope by nudging an input up by a small step h and down by h, then dividing the change in the output by 2h. Its error shrinks like h squared. |
| Activation curve | A graph of an activation function: the weighted sum z along the bottom and the neuron's output up the side; a dot marks where the neuron is right now. |
| Selection linking | Clicking something in one view (a car, a neuron) highlights the related thing in another view. |
| Baseline | A simple reference controller every later method is compared against; here the hand-wired neuron. |
| Positive feedback | An effect that makes its own cause bigger: with flipped weights the car steers towards the nearer wall, which makes that wall nearer still. |
| Negative control | A run deliberately set up to fail (zero weights, flipped signs) to show that the success was caused by the thing being tested. |
| Driving Exam (preview) | The fixed set of starts and limits every controller is judged on: 10 starts, 2 laps, 90 s. Built properly in Phase 5. |
| Interquartile range (IQR) | The spread of the middle half of the results (25th to 75th percentile); reported with the median instead of a single best run. |
