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
| Parameters | All weights and biases of a network. |
| Initialisation | How the parameters are set before learning starts. |
| Observation | The numbers the car's network receives each step (rays, speed, heading error). |
| Action | The numbers the network outputs (steer, throttle). |
| Policy | Any rule that turns observations into actions. A network is one kind of policy. |
| Episode | One drive from start until crash, time limit, stall or finish. |
| Loss | A number measuring how wrong predictions are; supervised learning makes it smaller. |
| MSE | Mean squared error: the average of (prediction − target)². |
| Derivative / gradient | How much an output changes when an input or parameter changes a tiny bit. |
| Chain rule | To find how A affects C through B, multiply (how A affects B) × (how B affects C). |
| Backpropagation | Applying the chain rule backwards through the network to get every parameter's gradient. |
| Gradient check | Comparing backprop's gradients with ones estimated by nudging each parameter; catches bugs. |
| Gradient descent | Repeatedly moving parameters a small step opposite to the gradient to reduce the loss. |
| Learning rate | The size of each gradient-descent step. |
| Epoch | One pass through the whole training dataset. |
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
| Linter / formatter | Tools (ruff) that flag likely mistakes and rewrite code into one consistent style. |
