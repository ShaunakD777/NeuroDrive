# NeuroDrive — Project Concept

## 1. What NeuroDrive is

NeuroDrive is an interactive laboratory for understanding how neural networks learn. It is built
around a top-down 2D racing simulator. Every car is driven by a small neural network that reads the
car's sensors and decides how to steer and accelerate.

It is **not** "a racing game with a neural network attached". It is a platform with three linked
views that together answer one question:

> **How do different learning algorithms turn the same inputs and driving environment into better
> decisions, and what can their successes and failures teach us about AI?**

The project is also a personal learning journey: from a single neuron, to multilayer networks, to
backpropagation and optimizers, to evolution, reinforcement learning and hybrids. Every concept is
implemented, visualised and measured.

**Guiding principle: Build it, visualise it, experiment with it, measure it.**

## 2. The three linked views

| View | Question it answers | What it shows |
|---|---|---|
| **Environment** | *What does the car do?* | Track, cars, sensor rays, current actions, decision trails, replays |
| **Neural Network** | *Why did it do that?* | Neurons, layers, weights, live activations, contribution of each signal to a chosen output |
| **Experiment** | *How did it come to do that?* | Training runs, checkpoints over time, learning curves, Driving Exam scores, comparisons |

The views share one selection: **Run → Checkpoint → Car → Neuron → Moment in time.**

- **Selecting a car** reveals its neural network, live, for the current frame.
- **Selecting an output neuron** (steer or throttle) explains that decision: which hidden neurons
  and which sensors pushed it, how sensitive it is to each sensor, how it would change if a sensor
  read differently ("what-if"), and how it varied along the car's trail.
- **Selecting a training run** shows how the network and its driving changed over time: scrub a
  timeline of checkpoints, see the weights at each point and what changed since the previous one,
  replay that checkpoint's drive, and see where it sits on every chart.

## 3. Same inputs, same environment, different feedback

Every learning method in NeuroDrive receives **the same observations** (5 distance rays, speed,
heading relative to the track) and produces **the same actions** (steer, throttle). What differs is
**the kind of feedback it learns from**. The environment exposes three feedback "taps":

| Tap | What it provides | Who uses it |
|---|---|---|
| **Teacher label** | "For this exact situation the expert would steer −0.4 and throttle +0.8" | Supervised / imitation learning (backpropagation on a differentiable loss) |
| **Episode summary** | One score for a whole drive: "reached 63 % of the lap, then crashed" | Random search, genetic algorithms, evolution strategies, NEAT (fitness) |
| **Step reward** | A small number after every step: "+0.2 for progress", "−5 for crashing" | Reinforcement learning (policy gradient, actor-critic, PPO) |

Consequences that shape the whole project:

- **Backpropagation needs a differentiable objective** and a per-example learning signal. In
  NeuroDrive that comes from teacher labels (imitation). The teacher may be a scripted expert
  driver, the user driving with the keyboard, or later a champion network.
- **Genetic algorithms need no gradient at all.** They only need to rank whole drives.
- **Reinforcement learning learns from interaction.** Rewards arrive step by step, but their
  consequences are delayed, which creates the credit-assignment problem.
- **Loss, fitness and reward are not interchangeable.** Each is shown only within its own method.
  The only common yardstick across methods is the **Driving Exam**: the same tracks, start
  positions, time limits and driving metrics for every network, however it was trained.
- Every network is differentiable even if it was not *trained* with gradients. So gradients can
  always be used to *explain* a decision.

## 4. Objectives

1. **Learn neural networks from first principles:** neurons, weights, biases, activations, layers,
   forward propagation, losses, derivatives, chain rule, backpropagation, optimizers, initialisation.
2. **Explore learning algorithms from basic to advanced:** supervised imitation, SGD / momentum /
   RMSProp / Adam, random search, genetic algorithms, evolution strategies, NEAT, REINFORCE,
   actor-critic, PPO, and hybrids. They are sequenced by prerequisites.
3. **Make learning visible:** follow a sensor reading through the network to a steering decision,
   and watch learning change that decision.
4. **Compare algorithms scientifically:** controlled, reproducible, multi-seed experiments with
   explicit budgets. No universal winner is declared without evidence.
5. **Stay extensible:** new algorithms, architectures, tracks and fitness/reward functions can be
   added without rewriting the environment or the views.

## 5. Learning stages

| Stage | Concepts | Question the user can answer at the end |
|---|---|---|
| 1. The neuron | Weighted sum, bias, activation functions, output ranges | Why does a sigmoid steering neuron always turn the same way? |
| 2. Networks | Layers, forward propagation, parameter count, initialisation, nonlinearity | Why can't an all-zero network learn different features? |
| 3. Learning from error | Loss, derivatives, chain rule, backpropagation, gradient checking | Walk backprop through a 2-2-1 network by hand. |
| 4. Optimisation | Learning rate, minibatches, SGD, momentum, RMSProp, Adam, overfitting | What happens with a too-high learning rate, and why? |
| 5. Experiment science | Baselines, seeds, held-out tests, budgets, median and spread | Is this improvement real or luck? |
| 6. Imitation | Demonstrations, imbalance, distribution shift, DAgger | Why can a model with tiny loss still crash? |
| 7. Evolution | Population, genome, fitness, selection, crossover, mutation, elitism | How can weights improve without any gradient? |
| 8. First comparison | Fair budgets, confounders | Which learned faster *per unit of experience*, and which generalised? |
| 9. Advanced evolution | Evolution strategies, NEAT | How is ES secretly related to gradient descent? |
| 10. Reinforcement learning | MDPs, return, discounting, policy gradient, critic, advantage, PPO | How does a per-step reward differ from a whole-run fitness? |
| 11. Hybrids | Distillation, pretrain-then-fine-tune | When does combining methods beat each alone? |

## 6. What success looks like

- An interactive platform where a user can watch cars drive, click any car to see its network,
  click an output to understand a decision, and scrub a training run to see learning happen.
- A set of written studies (see `EXPERIMENT_PROTOCOL.md`) that answer the central question with
  measured evidence, including documented failures.
- A user who can explain every algorithm in the project in their own words, with the maths.

The goal is not just a car that completes a lap. The goal is to understand *how* and *why*
different kinds of learning produce, or fail to produce, good decisions.
