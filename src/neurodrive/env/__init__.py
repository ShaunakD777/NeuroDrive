"""The driving world: track, car, sensors, episodes.

``env`` knows nothing about neural networks. It imports ``core/rng.py`` only; ``env/policies.py``
is the one exception that adapts ``core/nn`` networks (docs/ARCHITECTURE.md §5).
"""
