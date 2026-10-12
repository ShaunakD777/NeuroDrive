"""NeuroDrive: a 2D racing simulator whose cars are driven by neural networks.

The package is split into layers (see docs/ARCHITECTURE.md §5): ``core`` (maths with no
knowledge of driving), ``env`` (the driving world), ``learn`` (learning algorithms),
``experiment`` (runs, exams, statistics), ``explain`` (why a network acted as it did) and
``server`` (the bridge to the web viewer).
"""

__version__ = "0.1.0"
