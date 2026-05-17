"""Base materials for Case 1.

Players are expected to build phase-1 and phase-2 experiments on top of the
modules exposed here:

* :mod:`traffic_model` — the shared simulation engine.
* :mod:`starter_simulation` — runnable subcritical baseline.
* :mod:`plotting` — small matplotlib helpers used by the reference figures.
"""

from .plotting import log_hist, plot_dual_axis, plot_lines  # noqa: F401
from .traffic_model import TrafficCascadeSystem, TrafficParams, TrafficRunResult  # noqa: F401
