"""Superposition-interference instrument for the Water Pattern Tool.

Measures the critical exponent β of the residual stream's signed
interference vs. feature load. Two measurements:

- INTERNAL β: interference among a selected feature cluster.
- CROSS-FIELD β: interference that cluster receives from the broader
  feature population, reported against TWO nulls:
    - uniform null (random k-feature subsets), and
    - magnitude-matched null (random k-feature subsets whose per-
      feature magnitude-bin histogram matches the selected set's,
      with selected features excluded).

The matched null is the load-bearing baseline; the uniform null is
kept for contrast. The headline metric is cross_beta minus the matched
null mean. See experiment.py for the full method and guards.

For background see the briefing in the parallel session's repo and
the SESSION_NOTES.md addendum on this branch.
"""

from .superposition import (
    SuperpositionField,
    random_field,
    coherent_field,
    orthogonal_field,
)
from .experiment import (
    view1_experiment,
    select_crosslingual_features,
    magnitude_matched_null_beta,
    targeted_beta,
)
