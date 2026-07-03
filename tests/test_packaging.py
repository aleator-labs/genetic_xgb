"""Packaging invariants: the installed package must advertise its inline types.

``py.typed`` (PEP 561) is what lets mypy consume genetic_xgb's annotations in
downstream projects. This asserts the marker exists next to the *imported*
package. Under the dev environment's editable install that checks the source
tree; it becomes a true wheel-content check only when the suite runs against an
installed wheel — the CI packaging job planned in the roadmap (spec section
1.0). Until that job exists, wheel content is unverified by CI.
"""

from __future__ import annotations

from pathlib import Path

import genetic_xgb


def test_py_typed_marker_ships_with_the_package() -> None:
    package_dir = Path(genetic_xgb.__file__).resolve().parent
    assert (package_dir / "py.typed").is_file()
