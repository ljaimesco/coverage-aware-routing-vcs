"""Coverage penalty J (Eq. penalty) on a grid, with distances to route polylines,
and the capped per-cell values phi_c(r) used by the coverage gain F (Sec. V-B)."""
from __future__ import annotations

import numpy as np

from .geometry import point_polyline_dist2


class CoverageGrid:
    """Square cells over a bounding box; cell centers and areas.

    The cap dbar defaults to the diagonal of the box, which bounds every distance between
    a cell center and a route inside the box, so the cap never truncates (Sec. V-B).
    """

    def __init__(self, bbox, cell: float, dbar: float | None = None, road_points=None,
                 buffer_m: float | None = None):
        """If road_points and buffer_m are given, keep only cells whose center lies within
        buffer_m of the road network (the coverage domain); otherwise use the whole box.
        Cell centers are anchored at the box corner, so 50, 100 and 200 m grids are nested."""
        xmin, ymin, xmax, ymax = bbox
        xs = np.arange(xmin + cell / 2, xmax, cell)
        ys = np.arange(ymin + cell / 2, ymax, cell)
        X, Y = np.meshgrid(xs, ys)
        self.centers = np.column_stack([X.ravel(), Y.ravel()])
        self.n_box_cells = len(self.centers)
        if road_points is not None and buffer_m is not None:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(road_points).query(self.centers, distance_upper_bound=buffer_m)
            self.centers = self.centers[np.isfinite(dist)]
        self.area = np.full(len(self.centers), cell * cell)
        self.cell = cell
        diag = float(np.hypot(xmax - xmin, ymax - ymin))
        self.dbar2 = (dbar if dbar is not None else diag) ** 2

    def phi(self, poly: np.ndarray) -> np.ndarray:
        """Capped squared distance from every cell center to the polyline."""
        return np.minimum(point_polyline_dist2(self.centers, poly), self.dbar2)

    def penalty(self, phis) -> float:
        """J of a route set given the per-route phi arrays (capped at dbar^2)."""
        if len(phis) == 0:
            return float(np.sum(self.area * self.dbar2))
        return float(np.sum(self.area * np.min(np.vstack(phis), axis=0)))
