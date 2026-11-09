"""Road network loading and shortest-path utilities (no TraCI needed).

Routes are lists of junction ids; each consecutive pair is joined by the shortest
passenger edge between them. Route geometry is the concatenation of the edge
shapes, so coverage is measured to the actual road polylines.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import sumolib


@dataclass
class RoadNetwork:
    """Directed road graph with edge lengths and polyline geometries."""

    nodes: list                                     # junction ids, fixed order
    index: dict                                     # junction id -> int
    coords: np.ndarray                              # (n, 2) junction coordinates
    out_adj: list                                   # out_adj[u] = [(v, length, edge_id), ...]
    in_adj: list                                    # in_adj[v]  = [(u, length, edge_id), ...]
    edge_shape: dict = field(default_factory=dict)  # (u, v) -> (k, 2) polyline
    edge_len: dict = field(default_factory=dict)    # (u, v) -> length
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)              # xmin, ymin, xmax, ymax

    @classmethod
    def from_sumo(cls, net_file: str, vclass: str = "passenger") -> "RoadNetwork":
        net = sumolib.net.readNet(net_file)
        ids = sorted(n.getID() for n in net.getNodes())          # deterministic order
        index = {j: i for i, j in enumerate(ids)}
        coords = np.array([net.getNode(j).getCoord()[:2] for j in ids], dtype=float)
        out_adj = [[] for _ in ids]
        in_adj = [[] for _ in ids]
        edge_shape, edge_len = {}, {}
        for e in sorted(net.getEdges(), key=lambda e: e.getID()):
            if e.getFunction() == "internal" or not e.allows(vclass):
                continue
            u, v = index[e.getFromNode().getID()], index[e.getToNode().getID()]
            if u == v:
                continue
            length = float(e.getLength())
            key = (u, v)
            if key in edge_len and edge_len[key] <= length:      # keep the shortest parallel edge
                continue
            edge_len[key] = length
            edge_shape[key] = np.array([p[:2] for p in e.getShape()], dtype=float)
        for (u, v), length in edge_len.items():
            out_adj[u].append((v, length))
            in_adj[v].append((u, length))
        xmin, ymin, xmax, ymax = net.getBoundary()
        return cls(ids, index, coords, out_adj, in_adj, edge_shape, edge_len,
                   (float(xmin), float(ymin), float(xmax), float(ymax)))

    # ---------- shortest paths ----------
    def dijkstra(self, source: int, reverse: bool = False):
        """Single-source shortest paths. Ties are broken by node index (deterministic).

        Returns (dist, pred) arrays; pred[v] = previous node on the path (forward) or
        next node toward the source (reverse); -1 if unreachable.
        """
        adj = self.in_adj if reverse else self.out_adj
        n = len(self.nodes)
        dist = np.full(n, np.inf)
        pred = np.full(n, -1, dtype=np.int64)
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v] or (nd == dist[v] and u < pred[v]):
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, pred

    @staticmethod
    def path_from_pred(pred, source: int, target: int) -> list:
        """Forward path source -> target from a forward predecessor array."""
        path = [target]
        while path[-1] != source:
            p = int(pred[path[-1]])
            if p < 0:
                return []
            path.append(p)
        return path[::-1]

    @staticmethod
    def path_to_target(pred_rev, start: int, target: int) -> list:
        """Forward path start -> target from a reverse (toward target) predecessor array."""
        path = [start]
        while path[-1] != target:
            p = int(pred_rev[path[-1]])
            if p < 0:
                return []
            path.append(p)
        return path

    def route_length(self, route: list) -> float:
        return float(sum(self.edge_len[(route[k], route[k + 1])] for k in range(len(route) - 1)))

    def route_polyline(self, route: list) -> np.ndarray:
        """Concatenated edge geometries of a route, shape (k, 2)."""
        parts = []
        for k in range(len(route) - 1):
            shp = self.edge_shape[(route[k], route[k + 1])]
            parts.append(shp if not parts else shp[1:])
        return np.vstack(parts) if parts else self.coords[route[:1]]

    def largest_scc(self) -> np.ndarray:
        """Node indices of the largest strongly connected component."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        rows = [u for u, nb in enumerate(self.out_adj) for _ in nb]
        cols = [e[0] for nb in self.out_adj for e in nb]
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.nodes),) * 2)
        _, lab = connected_components(A, directed=True, connection="strong")
        return np.flatnonzero(lab == np.bincount(lab).argmax())

    def road_points(self, step: float = 25.0) -> np.ndarray:
        """Points along every kept edge shape, at most `step` meters apart."""
        from .geometry import resample
        return np.vstack([resample(shp, step) if len(shp) > 1 else shp for shp in self.edge_shape.values()])

    def nodes_within(self, center: np.ndarray, radius: float) -> np.ndarray:
        d = np.linalg.norm(self.coords - center[None, :], axis=1)
        return np.flatnonzero(d <= radius)
