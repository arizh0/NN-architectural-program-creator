from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import logging
import textwrap

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "layouts"

# Rectangular layout geometry
RECT_WIDTH = 15.0
EDGE_MARGIN = 1.0

MIN_CLEARANCE = 1.0
REL_W_SCALE = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Access-level color coding (single source of truth)
# -----------------------------
ACCESS_LEVEL_ORDER = [
    "PUBLIC_OPEN",
    "PUBLIC_GUIDED",
    "STAFF_GENERAL",
    "TECH_AUTHORIZED",
    "STAFF_RESTRICTED",
    "HAZARD_RESTRICTED",
]

ACCESS_LEVEL_TO_HEX = {
    # Public-facing (cool)
    "PUBLIC_OPEN": "#2E86FF",        # strong blue
    "PUBLIC_GUIDED": "#39C6D6",      # cyan
    # Staff (warm)
    "STAFF_GENERAL": "#F2C94C",      # yellow
    "TECH_AUTHORIZED": "#F2994A",    # orange
    # Restricted (hot)
    "STAFF_RESTRICTED": "#EB5757",   # red
    "HAZARD_RESTRICTED": "#8B1E3F",  # dark red / maroon
    # Fallback
    "UNKNOWN": "#BDBDBD",            # grey
}

ACCESS_LEVEL_TO_NUM = {
    "PUBLIC_OPEN": 1.0,
    "PUBLIC_GUIDED": 0.7,
    "STAFF_GENERAL": 0.3,
    "TECH_AUTHORIZED": 0.1,
    "STAFF_RESTRICTED": 0.0,
    "HAZARD_RESTRICTED": 0.0,
}


def access_color(access: str) -> Tuple[float, float, float]:
    key = str(access).strip().upper() if access is not None else "UNKNOWN"
    hx = ACCESS_LEVEL_TO_HEX.get(key, ACCESS_LEVEL_TO_HEX["UNKNOWN"])
    return mcolors.to_rgb(hx)


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return mcolors.to_hex(rgb, keep_alpha=False)


def rel_luminance(rgb: Tuple[float, float, float]) -> float:
    # Simple luminance proxy for choosing black/white label text.
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@dataclass
class SpaceGeom:
    name: str
    is_circle: bool
    w: float
    h: float
    r: float
    area: float
    access: str
    public_role: float
    is_blocker: bool = False
    blocker_reason: str = ""


# -----------------------------
# Helpers
# -----------------------------
def max_cross_depth() -> float:
    return RECT_WIDTH - 2.0 * EDGE_MARGIN


def is_layout_blocker_rect(h: float, tol_frac: float = 0.92) -> bool:
    return h >= tol_frac * max_cross_depth()


# -----------------------------
# Data loading
# -----------------------------
def load_data(spaces_csv: Path, rel_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(spaces_csv), pd.read_csv(rel_csv)


def build_space_geoms(
    spaces_df: pd.DataFrame,
    area_rescale_tol: float = 0.20
) -> List[SpaceGeom]:
    """
    Rectangles only. If Length_m * Depth_m does not match NetArea_m2 within tolerance,
    rescale w/h (preserving aspect) so that w*h == area.
    """
    geoms: List[SpaceGeom] = []
    for _, row in spaces_df.iterrows():
        name = str(row["SpaceName"])

        w = float(row["Length_m"])
        h = float(row["Depth_m"])
        area = float(row["NetArea_m2"])

        access = str(row.get("AccessLevel", "UNKNOWN"))
        public_role = float(row.get("PublicInterfaceRole", 0.0))

        # Safety: avoid nonsense
        w = max(w, 0.01)
        h = max(h, 0.01)
        area = max(area, 0.01)

        # If dims disagree with area, rescale to area (preserve aspect)
        wh = w * h
        rel_err = abs(wh - area) / max(area, 1e-9)
        if rel_err > area_rescale_tol:
            scale = math.sqrt(area / max(wh, 1e-9))
            w *= scale
            h *= scale

        is_blocker = is_layout_blocker_rect(h)
        blocker_reason = "cross_section_near_full_depth" if is_blocker else ""

        geoms.append(SpaceGeom(
            name=name,
            is_circle=False,
            w=w,
            h=h,
            r=0.0,
            area=area,
            access=access,
            public_role=public_role,
            is_blocker=is_blocker,
            blocker_reason=blocker_reason
        ))
    return geoms


def align_relationship_matrix(rel_df: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    row_names = rel_df["SpaceName"].astype(str).tolist()
    col_names = [c for c in rel_df.columns if c != "SpaceName"]
    M = rel_df.drop(columns=["SpaceName"]).to_numpy(dtype=float)
    return row_names, col_names, M


def subset_by_names(
    geoms_all: List[SpaceGeom],
    rel_rows: List[str],
    rel_cols: List[str],
    M: np.ndarray,
    chosen_names: List[str],
) -> Tuple[List[SpaceGeom], List[str], np.ndarray]:
    name_to_geom = {g.name: g for g in geoms_all}
    col_map = {nm: j for j, nm in enumerate(rel_cols)}
    chosen = [nm for nm in chosen_names if nm in name_to_geom and nm in col_map and nm in rel_rows]
    row_idx = [rel_rows.index(nm) for nm in chosen]
    col_idx = [col_map[nm] for nm in chosen]
    M_sub = M[np.ix_(row_idx, col_idx)]
    geoms_sub = [name_to_geom[nm] for nm in chosen]
    return geoms_sub, chosen, M_sub


def random_initial_subset(
    geoms_all: List[SpaceGeom],
    rel_rows: List[str],
    rel_cols: List[str],
    M: np.ndarray,
    n: int,
    seed: int
) -> List[str]:
    rng = np.random.default_rng(seed)
    name_to_geom = {g.name: g for g in geoms_all}
    rel_col_set = set(rel_cols)
    valid = [nm for nm in rel_rows if (nm in name_to_geom and nm in rel_col_set)]
    n = min(n, len(valid))
    return rng.choice(valid, size=n, replace=False).tolist()


def sparsify_edges(M: np.ndarray, topk: int = 10, abs_thresh: float = 3.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = M.shape[0]
    srcs, dsts, ws = [], [], []
    for i in range(n):
        row = M[i, :]
        idx = np.argsort(-np.abs(row))
        kept = 0
        for j in idx:
            if i == j:
                continue
            if abs(row[j]) < abs_thresh:
                continue
            srcs.append(i)
            dsts.append(j)
            ws.append(float(row[j]))
            kept += 1
            if kept >= topk:
                break
    return np.array(srcs, np.int64), np.array(dsts, np.int64), np.array(ws, np.float32)


# -----------------------------
# GNN init
# -----------------------------
class SimpleMPNN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, layers: int = 3):
        super().__init__()
        self.layers = layers
        self.lin_in = nn.Linear(in_dim, hid_dim)
        self.msg = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(layers)])
        self.act = nn.ReLU()
        self.out_s = nn.Linear(hid_dim, 1)
        self.out_side = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.lin_in(x))
        att = torch.sigmoid(w).unsqueeze(-1)
        for k in range(self.layers):
            m = self.msg[k](h)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, att * m.index_select(0, src))
            h = self.act(self.upd[k](h + agg))
        s = torch.sigmoid(self.out_s(h)).squeeze(-1)
        side = torch.tanh(self.out_side(h)).squeeze(-1)
        return s, side


def build_node_features(geoms: List[SpaceGeom]) -> torch.Tensor:
    feats = []
    for g in geoms:
        aspect = g.w / max(g.h, 1e-6)
        feats.append([
            math.log(max(g.area, 1e-6)),
            math.log(max(g.w, 1e-6)),
            math.log(max(g.h, 1e-6)),
            aspect,
            0.0,  # circles disabled
            float(g.public_role),
        ])
    return torch.tensor(feats, dtype=torch.float32, device=DEVICE)


def train_gnn_init(node_x: torch.Tensor, src: np.ndarray, dst: np.ndarray, w: np.ndarray,
                   epochs: int, lr: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    src_t = torch.tensor(src, dtype=torch.long, device=DEVICE)
    dst_t = torch.tensor(dst, dtype=torch.long, device=DEVICE)
    w_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    model = SimpleMPNN(in_dim=node_x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos_radius = 0.08
    neg_margin = 0.25

    for _ in range(epochs):
        opt.zero_grad()
        s, side = model(node_x, src_t, dst_t, w_t)
        ds = torch.abs(s.index_select(0, src_t) - s.index_select(0, dst_t))
        wpos = torch.relu(w_t)
        wneg = torch.relu(-w_t)
        pos_loss = (wpos * torch.relu(ds - pos_radius) ** 2).mean()
        neg_loss = (wneg * torch.relu(neg_margin - ds) ** 2).mean()
        diff = s.unsqueeze(0) - s.unsqueeze(1)
        repel = torch.exp(-torch.abs(diff) / 0.08).mean()
        side_prod = side.index_select(0, src_t) * side.index_select(0, dst_t)
        side_loss = (wpos * torch.relu(0.2 - side_prod) ** 2).mean()
        loss = pos_loss + neg_loss + 0.2 * repel + 0.2 * side_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        s, side = model(node_x, src_t, dst_t, w_t)

    s_np = s.cpu().numpy()
    side_np = np.where(side.cpu().numpy() >= 0.0, 1.0, -1.0)
    return s_np, side_np


# -----------------------------
# Geometry helpers
# -----------------------------
def compute_r_bounds(geoms: List[SpaceGeom]) -> torch.Tensor:
    rb = [0.5 * math.hypot(g.w, g.h) for g in geoms]
    return torch.tensor(rb, dtype=torch.float32, device=DEVICE)


def compute_half_extents(geoms: List[SpaceGeom]) -> Tuple[torch.Tensor, torch.Tensor]:
    half_w = torch.tensor([0.5 * g.w for g in geoms], dtype=torch.float32, device=DEVICE)
    half_h = torch.tensor([0.5 * g.h for g in geoms], dtype=torch.float32, device=DEVICE)
    return half_w, half_h


# -----------------------------
# Penalties (RECTANGLES ONLY)
# -----------------------------
def collision_penalty_rects_exact(
    x: torch.Tensor, y: torch.Tensor,
    half_w: torch.Tensor, half_h: torch.Tensor,
    clearance: float = 0.0
) -> torch.Tensor:
    dx = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    dy = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))

    sum_hw = half_w.unsqueeze(0) + half_w.unsqueeze(1) + clearance
    sum_hh = half_h.unsqueeze(0) + half_h.unsqueeze(1) + clearance

    ox = torch.relu(sum_hw - dx)
    oy = torch.relu(sum_hh - dy)

    inter = ox * oy
    n = x.shape[0]
    mask = torch.triu(torch.ones((n, n), device=x.device), diagonal=1)
    return ((inter ** 2) * mask).sum() / (mask.sum() + 1e-9)


def bounds_penalty_soft(
    x: torch.Tensor, y: torch.Tensor,
    half_w: torch.Tensor, half_h: torch.Tensor,
    L: float,
    outside_allow: float
) -> torch.Tensor:
    """
    Soft bounds with an allowed overshoot band (outside_allow).
    Inside the rectangular envelope +/- outside_allow => no penalty.
    """
    top = (RECT_WIDTH * 0.5) + outside_allow
    bot = (-RECT_WIDTH * 0.5) - outside_allow
    left = 0.0 - outside_allow
    right = L + outside_allow

    px0 = torch.relu((left + EDGE_MARGIN + half_w) - x) ** 2
    px1 = torch.relu(x - (right - EDGE_MARGIN - half_w)) ** 2
    py0 = torch.relu((bot + EDGE_MARGIN + half_h) - y) ** 2
    py1 = torch.relu(y - (top - EDGE_MARGIN - half_h)) ** 2
    return (px0 + px1 + py0 + py1).mean()


def border_attraction_penalty(y: torch.Tensor, half_h: torch.Tensor, band: float) -> torch.Tensor:
    target = (RECT_WIDTH * 0.5) - EDGE_MARGIN - half_h - band
    target = torch.clamp(target, min=0.0)
    return ((torch.abs(y) - target) ** 2).mean()


def prefix_keepout_penalty(x: torch.Tensor, half_w: torch.Tensor, keepout_end_x: float, exempt_mask: torch.Tensor) -> torch.Tensor:
    need = torch.relu((keepout_end_x + EDGE_MARGIN + half_w) - x) ** 2
    need = torch.where(exempt_mask, torch.zeros_like(need), need)
    return need.mean()


def map_weight_to_target_distance(
    w: torch.Tensor,
    min_sep: torch.Tensor,
    L: float,
    n: int
) -> torch.Tensor:
    base = (L - 2.0 * EDGE_MARGIN) / max(n, 1)
    far_extra = torch.tensor(min(2.5 * base, 0.8 * L), device=w.device, dtype=w.dtype)
    near_extra = torch.tensor(0.25 * base, device=w.device, dtype=w.dtype)
    closeness = torch.sigmoid((REL_W_SCALE * w) / 2.0)
    return min_sep + (closeness * near_extra) + ((1.0 - closeness) * far_extra)


def relationship_penalty(
    x: torch.Tensor,
    y: torch.Tensor,
    r_bound: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    L: float,
    min_clearance: float = 1.0
) -> torch.Tensor:
    dx = x.index_select(0, src) - x.index_select(0, dst)
    dy = y.index_select(0, src) - y.index_select(0, dst)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-9)

    min_sep = r_bound.index_select(0, src) + r_bound.index_select(0, dst) + min_clearance
    target = map_weight_to_target_distance(w, min_sep=min_sep, L=L, n=int(x.shape[0]))

    w_scaled = REL_W_SCALE * w
    wpos = torch.relu(w_scaled)
    wneg = torch.relu(-w_scaled)

    close_loss = (wpos * torch.relu(dist - target) ** 2).mean()
    far_loss = (wneg * torch.relu(target - dist) ** 2).mean()
    return close_loss + far_loss


def even_spacing_penalty(x: torch.Tensor, L: float, x_lo: Optional[float] = None) -> torch.Tensor:
    xs, _ = torch.sort(x)
    n = xs.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    lo = (EDGE_MARGIN + 2.0) if x_lo is None else max(EDGE_MARGIN + 2.0, x_lo + 2.0)
    hi = L - EDGE_MARGIN - 2.0
    if hi <= lo:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    target = torch.linspace(lo, hi, steps=n, device=x.device, dtype=x.dtype)
    return ((xs - target) ** 2).mean()


def blocker_anchor_penalty(
    x: torch.Tensor, y: torch.Tensor,
    block_idxs_t: torch.Tensor,
    x_fixed: torch.Tensor, y_fixed: torch.Tensor
) -> torch.Tensor:
    if block_idxs_t.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    xb = x.index_select(0, block_idxs_t)
    yb = y.index_select(0, block_idxs_t)
    return ((xb - x_fixed) ** 2 + (yb - y_fixed) ** 2).mean()


# -----------------------------
# Corridor geometry
# -----------------------------
def build_corridor_segments(polylines: List[List[Tuple[float, float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    segA, segB = [], []
    for poly in polylines:
        if len(poly) < 2:
            continue
        for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
            segA.append((ax, ay))
            segB.append((bx, by))
    if not segA:
        segA, segB = [(0.0, 0.0)], [(0.0, 0.0)]
    A = torch.tensor(segA, dtype=torch.float32, device=DEVICE)
    B = torch.tensor(segB, dtype=torch.float32, device=DEVICE)
    return A, B


def softmin_dist_points_to_segments(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    v = (B - A)
    vv = (v * v).sum(dim=1).clamp_min(1e-9)
    w = P.unsqueeze(1) - A.unsqueeze(0)
    t = (w * v.unsqueeze(0)).sum(dim=2) / vv.unsqueeze(0)
    t = t.clamp(0.0, 1.0)
    proj = A.unsqueeze(0) + t.unsqueeze(2) * v.unsqueeze(0)
    d2 = ((P.unsqueeze(1) - proj) ** 2).sum(dim=2)
    d = torch.sqrt(d2 + 1e-9)
    return -tau * torch.logsumexp(-d / tau, dim=1)


def rect_boundary_points(x: torch.Tensor, y: torch.Tensor, half_w: torch.Tensor, half_h: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.stack([x - half_w, y - half_h], dim=1),
        torch.stack([x - half_w, y + half_h], dim=1),
        torch.stack([x + half_w, y - half_h], dim=1),
        torch.stack([x + half_w, y + half_h], dim=1),
        torch.stack([x,          y - half_h], dim=1),
        torch.stack([x,          y + half_h], dim=1),
        torch.stack([x - half_w, y         ], dim=1),
        torch.stack([x + half_w, y         ], dim=1),
    ], dim=1)


def corridor_penalty_rects(
    P: torch.Tensor,
    half_w: torch.Tensor, half_h: torch.Tensor,
    A: torch.Tensor, B: torch.Tensor,
    corridor_w: float, margin: float = 1.0, tau: float = 0.5
) -> torch.Tensor:
    x = P[:, 0]
    y = P[:, 1]
    pts = rect_boundary_points(x, y, half_w, half_h)  # [N,8,2]
    N = x.shape[0]
    pts_flat = pts.reshape(N * 8, 2)
    d_pts = softmin_dist_points_to_segments(pts_flat, A, B, tau=tau).reshape(N, 8)
    d_min = d_pts.min(dim=1).values
    need = corridor_w * 0.5 + margin
    pen = torch.relu(need - d_min) ** 2
    return pen.mean()


# -----------------------------
# A* corridor update for the rectangular planning envelope
# -----------------------------
def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if (not grid[sy, sx]) or (not grid[gy, gx]):
        return None

    def heur(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap = []
    gscore = {start: 0.0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    heapq.heappush(open_heap, (heur(start, goal), 0.0, start))
    visited = set()

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)

        if cur == goal:
            path = []
            node: Optional[Tuple[int, int]] = cur
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            return path[::-1]

        cx, cy = cur
        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if not grid[ny, nx]:
                continue
            step = math.hypot(dx, dy)
            ng = g + step
            if ng < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                nf = ng + heur((nx, ny), goal)
                heapq.heappush(open_heap, (nf, ng, (nx, ny)))
    return None


def nearest_free_cell(grid: np.ndarray, cell: Tuple[int, int], max_r: int = 120) -> Optional[Tuple[int, int]]:
    h, w = grid.shape
    cx, cy = cell
    if 0 <= cx < w and 0 <= cy < h and grid[cy, cx]:
        return cell
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx]:
                    return (nx, ny)
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx]:
                    return (nx, ny)
    return None


def simplify_polyline(poly: List[Tuple[float, float]], eps: float = 1.0) -> List[Tuple[float, float]]:
    if len(poly) < 3:
        return poly
    pts = np.array(poly, dtype=float)

    def point_segment_distance(px, py, ax, ay, bx, by) -> float:
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = (wx * vx + wy * vy) / vv
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx = ax + t * vx
        cy = ay + t * vy
        return math.hypot(px - cx, py - cy)

    def rdp(a: int, b: int, idxs: List[int]) -> List[int]:
        ax, ay = pts[a]
        bx, by = pts[b]
        max_d = -1.0
        max_i = None
        for i in idxs:
            px, py = pts[i]
            d = point_segment_distance(px, py, ax, ay, bx, by)
            if d > max_d:
                max_d = d
                max_i = i
        if max_d > eps and max_i is not None:
            left = [i for i in idxs if a < i < max_i]
            right = [i for i in idxs if max_i < i < b]
            return rdp(a, max_i, left)[:-1] + rdp(max_i, b, right)
        return [a, b]

    keep = rdp(0, len(pts) - 1, list(range(1, len(pts) - 1)))
    keep = sorted(set(keep))
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in keep]


def chaikin_smooth(poly: List[Tuple[float, float]], refinements: int = 3) -> List[Tuple[float, float]]:
    if len(poly) < 3:
        return poly
    pts = np.array(poly, dtype=float)
    for _ in range(refinements):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = np.array(new_pts, dtype=float)
    return [(float(x), float(y)) for x, y in pts]


def build_occupancy_grid(
    geoms: List[SpaceGeom],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    corridor_w: float,
    res: float,
    extra_margin: float
) -> Tuple[np.ndarray, float, float]:
    x_min = 0.0
    y_min = -RECT_WIDTH * 0.5
    W = int(math.ceil(L / res)) + 1
    H = int(math.ceil(RECT_WIDTH / res)) + 1
    grid = np.ones((H, W), dtype=bool)

    inflate = corridor_w * 0.5 + extra_margin

    x_block = int(math.ceil(EDGE_MARGIN / res))
    y_block = int(math.ceil(EDGE_MARGIN / res))
    grid[:, :x_block] = False
    grid[:, -x_block:] = False
    grid[:y_block, :] = False
    grid[-y_block:, :] = False

    for i, g in enumerate(geoms):
        cx = float(x[i])
        cy = float(y[i])

        hx = g.w * 0.5 + inflate
        hy = g.h * 0.5 + inflate
        xmin, xmax = cx - hx, cx + hx
        ymin, ymax = cy - hy, cy + hy

        gx0 = max(0, int(math.floor((xmin - x_min) / res)))
        gx1 = min(W - 1, int(math.ceil((xmax - x_min) / res)))
        gy0 = max(0, int(math.floor((ymin - y_min) / res)))
        gy1 = min(H - 1, int(math.ceil((ymax - y_min) / res)))
        grid[gy0:gy1 + 1, gx0:gx1 + 1] = False

    return grid, x_min, y_min


def world_to_cell(xw: float, yw: float, x_min: float, y_min: float, res: float) -> Tuple[int, int]:
    gx = int(round((xw - x_min) / res))
    gy = int(round((yw - y_min) / res))
    return gx, gy


def cell_to_world(gx: int, gy: int, x_min: float, y_min: float, res: float) -> Tuple[float, float]:
    return x_min + gx * res, y_min + gy * res


def polyline_from_path(cells: List[Tuple[int, int]], x_min: float, y_min: float, res: float) -> List[Tuple[float, float]]:
    return [cell_to_world(gx, gy, x_min, y_min, res) for gx, gy in cells]


def choose_branch_targets(geoms: List[SpaceGeom], M: np.ndarray, k: int) -> List[int]:
    public = np.array([g.public_role for g in geoms], dtype=float)
    pos_deg = np.maximum(M, 0.0).sum(axis=1)
    access = np.array([ACCESS_LEVEL_TO_NUM.get(str(g.access).strip().upper(), 0.0) for g in geoms], dtype=float)

    def z(v: np.ndarray) -> np.ndarray:
        s = np.std(v)
        return np.zeros_like(v) if s < 1e-9 else (v - np.mean(v)) / s

    score = 1.5 * z(public) + 0.8 * z(access) + 1.0 * z(pos_deg)
    order = np.argsort(-score)
    return order[:k].tolist()


def blocker_fixed_positions(geoms: List[SpaceGeom], L: float) -> Tuple[np.ndarray, np.ndarray, List[int], float]:
    n = len(geoms)
    x_fix = np.zeros(n, dtype=float)
    y_fix = np.zeros(n, dtype=float)

    block_idxs = [i for i, g in enumerate(geoms) if g.is_blocker]
    if not block_idxs:
        return x_fix, y_fix, [], EDGE_MARGIN + 1.0

    block_idxs.sort(key=lambda i: geoms[i].w, reverse=True)

    x_cursor = EDGE_MARGIN + 2.0
    for i in block_idxs:
        hw = 0.5 * geoms[i].w
        x_center = x_cursor + hw
        x_fix[i] = x_center
        y_fix[i] = 0.0
        x_cursor = x_center + hw + MIN_CLEARANCE + 2.0

    keepout_end = x_cursor
    keepout_end = min(keepout_end, 0.25 * L)
    return x_fix, y_fix, block_idxs, keepout_end


def update_corridor_network(
    geoms: List[SpaceGeom],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    M: np.ndarray,
    corridor_w: float,
    grid_res: float,
    n_branches: int,
    smooth_refinements: int,
    start_x: float
) -> Optional[List[List[Tuple[float, float]]]]:
    grid, x_min, y_min = build_occupancy_grid(
        geoms=geoms, x=x, y=y, L=L,
        corridor_w=corridor_w, res=grid_res, extra_margin=1.0
    )

    sx = float(max(start_x, EDGE_MARGIN + 1.0))
    gx = float(L - EDGE_MARGIN - 1.0)

    start = world_to_cell(sx, 0.0, x_min, y_min, grid_res)
    goal = world_to_cell(gx, 0.0, x_min, y_min, grid_res)

    start = nearest_free_cell(grid, start, max_r=220)
    goal = nearest_free_cell(grid, goal, max_r=220)
    if start is None or goal is None:
        return None

    trunk_cells = astar(grid, start, goal)
    if trunk_cells is None or len(trunk_cells) < 2:
        return None

    trunk = polyline_from_path(trunk_cells, x_min, y_min, grid_res)
    trunk = simplify_polyline(trunk, eps=1.5 * grid_res)
    trunk = chaikin_smooth(trunk, refinements=smooth_refinements)

    polylines: List[List[Tuple[float, float]]] = [trunk]

    targets = choose_branch_targets(geoms, M, k=n_branches)
    if not targets:
        return polylines

    stride = max(1, len(trunk_cells) // 500)
    trunk_sub = trunk_cells[::stride]

    for ti in targets:
        tgt = world_to_cell(float(x[ti]), float(y[ti]), x_min, y_min, grid_res)
        tgt = nearest_free_cell(grid, tgt, max_r=220)
        if tgt is None:
            continue

        best = None
        bestd = 1e18
        for c in trunk_sub:
            dx = c[0] - tgt[0]
            dy = c[1] - tgt[1]
            d = dx * dx + dy * dy
            if d < bestd:
                bestd = d
                best = c
        if best is None:
            continue

        branch_cells = astar(grid, best, tgt)
        if branch_cells is None or len(branch_cells) < 2:
            continue

        branch = polyline_from_path(branch_cells, x_min, y_min, grid_res)
        branch = simplify_polyline(branch, eps=1.5 * grid_res)
        branch = chaikin_smooth(branch, refinements=max(1, smooth_refinements - 1))
        polylines.append(branch)

    return polylines


# -----------------------------
# Deterministic final "no-overlap" pass
# -----------------------------
def deoverlap_rects_inplace(
    x: np.ndarray, y: np.ndarray,
    half_w: np.ndarray, half_h: np.ndarray,
    L: float,
    outside_allow: float,
    max_iters: int = 600,
    step_cap: float = 2.0
) -> None:
    top = (RECT_WIDTH * 0.5) + outside_allow
    bot = (-RECT_WIDTH * 0.5) - outside_allow
    left = 0.0 - outside_allow
    right = L + outside_allow

    n = len(x)
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                ox = (half_w[i] + half_w[j] + MIN_CLEARANCE) - abs(dx)
                oy = (half_h[i] + half_h[j] + MIN_CLEARANCE) - abs(dy)
                if ox > 0.0 and oy > 0.0:
                    moved = True
                    if ox < oy:
                        s = 1.0 if dx >= 0 else -1.0
                        push = min(step_cap, 0.5 * ox + 1e-3)
                        x[i] -= s * push
                        x[j] += s * push
                    else:
                        s = 1.0 if dy >= 0 else -1.0
                        push = min(step_cap, 0.5 * oy + 1e-3)
                        y[i] -= s * push
                        y[j] += s * push

        x[:] = np.clip(x, left + EDGE_MARGIN + half_w, right - EDGE_MARGIN - half_w)
        y[:] = np.clip(y, bot + EDGE_MARGIN + half_h, top - EDGE_MARGIN - half_h)

        if not moved:
            return


# -----------------------------
# Solver
# -----------------------------
def solve_layout(
    geoms: List[SpaceGeom],
    src: np.ndarray, dst: np.ndarray, w: np.ndarray,
    L: float,
    corridor_polylines: List[List[Tuple[float, float]]],
    corridor_w: float,
    x_init: Optional[np.ndarray],
    y_init: Optional[np.ndarray],
    steps: int,
    lr: float,
    lambda_overlap: float,
    lambda_bounds: float,
    lambda_corridor: float,
    lambda_rel: float,
    lambda_even: float,
    lambda_border: float,
    lambda_prefix: float,
    lambda_blocker_anchor: float,
    border_band: float,
    keepout_end_x: float,
    blocker_idxs: List[int],
    x_fixed_all: np.ndarray,
    y_fixed_all: np.ndarray,
    outside_allow: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)

    n = len(geoms)
    src_t = torch.tensor(src, dtype=torch.long, device=DEVICE)
    dst_t = torch.tensor(dst, dtype=torch.long, device=DEVICE)
    w_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    r_bound = compute_r_bounds(geoms)
    half_w, half_h = compute_half_extents(geoms)
    A, B = build_corridor_segments(corridor_polylines)

    blocker_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    if blocker_idxs:
        blocker_mask[torch.tensor(blocker_idxs, dtype=torch.long, device=DEVICE)] = True

    lo_chk = max(EDGE_MARGIN + 2.0, keepout_end_x + 2.0)
    hi_chk = L - EDGE_MARGIN - 2.0
    if hi_chk <= lo_chk:
        keepout_end_x = EDGE_MARGIN + 1.0

    if x_init is None or y_init is None:
        rng = np.random.default_rng(seed)

        lo = max(EDGE_MARGIN + 2.0, keepout_end_x + 2.0)
        hi = L - EDGE_MARGIN - 2.0
        x0 = rng.uniform(lo, hi, size=n)

        sign = rng.choice([-1.0, 1.0], size=n)
        target = (RECT_WIDTH * 0.5) - EDGE_MARGIN - (np.array([0.5 * g.h for g in geoms])) - border_band
        target = np.clip(target, 0.0, None)
        y0 = sign * target + rng.normal(0.0, 0.8, size=n)

        if blocker_idxs:
            for i in blocker_idxs:
                x0[i] = x_fixed_all[i]
                y0[i] = y_fixed_all[i]
    else:
        x0 = np.asarray(x_init, dtype=float)
        y0 = np.asarray(y_init, dtype=float)

    x = torch.tensor(x0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    y = torch.tensor(y0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x, y], lr=lr)

    block_idxs_t = torch.tensor(blocker_idxs, dtype=torch.long, device=DEVICE) if blocker_idxs else torch.tensor([], dtype=torch.long, device=DEVICE)
    x_fixed_t = torch.tensor([x_fixed_all[i] for i in blocker_idxs], dtype=torch.float32, device=DEVICE) if blocker_idxs else torch.tensor([], dtype=torch.float32, device=DEVICE)
    y_fixed_t = torch.tensor([y_fixed_all[i] for i in blocker_idxs], dtype=torch.float32, device=DEVICE) if blocker_idxs else torch.tensor([], dtype=torch.float32, device=DEVICE)

    for _ in range(steps):
        opt.zero_grad()

        ov = collision_penalty_rects_exact(x, y, half_w, half_h, clearance=MIN_CLEARANCE)
        bd = bounds_penalty_soft(x, y, half_w, half_h, L=L, outside_allow=outside_allow)
        P = torch.stack([x, y], dim=1)
        co = corridor_penalty_rects(P, half_w, half_h, A, B, corridor_w=corridor_w, margin=1.0, tau=0.5)
        rl = relationship_penalty(x, y, r_bound, src_t, dst_t, w_t, L=L, min_clearance=MIN_CLEARANCE)
        ev = even_spacing_penalty(x, L=L, x_lo=keepout_end_x)
        bo = border_attraction_penalty(y, half_h, band=border_band)
        pr = prefix_keepout_penalty(x, half_w, keepout_end_x=keepout_end_x, exempt_mask=blocker_mask)
        an = blocker_anchor_penalty(x, y, block_idxs_t, x_fixed_t, y_fixed_t)

        loss = (
            (lambda_overlap * ov)
            + (lambda_bounds * bd)
            + (lambda_corridor * co)
            + (lambda_rel * rl)
            + (lambda_even * ev)
            + (lambda_border * bo)
            + (lambda_prefix * pr)
            + (lambda_blocker_anchor * an)
        )
        loss.backward()
        opt.step()

    return x.detach().cpu().numpy(), y.detach().cpu().numpy(), keepout_end_x


# -----------------------------
# Outputs
# -----------------------------
def _wrap_label_for_rect(label: str, max_chars_per_line: int, max_lines: int = 3) -> str:
    label = label.strip()
    if max_chars_per_line <= 4:
        return label[:max(1, max_chars_per_line)]
    lines = textwrap.wrap(label, width=max_chars_per_line, break_long_words=False, break_on_hyphens=True)
    if not lines:
        return label
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # Make truncation explicit if we dropped words
        if len(" ".join(lines)) < len(label):
            last = lines[-1]
            if len(last) >= 1:
                lines[-1] = (last[:-1] + "…") if len(last) > 1 else "…"
    return "\n".join(lines)


def draw_layout(
    geoms: List[SpaceGeom],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    corridor_polylines: List[List[Tuple[float, float]]],
    corridor_w: float,
    out_png: Path,
    out_width_px: int,
    out_dpi: int,
    outside_allow: float,
    show_access_legend: bool = True
) -> None:
    fig_w_in = out_width_px / out_dpi
    min_h_px = 3000
    est_h_px = int(out_width_px * ((RECT_WIDTH + 2 * outside_allow) / max(L, 1e-6)) * 3.0)
    fig_h_in = max(min_h_px, est_h_px) / out_dpi

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=out_dpi)
    ax.set_facecolor("white")

    envelope_rect = mpatches.Rectangle(
        (0, -RECT_WIDTH / 2), L, RECT_WIDTH,
        fill=False, edgecolor="black", linewidth=1.4, zorder=1
    )
    ax.add_patch(envelope_rect)

    tube_lw = 2.4
    core_lw = 1.0
    for poly in corridor_polylines:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        ax.plot(xs, ys, linewidth=tube_lw, alpha=0.18, zorder=2, solid_capstyle="round")
        ax.plot(xs, ys, linewidth=core_lw, alpha=0.9, zorder=3, solid_capstyle="round")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-outside_allow - 2, L + outside_allow + 2)
    ax.set_ylim(-RECT_WIDTH / 2 - outside_allow - 2, RECT_WIDTH / 2 + outside_allow + 2)
    ax.axis("off")

    # Ensure transforms are fully initialized (needed for pixel-based label sizing).
    fig.canvas.draw()

    # Draw spaces + labels (LABEL EVERY SPACE)
    for i, g in enumerate(geoms):
        rgb = access_color(g.access)
        fc = (*rgb, 0.62)

        hatch = "///" if g.is_blocker else None

        rect = mpatches.Rectangle(
            (x[i] - g.w / 2, y[i] - g.h / 2),
            g.w, g.h,
            facecolor=fc,
            edgecolor=(0, 0, 0, 0.72),
            linewidth=0.9,
            hatch=hatch,
            zorder=4
        )
        ax.add_patch(rect)

        # Compute rectangle size in pixels to pick a readable font size and wrap width.
        p0 = ax.transData.transform((x[i] - g.w / 2, y[i] - g.h / 2))
        p1 = ax.transData.transform((x[i] + g.w / 2, y[i] + g.h / 2))
        rect_w_px = max(1.0, abs(p1[0] - p0[0]))
        rect_h_px = max(1.0, abs(p1[1] - p0[1]))

        # Font sizing: convert px -> pt using dpi. Cap to avoid huge labels on large rooms.
        min_side_px = min(rect_w_px, rect_h_px)
        fs_pt = (0.28 * min_side_px) * (72.0 / out_dpi)
        fs_pt = float(np.clip(fs_pt, 2.2, 10.0))

        # Approximate max chars per line from available width.
        # Typical glyph width ~0.55*fontsize for sans fonts (rough but stable).
        char_px = max(1.0, 0.55 * fs_pt * (out_dpi / 72.0))
        max_chars = int(np.clip(rect_w_px / char_px, 6, 34))

        label = _wrap_label_for_rect(g.name, max_chars_per_line=max_chars, max_lines=3)

        txt_color = "black" if rel_luminance(rgb) > 0.62 else "white"

        ax.text(
            x[i], y[i], label,
            fontsize=fs_pt,
            ha="center", va="center",
            color=txt_color,
            zorder=5,
            clip_on=True,
            linespacing=0.9,
        )

    # Access legend (optional, but useful for booklet clarity)
    if show_access_legend:
        present = []
        for g in geoms:
            key = str(g.access).strip().upper()
            if key not in present:
                present.append(key)

        # Order by your defined hierarchy, then any unknowns.
        ordered = [k for k in ACCESS_LEVEL_ORDER if k in present]
        for k in present:
            if k not in ordered:
                ordered.append(k)

        handles = []
        labels = []
        for k in ordered:
            rgb = access_color(k)
            handles.append(mpatches.Patch(facecolor=(*rgb, 0.85), edgecolor=(0, 0, 0, 0.35)))
            labels.append(k)

        leg = ax.legend(
            handles, labels,
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            borderpad=0.6,
            labelspacing=0.5,
            fontsize=7.5
        )
        leg.set_zorder(10)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_png, dpi=out_dpi)
    plt.close(fig)


def save_corridor_json(polylines: List[List[Tuple[float, float]]], out_json: Path) -> None:
    data = []
    for i, poly in enumerate(polylines):
        data.append({
            "type": "trunk" if i == 0 else "branch",
            "points": [{"x": float(px), "y": float(py)} for px, py in poly]
        })
    out_json.write_text(json.dumps({"corridor_network": data}, indent=2), encoding="utf-8")


# -----------------------------
# Fit evaluation + dropping
# -----------------------------
def feasibility_metrics(
    geoms: List[SpaceGeom],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    corridor: List[List[Tuple[float, float]]],
    corridor_w: float,
    grid_res: float,
    start_x: float,
    outside_allow: float
) -> Dict[str, float]:
    half_w, half_h = compute_half_extents(geoms)
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    ov = float(collision_penalty_rects_exact(xt, yt, half_w, half_h, clearance=MIN_CLEARANCE).detach().cpu().numpy())
    bd = float(bounds_penalty_soft(xt, yt, half_w, half_h, L=L, outside_allow=outside_allow).detach().cpu().numpy())

    A, B = build_corridor_segments(corridor)
    P = torch.stack([xt, yt], dim=1)
    co = float(corridor_penalty_rects(P, half_w, half_h, A, B, corridor_w=corridor_w, margin=1.0, tau=0.5).detach().cpu().numpy())

    trunk_ok = 0.0
    new_corridor = update_corridor_network(
        geoms=geoms, x=x, y=y, L=L, M=np.zeros((len(geoms), len(geoms))),
        corridor_w=corridor_w, grid_res=grid_res, n_branches=0, smooth_refinements=1,
        start_x=start_x
    )
    if new_corridor is not None and len(new_corridor[0]) >= 2:
        trunk_ok = 1.0

    return {"overlap": ov, "bounds": bd, "corridor": co, "trunk_ok": trunk_ok}


def importance_score(geoms: List[SpaceGeom], M: np.ndarray) -> np.ndarray:
    public = np.array([g.public_role for g in geoms], dtype=float)
    pos_deg = np.maximum(M, 0.0).sum(axis=1)
    access = np.array([ACCESS_LEVEL_TO_NUM.get(str(g.access).strip().upper(), 0.0) for g in geoms], dtype=float)

    def z(v: np.ndarray) -> np.ndarray:
        s = np.std(v)
        return np.zeros_like(v) if s < 1e-9 else (v - np.mean(v)) / s

    return 1.5 * z(public) + 1.0 * z(pos_deg) + 0.8 * z(access)


def per_space_violation_score(
    geoms: List[SpaceGeom],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    corridor: List[List[Tuple[float, float]]],
    corridor_w: float,
    outside_allow: float
) -> np.ndarray:
    n = len(geoms)
    half_w = np.array([0.5 * g.w for g in geoms], dtype=float)
    half_h = np.array([0.5 * g.h for g in geoms], dtype=float)

    top = (RECT_WIDTH * 0.5) + outside_allow
    bot = (-RECT_WIDTH * 0.5) - outside_allow
    left = 0.0 - outside_allow
    right = L + outside_allow

    bx = np.maximum(0.0, (left + EDGE_MARGIN + half_w) - x) + np.maximum(0.0, x - (right - EDGE_MARGIN - half_w))
    by = np.maximum(0.0, (bot + EDGE_MARGIN + half_h) - y) + np.maximum(0.0, y - (top - EDGE_MARGIN - half_h))
    b = bx + by

    A, B = build_corridor_segments(corridor)
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    half_w_t = torch.tensor(half_w, dtype=torch.float32, device=DEVICE)
    half_h_t = torch.tensor(half_h, dtype=torch.float32, device=DEVICE)

    pts = rect_boundary_points(xt, yt, half_w_t, half_h_t)
    pts_flat = pts.reshape(n * 8, 2)
    d_pts = softmin_dist_points_to_segments(pts_flat, A, B, tau=0.5).reshape(n, 8).detach().cpu().numpy()
    d_min = d_pts.min(axis=1)
    need = (corridor_w * 0.5 + 1.0)
    c = np.maximum(0.0, need - d_min)

    o = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = abs(x[j] - x[i])
            dy = abs(y[j] - y[i])
            ox = (half_w[i] + half_w[j] + MIN_CLEARANCE) - dx
            oy = (half_h[i] + half_h[j] + MIN_CLEARANCE) - dy
            if ox > 0.0 and oy > 0.0:
                o[i] += (ox * oy)

    return 2.0 * b + 2.0 * c + 2.5 * o


def choose_drop_index(
    geoms: List[SpaceGeom],
    M: np.ndarray,
    violation: np.ndarray,
    protect_top_frac: float = 0.25
) -> int:
    imp = importance_score(geoms, M)
    n = len(geoms)
    protect_n = max(1, int(math.floor(protect_top_frac * n)))
    protected = set(np.argsort(-imp)[:protect_n].tolist())

    candidates = [i for i in range(n) if i not in protected]
    if not candidates:
        return int(np.argmax(violation))

    cand_scores = [(violation[i], i) for i in candidates]
    cand_scores.sort(reverse=True)
    return int(cand_scores[0][1])


# -----------------------------
# Main pipeline
# -----------------------------
def cross_section_fit_ok(geom: SpaceGeom) -> bool:
    return geom.h <= max_cross_depth()


def run_pipeline_for_set(
    geoms_all: List[SpaceGeom],
    rel_rows: List[str],
    rel_cols: List[str],
    M_full: np.ndarray,
    chosen_names: List[str],
    L: float,
    args
):
    geoms, names, M = subset_by_names(geoms_all, rel_rows, rel_cols, M_full, chosen_names)
    src, dst, w = sparsify_edges(M, topk=10, abs_thresh=3.5)

    x_fixed_all, y_fixed_all, blocker_idxs, keepout_end_x = blocker_fixed_positions(geoms, L=L)
    start_x = (keepout_end_x + 1.0) if blocker_idxs else (EDGE_MARGIN + 1.0)
    start_x = max(start_x, EDGE_MARGIN + 1.0)

    node_x = build_node_features(geoms)
    s_init, side_init = train_gnn_init(node_x, src, dst, w, epochs=args.gnn_epochs, lr=args.gnn_lr, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1)

    lo = max(EDGE_MARGIN + 2.0, keepout_end_x + 2.0)
    hi = L - EDGE_MARGIN - 2.0
    if hi <= lo:
        keepout_end_x = EDGE_MARGIN + 1.0
        lo = EDGE_MARGIN + 2.0
        hi = L - EDGE_MARGIN - 2.0

    x0 = rng.uniform(lo, hi, size=len(geoms))

    target = (RECT_WIDTH * 0.5) - EDGE_MARGIN - (np.array([0.5 * g.h for g in geoms])) - args.border_band
    target = np.clip(target, 0.0, None)
    y0 = side_init * target + rng.normal(0.0, 0.8, size=len(geoms))

    if blocker_idxs:
        for i in blocker_idxs:
            x0[i] = x_fixed_all[i]
            y0[i] = y_fixed_all[i]

    corridor = [[(start_x, 0.0), (L - EDGE_MARGIN - 1.0, 0.0)]]

    for it in range(args.coevolve_iters):
        x0, y0, keepout_end_x = solve_layout(
            geoms=geoms, src=src, dst=dst, w=w,
            L=L,
            corridor_polylines=corridor,
            corridor_w=args.corridor_w,
            x_init=x0, y_init=y0,
            steps=args.solver_steps,
            lr=args.solver_lr,
            lambda_overlap=args.lambda_overlap,
            lambda_bounds=args.lambda_bounds,
            lambda_corridor=args.lambda_corridor,
            lambda_rel=args.lambda_rel,
            lambda_even=args.lambda_even,
            lambda_border=args.lambda_border,
            lambda_prefix=args.lambda_prefix,
            lambda_blocker_anchor=args.lambda_blocker_anchor,
            border_band=args.border_band,
            keepout_end_x=keepout_end_x,
            blocker_idxs=blocker_idxs,
            x_fixed_all=x_fixed_all,
            y_fixed_all=y_fixed_all,
            outside_allow=args.outside_allow,
            seed=args.seed + it
        )

        start_x = (keepout_end_x + 1.0) if blocker_idxs else (EDGE_MARGIN + 1.0)

        new_corridor = update_corridor_network(
            geoms=geoms, x=x0, y=y0, L=L, M=M,
            corridor_w=args.corridor_w,
            grid_res=args.grid_res,
            n_branches=args.n_branches,
            smooth_refinements=args.smooth_refinements,
            start_x=start_x
        )
        if new_corridor is not None:
            corridor = new_corridor

    half_w_np = np.array([0.5 * g.w for g in geoms], dtype=float)
    half_h_np = np.array([0.5 * g.h for g in geoms], dtype=float)
    deoverlap_rects_inplace(x0, y0, half_w_np, half_h_np, L=L, outside_allow=args.outside_allow)

    new_corridor = update_corridor_network(
        geoms=geoms, x=x0, y=y0, L=L, M=M,
        corridor_w=args.corridor_w,
        grid_res=args.grid_res,
        n_branches=args.n_branches,
        smooth_refinements=args.smooth_refinements,
        start_x=start_x
    )
    if new_corridor is not None:
        corridor = new_corridor

    metrics = feasibility_metrics(
        geoms=geoms, x=x0, y=y0, L=L,
        corridor=corridor,
        corridor_w=args.corridor_w,
        grid_res=args.grid_res,
        start_x=start_x,
        outside_allow=args.outside_allow
    )

    return geoms, names, x0, y0, corridor, metrics, M, start_x


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit a subset of spaces into a rectangular layout envelope using a message-passing network."
    )
    ap.add_argument("--spaces-csv", type=Path, default=DEFAULT_DATA_DIR / "spaces.csv")
    ap.add_argument("--relationships-csv", type=Path, default=DEFAULT_DATA_DIR / "relationships.csv")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n_spaces", type=int, default=50)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)

    ap.add_argument("--rect-length", type=float, required=True)
    ap.add_argument("--corridor_w", type=float, default=4.0)

    ap.add_argument("--coevolve_iters", type=int, default=4)

    ap.add_argument("--grid_res", type=float, default=0.2)

    ap.add_argument("--n_branches", type=int, default=4)
    ap.add_argument("--smooth_refinements", type=int, default=3)

    ap.add_argument("--gnn_epochs", type=int, default=120)
    ap.add_argument("--gnn_lr", type=float, default=2e-3)

    ap.add_argument("--solver_steps", type=int, default=900)
    ap.add_argument("--solver_lr", type=float, default=6e-2)

    ap.add_argument("--lambda_overlap", type=float, default=4000.0)
    ap.add_argument("--lambda_bounds", type=float, default=120.0)

    ap.add_argument("--lambda_corridor", type=float, default=140.0)
    ap.add_argument("--lambda_rel", type=float, default=2.0)
    ap.add_argument("--lambda_even", type=float, default=20.0)

    ap.add_argument("--lambda_border", type=float, default=65.0)
    ap.add_argument("--lambda_prefix", type=float, default=260.0)
    ap.add_argument("--lambda_blocker_anchor", type=float, default=2500.0)
    ap.add_argument("--border_band", type=float, default=1.5)

    ap.add_argument("--outside_allow", type=float, default=6.0)

    ap.add_argument("--drop_until_fit", action="store_true")
    ap.add_argument("--min_spaces", type=int, default=20)

    ap.add_argument("--eps_overlap", type=float, default=1e-9)
    ap.add_argument("--eps_bounds", type=float, default=1e-3)
    ap.add_argument("--eps_corridor", type=float, default=1e-3)

    ap.add_argument("--out_width_px", type=int, default=8000)
    ap.add_argument("--out_dpi", type=int, default=300)

    ap.add_argument("--area_rescale_tol", type=float, default=0.20)

    # New: allow turning legend off if you want a cleaner booklet page
    ap.add_argument("--no_access_legend", action="store_true")

    args = ap.parse_args()

    logger.info(f"Starting run with args {args}")
    L = float(args.rect_length)
    outdir = args.outdir.resolve()
    out_layout_csv = outdir / "layout_result.csv"
    out_corridor_json = outdir / "corridor_network.json"
    out_png = outdir / "layout.png"
    out_dropped_csv = outdir / "dropped_spaces.csv"

    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data")
    spaces_df, rel_df = load_data(args.spaces_csv, args.relationships_csv)
    geoms_all = build_space_geoms(spaces_df, area_rescale_tol=args.area_rescale_tol)
    rel_rows, rel_cols, M_full = align_relationship_matrix(rel_df)

    chosen_names = random_initial_subset(geoms_all, rel_rows, rel_cols, M_full, n=args.n_spaces, seed=args.seed)

    dropped_log = []

    def geom_by_name(nm: str) -> Optional[SpaceGeom]:
        for g in geoms_all:
            if g.name == nm:
                return g
        return None

    pre_drops = []
    for nm in list(chosen_names):
        g = geom_by_name(nm)
        if g is None:
            continue
        if not cross_section_fit_ok(g):
            pre_drops.append(nm)

    if pre_drops:
        for nm in pre_drops:
            dropped_log.append({
                "SpaceName": nm,
                "reason": "cross_section_impossible",
                "violation_score": float("inf"),
                "metrics_overlap": None,
                "metrics_bounds": None,
                "metrics_corridor": None,
                "trunk_ok": None,
                "remaining_before_drop": len(chosen_names)
            })
        chosen_names = [nm for nm in chosen_names if nm not in set(pre_drops)]
        logger.info(f"Pre-check dropped {len(pre_drops)} spaces that cannot fit cross-section")

    logger.info(f"Fitting {len(chosen_names)} spaces into rectangular envelope")

    while True:
        geoms, names, x, y, corridor, metrics, M_sub, start_x = run_pipeline_for_set(
            geoms_all=geoms_all,
            rel_rows=rel_rows,
            rel_cols=rel_cols,
            M_full=M_full,
            chosen_names=chosen_names,
            L=L,
            args=args
        )

        feasible = (
            metrics["trunk_ok"] >= 0.5 and
            metrics["overlap"] <= args.eps_overlap and
            metrics["bounds"] <= args.eps_bounds and
            metrics["corridor"] <= args.eps_corridor
        )

        if feasible or (not args.drop_until_fit):
            break

        if len(chosen_names) <= args.min_spaces:
            break

        viol = per_space_violation_score(geoms, x, y, L, corridor, args.corridor_w, outside_allow=args.outside_allow)
        drop_i = choose_drop_index(geoms, M_sub, viol, protect_top_frac=0.25)
        drop_name = names[drop_i]
        logger.info(f"Infeasible fit: dropping {drop_name} and retrying")

        dropped_log.append({
            "SpaceName": drop_name,
            "reason": "infeasible_fit_drop",
            "violation_score": float(viol[drop_i]),
            "metrics_overlap": metrics["overlap"],
            "metrics_bounds": metrics["bounds"],
            "metrics_corridor": metrics["corridor"],
            "trunk_ok": metrics["trunk_ok"],
            "remaining_before_drop": len(chosen_names)
        })

        chosen_names = [nm for nm in chosen_names if nm != drop_name]

    logger.info("Saving outputs")

    colors_rgb = [access_color(g.access) for g in geoms]
    colors_hex = [rgb_to_hex(c) for c in colors_rgb]

    out = pd.DataFrame({
        "SpaceName": names,
        "x_m": x,
        "y_m": y,
        "shape": ["rect"] * len(names),
        "w_m": [g.w for g in geoms],
        "h_m": [g.h for g in geoms],
        "NetArea_m2": [g.area for g in geoms],
        "AccessLevel": [g.access for g in geoms],
        "PublicInterfaceRole": [g.public_role for g in geoms],
        "is_blocker": [bool(g.is_blocker) for g in geoms],
        "blocker_reason": [g.blocker_reason for g in geoms],
        "color_hex": colors_hex,
        "corridor_start_x_m": [float(start_x)] * len(names),
        "outside_allow_m": [float(args.outside_allow)] * len(names),
    })
    out.to_csv(out_layout_csv, index=False)

    save_corridor_json(corridor, out_corridor_json)

    draw_layout(
        geoms=geoms,
        x=x,
        y=y,
        L=L,
        corridor_polylines=corridor,
        corridor_w=args.corridor_w,
        out_png=out_png,
        out_width_px=args.out_width_px,
        out_dpi=args.out_dpi,
        outside_allow=args.outside_allow,
        show_access_legend=(not args.no_access_legend)
    )

    dropped_df = pd.DataFrame(dropped_log)
    dropped_df.to_csv(out_dropped_csv, index=False)

    print(f"Wrote: {out_layout_csv.resolve()}")
    print(f"Wrote: {out_corridor_json.resolve()}")
    print(f"Wrote: {out_png.resolve()}")
    print(f"Wrote: {out_dropped_csv.resolve()}")
    print(f"Final spaces kept: {len(names)} | Dropped: {len(dropped_log)}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
