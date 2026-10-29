"""
field_map.py
Grid representation of the field - the rover drives BETWEEN the plant rows.

Layout
    Plants sit in rows. Rows are paired, and the rover drives along the aisle
    (lane) between the two rows of a pair, scanning the plant on its left and the
    plant on its right at every stop. At the end of a lane it turns through the
    headland (open strip at the field edge) and enters the next lane.

        plant row 0   o  o  o  o  o
        aisle 1     >>>>>>>>>>>>>>>>>>>>   (rover lane, scans rows 0 and 1)
        plant row 1   o  o  o  o  o
        plant row 2   o  o  o  o  o
        aisle 2     <<<<<<<<<<<<<<<<<<<<   (scans rows 2 and 3)
        plant row 3   o  o  o  o  o

Colours
    green  = healthy            red    = disease detected
    purple = dry soil           amber  = model not confident (manual check)
    grey   = not scanned yet    blue line = rover path (in the aisles)
    dashed boxes = zones (Alpha, Beta, Gamma ...)

Usage (software simulation or real rover - same calls):
    grid = FieldGrid(rows=8, cols=12)
    for stop in grid.scan_plan():
        grid.move_to(stop)
        for (r, c) in stop["plants"]:
            report = agent.diagnose(photo_of_plant(r, c))
            grid.record(r, c, report, image_path, moisture=moisture_percent)
    grid.render().savefig("field_map.png")
"""

import json
import math
import textwrap
from enum import Enum

import matplotlib

matplotlib.use("Agg")  # render to image; works inside Gradio and plain scripts
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle


class CellState(str, Enum):
    UNSCANNED = "unscanned"
    HEALTHY = "healthy"
    DISEASED = "diseased"
    UNCERTAIN = "uncertain"
    DRY = "dry"


COLORS = {
    CellState.UNSCANNED: "#d1d5db",
    CellState.HEALTHY: "#4ade80",
    CellState.DISEASED: "#ef4444",
    CellState.UNCERTAIN: "#fbbf24",
    CellState.DRY: "#c084fc",
}
PATH_COLOR = "#1d4ed8"
GROUND_COLOR = "#f3ecdc"
AISLE_COLOR = "#e0d0a8"
DEFAULT_DRY_THRESHOLD = 35  # soil moisture % below this = dry

# drawing geometry (plant cell = 1 x 1)
AISLE_W = 0.9    # width of the lane between the two rows of a pair
BED_GAP = 0.25   # small gap between neighbouring pairs
HEADLAND = 0.45  # x-offset of the turning lane at the field edge


def layout_row_y(r):
    """Top y-coordinate of plant row r, including the aisles above it."""
    return r + AISLE_W * math.ceil(r / 2) + BED_GAP * (r // 2)


def classify_report(report, moisture=None, dry_threshold=DEFAULT_DRY_THRESHOLD):
    """Turn an agent report (+ optional soil moisture) into a cell state.
    Uses the raw predicted class name (not the knowledge base), so a class
    that is missing from disease_remedies.json is still marked correctly."""
    if report.get("is_uncertain"):
        return CellState.UNCERTAIN
    top_class = report["top_prediction"]["class"]
    if top_class.lower().endswith("healthy"):
        if moisture is not None and moisture < dry_threshold:
            return CellState.DRY
        return CellState.HEALTHY
    return CellState.DISEASED


class FieldGrid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [
            [
                {"state": CellState.UNSCANNED, "disease": None, "class": None,
                 "severity": None, "confidence": None, "moisture": None, "image": None}
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]
        self.zones = {}            # name -> (row_start, row_end, col_start, col_end), end exclusive
        self.trail = []            # (x, y) points the rover has travelled, in drawing units
        self.rover = None          # current (x, y)
        self.current_plants = []   # plants scanned at the current stop

    # ---------- geometry ----------
    @property
    def n_aisles(self):
        return (self.rows + 1) // 2

    def aisle_y(self, k):
        """y of the centre line of aisle k (between plant rows 2k and 2k+1)."""
        return layout_row_y(2 * k) + 1 + AISLE_W / 2

    def total_height(self):
        h = layout_row_y(self.rows - 1) + 1
        if self.rows % 2 == 1:   # last single row still gets a lane below it
            h += AISLE_W
        return h

    # ---------- rover route ----------
    def scan_plan(self):
        """Stops in visiting order. Aisle 0 left->right, aisle 1 right->left, ...
        Each stop scans the plant on each side of the aisle (one side only for an odd last row)."""
        stops = []
        for k in range(self.n_aisles):
            cols = range(self.cols) if k % 2 == 0 else range(self.cols - 1, -1, -1)
            side_rows = [r for r in (2 * k, 2 * k + 1) if r < self.rows]
            for c in cols:
                stops.append({"aisle": k, "col": c, "x": c + 0.5, "y": self.aisle_y(k),
                              "plants": [(r, c) for r in side_rows]})
        return stops

    def move_to(self, stop):
        """Move the rover to a stop, drawing the headland turn when it changes aisle."""
        pts = []
        if not self.trail:
            pts.append((-HEADLAND, stop["y"]))          # enters from the left headland
        else:
            last_x, last_y = self.trail[-1]
            if abs(last_y - stop["y"]) > 1e-9:          # new aisle: go round the field edge
                side = self.cols + HEADLAND if last_x > self.cols / 2 else -HEADLAND
                pts += [(side, last_y), (side, stop["y"])]
        pts.append((stop["x"], stop["y"]))
        self.trail.extend(pts)
        self.rover = (stop["x"], stop["y"])
        self.current_plants = list(stop["plants"])

    # ---------- recording results ----------
    def record(self, row, col, report, image_path=None, moisture=None,
               dry_threshold=DEFAULT_DRY_THRESHOLD):
        state = classify_report(report, moisture, dry_threshold)
        self.cells[row][col].update({
            "state": state,
            "disease": report.get("disease"),
            "class": report["top_prediction"]["class"],
            "severity": report.get("severity"),
            "confidence": report["top_prediction"]["confidence"],
            "moisture": moisture,
            "image": image_path,
        })
        return state

    # ---------- queries ----------
    def counts(self):
        counts = {s: 0 for s in CellState}
        for row in self.cells:
            for cell in row:
                counts[cell["state"]] += 1
        return counts

    def attention_cells(self):
        """Plants that need the farmer's attention: disease, dry soil, or unsure."""
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                s = cell["state"]
                if s == CellState.DISEASED or s == CellState.UNCERTAIN:
                    out.append([r, c, s.value, cell["disease"], cell["severity"],
                                f"{cell['confidence'] * 100:.0f}%"])
                elif s == CellState.DRY:
                    out.append([r, c, s.value, f"Low soil moisture ({cell['moisture']:.0f}%)",
                                "-", f"{cell['confidence'] * 100:.0f}%"])
        return out

    def summary_text(self):
        n = self.counts()
        total = self.rows * self.cols
        scanned = total - n[CellState.UNSCANNED]
        pct = (n[CellState.DISEASED] / scanned * 100) if scanned else 0
        return (f"Scanned {scanned}/{total}  |  Healthy: {n[CellState.HEALTHY]}  |  "
                f"Diseased: {n[CellState.DISEASED]} ({pct:.0f}%)  |  "
                f"Dry soil: {n[CellState.DRY]}  |  Manual check: {n[CellState.UNCERTAIN]}")

    def save_json(self, path):
        data = {
            "rows": self.rows, "cols": self.cols, "zones": self.zones,
            "rover_trail": self.trail,
            "cells": [[{**c, "state": c["state"].value} for c in row] for row in self.cells],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ---------- drawing ----------
    def render(self, title="Field map"):
        W, H = self.cols, self.total_height()
        margin = 0.9
        fig, ax = plt.subplots(figsize=(max(7, (W + 2 * margin) * 0.85), max(5, H * 0.85 + 1.8)))

        # ground, aisle lanes and headland strips
        ax.add_patch(Rectangle((-margin, -0.3), W + 2 * margin, H + 0.6,
                               facecolor=GROUND_COLOR, edgecolor="none", zorder=0))
        for k in range(self.n_aisles):
            ax.add_patch(Rectangle((-margin, self.aisle_y(k) - AISLE_W / 2), W + 2 * margin, AISLE_W,
                                   facecolor=AISLE_COLOR, edgecolor="none", zorder=0.5))
        for x0 in (-margin, W):
            ax.add_patch(Rectangle((x0, -0.3), margin, H + 0.6,
                                   facecolor=AISLE_COLOR, edgecolor="none", zorder=0.5))

        # plants
        label_cells = self.rows <= 8 and self.cols <= 12
        for r in range(self.rows):
            y = layout_row_y(r)
            for c in range(self.cols):
                cell = self.cells[r][c]
                s = cell["state"]
                ax.add_patch(FancyBboxPatch((c + 0.07, y + 0.07), 0.86, 0.86,
                                            boxstyle="round,pad=0,rounding_size=0.18",
                                            facecolor=COLORS[s], edgecolor="white",
                                            linewidth=1.2, zorder=1))
                if label_cells and s in (CellState.DISEASED, CellState.UNCERTAIN, CellState.DRY):
                    if s == CellState.DRY:
                        name = f"Dry {cell['moisture']:.0f}%"
                    elif s == CellState.UNCERTAIN:
                        name = f"? {cell['disease'] or ''}"
                    else:
                        name = cell["disease"] or "?"
                    ax.text(c + 0.5, y + 0.5, textwrap.fill(name, 9, max_lines=2, placeholder="…"),
                            ha="center", va="center", fontsize=6.3, color="#1f1f1f",
                            fontweight="bold", zorder=2)

        # rover path (in the aisles) and current position
        if self.trail:
            xs, ys = [p[0] for p in self.trail], [p[1] for p in self.trail]
            ax.plot(xs, ys, color="white", linewidth=5.5, zorder=3, solid_capstyle="round",
                    solid_joinstyle="round")
            ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.8, zorder=4, solid_capstyle="round",
                    solid_joinstyle="round")
            ax.plot(xs[0], ys[0], marker="s", markersize=10, color="white",
                    markeredgecolor=PATH_COLOR, markeredgewidth=2, zorder=5)
        if self.rover:
            rx, ry = self.rover
            for (r, c) in self.current_plants:   # camera "looks" at the plants beside the rover
                cx, cy = c + 0.5, layout_row_y(r) + 0.5
                ax.plot([rx, cx], [ry, cy], color=PATH_COLOR, linewidth=1.6, alpha=0.75, zorder=5)
                ax.add_patch(Circle((cx, cy), 0.52, fill=False, edgecolor=PATH_COLOR,
                                    linewidth=2.2, zorder=6))
            ax.plot(rx, ry, marker="o", markersize=14, color=PATH_COLOR,
                    markeredgecolor="white", markeredgewidth=2.5, zorder=7)

        # zone outlines
        for name, (r0, r1, c0, c1) in self.zones.items():
            y0, y1 = layout_row_y(r0), layout_row_y(r1 - 1) + 1
            ax.add_patch(Rectangle((c0, y0), c1 - c0, y1 - y0, fill=False, edgecolor="#111827",
                                   linewidth=2.2, linestyle="--", zorder=8))
            ax.text(c0 + 0.12, y0 + 0.1, f"Grid {name}", ha="left", va="top", fontsize=8,
                    fontweight="bold", color="#111827", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, linewidth=0))

        ax.set_xlim(-margin, W + margin)
        ax.set_ylim(H + 0.3, -0.3)  # row 0 at the top
        ax.set_aspect("equal")
        ax.set_xticks([c + 0.5 for c in range(self.cols)])
        ax.set_xticklabels([str(c) for c in range(self.cols)], fontsize=8)
        ax.set_yticks([layout_row_y(r) + 0.5 for r in range(self.rows)])
        ax.set_yticklabels([f"Row {r}" for r in range(self.rows)], fontsize=7.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\n{self.summary_text()}", fontsize=9)

        legend = [
            Patch(facecolor=COLORS[CellState.HEALTHY], label="Healthy"),
            Patch(facecolor=COLORS[CellState.DISEASED], label="Disease detected"),
            Patch(facecolor=COLORS[CellState.DRY], label="Dry soil"),
            Patch(facecolor=COLORS[CellState.UNCERTAIN], label="Low confidence"),
            Patch(facecolor=COLORS[CellState.UNSCANNED], label="Not scanned"),
            Patch(facecolor=AISLE_COLOR, label="Aisle (rover lane)"),
            Line2D([0], [0], color=PATH_COLOR, linewidth=3, label="Rover path"),
        ]
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                  ncol=4, fontsize=8, frameon=False)
        fig.tight_layout()
        return fig
