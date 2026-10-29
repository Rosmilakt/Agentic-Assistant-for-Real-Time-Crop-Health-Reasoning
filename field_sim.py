"""
field_sim.py
Simulation model for the rover's autonomous run (no hardware needed).

Where the field comes from (SCENARIOS)
  - "Test dataset: each zone = one class"  -> every zone is filled mostly with one
        class from YOUR test dataset (e.g. Grid Beta = Late blight), plus a few strays.
  - "Test dataset: random sample"          -> every cell takes a random test image
        (the dataset's natural mix of classes).
  - "Simulated: ..."                       -> invented zone conditions (healthy / diseased /
        dry) using dataset images, with a simulated soil-moisture sensor.
  In the dataset modes the TRUE label of every cell comes from the image's folder name,
  so the report can compare what the model detected with the real answer.

What happens during the run
  - The rover drives along the aisles BETWEEN the plant rows and scans the plant on each
    side at every stop (the REAL CropAgent diagnoses each image).
  - Soil moisture is simulated; below the threshold a healthy-looking plant is "Dehydrated".
  - At the end an Agricultural Condition Report is compiled: field summary analytics +
    a per-zone table (status, threat, recommendation, ground truth).

Each plant cell is one sampling point that stands for a patch of plants
(plants_per_cell), so "plants analyzed" = scanned cells x plants_per_cell.

Swap-in for real hardware later: replace the dataset image / simulated moisture with the
rover's camera frame and soil sensor reading; the grid and report code stay the same.
"""

import csv
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from field_map import FieldGrid, CellState, DEFAULT_DRY_THRESHOLD, layout_row_y

IMG_EXT = (".jpg", ".jpeg", ".png")
GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"]

STATUS_LABEL = {
    CellState.HEALTHY: "Healthy",
    CellState.DISEASED: "Stressed",
    CellState.DRY: "Dehydrated",
    CellState.UNCERTAIN: "Needs manual check",
}

SCENARIOS = {
    "Test dataset: each zone = one class": "dataset_zones",
    "Test dataset: random sample (natural mix)": "dataset_random",
    "Simulated: mostly healthy + 1 diseased + 1 dry zone": "example",
    "Simulated: random zone conditions": "random",
}
SOURCE_TEXT = {
    "dataset_zones": "test dataset - each zone filled mostly with one class",
    "dataset_random": "test dataset - random sample of images",
    "example": "simulated zones (healthy / diseased / dry)",
    "random": "simulated zones (random conditions)",
}

ACTION_VERBS = {"apply", "spray", "remove", "improve", "prune", "use", "destroy", "avoid",
                "introduce", "disinfect", "wash", "plant", "rotate", "mulch", "control",
                "treat", "increase", "ensure"}

TABLE_HEADERS = [
    "Zone Coordinates (Center)",
    "Dominant Status",
    "Primary Threat Detected",
    "Actionable Recommendation",
    "Avg Soil Moisture",
    "Ground Truth (dataset labels)",
]


@dataclass
class SimConfig:
    rows: int = 8
    cols: int = 12
    zone_rows: int = 2
    zone_cols: int = 3
    cell_size_m: float = 18.0      # side of one cell in metres
    plants_per_cell: int = 150     # plants represented by one sampling point
    origin_lat: float = 10.8500    # north-west corner of the field
    origin_lon: float = 75.9515
    scenario: str = "dataset_zones"
    crop: str = "Tomato"
    dry_threshold: float = DEFAULT_DRY_THRESHOLD
    seed: Optional[int] = None


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------
def is_healthy_class(cls):
    return cls.lower().endswith("healthy")


def pretty_class(cls):
    """'Tomato___Late_blight' -> 'Late blight'"""
    return cls.split("___")[-1].replace("_", " ").strip()


def collect_by_class(folder, crop=""):
    """Scan a PlantVillage-style folder tree -> {class_folder_name: [image paths]}.
    The class folder name is the ground-truth label."""
    by_class = {}
    for root, _, files in os.walk(folder):
        cls = os.path.basename(root)
        if crop and not cls.lower().startswith(crop.lower()):
            continue
        imgs = [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXT)]
        if imgs:
            by_class.setdefault(cls, []).extend(imgs)
    return by_class


def split_classes(by_class):
    """-> (all healthy images, {disease_class: images})"""
    healthy = [p for c, imgs in by_class.items() if is_healthy_class(c) for p in imgs]
    diseased = {c: imgs for c, imgs in by_class.items() if not is_healthy_class(c)}
    return healthy, diseased


def collect_images(folder, crop=""):
    return split_classes(collect_by_class(folder, crop))


# ----------------------------------------------------------------------------
# zones + GPS
# ----------------------------------------------------------------------------
def make_zones(rows, cols, zone_rows, zone_cols):
    """Split the grid into zone_rows x zone_cols blocks named Alpha, Beta, ..."""
    # row boundaries snap to even numbers so each aisle (a pair of rows) stays inside one zone
    row_edges = [min(rows, 2 * round(i * rows / zone_rows / 2)) for i in range(zone_rows + 1)]
    row_edges[-1] = rows
    if any(b <= a for a, b in zip(row_edges, row_edges[1:])):   # too few rows: plain split
        row_edges = [round(i * rows / zone_rows) for i in range(zone_rows + 1)]
    col_edges = [round(i * cols / zone_cols) for i in range(zone_cols + 1)]
    zones, k = {}, 0
    for i in range(zone_rows):
        for j in range(zone_cols):
            name = GREEK[k] if k < len(GREEK) else f"Z{k + 1}"
            zones[name] = (row_edges[i], row_edges[i + 1], col_edges[j], col_edges[j + 1])
            k += 1
    return zones


def zone_center_gps(bounds, cfg):
    """Centre of a zone as (lat, lon). Rows run south, columns run east from the origin."""
    r0, r1, c0, c1 = bounds
    dy = ((layout_row_y(r0) + layout_row_y(r1 - 1) + 1) / 2) * cfg.cell_size_m  # metres south
    dx = ((c0 + c1) / 2) * cfg.cell_size_m                                      # metres east
    lat = cfg.origin_lat - dy / 111_320
    lon = cfg.origin_lon + dx / (111_320 * math.cos(math.radians(cfg.origin_lat)))
    return lat, lon


# ----------------------------------------------------------------------------
# the simulator
# ----------------------------------------------------------------------------
P_RANGE = {"healthy": (0.80, 0.97), "disease": (0.55, 0.85), "dry": (0.55, 0.85)}
MOISTURE = {  # (mean, std, min, max) of the simulated soil sensor, in %
    "healthy": (62, 8, 40, 95),
    "disease": (55, 10, 20, 90),
    "dry": (20, 7, 3, 33),
}


class FieldSimulator:
    def __init__(self, agent, healthy_imgs, disease_imgs, cfg, by_class=None):
        self.agent = agent
        self.cfg = cfg
        self.healthy_imgs = healthy_imgs
        self.disease_imgs = disease_imgs
        self.by_class = by_class or {**disease_imgs, "healthy": healthy_imgs}
        self.rng = random.Random(cfg.seed)

        self.grid = FieldGrid(cfg.rows, cfg.cols)
        self.zones = make_zones(cfg.rows, cfg.cols, cfg.zone_rows, cfg.zone_cols)
        self.grid.zones = self.zones

        self._used = set()
        self.zone_plan = {}
        self.truth = {}         # (row, col) -> (kind, class)   kind: healthy / disease / dry
        self.cell_images = {}   # (row, col) -> image path
        self.matches = 0

        if cfg.scenario in ("dataset_zones", "dataset_random"):
            self._build_from_dataset()
        else:
            self._build_from_scenario()

    def _pick(self, pool):
        """Random image from pool, avoiding repeats where possible."""
        img = self.rng.choice(pool)
        for _ in range(20):
            if img not in self._used:
                break
            img = self.rng.choice(pool)
        self._used.add(img)
        return img

    # ----- grid built from the test dataset -----
    def _build_from_dataset(self):
        classes = list(self.by_class)
        flat = [(c, p) for c, imgs in self.by_class.items() for p in imgs]
        healthy_cls = [c for c in classes if is_healthy_class(c)]
        other_cls = [c for c in classes if not is_healthy_class(c)]

        def put(r, c, cls, img):
            self.truth[(r, c)] = ("healthy" if is_healthy_class(cls) else "disease", cls)
            self.cell_images[(r, c)] = img
            self._used.add(img)

        if self.cfg.scenario == "dataset_random":
            for name in self.zones:
                self.zone_plan[name] = {"cond": None, "cls": None}
            for r in range(self.cfg.rows):
                for c in range(self.cfg.cols):
                    cls, img = self.rng.choice(flat)
                    put(r, c, cls, img)
            return

        # dataset_zones: each zone gets one primary class (first zone = a healthy class)
        pool = []
        for i, name in enumerate(self.zones):
            if i == 0 and healthy_cls:
                primary = self.rng.choice(healthy_cls)
            else:
                if not pool:
                    pool = (other_cls or classes)[:]
                    self.rng.shuffle(pool)
                primary = pool.pop()
            self.zone_plan[name] = {"cond": None, "cls": primary}

            r0, r1, c0, c1 = self.zones[name]
            purity = self.rng.uniform(0.75, 0.95)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if self.rng.random() < purity:
                        put(r, c, primary, self._pick(self.by_class[primary]))
                    else:
                        cls, img = self.rng.choice(flat)
                        put(r, c, cls, img)

    # ----- grid built from an invented scenario -----
    def _build_from_scenario(self):
        classes = list(self.disease_imgs)
        base = ["healthy", "disease", "dry"]
        for i, name in enumerate(self.zones):
            if self.cfg.scenario == "example":
                cond = base[i] if i < len(base) else "healthy"
            else:
                cond = self.rng.choice(base)
            self.zone_plan[name] = {"cond": cond,
                                    "cls": self.rng.choice(classes) if cond == "disease" else None}

        for name, (r0, r1, c0, c1) in self.zones.items():
            plan = self.zone_plan[name]
            cond = plan["cond"]
            p = self.rng.uniform(*P_RANGE[cond])
            others = [k for k in ("disease", "dry") if k != cond]
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if self.rng.random() < p:
                        kind = cond
                    elif cond == "healthy":
                        kind = self.rng.choice(others)
                    else:
                        kind = "healthy" if self.rng.random() < 0.6 else self.rng.choice(others)

                    cls = None
                    if kind == "disease":
                        if cond == "disease" and self.rng.random() < 0.85:
                            cls = plan["cls"]
                        else:
                            cls = self.rng.choice(classes)
                    self.truth[(r, c)] = (kind, cls)
                    pool = self.disease_imgs[cls] if kind == "disease" else self.healthy_imgs
                    self.cell_images[(r, c)] = self._pick(pool)

    def _moisture(self, kind):
        mean, std, lo, hi = MOISTURE[kind]
        return min(hi, max(lo, self.rng.gauss(mean, std)))

    def scan_iter(self):
        """Drive down the aisles. At every stop the rover scans the plant on each side of it.
        Yields (stop, [(row, col, state), ...]) after each stop."""
        expected_state = {"healthy": CellState.HEALTHY, "disease": CellState.DISEASED,
                          "dry": CellState.DRY}
        for stop in self.grid.scan_plan():
            self.grid.move_to(stop)
            results = []
            for (r, c) in stop["plants"]:
                kind, _ = self.truth[(r, c)]
                img = self.cell_images[(r, c)]
                moisture = self._moisture(kind)

                report = self.agent.diagnose(img)
                state = self.grid.record(r, c, report, img, moisture, self.cfg.dry_threshold)
                if state == expected_state[kind]:
                    self.matches += 1
                results.append((r, c, state))
            yield stop, results


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def _disease_info(agent, cls):
    info = agent._lookup(cls)
    if info.get("disease") == "Unrecognized class":
        pretty = cls.replace("___", " - ").replace("_", " ")
        return {"name": pretty, "type": None, "severity": "unknown", "urgency": "",
                "action": "No remedy on file - send a leaf sample for manual inspection."}
    sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", info["remedy"].strip())
    # prefer the first sentence that tells the farmer what to DO (starts with an action verb)
    action = next((x for x in sentences if x.split()[0].lower() in ACTION_VERBS), sentences[0])
    disease_type = (info.get("type") or "").split(" (")[0]   # "oomycete (water mold)" -> "oomycete"
    return {"name": info["disease"], "type": disease_type, "severity": info.get("severity"),
            "urgency": info.get("urgency", ""), "action": action}


def _truth_text(keys):
    """keys = [(kind, class), ...] of the scanned cells in a zone -> e.g. 'Stressed - Late blight (78%)'"""
    kinds = Counter(k for k, _ in keys)
    top_kind, n = kinds.most_common(1)[0]
    label = {"healthy": "Healthy", "disease": "Stressed", "dry": "Dehydrated"}[top_kind]
    if top_kind == "disease":
        top_cls = Counter(c for k, c in keys if k == "disease").most_common(1)[0][0]
        label += f" - {pretty_class(top_cls)}"
    return f"{label} ({n / len(keys) * 100:.0f}%)"


def _analyse_zone(name, bounds, grid, cfg, agent, truth=None):
    r0, r1, c0, c1 = bounds
    coords = [(r, c) for r in range(r0, r1) for c in range(c0, c1)
              if grid.cells[r][c]["state"] != CellState.UNSCANNED]
    if not coords:
        return None
    cells = [grid.cells[r][c] for r, c in coords]

    counts = Counter(cell["state"] for cell in cells)
    dominant, n_dom = counts.most_common(1)[0]
    pct = n_dom / len(cells) * 100
    moist = [cell["moisture"] for cell in cells if cell["moisture"] is not None]
    avg_moist = sum(moist) / len(moist) if moist else None

    disease_counter = Counter(cell["class"] for cell in cells if cell["state"] == CellState.DISEASED)
    lat, lon = zone_center_gps(bounds, cfg)

    if dominant == CellState.DISEASED:
        info = _disease_info(agent, disease_counter.most_common(1)[0][0])
        kind = f" ({info['type'].capitalize()})" if info["type"] and info["type"] not in ("none", "unknown") else ""
        threat = f"{info['name']}{kind}"
        rec = info["action"]
        if info["severity"] in ("critical", "high") and info["urgency"]:
            rec += f" ({info['urgency']})"
    elif dominant == CellState.DRY:
        threat = "Under-watering / Soil Dryness"
        rec = "Check drip irrigation line for blockages and irrigate this zone."
    elif dominant == CellState.UNCERTAIN:
        threat = "Unclear (low model confidence)"
        rec = "Re-scan this zone or inspect it manually."
    else:  # healthy
        notes = []
        if counts[CellState.DISEASED]:
            info = _disease_info(agent, disease_counter.most_common(1)[0][0])
            notes.append(f"{info['name']} in {counts[CellState.DISEASED]} cell(s)")
        if counts[CellState.DRY]:
            notes.append(f"dry soil in {counts[CellState.DRY]} cell(s)")
        threat = "None" if not notes else "None (minor: " + "; ".join(notes) + ")"
        rec = "Standard irrigation schedule." + (" Spot-check the minor cells." if notes else "")

    return {
        "zone": name,
        "lat": round(lat, 4), "lon": round(lon, 4),
        "coords": f"Grid {name}: {lat:.4f}, {lon:.4f}",
        "status": STATUS_LABEL[dominant],
        "status_pct": round(pct, 1),
        "threat": threat,
        "recommendation": rec,
        "avg_moisture": None if avg_moist is None else round(avg_moist, 1),
        "ground_truth": _truth_text([truth[rc] for rc in coords]) if truth else "-",
        "cells_scanned": len(cells),
        "counts": {s.value: counts.get(s, 0) for s in CellState if s != CellState.UNSCANNED},
    }


def build_report(grid, cfg, agent, matches=None, truth=None):
    counts = grid.counts()
    total = grid.rows * grid.cols
    scanned = total - counts[CellState.UNSCANNED]
    area_ha = scanned * cfg.cell_size_m ** 2 / 10_000
    plants = scanned * cfg.plants_per_cell
    pct = lambda n: round(n / scanned * 100, 1) if scanned else 0.0

    summary = {
        "mode": "SIMULATION (no rover hardware)",
        "source": SOURCE_TEXT.get(cfg.scenario, cfg.scenario),
        "generated": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "area_scanned_ha": round(area_ha, 2),
        "plants_analyzed": plants,
        "health_score_pct": pct(counts[CellState.HEALTHY]),
        "diseased_pct": pct(counts[CellState.DISEASED]),
        "dry_pct": pct(counts[CellState.DRY]),
        "manual_check_pct": pct(counts[CellState.UNCERTAIN]),
        "cells_scanned": scanned,
        "cells_total": total,
        "simulation_accuracy_pct": pct(matches) if matches is not None else None,
    }
    zones = [z for z in (_analyse_zone(n, b, grid, cfg, agent, truth) for n, b in grid.zones.items()) if z]
    return {"summary": summary, "zones": zones, "config": asdict(cfg)}


def summary_markdown(report):
    s = report["summary"]
    lines = [
        "### Agricultural Condition Report",
        f"*{s['mode']} - generated {s['generated']}*  \n*Grid source: {s['source']}*",
        "",
        "#### Field Summary Analytics",
        f"- **Total area scanned:** {s['area_scanned_ha']} hectares",
        f"- **Total plants analyzed:** {s['plants_analyzed']:,}",
        f"- **Overall field health score:** {s['health_score_pct']}% healthy",
        f"- Disease detected: {s['diseased_pct']}%  |  Dry soil: {s['dry_pct']}%  |  "
        f"Needs manual check: {s['manual_check_pct']}%",
    ]
    if s["simulation_accuracy_pct"] is not None:
        lines.append(f"- *Check against true labels:* the agent's call matched the true condition in "
                     f"{s['simulation_accuracy_pct']}% of cells")
    return "\n".join(lines)


def zone_table_rows(report):
    rows = []
    for z in report["zones"]:
        moist = "-" if z["avg_moisture"] is None else f"{z['avg_moisture']:.0f}%"
        rows.append([z["coords"], f"{z['status']} ({z['status_pct']:.0f}%)",
                     z["threat"], z["recommendation"], moist, z["ground_truth"]])
    return rows


def save_report(report, out_dir):
    """Writes the report as JSON, CSV (zone table) and Markdown. Returns the file paths."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"condition_report_{stamp}")

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = zone_table_rows(report)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_HEADERS)
        w.writerows(rows)

    md = [summary_markdown(report), "", "#### Zone Analysis", "",
          "| " + " | ".join(TABLE_HEADERS) + " |",
          "|" + "---|" * len(TABLE_HEADERS)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return [base + ".md", base + ".csv", base + ".json"]
