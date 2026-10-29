"""
field_app.py
Field-scan SIMULATION app (no hardware needed).

A virtual rover walks a serpentine path over a zoned field grid. At each cell it
"photographs" a leaf (a random image from your PlantVillage folder) and reads a
simulated soil-moisture sensor. Your real CropAgent diagnoses every photo, the map
updates live, and at the end an Agricultural Condition Report is produced
(field summary + per-zone status, threat and recommendation) that you can download.

Usage:
    python field_app.py
"""

import os
import time

import gradio as gr
import matplotlib.pyplot as plt

from agent import CropAgent
from field_sim import (SimConfig, FieldSimulator, collect_by_class, split_classes, build_report,
                       summary_markdown, zone_table_rows, save_report,
                       SCENARIOS, TABLE_HEADERS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model and knowledge base...")
agent = CropAgent()
print("Ready.")


def resolve_folder(folder):
    folder = (folder or "").strip().strip('"')
    if os.path.isabs(folder) and os.path.isdir(folder):
        return folder
    for base in (os.getcwd(), BASE_DIR):
        candidate = os.path.join(base, folder)
        if os.path.isdir(candidate):
            return candidate
    return None


def run_scan(folder, crop, rows, cols, zone_rows, zone_cols, scenario_label,
             cell_size, plants_per_cell, delay, seed):
    rows, cols = int(rows), int(cols)
    zone_rows, zone_cols = min(int(zone_rows), max(1, rows // 2)), min(int(zone_cols), cols)

    folder_path = resolve_folder(folder)
    if folder_path is None:
        yield None, f"**Folder not found:** `{folder}`. Check the image folder path.", [], None
        return

    by_class = collect_by_class(folder_path, (crop or "").strip())
    healthy, diseased = split_classes(by_class)
    scenario = SCENARIOS[scenario_label]
    from_dataset = scenario.startswith("dataset")
    if not by_class or (not from_dataset and (not healthy or not diseased)):
        yield None, (f"**Not enough images for crop '{crop}'.** Found {len(by_class)} class folders "
                     f"({len(healthy)} healthy images, {len(diseased)} diseased classes). "
                     f"Try another crop name or clear the filter."), [], None
        return

    cfg = SimConfig(rows=rows, cols=cols, zone_rows=zone_rows, zone_cols=zone_cols,
                    cell_size_m=float(cell_size), plants_per_cell=int(plants_per_cell),
                    scenario=scenario, crop=(crop or "").strip(),
                    seed=None if seed is None else int(seed))
    sim = FieldSimulator(agent, healthy, diseased, cfg, by_class)

    fig = sim.grid.render("Field map - scan starting")
    yield fig, "_Rover starting the run..._", [], None
    plt.close(fig)

    for stop, _ in sim.scan_iter():
        fig = sim.grid.render(f"Field map - rover in aisle {stop['aisle'] + 1}, position {stop['col']}")
        yield fig, f"_Scanning..._  {sim.grid.summary_text()}", [], None
        plt.close(fig)
        time.sleep(delay)

    report = build_report(sim.grid, cfg, agent, sim.matches, sim.truth)
    files = save_report(report, os.path.join(BASE_DIR, "reports"))
    fig = sim.grid.render("Field map - scan complete")
    yield fig, summary_markdown(report), zone_table_rows(report), files
    plt.close(fig)


with gr.Blocks(title="Field Scan Simulation") as demo:
    gr.Markdown("# 🚜 Field Scan Simulation")
    gr.Markdown(
        "The field grid is built from **your test dataset**: every plant cell takes a real image and its true "
        "label from the folder name. A virtual rover drives along the aisles **between the plant rows**, "
        "scans the plant on each side with your trained model, then compiles an "
        "**Agricultural Condition Report** and checks it against the true labels.  \n"
        "🟩 healthy · 🟥 disease · 🟪 dry soil · 🟨 low confidence · 🔵 rover path"
    )

    with gr.Row():
        with gr.Column(scale=1):
            folder = gr.Textbox(label="Test dataset folder", value="data/val",
                                info="One sub-folder per class, e.g. data/val (use your test folder if you have one)")
            crop = gr.Textbox(label="Crop (optional filter)", value="Tomato",
                              info="Only classes starting with this, e.g. Tomato, Apple, Potato")
            scenario = gr.Dropdown(list(SCENARIOS), value=list(SCENARIOS)[0], label="How to build the grid")
            rows = gr.Slider(4, 12, value=8, step=1, label="Grid rows")
            cols = gr.Slider(4, 14, value=12, step=1, label="Grid columns")
            zone_rows = gr.Slider(1, 3, value=2, step=1, label="Zone rows")
            zone_cols = gr.Slider(1, 3, value=3, step=1, label="Zone columns")
            with gr.Accordion("Advanced", open=False):
                cell_size = gr.Number(value=18, label="Cell size (metres)")
                plants = gr.Number(value=150, label="Plants represented by each sampling point")
                delay = gr.Slider(0, 1.0, value=0.1, step=0.05, label="Delay per cell (seconds)")
                seed = gr.Number(value=None, precision=0, label="Random seed (blank = different every run)")
            start = gr.Button("▶ Start scan", variant="primary")
        with gr.Column(scale=2):
            map_plot = gr.Plot(label="Field map")

    summary = gr.Markdown()
    gr.Markdown("#### Zone Analysis")
    table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True)
    downloads = gr.File(label="Download report (Markdown / CSV / JSON)", file_count="multiple")

    start.click(run_scan,
                inputs=[folder, crop, rows, cols, zone_rows, zone_cols, scenario,
                        cell_size, plants, delay, seed],
                outputs=[map_plot, summary, table, downloads])

if __name__ == "__main__":
    demo.launch()
