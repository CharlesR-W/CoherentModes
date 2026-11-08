#!/usr/bin/env python3
"""Inline MOCKUP data exercising the generic CellsDevelopmental-derived template."""
from pathlib import Path
from report import write_report


def main():
    spec = {
        "title": "MOCKUP · interactive report template",
        "subtitle": "Synthetic data only: check rendering and controls, not scientific claims. "
                    "Three arbitrary checkpoints; no training run, seeds, or uncertainty bands.",
        "checkpoints": [0, 10, 20],
        "sections": [{
            "title": "Three browser-drawn views",
            "caption": "The checkpoint slider updates the scatter and heatmap from embedded numbers. "
                       "The line plot is a static overview. Axes and color ranges remain fixed.",
            "figures": [{
                "title": "MOCKUP · 3D coordinates at the selected checkpoint",
                "caption": "Drag to rotate; hover to inspect each synthetic point. "
                           "Coordinates are constructed directly, not an embedding or measured representation.",
                "frames": [[{"type": "scatter3d", "mode": "markers", "name": "Synthetic points",
                             "x": [0, 1, 2, 3], "y": [1, 0, 2, 1],
                             "z": [i * .2, 1 + i * .2, 2 - i * .2, 1],
                             "marker": {"size": 6, "color": "#4477aa"}}] for i in range(3)],
                "layout": {"margin": {"l": 0, "r": 0, "t": 0, "b": 0}, "scene": {"camera": {"eye": {"x": 2.6, "y": 2.6, "z": 2.2}}, **{axis: {"title": {"text": f"{axis[0]} (a.u.)"},
                                            "range": [-.5, 3.5]} for axis in ("xaxis", "yaxis", "zaxis")}}},
            }, {
                "title": "MOCKUP · matrix values at the selected checkpoint",
                "caption": "Rows and columns index a synthetic 3 × 3 matrix. "
                           "A fixed 0–6 color scale makes changes comparable; no values are capped.",
                "frames": [[{"type": "heatmap", "x": [0, 1, 2], "y": [0, 1, 2],
                             "z": [[i, 1+i, 2+i], [1+i, 2+i, 3+i], [2+i, 3+i, 4+i]],
                             "colorscale": "Viridis", "zmin": 0, "zmax": 6,
                             "colorbar": {"title": {"text": "Value (a.u.)"}}}] for i in range(3)],
                "layout": {"xaxis": {"title": {"text": "Column index"}, "dtick": 1, "range": [-.5, 2.5]},
                           "yaxis": {"title": {"text": "Row index"}, "dtick": 1, "range": [-.5, 2.5]}},
            }, {
                "title": "MOCKUP · scalar readout across all checkpoints",
                "caption": "Each point is an invented scalar, joined only as a visual guide. "
                           "The decreasing values demonstrate a line trace, not measured convergence.",
                "data": [{"type": "scatter", "mode": "lines+markers", "name": "Synthetic readout",
                          "x": [0, 10, 20], "y": [1, .6, .3]}],
                "layout": {"xaxis": {"title": {"text": "Checkpoint (step)"}, "range": [0, 20]},
                           "yaxis": {"title": {"text": "Readout (arbitrary units)"}, "range": [0, 1.1]}},
            }],
        }],
    }
    print(write_report(spec, Path(__file__).with_name("example.html")))


if __name__ == "__main__":
    main()
