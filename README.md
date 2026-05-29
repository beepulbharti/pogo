# POGO

This is the code release for the paper *Parameter-Free and Group Conditional Online Conformal Prediction*

by [Beepul Bharti](https://beepulbharti.github.io), Ambar Pal, [Jacopo Teneggi](https://jacopoteneggi.github.io/) and [Jeremias Sulam](https://jsulam.github.io/).

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for Python environment and dependency management.

Install `uv` if needed:

```bash
pip install uv
```

Then, from the root of the repository, install dependencies:

```bash
uv sync
```

---

## Synthetic experiments

Run the following commands from the root of the repository:

```bash
cd synthetic
uv run run_all.py
uv run generate_figures.py --num-groups 50
```

To generate figures for a different number of groups, replace `50` with the desired value of `k`:

```bash
uv run generate_figures.py --num-groups <k>
```

---

## Stock market experiments

Run the following commands from the root of the repository:

```bash
cd stocks
uv run run_experiment.py --stock <stock_name>
uv run generate_figures.py --stock <stock_name>
```

For example, to run the experiment and generate figures for the AAPL stock, run the following commands

```bash
uv run run_experiment.py --stock AAPL
uv run generate_figures.py --stock AAPL
```

---

## MIMIC-IV experiments

Run the following commands from the root of the repository:

```bash
cd mimic
uv run run_experiment.py
uv run generate_figures.py
```

---

## References
