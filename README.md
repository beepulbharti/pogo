# POGO

This is the code release for the paper *Multiaccuracy and Multicalibration with Proxy Groups*

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
uv run getSandP.py
uv run forecast.py
uv run generate_figures.py
```

To run forecasts for a specific stock, replace the forecast command with:

```bash
uv run forecast.py --stock <stock_name>
```

For example:

```bash
uv run forecast.py --stock AAPL
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
