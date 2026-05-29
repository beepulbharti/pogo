# POGO

This is the code release for the paper *Multiaccuracy and Multicalibration with Proxy Groups*

by [Beepul Bharti](https://beepulbharti.github.io), Ambar Pal, [Jacopo Teneggi](https://jacopoteneggi.github.io/)  and [Jeremias Sulam](https://jsulam.github.io/).
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

From the root of the repository:

```bash
cd synthetic
```

Run all synthetic experiments:

```bash
uv run run_all.py
```

Generate figures:

```bash
uv run generate_figures.py --num-groups <k>
```

For example:

```bash
uv run generate_figures.py --num-groups 50
```

---

## Stock market experiments

From the root of the repository:

```bash
cd stocks
```

Download/process S&P data:

```bash
uv run getSandP.py
```

Run forecasts for all stocks:

```bash
uv run forecast.py
```

Or run forecasts for a specific stock:

```bash
uv run forecast.py --stock <stock_name>
```

For example:

```bash
uv run forecast.py --stock AAPL
```

Generate figures:

```bash
uv run generate_figures.py
```

---

## MIMIC-IV experiments

From the root of the repository:

```bash
cd mimic
```

Run the experiment:

```bash
uv run run_experiment.py
```

Generate figures:

```bash
uv run generate_figures.py
```