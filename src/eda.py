"""
eda.py — exploratory figures for Problem 1(b). Saves PNGs to report/figures/.

Run from the repo root:
    python -m src.eda
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from . import data_prep

FIG_DIR = "report/figures"
BLUE, ORANGE, GRID, INK = "#3266AD", "#D9822B", "#E3E3E3", "#333333"
SEED = data_prep.SEED


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[eda] saved {path}")


def price_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    p = df["price"].to_numpy()
    cap = np.quantile(p, 0.99)
    ax[0].hist(p[p <= cap], bins=60, color=BLUE)
    _style(ax[0], f"price (AZN), up to the 99th percentile ({cap:,.0f})", "listings")
    ax[0].set_title("price", fontsize=10, color=INK)
    ax[1].hist(np.log(p), bins=60, color=BLUE)
    _style(ax[1], "log(price)", "listings")
    ax[1].set_title("log(price)", fontsize=10, color=INK)
    _save(fig, "eda_price_distribution.png")


def area_vs_price(df):
    d = df.sample(min(len(df), 20000), random_state=SEED)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(d["area_m2"], d["price"], s=3, alpha=0.25, color=BLUE, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "area (m², log scale)", "price (AZN, log scale)")
    _save(fig, "eda_area_vs_price.png")


def rooms_vs_price(df):
    d = df.dropna(subset=["rooms"])
    r = d["rooms"].clip(upper=6).astype(int)
    levels = sorted(r.unique())
    med = [d.loc[r == k, "price"].median() for k in levels]
    cnt = [int((r == k).sum()) for k in levels]
    labels = [f"{k}+" if k == 6 else str(k) for k in levels]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.bar(labels, med, color=BLUE, width=0.6)
    for x, c in zip(range(len(levels)), cnt):
        ax.text(x, 0, f"n={c:,}", ha="center", va="bottom", fontsize=6.5, color="white", rotation=90)
    _style(ax, "rooms (6+ grouped)", "median price (AZN)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(False, axis="x")
    _save(fig, "eda_rooms_vs_price.png")


def location_effect(df):
    d = df.sample(min(len(df), 30000), random_state=SEED)
    lo_x, hi_x = np.quantile(d["lng"], [0.01, 0.99])
    lo_y, hi_y = np.quantile(d["lat"], [0.01, 0.99])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sc = ax.scatter(d["lng"], d["lat"], c=np.log(d["price"]), s=3, alpha=0.5,
                    cmap="Blues", linewidths=0)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    _style(ax, "longitude", "latitude")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log(price)", color=INK)
    cb.outline.set_visible(False)
    _save(fig, "eda_location_price.png")


def tier_balance(df):
    X, y, _ = data_prep.make_features(df)
    _, ytr, _, yva, _, yte = data_prep.train_val_test_split(X, y)
    ttr, thr = data_prep.make_tier_label(ytr)
    tva, _ = data_prep.make_tier_label(yva, thr)
    tte, _ = data_prep.make_tier_label(yte, thr)
    names = ["train", "val", "test"]
    std = [int((t == 0).sum()) for t in (ttr, tva, tte)]
    prem = [int((t == 1).sum()) for t in (ttr, tva, tte)]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.bar(x - 0.2, std, 0.38, color=BLUE, label="standard (0)")
    ax.bar(x + 0.2, prem, 0.38, color=ORANGE, label="premium (1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    _style(ax, f"split (tier threshold = {thr:,.0f} AZN)", "listings")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "eda_tier_balance.png")
    print(f"[eda] premium share train/val/test = {ttr.mean():.3f} / {tva.mean():.3f} / {tte.mean():.3f}")


def main():
    df = data_prep.clean(data_prep.load_raw())
    price_distribution(df)
    area_vs_price(df)
    rooms_vs_price(df)
    location_effect(df)
    tier_balance(df)


if __name__ == "__main__":
    main()