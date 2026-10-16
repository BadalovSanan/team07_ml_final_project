"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# Columns that LEAK the target (derived from price). Drop these before training.
# TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",     # "3 450 AZN/m²": price restated per m²
    "total_price",    # equals `price` in 99.99% of rows
]


# Other columns dropped in clean(): identifiers, URLs, scrape bookkeeping, free text,
# post-listing popularity/promotion, currency (always AZN), English duplicates of the
# Azerbaijani columns we keep, and `Binanın növü` (only 154 non-null values).
DROP_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url", "img_url",
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "day_y", "hour_x", "hour_y",
    "updated", "owner_name", "owner_title", "shop_name", "shop_title", "address",
    "currency_x", "currency_y", "views", "vip", "featured", "products_label",
    "description", "attributes", "extra_info", "city_when",
    "repair", "bill_of_sale", "mortgage", "Binanın növü",
]

# Outlier rules: fixed bounds (not computed from the data, so no leakage).
PRICE_MIN, PRICE_MAX = 3_000, 10_000_000      # AZN
AREA_MIN, AREA_MAX = 10, 50_000               # m²
ROOMS_MAX = 20
LAT_RANGE = (38.0, 42.0)                      # Azerbaijan bounding box
LNG_RANGE = (44.5, 51.0)
SOT_TO_M2 = 100.0                             # 1 sot = 100 m²


def _num(s):
    """First number in a string like '1 200 m²' / '2,5 sot' (spaces and commas fixed)."""
    t = re.sub(r"\s+", "", s.lower()).replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", t)
    return t, (float(m.group()) if m else np.nan)


def _area_m2(s):
    """'145 m²' -> 145 ; '3 sot' -> 300 ; unparseable -> NaN."""
    if not isinstance(s, str):
        return np.nan
    t, v = _num(s)
    return v * SOT_TO_M2 if "sot" in t else v


def _land_sot(s):
    """'2.5 sot' -> 2.5 ; missing -> NaN."""
    return _num(s)[1] if isinstance(s, str) else np.nan


def _floor(s):
    """'7 / 9' -> (7, 9) ; missing/odd -> (NaN, NaN)."""
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", s) if isinstance(s, str) else None
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    df = df.copy()
    n0 = len(df)

    def step(msg, before):
        print(f"[clean] {msg}: removed {before - len(df):,}")

    # types: numeric columns arrive as strings ("145 m²", "7 / 9", "2.5 sot")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area_m2"] = df["Sahə"].map(_area_m2)
    df["land_sot"] = df["Torpaq sahəsi"].map(_land_sot)
    fl = df["Mərtəbə"].map(_floor)
    df["floor"] = [f[0] for f in fl]
    df["n_floors"] = [f[1] for f in fl]
    df["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")

    b = len(df)
    df = df.dropna(subset=["price", "area_m2", "lat", "lng"])
    step("missing/unparseable price, area or coordinates", b)

    # duplicates: the dump has no exact duplicate rows/ids, so match on listing content
    b = len(df)
    df = df.drop_duplicates(subset=["price", "Sahə", "Otaq sayı", "Mərtəbə", "lat",
                                    "lng", "Kateqoriya", "location", "description"])
    step("content duplicates", b)

    # outliers (all prices are AZN, no currency conversion needed)
    b = len(df)
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    step(f"price outside [{PRICE_MIN:,}, {PRICE_MAX:,}] AZN", b)
    b = len(df)
    df = df[(df["area_m2"] >= AREA_MIN) & (df["area_m2"] <= AREA_MAX)]
    step(f"area outside [{AREA_MIN}, {AREA_MAX:,}] m²", b)
    b = len(df)
    df = df[df["rooms"].isna() | (df["rooms"] <= ROOMS_MAX)]
    step(f"rooms > {ROOMS_MAX}", b)
    b = len(df)
    df = df[df["lat"].between(*LAT_RANGE) & df["lng"].between(*LNG_RANGE)]
    step("coordinates outside Azerbaijan", b)

    # drop leakage + unused columns, then translate the Azerbaijani names we keep
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS + DROP_COLUMNS if c in df.columns])
    df = df.drop(columns=["Sahə", "Torpaq sahəsi", "Mərtəbə", "Otaq sayı"])
    df = df.rename(columns={"Kateqoriya": "category", "Təmir": "repair",
                            "Çıxarış": "bill_of_sale", "İpoteka": "mortgage"})
    df = df.reset_index(drop=True)
    print(f"[clean] rows: {n0:,} -> {len(df):,} | leakage columns dropped: {LEAKAGE_COLUMNS}")
    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    # Missing floor/rooms/land is mostly structural (a house has no floor), so fill with
    # a fixed 0 (+ an indicator) instead of a statistic learned from the data.
    cols = {
        "area_m2": df["area_m2"],
        "land_sot": df["land_sot"].fillna(0),
        "rooms": df["rooms"].fillna(0),
        "rooms_missing": df["rooms"].isna(),
        "floor": df["floor"].fillna(0),
        "n_floors": df["n_floors"].fillna(0),
        "floor_missing": df["floor"].isna(),
        "lat": df["lat"],              # location enters through lat/lng
        "lng": df["lng"],
        "repair_yes": df["repair"] == "var",
        "repair_no": df["repair"] == "yoxdur",      # NaN -> both 0
        "bill_of_sale": df["bill_of_sale"] == "var",
        "mortgage": df["mortgage"] == "var",
        "is_baku": df["city"] == "bakı",
    }
    for cat in ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi",
                "Torpaq", "Obyekt", "Ofis", "Qaraj"]:   # fixed list -> same columns in every split
        cols[f"cat_{cat}"] = df["category"] == cat
    names = list(cols)
    X = np.column_stack([np.asarray(cols[n], dtype=float) for n in names])
    y = df["price"].to_numpy(dtype=float)
    return X, y, names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    # Stratified on above/below the overall median price so the premium share is the
    # same in every split. This only decides which rows go where; the real tier
    # threshold is still the TRAIN median (make_tier_label on the train prices).
    X, y = np.asarray(X), np.asarray(y)
    strata = (y > np.median(y)).astype(int)
    rng = np.random.default_rng(seed)
    parts = {"tr": [], "val": [], "te": []}
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_te, n_val = int(round(len(idx) * test_size)), int(round(len(idx) * val_size))
        parts["te"].append(idx[:n_te])
        parts["val"].append(idx[n_te:n_te + n_val])
        parts["tr"].append(idx[n_te + n_val:])
    tr, val, te = (rng.permutation(np.concatenate(parts[k])) for k in ("tr", "val", "te"))
    return X[tr], y[tr], X[val], y[val], X[te], y[te]


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0          # constant column: centre only
    return tuple((X - mu) / sd for X in (X_tr, *others))