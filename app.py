import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import difflib
import base64
import json
import requests

st.set_page_config(page_title="Macro Satiety Optimizer (MVP)", layout="centered")

# ----------------------------
# GitHub-backed persistence
# ----------------------------
def _gh_headers():
    return {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }

def gh_get_file(path: str) -> tuple[str | None, str | None]:
    """Return (content_text, sha). If file doesn't exist, returns (None, None)."""
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(), timeout=20)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]

def gh_put_file(path: str, content_text: str, message: str, sha: str | None = None) -> None:
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=_gh_headers(), json=payload, timeout=20)
    r.raise_for_status()

MEALS_GH_PATH = "data/meals.json"

def load_meals() -> dict:
    txt, _ = gh_get_file(MEALS_GH_PATH)
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        return {}

def save_meals(meals: dict) -> None:
    _, sha = gh_get_file(MEALS_GH_PATH)
    new_txt = json.dumps(meals, indent=2)
    gh_put_file(MEALS_GH_PATH, new_txt, "Update meals", sha=sha)

def require_password():
    password = st.secrets.get("APP_PASSWORD")

    if not password:
      return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("## 🔐 Macro Volume Prioritiser")
        st.markdown("Enter password to continue")

        entered = st.text_input("Password", type="password")

        if st.button("Login"):
            if entered == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

        st.stop()

require_password()

# ----------------------------
# Data loading + validation
# ----------------------------
@st.cache_data
def load_foods(path: str = "foods.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "food", "category", "swap_group", "typical_serving_g",
        "kcal_per_100g", "protein_per_100g", "carbs_per_100g",
        "fat_per_100g", "fibre_per_100g",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"foods.csv missing columns: {sorted(missing)}")

    num_cols = [
        "typical_serving_g", "kcal_per_100g", "protein_per_100g", "carbs_per_100g",
        "fat_per_100g", "fibre_per_100g"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "kcal_per_serving" in df.columns:
        df["kcal_per_serving"] = pd.to_numeric(df["kcal_per_serving"], errors="coerce")
    else:
        df["kcal_per_serving"] = np.nan

    need_calc = df["kcal_per_serving"].isna()
    df.loc[need_calc, "kcal_per_serving"] = (
        df.loc[need_calc, "kcal_per_100g"] * df.loc[need_calc, "typical_serving_g"] / 100.0
    )

    bad_rows = df[
        df["food"].isna() |
        df["swap_group"].isna() |
        df["typical_serving_g"].isna() |
        df["kcal_per_100g"].isna()
    ]
    if len(bad_rows) > 0:
        example = bad_rows.head(5)[["food", "swap_group", "typical_serving_g", "kcal_per_100g"]]
        raise ValueError(
            "foods.csv has missing/invalid values in key columns. "
            "Check these example rows:\n" + example.to_string(index=False)
        )

    df = df[df["kcal_per_100g"] > 0].copy()
    return df

def norm_food_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

# ----------------------------
# User foods
# ----------------------------
USER_FOODS_GH_PATH = "data/user_foods.csv"

def _user_foods_empty_df() -> pd.DataFrame:
    cols = [
        "food","category","swap_group","typical_serving_g","kcal_per_serving",
        "kcal_per_100g","protein_per_100g","carbs_per_100g","fat_per_100g","fibre_per_100g",
        "unit_label"
    ]
    return pd.DataFrame(columns=cols)

def load_user_foods() -> pd.DataFrame:
    txt, _ = gh_get_file(USER_FOODS_GH_PATH)
    if not txt:
        return _user_foods_empty_df()

    df = pd.read_csv(pd.io.common.StringIO(txt))

    # Coerce numerics safely
    num_cols = [
        "typical_serving_g","kcal_per_serving","kcal_per_100g",
        "protein_per_100g","carbs_per_100g","fat_per_100g","fibre_per_100g"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "unit_label" not in df.columns:
        df["unit_label"] = ""

    # Fill kcal_per_serving if missing
    if "kcal_per_serving" not in df.columns:
        df["kcal_per_serving"] = np.nan

    need = df["kcal_per_serving"].isna()
    if "kcal_per_100g" in df.columns and "typical_serving_g" in df.columns:
        df.loc[need, "kcal_per_serving"] = df.loc[need, "kcal_per_100g"] * df.loc[need, "typical_serving_g"] / 100.0

    return df

def append_user_food(row: dict, update_if_exists: bool = False):
    # Load current CSV from GitHub
    txt, sha = gh_get_file(USER_FOODS_GH_PATH)

    if txt:
        df = pd.read_csv(pd.io.common.StringIO(txt))
    else:
        df = _user_foods_empty_df()

    if "unit_label" not in df.columns:
        df["unit_label"] = ""

    def _norm(s): 
        return " ".join(str(s).strip().lower().split())

    row_food_norm = _norm(row["food"])
    if df.empty:
        idx = None
    else:
        matches = df["food"].apply(_norm) == row_food_norm
        idx = df.index[matches].tolist()
        idx = idx[0] if idx else None

    # If exists
    if idx is not None:
        if not update_if_exists:
            raise ValueError("That food already exists in your user list.")

        # Detect whether anything is actually different
        def _as_str(x):
            return "" if pd.isna(x) else str(x)

        changed_fields = []
        for k, v in row.items():
            if k not in df.columns:
                continue
            old = df.at[idx, k]
            if _as_str(old) != _as_str(v):
                changed_fields.append(k)

        if not changed_fields:
            
            return

        # Overwrite the row
        for k, v in row.items():
            if k in df.columns:
                df.at[idx, k] = v

        msg = f"Update user food: {row['food']} ({', '.join(changed_fields)})"
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        msg = f"Add user food: {row['food']}"

    # Save back to GitHub
    out = df.to_csv(index=False)
    gh_put_file(USER_FOODS_GH_PATH, out, msg, sha=sha)

def find_existing_food(df: pd.DataFrame, food_name: str):
    def _norm(s): return " ".join(str(s).strip().lower().split())
    target = _norm(food_name)
    if df.empty or "food" not in df.columns:
        return None
    matches = df["food"].apply(_norm) == target
    idxs = df.index[matches].tolist()
    return idxs[0] if idxs else None

def diff_food_rows(old_row: dict, new_row: dict) -> list[str]:
    changed = []
    for k, v in new_row.items():
        if k not in old_row:
            continue
        old = old_row.get(k)
        # compare as strings to avoid float formatting pain
        if ("" if pd.isna(old) else str(old)) != ("" if pd.isna(v) else str(v)):
            changed.append(k)
    return changed
# ----------------------------
# Core calculations
# ----------------------------
def satiety_score_row(row) -> float:
    kcal = max(float(row["kcal_per_100g"]), 1e-6)
    protein = float(row["protein_per_100g"])
    fibre = float(row["fibre_per_100g"])

    protein_per_kcal = protein / kcal
    fibre_per_kcal = fibre / kcal
    energy_density_penalty = kcal / 100.0

    return (protein_per_kcal * 100) + (fibre_per_kcal * 200) - (energy_density_penalty * 1.5)

def macros_for_grams(row, grams: float):
    factor = grams / 100.0
    kcal = float(row["kcal_per_100g"]) * factor
    p = float(row["protein_per_100g"]) * factor
    c = float(row["carbs_per_100g"]) * factor
    f = float(row["fat_per_100g"]) * factor
    fibre = float(row["fibre_per_100g"]) * factor
    return kcal, p, c, f, fibre

def build_why(row, chosen_kcal_per_100g, chosen_p_per_kcal, chosen_fibre_per_kcal, chosen_grams) -> str:
    reasons = []

    # Volume
    if row["grams_suggested"] >= chosen_grams * 1.30:
        extra = row["grams_suggested"] - chosen_grams
        reasons.append(f"More volume (+{int(round(extra))}g)")

    # Lower energy density
    if row["kcal_per_100g"] <= chosen_kcal_per_100g * 0.85:
        reasons.append("Lower kcal/100g")

    # Protein density per calorie
    kcal = max(float(row["kcal_per_100g"]), 1e-6)
    p_per_kcal = float(row["protein_per_100g"]) / kcal
    if p_per_kcal >= chosen_p_per_kcal * 1.15:
        reasons.append("Higher protein/kcal")

    # Fibre density per calorie
    fibre_per_kcal = float(row["fibre_per_100g"]) / kcal
    if fibre_per_kcal >= chosen_fibre_per_kcal * 1.15 and float(row["fibre_per_100g"]) > 0:
        reasons.append("Higher fibre/kcal")

    if not reasons:
        reasons.append("Closest macro match")

    # Keep it short
    return ", ".join(reasons[:3])

def find_swaps(
    df: pd.DataFrame,
    chosen_food: str,
    grams: float,
    focus: str,
    kcal_tol_pct: float,
    macro_tol_pct: float,
    top_n: int,
    max_servings: float,
    prefer_same_group: bool,
    match_mode: str,
):
    chosen = df.loc[df["food"] == chosen_food].iloc[0]
    chosen_group = chosen["swap_group"]

    chosen_kcal, chosen_p, chosen_c, chosen_f, chosen_fibre = macros_for_grams(chosen, grams)

    kcal_low = chosen_kcal * (1 - kcal_tol_pct / 100)
    kcal_high = chosen_kcal * (1 + kcal_tol_pct / 100)

    # chosen density references (for "why")
    chosen_kcal_per_100g = float(chosen["kcal_per_100g"])
    chosen_p_per_kcal = float(chosen["protein_per_100g"]) / max(chosen_kcal_per_100g, 1e-6)
    chosen_fibre_per_kcal = float(chosen["fibre_per_100g"]) / max(chosen_kcal_per_100g, 1e-6)

    candidates = df.copy()
    candidates["satiety_score"] = candidates.apply(satiety_score_row, axis=1)

        # ---- Remove duplicates + exclude the chosen food (robust to case/whitespace) ----
    candidates["food_norm"] = candidates["food"].apply(norm_food_name)
    chosen_norm = norm_food_name(chosen_food)

    # Drop duplicate foods (e.g. "Asparagus" and "asparagus ")
    candidates = candidates.drop_duplicates(subset=["food_norm"], keep="first")

    # Exclude the chosen food from suggestions
    candidates = candidates[candidates["food_norm"] != chosen_norm].copy()

    # grams needed to match chosen calories
    # -------------------------
# Matching logic
# -------------------------

    if match_mode.startswith("Calorie-match"):
        # Match calories exactly (current behaviour)
        candidates["grams_suggested"] = (chosen_kcal / candidates["kcal_per_100g"]) * 100.0

    else:
        # Serving-based matching
        serving_options = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        serving_options = serving_options[serving_options <= max_servings]

        focus_key = {
            "protein": "protein_per_100g",
            "carbs": "carbs_per_100g",
            "fat": "fat_per_100g"
        }[focus]

        target_macro = {
            "protein": chosen_p,
            "carbs": chosen_c,
            "fat": chosen_f
        }[focus]

        best_grams = []
        best_focus_diff = []

        for _, row in candidates.iterrows():
            grams_trials = row["typical_serving_g"] * serving_options
            factor = grams_trials / 100.0
            focus_trials = row[focus_key] * factor
            diff = np.abs(focus_trials - target_macro)

            i_best = int(np.argmin(diff))
            best_grams.append(float(grams_trials[i_best]))
            best_focus_diff.append(float(diff[i_best]))

        candidates["grams_suggested"] = best_grams


    # Portion guardrails via servings
    candidates["servings_suggested"] = candidates["grams_suggested"] / candidates["typical_serving_g"]


    # Compute macros at kcal-matched grams
    def compute_at_match(row):
        g = float(row["grams_suggested"])
        kcal, p, c, f, fibre = macros_for_grams(row, g)
        return pd.Series({"kcal": kcal, "p": p, "c": c, "f": f, "fibre": fibre})

    computed = candidates.apply(compute_at_match, axis=1)
    candidates = pd.concat([candidates, computed], axis=1)

    # Filter by calorie tolerance
    candidates = candidates[(candidates["kcal"] >= kcal_low) & (candidates["kcal"] <= kcal_high)]

    # Prefer same swap_group (only if we still have enough options)
    if prefer_same_group:
        same_group = candidates[candidates["swap_group"] == chosen_group]
        if len(same_group) >= top_n:
            candidates = same_group

    # Macro focus matching
    focus_col = {"protein": "p", "carbs": "c", "fat": "f"}[focus]
    target_macro = {"protein": chosen_p, "carbs": chosen_c, "fat": chosen_f}[focus]

    if target_macro < 1e-6:
        candidates["focus_diff_pct"] = 0.0
    else:
        candidates["focus_diff_pct"] = (candidates[focus_col] - target_macro).abs() / target_macro * 100.0

    candidates = candidates[candidates["focus_diff_pct"] <= macro_tol_pct]

    # Macro drift (% difference vs chosen item at calorie-matched grams)
    candidates["protein_diff_pct"] = (candidates["p"] - chosen_p).abs() / max(chosen_p, 1e-6) * 100.0
    candidates["carbs_diff_pct"] = (candidates["c"] - chosen_c).abs() / max(chosen_c, 1e-6) * 100.0
    candidates["fat_diff_pct"] = (candidates["f"] - chosen_f).abs() / max(chosen_f, 1e-6) * 100.0

    candidates["macro_drift_pct"] = candidates[["protein_diff_pct", "carbs_diff_pct", "fat_diff_pct"]].max(axis=1)

    # Non-focus drift (worst drift of the OTHER two macros)
    if focus == "protein":
        candidates["non_focus_drift_pct"] = candidates[["carbs_diff_pct", "fat_diff_pct"]].max(axis=1)
    elif focus == "carbs":
        candidates["non_focus_drift_pct"] = candidates[["protein_diff_pct", "fat_diff_pct"]].max(axis=1)
    else:
        candidates["non_focus_drift_pct"] = candidates[["protein_diff_pct", "carbs_diff_pct"]].max(axis=1)

    # Why this swap
    candidates["why"] = candidates.apply(
        lambda r: build_why(
            r,
            chosen_kcal_per_100g=chosen_kcal_per_100g,
            chosen_p_per_kcal=chosen_p_per_kcal,
            chosen_fibre_per_kcal=chosen_fibre_per_kcal,
            chosen_grams=grams,
        ),
        axis=1
    )

    # Prefer higher volume at same calories
    candidates["volume_bonus"] = candidates["grams_suggested"]

    # Final ranking
    candidates = candidates.sort_values(
        by=["focus_diff_pct", "satiety_score", "volume_bonus"],
        ascending=[True, False, False]
    )

    return {
        "chosen": {
            "food": chosen_food,
            "swap_group": chosen_group,
            "grams": grams,
            "kcal": chosen_kcal,
            "p": chosen_p,
            "c": chosen_c,
            "f": chosen_f,
            "fibre": chosen_fibre,
        },
        "swaps": candidates.head(top_n)
    }

def item_to_grams(df: pd.DataFrame, food: str, amount, unit) -> float:
    # Defaults
    if unit not in ("grams", "servings"):
        unit = "grams"
    try:
        amount = float(amount)
    except Exception:
        amount = 0.0

    if amount <= 0:
        return 0.0

    # Guard: food must exist
    match = df.loc[df["food"] == food]
    if match.empty:
        return 0.0

    serving_g = float(match.iloc[0]["typical_serving_g"])
    return amount * serving_g if unit == "servings" else amount

def meal_totals(df: pd.DataFrame, items: list[dict]) -> dict:
    tot = {"kcal":0.0,"p":0.0,"c":0.0,"f":0.0,"fibre":0.0}
    for it in items:
        g = item_to_grams(df, it["food"], it["amount"], it["unit"])
        row = df.loc[df["food"] == it["food"]].iloc[0]
        kcal, p, c, f, fibre = macros_for_grams(row, g)
        tot["kcal"] += kcal; tot["p"] += p; tot["c"] += c; tot["f"] += f; tot["fibre"] += fibre
    return tot
# ----------------------------
# UI
# ----------------------------
st.title("Macro Volume Prioritiser (MVP)")
st.caption("Help swap calorie-dense, low volume foods for higher-volume, more filling alternatives while staying close on calories/macros.")

# Load foods
base_df = load_foods()
user_df = load_user_foods()

if user_df is not None and not user_df.empty:
    merged = pd.concat([base_df, user_df], ignore_index=True)
    merged["food_norm"] = merged["food"].apply(norm_food_name)
    merged = merged.drop_duplicates(subset=["food_norm"], keep="first").drop(columns=["food_norm"])
    df = merged
else:
    df = base_df

st.sidebar.markdown(
"""
### How this works
We calorie-match your food, then rank alternatives by:
- Macro similarity
- Satiety score
- Portion realism
"""
)

with st.sidebar:
    st.header("Settings")

    kcal_tol_pct = st.slider(
        "Calorie tolerance (%)",
        2, 30, 10,
        help="""
How far the swap can deviate from your original calories.

Example:
If your food is 300 kcal and tolerance is 10%,
the swap can be between 270–330 kcal.

Lower = stricter matching.
Higher = more swap options.
"""
    )

    focus = st.selectbox(
        "Macro to prioritise matching",
        ["protein", "carbs", "fat"],
        help="""
Which macro should stay closest to your original food?

Example:
If you choose 'protein', swaps will try to keep protein grams
similar while adjusting carbs/fat.
"""
    )

    macro_tol_pct = st.slider(
        "Macro focus tolerance (%)",
        5, 35, 25,
        help="""
How close the chosen macro must remain.

Example:
If your food has 20g protein and tolerance is 15%,
the swap must stay within 17–23g protein.
"""
    )

    match_mode = st.radio(
        "Swap matching mode",
        ["Calorie-match (strict)", "Serving-match (real-world)"],
        help="""
    Calorie-match: scales grams to closely match calories.

    Serving-match: suggests swaps in realistic serving sizes and shows macro differences.
    """
    )

    st.divider()

    prefer_same_group = st.checkbox(
        "Prefer swaps from same swap group",
        value=True,
        help="""
Keeps swaps in the same logical category.

Example:
If swapping salmon, it prefers other fatty proteins
instead of jumping to yoghurt or rice.
"""
    )

    max_servings = st.slider(
        "Max suggested servings",
        1.0, 6.0, 4.0, 0.5,
        help="""
Limits how large a portion the swap can suggest.

Example:
If typical serving of potatoes is 250g,
and max servings is 4,
the app won't suggest more than 1000g.
"""
    )

    top_n = st.slider(
        "Number of suggestions",
        3, 15, 5,
        help="How many swap options to display."
    )

# If a meal is loaded, allow picking which item to optimise
if st.session_state.get("meal_items"):
    st.subheader("Quick load from meal")
    meal_df = pd.DataFrame(st.session_state.meal_items)
    pick_idx = st.selectbox(
        "Pick an item to optimise",
        options=list(range(len(meal_df))),
        format_func=lambda i: f"{meal_df.loc[i,'food']} ({meal_df.loc[i,'amount']} {meal_df.loc[i,'unit']})"
    )

    picked = st.session_state.meal_items[pick_idx]
    # Pre-fill your main inputs
    food = picked["food"]
    if picked["unit"] == "servings":
        # your UI already supports servings; set defaults
        input_mode = "Servings"
        servings = float(picked["amount"])
        grams = item_to_grams(df, food, servings, "servings")
    else:
        input_mode = "Grams"
        grams = float(picked["amount"])
        servings = grams / float(df.loc[df["food"]==food].iloc[0]["typical_serving_g"])
st.subheader("1) Choose what you're currently eating")

col1, col2 = st.columns([2, 1])

with col1:
    foods_list = sorted(df["food"].unique())
    food = st.selectbox("Food", foods_list)

selected_row = df.loc[df["food"] == food].iloc[0]
typical_serving = selected_row["typical_serving_g"]
unit_label = selected_row.get("unit_label", "")

with col1:
    input_mode = st.radio(
        "Enter amount as:",
        ["Grams", "Servings"],
        horizontal=True
    )

    if input_mode == "Grams":
        grams = st.number_input(
            "Grams",
            min_value=1.0,
            max_value=2000.0,
            value=float(typical_serving),
            step=10.0
        )
        servings = grams / typical_serving
    else:
        servings = st.number_input(
            "Servings",
            min_value=0.5,
            max_value=20.0,
            value=1.0,
            step=0.5
        )
        grams = servings * typical_serving

with col2:
    st.markdown("### Serving info")

    if unit_label:
        st.write(f"1 serving ≈ **{int(typical_serving)}g**")
        st.write(f"≈ {unit_label}")
    else:
        st.write(f"1 serving = **{int(typical_serving)}g**")

    st.write(f"Current amount:")
    st.write(f"**{grams:.0f}g**")
    st.write(f"≈ {servings:.2f} servings")


if st.button("Suggest higher-satiety swaps"):
    result = find_swaps(
        df=df,
        chosen_food=food,
        grams=grams,
        focus=focus,
        kcal_tol_pct=kcal_tol_pct,
        macro_tol_pct=macro_tol_pct,
        top_n=top_n,
        max_servings=max_servings,
        prefer_same_group=prefer_same_group,
        match_mode=match_mode,
    )

    chosen = result["chosen"]
    swaps = result["swaps"]

    st.markdown("### Your current choice")
    st.write(
        f"**{chosen['food']} — {chosen['grams']:.0f}g**  \n"
        f"Group: **{chosen['swap_group']}**  \n"
        f"Calories: **{chosen['kcal']:.0f} kcal** | "
        f"P: **{chosen['p']:.1f}g** | C: **{chosen['c']:.1f}g** | "
        f"F: **{chosen['f']:.1f}g** | Fibre: **{chosen['fibre']:.1f}g**"
    )

    st.markdown("### Swap suggestions (calorie-matched)")
    if swaps.empty:
        st.warning(
            "No swaps found within your settings. Try:\n"
            "- Increasing calorie tolerance\n"
            "- Increasing macro focus tolerance\n"
            "- Increasing max suggested servings\n"
            "- Turning off 'prefer same swap group'"
        )
    else:
        # Drift warning (keeps trust high)
        if (swaps["non_focus_drift_pct"] > 30).any():
            st.info(
                "Some swaps match your chosen macro but drift a lot on the other macros. "
                "If you want stricter swaps, lower calorie tolerance and/or macro focus tolerance."
            )

        display_cols = [
            "food", "category", "swap_group",
            "typical_serving_g", "kcal_per_serving",
            "grams_suggested", "servings_suggested",
            "kcal", "p", "c", "f", "fibre",
            "satiety_score", "focus_diff_pct",
            "protein_diff_pct", "carbs_diff_pct", "fat_diff_pct",
            "macro_drift_pct", "non_focus_drift_pct",
            "why"
        ]
        display = swaps[display_cols].copy()

        display.rename(columns={
            "food": "Swap to",
            "category": "Category",
            "swap_group": "Swap group",
            "typical_serving_g": "Typical serving (g)",
            "kcal_per_serving": "Kcal per serving",
            "grams_suggested": "Suggested grams",
            "servings_suggested": "Suggested servings",
            "kcal": "Calories",
            "p": "Protein (g)",
            "c": "Carbs (g)",
            "f": "Fat (g)",
            "fibre": "Fibre (g)",
            "satiety_score": "Satiety score",
            "focus_diff_pct": f"{focus.title()} focus diff (%)",
            "protein_diff_pct": "Protein diff (%)",
            "carbs_diff_pct": "Carbs diff (%)",
            "fat_diff_pct": "Fat diff (%)",
            "macro_drift_pct": "Macro drift (%)",
            "non_focus_drift_pct": "Non-focus drift (%)",
            "why": "Why this swap"
        }, inplace=True)

        # Rounding
        for col in ["Typical serving (g)", "Kcal per serving", "Suggested grams", "Calories"]:
            display[col] = display[col].round(0).astype(int)

        display["Suggested servings"] = display["Suggested servings"].round(2)

        pct_cols = [
            f"{focus.title()} focus diff (%)", "Protein diff (%)", "Carbs diff (%)", "Fat diff (%)",
            "Macro drift (%)", "Non-focus drift (%)"
        ]
        for col in ["Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)", "Satiety score"] + pct_cols:
            if col in display.columns:
                display[col] = display[col].round(1)
                
        display = display.loc[:, ~display.columns.duplicated()].copy()

        st.dataframe(display, use_container_width=True)

        st.caption(
            "Satiety score is an MVP proxy (protein + fibre per calorie, penalised by calorie density). "
            "‘Macro drift’ shows how far the other macros move at calorie-matched portions."
        )

with st.sidebar:
    st.divider()
    with st.expander("🍽️ Saved meals", expanded=False):
        meals = load_meals()

        meal_names = ["(none)"] + sorted(meals.keys())
        selected_meal = st.selectbox("Load a meal", meal_names, key="selected_meal")

        # Initialise session meal state
        if "meal_items" not in st.session_state:
            st.session_state.meal_items = []

        if selected_meal != "(none)":
            if st.button("Load meal"):
                st.session_state.meal_items = meals[selected_meal]
                st.success(f"Loaded: {selected_meal}")

        # Editor for current meal items
        st.markdown("**Current meal items**")
        if not st.session_state.meal_items:
            st.info("No items yet – add some below.")
            editor_df = pd.DataFrame(columns=["food","amount","unit"])
            editor_df.loc[0] = [foods_list[0], 1.0, "servings"]
        else:
            editor_df = pd.DataFrame(st.session_state.meal_items)

        foods_list = sorted(df["food"].unique())
        edited = st.data_editor(
            editor_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "food": st.column_config.SelectboxColumn("food", options=foods_list),
                "amount": st.column_config.NumberColumn("amount", min_value=0.0, step=0.5),
                "unit": st.column_config.SelectboxColumn("unit", options=["grams","servings"]),
            },
            key="meal_editor"
        )

        # Keep session state in sync
        clean = edited.copy()
        clean = clean.dropna(subset=["food"])                 # must have food
        clean["unit"] = clean["unit"].fillna("grams")         # default unit
        clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce").fillna(0.0)
        clean = clean[clean["amount"] > 0]                    # ignore zero/blank amounts
        clean = clean[clean["unit"].isin(["grams", "servings"])]
        
        st.session_state.meal_items = clean.to_dict("records")

        # Show meal totals
        if st.session_state.meal_items:
            totals = meal_totals(df, st.session_state.meal_items)
            st.markdown(
                f"**Meal totals:** {totals['kcal']:.0f} kcal  \n"
                f"P {totals['p']:.1f}g • C {totals['c']:.1f}g • F {totals['f']:.1f}g • Fibre {totals['fibre']:.1f}g"
            )

        st.divider()

        # Save / delete
        save_name = st.text_input("Meal name to save", value="" if selected_meal=="(none)" else selected_meal)

        colA, colB = st.columns(2)
        with colA:
            if st.button("Save meal"):
                if not save_name.strip():
                    st.error("Enter a meal name.")
                elif not st.session_state.meal_items:
                    st.error("Add at least one item.")
                else:
                    meals[save_name.strip()] = st.session_state.meal_items
                    save_meals(meals)
                    st.success("Saved.")
                    st.rerun()

        with colB:
            if selected_meal != "(none)" and st.button("Delete meal"):
                meals.pop(selected_meal, None)
                save_meals(meals)
                st.session_state.meal_items = []
                st.success("Deleted.")
                st.rerun()


with st.sidebar:
    st.divider()
    with st.expander("➕ Add a new food to the database"):
        st.write("Add a food if it isn't in the list. Use per-100g values where possible.")

        new_food = st.text_input("Food name")
        new_category = st.selectbox("Category", ["veg","fruit","dairy","meat","fish","fat","carb","snack","supplement","other"])
        new_group = st.selectbox(
            "Swap group",
            ["veg_volume","fruit","protein_lean","protein_fatty","protein_lean_dairy","protein_veg",
             "carb_starchy","carb_grain","fat_oil","fat_spread","fat_wholefood","nuts_seeds",
             "snack_volume","snack_crispy","protein_supplement","other"]
        )

        typical_serving = st.number_input("Typical serving (g)", min_value=1.0, max_value=2000.0, value=150.0, step=10.0)

        kcal100 = st.number_input("kcal per 100g", min_value=0.0, value=100.0, step=1.0)
        p100 = st.number_input("protein per 100g (g)", min_value=0.0, value=10.0, step=0.1)
        c100 = st.number_input("carbs per 100g (g)", min_value=0.0, value=10.0, step=0.1)
        f100 = st.number_input("fat per 100g (g)", min_value=0.0, value=3.0, step=0.1)
        fib100 = st.number_input("fibre per 100g (g)", min_value=0.0, value=1.0, step=0.1)

        # --- Build row dict once ---
        row = None
        if new_food.strip() and kcal100 > 0:
            kcal_serv = kcal100 * typical_serving / 100.0
            row = {
                "food": new_food.strip(),
                "category": new_category,
                "swap_group": new_group,
                "typical_serving_g": float(typical_serving),
                "kcal_per_serving": float(kcal_serv),
                "kcal_per_100g": float(kcal100),
                "protein_per_100g": float(p100),
                "carbs_per_100g": float(c100),
                "fat_per_100g": float(f100),
                "fibre_per_100g": float(fib100),
                "unit_label": "",
            }

        # --- Save button ---
        if st.button("Save food"):
            if not new_food.strip():
                st.error("Please enter a food name.")
            elif kcal100 <= 0:
                st.error("kcal per 100g must be greater than 0.")
            else:
                uf = load_user_foods()
                idx = find_existing_food(uf, row["food"])

                if idx is None:
                    # Add new
                    try:
                        append_user_food(row, update_if_exists=False)
                        st.success("Added new food.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
                else:
                    # Exists -> check diff then prompt
                    old = uf.loc[idx].to_dict()
                    changed = diff_food_rows(old, row)

                    if not changed:
                        st.info("That food already exists and the values are identical. Nothing to update.")
                    else:
                        st.session_state.pending_food_update = row
                        st.session_state.pending_food_changed_fields = changed
                        st.session_state.pending_food_name = row["food"]

        # --- Prompt to update (keep INSIDE expander) ---
        if st.session_state.get("pending_food_update"):
            name = st.session_state.get("pending_food_name", "this food")
            fields = st.session_state.get("pending_food_changed_fields", [])

            st.warning(f"'{name}' already exists. Update it with the new values?")
            st.write("Fields that will change:", ", ".join(fields))

            colU, colC = st.columns(2)
            with colU:
                if st.button("✅ Update existing food", key="confirm_update_food"):
                    try:
                        append_user_food(st.session_state.pending_food_update, update_if_exists=True)
                        st.success("Food updated.")
                        st.session_state.pending_food_update = None
                        st.session_state.pending_food_changed_fields = None
                        st.session_state.pending_food_name = None
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with colC:
                if st.button("Cancel", key="cancel_update_food"):
                    st.session_state.pending_food_update = None
                    st.session_state.pending_food_changed_fields = None
                    st.session_state.pending_food_name = None
                    st.info("Update cancelled.")