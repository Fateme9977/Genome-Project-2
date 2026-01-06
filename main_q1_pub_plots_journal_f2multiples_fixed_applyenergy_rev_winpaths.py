import argparse
import logging
import sys
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import lsq_linear
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def apply_pub_style():
    """Matplotlib style tuned for journal-quality figures (Q1-ready).
    Notes:
    - Uses a serif family + STIX math for consistent scientific typography.
    - Uses subtle y-gridlines only (enabled per-axis via format_ax()).
    """
    plt.rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,

        # Lines/ticks
        'lines.linewidth': 2.0,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,

        # Layout / export
        'figure.dpi': 120,
        'savefig.dpi': 600,
        'figure.constrained_layout.use': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Grids off by default (enable selectively)
        'axes.grid': False,
    })


def format_ax(ax, grid: str | None = 'y', alpha: float = 0.18):
    """Consistent axis styling: clean spines + subtle grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    if grid:
        ax.grid(True, axis=grid, alpha=alpha, linewidth=0.7, zorder=0)

def save_figure(fig, basepath: Path, formats=('png','pdf'), dpi: int = 600):
    """Save a figure in multiple formats; tight bbox for publication export."""
    basepath = Path(basepath)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        out = basepath.with_suffix('.' + fmt)
        if fmt in ('png', 'jpg', 'jpeg', 'tif', 'tiff'):
            fig.savefig(out, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        else:
            fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# Configure logging
def setup_logging(outdir):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    logging.info("Logging initialized.")
    logging.info("Timestamps will be treated as naive/local; no timezone normalization applied.")

def parse_args():
    parser = argparse.ArgumentParser(description="Building Energy Balance Point Analysis")
    parser.add_argument('--demo', action='store_true', help="Run in demo mode using subset files")
    parser.add_argument('--outdir', type=str, default='output', help="Output directory")
    parser.add_argument('--bootstrap_iters', type=int, default=None,
                        help="Number of bootstrap iterations (default: 50 demo, 200 prod)")
    parser.add_argument('--bootstrap_block_size', type=int, default=None,
                        help="Moving-block bootstrap size in DAYS (default: 5 demo, 7 prod). Set 1 for IID bootstrap.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--max_buildings', type=int, default=None, help="Max buildings to process")
    parser.add_argument('--building_list', type=str, default=None,
                        help="Path to a text/CSV file containing building IDs (one per line or a column named building_id)")

    # Data paths (BDG2)
    parser.add_argument('--data_dir', type=str, default='data',
                        help="Base directory containing BDG2 data folder structure (default: data).")
    parser.add_argument('--electricity_file', type=str, default=None,
                        help="Path to electricity CSV. If not set, uses BDG2 default under data_dir.")
    parser.add_argument('--weather_file', type=str, default=None,
                        help="Path to weather CSV. If not set, uses BDG2 default under data_dir.")
    parser.add_argument('--metadata_file', type=str, default=None,
                        help="Path to metadata CSV. If not set, uses BDG2 default under data_dir.")
    parser.add_argument('--auto_download_data', action='store_true',
                        help="If missing, download BDG2 CSVs from GitHub into data_dir.")
    parser.add_argument('--github_branch', type=str, default='master',
                        help="Git branch for BDG2 raw downloads (default: master).")

    # Model selection stability / validation
    parser.add_argument('--model_selection_bootstrap_iters', type=int, default=None,
                        help="Bootstrap iterations for model-form stability (default: 20 demo, 50 prod).")
    parser.add_argument('--blocked_cv_folds', type=int, default=0,
                        help="If >0, run blocked time-series CV with this many folds (expensive).")

    # Plot controls (publication)
    parser.add_argument('--plot_trim_quantile', type=float, default=0.99,
                        help='Quantile for trimming outliers in F4/F5 (e.g., 0.99). Use 1.0 to disable trimming.')
    parser.add_argument('--plot_logy', action='store_true',
                        help='Also save log-y versions of F4/F5 (positive values only).')
    parser.add_argument('--plot_r2_min_usage', type=float, default=0.2,
                        help='Minimum R^2 to include in usage-stratified T_bp plot (F3).')
    parser.add_argument('--plot_usage_min_n', type=int, default=20,
                        help='Minimum number of buildings per usage category to plot (F3).')
    parser.add_argument('--plot_r2_min_examples', type=float, default=0.2,
                        help='Minimum R^2 preference when selecting example buildings (F1).')

    parser.add_argument('--fig_formats', type=str, default='png,pdf',
                        help='Comma-separated formats to save (default: png,pdf). Example: png,pdf,svg')
    parser.add_argument('--fig_dpi', type=int, default=600,
                        help='DPI for raster figure outputs (png/jpg/tif). Default: 600')

    args = parser.parse_args()

    if args.bootstrap_iters is None:
        args.bootstrap_iters = 50 if args.demo else 200

    if args.bootstrap_block_size is None:
        # weekly-ish blocks are a reasonable default for daily energy time series
        args.bootstrap_block_size = 5 if args.demo else 7


    if args.model_selection_bootstrap_iters is None:
        args.model_selection_bootstrap_iters = 20 if args.demo else 50

    return args

def load_building_list(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"building_list file not found: {path}")

    # If CSV: must have building_id column (or use first column)
    if p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p, dtype=str)
        if "building_id" in df.columns:
            ids = df["building_id"].dropna().astype(str).tolist()
        else:
            ids = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        # Treat as text: one id per line
        ids = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)

    # de-dup preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -------------------------
# Data path helpers (BDG2)
# -------------------------
BDG2_DEFAULT_RELPATHS = {
    'electricity_cleaned.csv': 'meters/cleaned/electricity_cleaned.csv',
    'metadata.csv': 'metadata/metadata.csv',
    'weather.csv': 'weather/weather.csv',
}

def _download_if_missing(local_path: Path, raw_url: str, auto_download: bool) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        return local_path

    if not auto_download:
        raise FileNotFoundError(
            f"Missing data file: {local_path}. "
            f"Either place the file there, set the --*_file arguments, "
            f"or rerun with --auto_download_data to fetch from GitHub."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {raw_url} -> {local_path} ...")
    urllib.request.urlretrieve(raw_url, local_path)
    return local_path


def _resolve_data_dir_maybe_repo_root(data_dir: Path) -> Path:
    """Allow --data_dir to point to either the BDG2 repo root or the BDG2 data/ folder."""
    data_dir = Path(data_dir)
    # If user passes repo root, normalize to <root>/data
    if (data_dir / "data").exists() and not (data_dir / "meters").exists() and not (data_dir / "weather").exists():
        return data_dir / "data"
    return data_dir


def _resolve_local_file(base: Path, preferred_relpath: str, alt_relpaths, glob_patterns) -> Path:
    """Resolve a local file path by trying:
    1) preferred relative path
    2) alternative relative paths
    3) recursive glob patterns under base

    Returns the first existing path, else returns base/preferred_relpath (non-existing).
    """
    base = Path(base)

    # 1) preferred relpath
    cand = base / preferred_relpath
    if cand.exists():
        return cand

    # 2) alt relpaths
    for rp in alt_relpaths:
        cand2 = base / rp
        if cand2.exists():
            logging.info(f"Using local data file found at: {cand2}")
            return cand2

    # 3) recursive search (pick the shortest path match for stability)
    matches = []
    for pat in glob_patterns:
        matches.extend(list(base.rglob(pat)))
    matches = [p for p in matches if p.is_file()]
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p)))
        logging.info(f"Using local data file found by search: {matches[0]}")
        return matches[0]

    return base / preferred_relpath


def resolve_bdg2_data_files(args):
    """Resolve electricity/weather/metadata paths; optionally auto-download from BDG2 GitHub.

    You may set --data_dir to either:
      - the BDG2 repo root (contains a 'data/' folder), OR
      - the BDG2 'data/' folder itself.

    Expected BDG2 data structure (inside the resolved data directory):
      - meters/cleaned/electricity_cleaned.csv (or .csv.gz)
      - weather/weather.csv (or .csv.gz)
      - metadata/metadata.csv (or .csv.gz)
    """
    base = _resolve_data_dir_maybe_repo_root(Path(args.data_dir))

    if args.demo:
        # Demo mode expects local small files (user-provided), but still allow explicit paths.
        elec = Path(args.electricity_file) if args.electricity_file else Path('electricity_head.csv')
        wea  = Path(args.weather_file) if args.weather_file else Path('weather_head.csv')
        meta = Path(args.metadata_file) if args.metadata_file else Path('metadata_head.csv')
        return elec, wea, meta

    # Production: prefer explicit file args; otherwise auto-resolve local files with fallbacks.
    if args.electricity_file:
        elec = Path(args.electricity_file)
    else:
        elec = _resolve_local_file(
            base=base,
            preferred_relpath=BDG2_DEFAULT_RELPATHS['electricity_cleaned.csv'],
            alt_relpaths=[
                "electricity_cleaned.csv",
                "meters/cleaned/electricity_cleaned.csv",
                "meters/cleaned/electricity_cleaned.csv.gz",
                "electricity_cleaned.csv.gz",
                "meters/cleaned/electricity_cleaned.parquet",
            ],
            glob_patterns=["electricity_cleaned.csv", "electricity_cleaned.csv.gz", "electricity_cleaned.parquet"],
        )

    if args.weather_file:
        wea = Path(args.weather_file)
    else:
        wea = _resolve_local_file(
            base=base,
            preferred_relpath=BDG2_DEFAULT_RELPATHS['weather.csv'],
            alt_relpaths=[
                "weather.csv",
                "weather/weather.csv",
                "weather/weather.csv.gz",
                "weather.csv.gz",
                "weather/weather.parquet",
            ],
            glob_patterns=["weather.csv", "weather.csv.gz", "weather.parquet"],
        )

    if args.metadata_file:
        meta = Path(args.metadata_file)
    else:
        meta = _resolve_local_file(
            base=base,
            preferred_relpath=BDG2_DEFAULT_RELPATHS['metadata.csv'],
            alt_relpaths=[
                "metadata.csv",
                "metadata/metadata.csv",
                "metadata/metadata.csv.gz",
                "metadata.csv.gz",
                "metadata/metadata.parquet",
            ],
            glob_patterns=["metadata.csv", "metadata.csv.gz", "metadata.parquet"],
        )

    # If requested, auto-download missing files from GitHub into the preferred locations.
    raw_base = f"https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/{args.github_branch}/data/"
    elec = _download_if_missing(elec, raw_base + BDG2_DEFAULT_RELPATHS['electricity_cleaned.csv'], args.auto_download_data)
    wea  = _download_if_missing(wea,  raw_base + BDG2_DEFAULT_RELPATHS['weather.csv'], args.auto_download_data)
    meta = _download_if_missing(meta, raw_base + BDG2_DEFAULT_RELPATHS['metadata.csv'], args.auto_download_data)

    return elec, wea, meta
    # Production defaults: BDG2 cleaned electricity + weather + metadata
    elec = Path(args.electricity_file) if args.electricity_file else (base / BDG2_DEFAULT_RELPATHS['electricity_cleaned.csv'])
    wea  = Path(args.weather_file) if args.weather_file else (base / BDG2_DEFAULT_RELPATHS['weather.csv'])
    meta = Path(args.metadata_file) if args.metadata_file else (base / BDG2_DEFAULT_RELPATHS['metadata.csv'])

    raw_base = f"https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/{args.github_branch}/data/"
    elec = _download_if_missing(elec, raw_base + BDG2_DEFAULT_RELPATHS['electricity_cleaned.csv'], args.auto_download_data)
    wea  = _download_if_missing(wea,  raw_base + BDG2_DEFAULT_RELPATHS['weather.csv'], args.auto_download_data)
    meta = _download_if_missing(meta, raw_base + BDG2_DEFAULT_RELPATHS['metadata.csv'], args.auto_download_data)

    return elec, wea, meta

def write_manuscript_snippets(outdir: Path) -> None:
    """Write small LaTeX/markdown snippets to reduce manuscript friction.

    Outputs:
      - nomenclature.md : compact variable/units table
      - equations.tex   : model equations (copy-paste)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nomen = [
        "| Symbol | Definition | Unit |",
        "|---|---|---|",
        "| $E_{b,h}$ | Electricity use of building $b$ at hour $h$ | kWh |",
        "| $E_{b,d}$ | Daily electricity use (sum over available hours) | kWh day$^{-1}$ |",
        "| $n_{E,d}$ | Available hourly electricity samples in day $d$ | h |",
        "| $T_{s,h}$ | Outdoor air temperature at site $s$ and hour $h$ | °C |",
        "| $T_{s,d}$ | Daily mean outdoor air temperature (mean over available hours) | °C |",
        "| $n_{T,d}$ | Available hourly temperature samples in day $d$ | h |",
        "| $T_{bp}$ | Change-point / balance-point temperature (model dependent) | °C |",
        r"| $\beta_0$ | Base load (intercept) | kWh day$^{-1}$ |",
        r"| $\beta_h$ | Heating sensitivity (slope below $T_{bp}$) | kWh day$^{-1}$ °C$^{-1}$ |",
        r"| $\beta_c$ | Cooling sensitivity (slope above $T_{bp}$) | kWh day$^{-1}$ °C$^{-1}$ |",
        "| $A$ | Floor area | m$^2$ |",
        r"| $\beta_{(\cdot),\mathrm{norm}}$ | Area-normalized coefficients ($\beta/A$) | kWh m$^{-2}$ day$^{-1}$ (and per-°C for slopes) |",
    ]
    (outdir / "nomenclature.md").write_text("\n".join(nomen) + "\n", encoding="utf-8")

    eqs = r"""% --- Balance-point / change-point model family (daily resolution) ---
% Daily aggregation (no scaling to 24 h; partial days retained if coverage threshold met)
E_{b,d} = \sum_{h \in d} E_{b,h}
\qquad
T_{s,d} = \frac{1}{n_{T,d}} \sum_{h \in d} T_{s,h}

% Model structures (non-negativity constraints: \beta_0, \beta_h, \beta_c \ge 0)
% M0: base-only
E_{b,d} = \beta_0 + \varepsilon_d

% Mh: heating-only hinge
E_{b,d} = \beta_0 + \beta_h\,\max(0,\,T_{bp}-T_{s,d}) + \varepsilon_d

% Mc: cooling-only hinge
E_{b,d} = \beta_0 + \beta_c\,\max(0,\,T_{s,d}-T_{bp}) + \varepsilon_d

% Mhc: two-sided hinge
E_{b,d} = \beta_0 + \beta_h\,\max(0,\,T_{bp}-T_{s,d}) + \beta_c\,\max(0,\,T_{s,d}-T_{bp}) + \varepsilon_d
"""
    (outdir / "equations.tex").write_text(eqs, encoding="utf-8")


def load_and_audit(electricity_file, weather_file, metadata_file):
    logging.info("--- Starting Data Load & Audit ---")
    
    # --- Load Metadata ---
    logging.info(f"Loading metadata from {metadata_file}...")
    meta_df = pd.read_csv(metadata_file, dtype={'building_id': str, 'site_id': str})
    logging.info(f"Metadata loaded: {meta_df.shape[0]} buildings.")
    
    # --- Load Weather ---
    logging.info(f"Loading weather from {weather_file}...")
    weather_df = pd.read_csv(weather_file, parse_dates=['timestamp'], dtype={'site_id': str})
    logging.info(f"Weather loaded: {weather_df.shape[0]} rows. Time range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    
    # Weather Duplicates
    weather_dup_mask = weather_df.duplicated(subset=['site_id', 'timestamp'], keep=False)
    weather_dup_count = weather_dup_mask.sum()
    if weather_dup_count > 0:
        logging.warning(f"Found {weather_dup_count} duplicate weather rows (site_id, timestamp). Collapsing by taking MEAN.")
        before_dedup = len(weather_df)
        weather_df = weather_df.groupby(['site_id', 'timestamp'], as_index=False).mean(numeric_only=True)
        logging.info(f"Weather rows collapsed: {before_dedup} -> {len(weather_df)} (Removed {before_dedup - len(weather_df)} rows).")
    else:
        logging.info("No duplicates found in weather data.")

    # --- Load Electricity (Wide) ---
    logging.info(f"Loading electricity from {electricity_file}...")
    elec_df = pd.read_csv(electricity_file, parse_dates=['timestamp'])
    
    logging.info(f"Electricity loaded: {elec_df.shape[0]} rows, {elec_df.shape[1]-1} buildings. Time range: {elec_df['timestamp'].min()} to {elec_df['timestamp'].max()}")

    # Electricity Duplicates (Timestamp)
    elec_dup_mask = elec_df.duplicated(subset=['timestamp'], keep=False)
    elec_dup_count = elec_dup_mask.sum()
    if elec_dup_count > 0:
        logging.warning(f"Found {elec_dup_count} duplicate electricity timestamps. Collapsing by taking MEAN row-wise.")
        before_dedup = len(elec_df)
        elec_df = elec_df.groupby('timestamp', as_index=False).mean(numeric_only=True)
        logging.info(f"Electricity rows collapsed: {before_dedup} -> {len(elec_df)} (Removed {before_dedup - len(elec_df)} rows).")
    else:
        logging.info("No duplicates found in electricity timestamps.")

    return elec_df, weather_df, meta_df

def preprocess_wide_and_join(elec_df, weather_df, meta_df, outdir):
    logging.info("--- Starting Preprocessing (Wide -> Daily) ---")
    
    # --- Preprocess Electricity (Wide) ---
    logging.info("Setting negative electricity values to NaN...")
    timestamp_col = elec_df['timestamp']
    data_cols = [c for c in elec_df.columns if c != 'timestamp']
    
    elec_vals = elec_df[data_cols].values
    neg_count = np.sum(elec_vals < 0)
    if neg_count > 0:
        logging.warning(f"Found {neg_count} negative electricity values. Setting to NaN.")
        elec_vals[elec_vals < 0] = np.nan
        elec_df[data_cols] = elec_vals
    
    elec_df = elec_df.set_index('timestamp')
    
    logging.info("Resampling electricity to Daily (Sum & Count)...")
    elec_daily_sum = elec_df.resample('D').sum(min_count=0)
    elec_daily_count = elec_df.resample('D').count()
    
    logging.info("Stacking electricity to long format...")
    E_day_long = elec_daily_sum.stack()
    n_hours_E_long = elec_daily_count.stack()
    
    elec_long = pd.DataFrame({'E_day': E_day_long, 'n_hours_E': n_hours_E_long}).reset_index()
    elec_long = elec_long.rename(columns={'level_1': 'building_id'})
    
    logging.info(f"Electricity Long Shape: {elec_long.shape}")

    # --- Preprocess Weather ---
    logging.info("Resampling weather to Daily (Mean & Count)...")

    # Avoid groupby.apply warnings and improve speed by using groupby + resample directly
    # Result has one row per (site_id, day)
    weather_daily = (
        weather_df.set_index('timestamp')
                 .groupby('site_id')['airTemperature']
                 .resample('D')
                 .agg(['mean', 'count'])
                 .reset_index()
                 .rename(columns={'mean': 'T_day', 'count': 'n_hours_T'})
    )

    logging.info(f"Weather Daily Shape: {weather_daily.shape}")

    # --- Join Data ---
    logging.info("Joining Electricity + Metadata + Weather...")
    
    elec_long['building_id'] = elec_long['building_id'].astype(str)
    
    total_buildings_elec = elec_long['building_id'].nunique()
    logging.info(f"Total buildings in electricity data: {total_buildings_elec}")
    
    merged_df = pd.merge(elec_long, meta_df, on='building_id', how='left')
    
    buildings_with_meta = merged_df[merged_df['site_id'].notna()]['building_id'].nunique()
    logging.info(f"Buildings surviving metadata join: {buildings_with_meta} (Lost: {total_buildings_elec - buildings_with_meta})")
    
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    weather_daily['timestamp'] = pd.to_datetime(weather_daily['timestamp'])
    
    final_df = pd.merge(merged_df, weather_daily, on=['site_id', 'timestamp'], how='left')
    
    # --- QC & Filtering ---
    logging.info("Applying coverage filters (n_hours_E >= 18 AND n_hours_T >= 18)...")
    
    initial_rows = len(final_df)
    valid_mask = (final_df['n_hours_E'] >= 18) & (final_df['n_hours_T'] >= 18)
    filtered_df = final_df[valid_mask].copy()
    
    dropped_rows = initial_rows - len(filtered_df)
    logging.info(f"Rows dropped due to coverage: {dropped_rows} ({dropped_rows/initial_rows*100:.1f}%)")
    
    buildings_passing_coverage = filtered_df['building_id'].nunique()
    logging.info(f"Buildings with at least one valid day: {buildings_passing_coverage}")
    
    try:
        parquet_path = outdir / 'daily_data.parquet'
        filtered_df.to_parquet(parquet_path)
        logging.info(f"Saved intermediate data to {parquet_path}")
    except ImportError:
        csv_path = outdir / 'daily_data.csv'
        filtered_df.to_csv(csv_path, index=False)
        logging.info(f"Parquet library missing, saved intermediate data to {csv_path}")
        
    qc_stats = {
        'total_buildings_elec': total_buildings_elec,
        'buildings_with_metadata': buildings_with_meta,
        'buildings_passing_coverage': buildings_passing_coverage,
        'total_rows_daily': initial_rows,
        'valid_rows_daily': len(filtered_df)
    }
        
    return filtered_df, qc_stats

def screen_buildings(daily_df, is_demo=False, max_buildings=None):
    logging.info("--- Starting Building Screening ---")
    
    screening_stats = []
    valid_buildings = []
    
    # Relaxed threshold for demo
    n_days_thresh = 5 if is_demo else 250
    if is_demo:
        logging.warning(f"DEMO MODE: Using relaxed screening threshold n_days >= {n_days_thresh}")
    
    grouped = daily_df.groupby('building_id')
    
    for b_id, df_b in grouped:
        n_days = len(df_b)
        T_min = df_b['T_day'].min()
        T_max = df_b['T_day'].max()
        T_range = T_max - T_min
        std_E = df_b['E_day'].std()
        median_E = df_b['E_day'].median()
        
        reasons = []
        if n_days < n_days_thresh:
            reasons.append(f"n_days({n_days}) < {n_days_thresh}")
        if T_range < 10:
            reasons.append(f"T_range({T_range:.1f}) < 10")
        if std_E <= 0 or np.isnan(std_E):
            reasons.append("std(E) <= 0")
        if median_E <= 0:
            reasons.append("median(E) <= 0")
            
        status = "kept" if not reasons else "removed"
        if status == "kept":
            valid_buildings.append(b_id)
            
        screening_stats.append({
            'building_id': b_id,
            'n_days': n_days,
            'T_range': T_range,
            'std_E': std_E,
            'median_E': median_E,
            'status': status,
            'removal_reason': "; ".join(reasons)
        })
        
        if max_buildings and len(valid_buildings) >= max_buildings:
            logging.info(f"Reached max_buildings limit ({max_buildings}). Stopping screening.")
            break
            
    screening_df = pd.DataFrame(screening_stats)
    logging.info(f"Screening complete. Kept {len(valid_buildings)} / {len(screening_stats)} buildings.")
    
    return valid_buildings, screening_df

def fit_single_model(X, E):
    res = lsq_linear(X, E, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    return res

def fit_model_candidates(df_bldg, tbp_step=0.5):
    """Select best change-point/balance-point model by AIC under non-negativity constraints.

    Key fixes vs. earlier version:
      - M0 (intercept-only) is fit ONCE (no TBP dependence); its T_bp is set to NaN.
      - One-/two-sided hinge models are grid-searched over TBP.
      - Degenerate feature cases (near-constant xh/xc) are skipped to avoid numerical artifacts.
    """
    T = df_bldg['T_day'].values
    E = df_bldg['E_day'].values
    n = len(E)

    # --- Fit M0 once (no TBP) ---
    X0 = np.ones((n, 1))
    res0 = fit_single_model(X0, E)
    sse0 = res0.cost * 2
    k0 = 1
    aic0 = n * np.log(sse0 / n) + 2 * k0 if sse0 > 0 else -np.inf

    y_pred0 = X0 @ res0.x
    sst = np.sum((E - np.mean(E))**2)
    r2_0 = 1 - sse0 / sst if sst > 0 else np.nan
    rmse0 = np.sqrt(sse0 / n)

    best_aic = aic0
    best_result = {
        'model_type': 'M0',
        'T_bp': np.nan,            # critical: TBP undefined for M0
        'beta_0': float(res0.x[0]),
        'beta_h': 0.0,
        'beta_c': 0.0,
        'RMSE': float(rmse0),
        'R2': float(r2_0) if np.isfinite(r2_0) else np.nan,
        'AIC': float(best_aic),
        'n_days': int(n),
    }

    # --- TBP grid for hinge models ---
    t_min = np.percentile(T, 5)
    t_max = np.percentile(T, 95)
    tbp_grid = np.arange(t_min, t_max + tbp_step, tbp_step)

    # small tolerance to treat features as degenerate
    eps_feat = 1e-10

    for tbp in tbp_grid:
        xh = np.maximum(0.0, tbp - T)
        xc = np.maximum(0.0, T - tbp)

        # Pre-check feature degeneracy (constant columns add no information and can trigger warnings)
        xh_ok = (np.nanmax(xh) - np.nanmin(xh)) > eps_feat
        xc_ok = (np.nanmax(xc) - np.nanmin(xc)) > eps_feat

        # Candidate models (skip if their hinge feature is degenerate)
        candidates = []
        if xh_ok:
            Xh = np.column_stack([np.ones(n), xh])
            candidates.append(('Mh', Xh, ['beta_0', 'beta_h']))
        if xc_ok:
            Xc = np.column_stack([np.ones(n), xc])
            candidates.append(('Mc', Xc, ['beta_0', 'beta_c']))
        if xh_ok and xc_ok:
            Xhc = np.column_stack([np.ones(n), xh, xc])
            candidates.append(('Mhc', Xhc, ['beta_0', 'beta_h', 'beta_c']))

        for model_name, X, param_names in candidates:
            res = fit_single_model(X, E)
            sse = res.cost * 2
            k = X.shape[1]
            aic = n * np.log(sse / n) + 2 * k if sse > 0 else -np.inf

            if aic < best_aic:
                best_aic = aic

                y_pred = X @ res.x
                r2 = 1 - sse / sst if sst > 0 else np.nan
                rmse = np.sqrt(sse / n)

                params = {p: 0.0 for p in ['beta_0', 'beta_h', 'beta_c']}
                for i, p_name in enumerate(param_names):
                    params[p_name] = float(res.x[i])

                best_result = {
                    'model_type': model_name,
                    'T_bp': float(tbp),
                    'beta_0': params['beta_0'],
                    'beta_h': params['beta_h'],
                    'beta_c': params['beta_c'],
                    'RMSE': float(rmse),
                    'R2': float(r2) if np.isfinite(r2) else np.nan,
                    'AIC': float(best_aic),
                    'n_days': int(n),
                }

    return best_result



def calendar_block_bootstrap_sample(df_bldg: pd.DataFrame, rng: np.random.Generator, block_size: int = 7,
                                   target_n_valid=None, max_blocks_factor: int = 10) -> pd.DataFrame:
    """Moving-block bootstrap sampled on the *calendar* (daily grid).

    Why this exists:
      - daily_df is filtered by coverage, so timestamps can have gaps.
      - A naive block bootstrap on the filtered rows treats non-consecutive calendar days as adjacent.
      - Here we sample blocks on a complete daily calendar index first, then drop invalid days.

    Returns:
      DataFrame with columns ['timestamp','T_day','E_day'] containing exactly target_n_valid valid rows
      (or fewer if the building has too little valid data).
    """
    df = df_bldg.sort_values('timestamp').copy()
    df = df[['timestamp', 'T_day', 'E_day']].copy()

    # Build full calendar index for this building
    t0 = pd.to_datetime(df['timestamp'].min()).normalize()
    t1 = pd.to_datetime(df['timestamp'].max()).normalize()
    if pd.isna(t0) or pd.isna(t1) or t1 < t0:
        return df.iloc[0:0].copy()

    full = pd.DataFrame({'timestamp': pd.date_range(t0, t1, freq='D')})
    full = full.merge(df, on='timestamp', how='left')

    valid_mask = full['E_day'].notna() & full['T_day'].notna()
    valid_idx = full.index[valid_mask].to_numpy()
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return df.iloc[0:0].copy()

    if target_n_valid is None:
        target_n_valid = n_valid
    target_n_valid = int(max(1, min(target_n_valid, n_valid)))

    block_size = int(block_size) if block_size is not None else 1
    block_size = max(1, min(block_size, len(full)))

    # IID bootstrap
    if block_size == 1:
        choice = rng.choice(valid_idx, size=target_n_valid, replace=True)
        return full.loc[choice, ['timestamp', 'T_day', 'E_day']].reset_index(drop=True)

    # Calendar moving-block bootstrap
    max_start = len(full) - block_size
    n_blocks_target = int(np.ceil(target_n_valid / max(1, block_size)))
    max_blocks = max(1, max_blocks_factor * n_blocks_target)

    collected = []
    n_collected = 0
    for _ in range(max_blocks):
        s = int(rng.integers(0, max_start + 1))
        block = full.iloc[s:s + block_size]
        block_v = block[block['E_day'].notna() & block['T_day'].notna()]
        if not block_v.empty:
            collected.append(block_v[['timestamp', 'T_day', 'E_day']])
            n_collected += len(block_v)
        if n_collected >= target_n_valid:
            break

    if n_collected < target_n_valid:
        # Fallback: IID on valid rows (rare; implies many missing days even after coverage screening)
        choice = rng.choice(valid_idx, size=target_n_valid, replace=True)
        return full.loc[choice, ['timestamp', 'T_day', 'E_day']].reset_index(drop=True)

    out = pd.concat(collected, ignore_index=True)
    out = out.iloc[:target_n_valid].reset_index(drop=True)
    return out

def bootstrap_uncertainty(df_bldg, selected_model_type, tbp_step, n_iters=50, rng=None, block_size: int = 7):
    """Bootstrap confidence intervals for TBP and coefficients.

    Journal-facing fix:
      - Uses a moving-block bootstrap by default (block_size=7 days), sampled on the calendar-day grid to preserve temporal dependence even with missing days.
      - For M0, TBP is undefined -> TBP CI is NaN, and only beta_0 is bootstrapped.

    Notes:
      - Set block_size=1 to recover IID (row-wise) bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Fast path: no bootstrap requested
    if n_iters is None or n_iters <= 0:
        return {
            'T_bp_CI_low': np.nan, 'T_bp_CI_high': np.nan,
            'beta_0_CI_low': np.nan, 'beta_0_CI_high': np.nan,
            'beta_h_CI_low': np.nan, 'beta_h_CI_high': np.nan,
            'beta_c_CI_low': np.nan, 'beta_c_CI_high': np.nan
        }
    # Sample on calendar (daily grid) to avoid treating gaps as adjacency
    df_b = df_bldg.sort_values('timestamp').reset_index(drop=True)
    n_valid = int(df_b[['T_day','E_day']].dropna().shape[0])
    if n_valid == 0:
        return {
            'T_bp_CI_low': np.nan, 'T_bp_CI_high': np.nan,
            'beta_0_CI_low': np.nan, 'beta_0_CI_high': np.nan,
            'beta_h_CI_low': np.nan, 'beta_h_CI_high': np.nan,
            'beta_c_CI_low': np.nan, 'beta_c_CI_high': np.nan
        }

    boot_rows = []

    for _ in range(n_iters):
        df_s = calendar_block_bootstrap_sample(df_b, rng=rng, block_size=block_size, target_n_valid=n_valid)

        T = df_s['T_day'].values
        E = df_s['E_day'].values
        nn = len(E)

        # M0: only intercept; TBP undefined
        if selected_model_type == 'M0':
            X = np.ones((nn, 1))
            res = fit_single_model(X, E)
            boot_rows.append({'T_bp': np.nan, 'beta_0': float(res.x[0]), 'beta_h': 0.0, 'beta_c': 0.0})
            continue

        # Grid search for T_bp on sample (fixed model structure)
        t_min = np.percentile(T, 5)
        t_max = np.percentile(T, 95)
        tbp_grid = np.arange(t_min, t_max + tbp_step, tbp_step)

        best_sse = np.inf
        best_params = None
        best_tbp = None

        eps_feat = 1e-10

        for tbp in tbp_grid:
            xh = np.maximum(0.0, tbp - T)
            xc = np.maximum(0.0, T - tbp)

            # construct X for selected model; skip degenerate features
            if selected_model_type == 'Mh':
                if (np.nanmax(xh) - np.nanmin(xh)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(nn), xh])
            elif selected_model_type == 'Mc':
                if (np.nanmax(xc) - np.nanmin(xc)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(nn), xc])
            elif selected_model_type == 'Mhc':
                if (np.nanmax(xh) - np.nanmin(xh)) <= eps_feat or (np.nanmax(xc) - np.nanmin(xc)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(nn), xh, xc])
            else:
                continue

            res = fit_single_model(X, E)
            sse = res.cost * 2
            if sse < best_sse:
                best_sse = sse
                best_params = res.x
                best_tbp = tbp

        if best_params is None:
            # fallback: treat as flat
            X = np.ones((nn, 1))
            res = fit_single_model(X, E)
            boot_rows.append({'T_bp': np.nan, 'beta_0': float(res.x[0]), 'beta_h': 0.0, 'beta_c': 0.0})
            continue

        params = {'T_bp': float(best_tbp), 'beta_0': 0.0, 'beta_h': 0.0, 'beta_c': 0.0}
        params['beta_0'] = float(best_params[0])
        if selected_model_type == 'Mh':
            params['beta_h'] = float(best_params[1])
        elif selected_model_type == 'Mc':
            params['beta_c'] = float(best_params[1])
        elif selected_model_type == 'Mhc':
            params['beta_h'] = float(best_params[1])
            params['beta_c'] = float(best_params[2])

        boot_rows.append(params)

    boot_df = pd.DataFrame(boot_rows)

    ci = {}
    for col in ['T_bp', 'beta_0', 'beta_h', 'beta_c']:
        if col in boot_df.columns and boot_df[col].notna().any():
            ci[f'{col}_CI_low'] = float(boot_df[col].quantile(0.025))
            ci[f'{col}_CI_high'] = float(boot_df[col].quantile(0.975))
        else:
            ci[f'{col}_CI_low'] = np.nan
            ci[f'{col}_CI_high'] = np.nan

    return ci

def classify_typology(row, mean_load):
    # Active Slope Rule:
    # 1. CI lower bound > 0
    # 2. Contribution C = beta * 10 > 0.05 * mean_load
    
    beta_h = row['beta_h']
    beta_c = row['beta_c']
    
    ci_h_low = row['beta_h_CI_low']
    ci_c_low = row['beta_c_CI_low']
    
    C_h = beta_h * 10
    C_c = beta_c * 10
    
    thresh = 0.05 * mean_load
    
    h_active = (ci_h_low > 0) and (C_h > thresh)
    c_active = (ci_c_low > 0) and (C_c > thresh)
    
    typology = "Base-dominated/Insensitive"
    
    if h_active and not c_active:
        typology = "Heating-dominated"
    elif c_active and not h_active:
        typology = "Cooling-dominated"
    elif h_active and c_active:
        # Dominance Check
        if C_h >= 1.25 * C_c:
            typology = "Heating-dominated"
        elif C_c >= 1.25 * C_h:
            typology = "Cooling-dominated"
        else:
            typology = "Mixed"
            
    return typology



def predict_from_params(model_type: str, T: np.ndarray, beta_0: float, beta_h: float, beta_c: float, T_bp: float = np.nan):
    """Vectorized prediction for the four model families."""
    T = np.asarray(T, dtype=float)
    if model_type == 'M0' or T_bp is None or np.isnan(T_bp):
        return np.full_like(T, float(beta_0), dtype=float)

    tbp = float(T_bp)
    if model_type == 'Mh':
        return float(beta_0) + float(beta_h) * np.maximum(0.0, tbp - T)
    if model_type == 'Mc':
        return float(beta_0) + float(beta_c) * np.maximum(0.0, T - tbp)
    if model_type == 'Mhc':
        return float(beta_0) + float(beta_h) * np.maximum(0.0, tbp - T) + float(beta_c) * np.maximum(0.0, T - tbp)

    raise ValueError(f"Unknown model_type: {model_type}")

def bootstrap_model_selection_stability(df_bldg: pd.DataFrame, tbp_step: float, n_iters: int,
                                       rng: np.random.Generator, block_size: int = 7,
                                       min_points: int = 60) -> dict:
    """Estimate stability of selected model form under bootstrap resampling.

    This is the journal-facing 'sanity check': if a building's data supports (say) Mhc,
    then a reasonable fraction of bootstrap resamples should re-select Mhc.
    """
    df_b = df_bldg.sort_values('timestamp').reset_index(drop=True)
    n_valid = int(df_b[['T_day','E_day']].dropna().shape[0])
    if n_valid < min_points or n_iters is None or n_iters <= 0:
        return {
            'msel_n_eff': 0,
            'msel_share_M0': np.nan,
            'msel_share_Mh': np.nan,
            'msel_share_Mc': np.nan,
            'msel_share_Mhc': np.nan,
            'msel_stability': np.nan,
            'msel_entropy': np.nan,
        }

    counts = {'M0': 0, 'Mh': 0, 'Mc': 0, 'Mhc': 0}
    n_eff = 0

    for _ in range(n_iters):
        df_s = calendar_block_bootstrap_sample(df_b, rng=rng, block_size=block_size, target_n_valid=n_valid)
        df_s = df_s.dropna(subset=['T_day', 'E_day'])
        if len(df_s) < min_points:
            continue

        r = fit_model_candidates(df_s, tbp_step)
        mt = r.get('model_type', None)
        if mt in counts:
            counts[mt] += 1
            n_eff += 1

    if n_eff == 0:
        return {
            'msel_n_eff': 0,
            'msel_share_M0': np.nan,
            'msel_share_Mh': np.nan,
            'msel_share_Mc': np.nan,
            'msel_share_Mhc': np.nan,
            'msel_stability': np.nan,
            'msel_entropy': np.nan,
        }

    shares = {k: v / n_eff for k, v in counts.items()}
    p = np.array([shares['M0'], shares['Mh'], shares['Mc'], shares['Mhc']], dtype=float)
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero))) if len(p_nonzero) else np.nan

    return {
        'msel_n_eff': int(n_eff),
        'msel_share_M0': float(shares['M0']),
        'msel_share_Mh': float(shares['Mh']),
        'msel_share_Mc': float(shares['Mc']),
        'msel_share_Mhc': float(shares['Mhc']),
        'msel_stability': float(np.max(p)),
        'msel_entropy': entropy,
    }

def blocked_time_series_cv(df_bldg: pd.DataFrame, tbp_step: float, n_folds: int,
                           min_points_per_fold: int = 60) -> dict:
    """Blocked (contiguous) time-series CV with model selection inside each training split.

    This is computationally expensive. Enable via --blocked_cv_folds.
    """
    if n_folds is None or n_folds <= 1:
        return {'cv_n_folds': 0, 'cv_rmse': np.nan, 'cv_r2': np.nan}

    df = df_bldg.sort_values('timestamp').dropna(subset=['T_day', 'E_day']).reset_index(drop=True)
    n = len(df)
    if n < n_folds * min_points_per_fold:
        return {'cv_n_folds': 0, 'cv_rmse': np.nan, 'cv_r2': np.nan}

    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)

    fold_rows = []
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx)
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        r = fit_model_candidates(df_train, tbp_step)
        yhat = predict_from_params(
            r['model_type'], df_test['T_day'].values,
            r.get('beta_0', np.nan), r.get('beta_h', 0.0), r.get('beta_c', 0.0), r.get('T_bp', np.nan)
        )
        y = df_test['E_day'].values.astype(float)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        sse = float(np.sum((y - yhat) ** 2))
        r2 = float(1 - sse / sst) if sst > 0 else np.nan

        fold_rows.append({'fold': k, 'rmse': rmse, 'r2': r2, 'n_test': len(df_test), 'model_type': r['model_type']})

    df_fold = pd.DataFrame(fold_rows)
    w = df_fold['n_test'].values.astype(float)
    w = w / np.sum(w)
    cv_rmse = float(np.sum(w * df_fold['rmse'].values))
    cv_r2 = float(np.sum(w * np.nan_to_num(df_fold['r2'].values, nan=0.0)))  # conservative weighting

    shares = df_fold['model_type'].value_counts(normalize=True).to_dict()
    return {
        'cv_n_folds': int(n_folds),
        'cv_rmse': cv_rmse,
        'cv_r2': cv_r2,
        'cv_share_M0': float(shares.get('M0', 0.0)),
        'cv_share_Mh': float(shares.get('Mh', 0.0)),
        'cv_share_Mc': float(shares.get('Mc', 0.0)),
        'cv_share_Mhc': float(shares.get('Mhc', 0.0)),
    }

def analyze_stability(df_bldg, selected_model_type, tbp_step, min_days=200):
    """Year-to-year stability of T_bp (2016 vs 2017) under fixed model structure.

    Fixes:
      - For M0, T_bp is undefined -> skip stability.
      - Avoid fitting degenerate hinge features (constant xh/xc) to prevent spurious TBP.
    """
    if selected_model_type == 'M0':
        return {}

    # Split by year
    df_2016 = df_bldg[df_bldg['timestamp'].dt.year == 2016]
    df_2017 = df_bldg[df_bldg['timestamp'].dt.year == 2017]

    if len(df_2016) < min_days or len(df_2017) < min_days:
        return {}

    eps_feat = 1e-10

    def fit_fixed_structure(df_sub):
        T = df_sub['T_day'].values
        E = df_sub['E_day'].values
        n = len(E)

        t_min = np.percentile(T, 5)
        t_max = np.percentile(T, 95)
        tbp_grid = np.arange(t_min, t_max + tbp_step, tbp_step)

        best_aic = np.inf
        best_params = None
        best_tbp = None

        for tbp in tbp_grid:
            xh = np.maximum(0.0, tbp - T)
            xc = np.maximum(0.0, T - tbp)

            if selected_model_type == 'Mh':
                if (np.nanmax(xh) - np.nanmin(xh)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(n), xh])
            elif selected_model_type == 'Mc':
                if (np.nanmax(xc) - np.nanmin(xc)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(n), xc])
            elif selected_model_type == 'Mhc':
                if (np.nanmax(xh) - np.nanmin(xh)) <= eps_feat or (np.nanmax(xc) - np.nanmin(xc)) <= eps_feat:
                    continue
                X = np.column_stack([np.ones(n), xh, xc])
            else:
                return None

            res = fit_single_model(X, E)
            sse = res.cost * 2
            k = X.shape[1]
            aic = n * np.log(sse / n) + 2 * k if sse > 0 else -np.inf

            if aic < best_aic:
                best_aic = aic
                best_tbp = float(tbp)
                best_params = res.x

        if best_params is None:
            return None
        return best_tbp, best_params

    res_2016 = fit_fixed_structure(df_2016)
    res_2017 = fit_fixed_structure(df_2017)

    if res_2016 and res_2017:
        tbp_16, _ = res_2016
        tbp_17, _ = res_2017
        return {
            'T_bp_2016': float(tbp_16),
            'T_bp_2017': float(tbp_17),
            'delta_T_bp': float(tbp_17 - tbp_16),
        }

    return {}

def plot_figures(results_df, daily_df, outdir, plot_trim_quantile=0.99, plot_logy=False, plot_r2_min_usage=0.2, plot_usage_min_n=20, plot_r2_min_examples=0.2, bootstrap_excursion_degC=10.0, contrib_frac=0.05, fig_formats=('png','pdf'), fig_dpi=600):
    """Generate journal-grade figures (PNG+vector formats by default).

    Notes:
      - For one-sided models (Mh/Mc), T_bp is best interpreted as onset/change-point temperature.
      - For Mhc, T_bp aligns more closely with a classical balance point temperature.
    """
    logging.info("Generating plots (publication style)...")
    apply_pub_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def active_subset(df, r2_min=None):
        out = df.copy()
        if 'model_type' in out.columns:
            out = out[out['model_type'] != 'M0']
        if 'typology' in out.columns:
            out = out[out['typology'] != "Base-dominated/Insensitive"]
        if r2_min is not None and 'R2' in out.columns:
            out = out[pd.to_numeric(out['R2'], errors='coerce') >= r2_min]
        return out


    def compute_slope_active_flags(df):
        """Return boolean flags for heating/cooling 'active' slopes used in typology rules.

        Active if:
          1) lower CI bound > 0, AND
          2) contribution = beta * bootstrap_excursion_degC exceeds contrib_frac * mean_load.
        Falls back to simple beta>0 if CI/mean_load are missing.
        """
        out = df.copy()
        h_active = pd.Series(False, index=out.index)
        c_active = pd.Series(False, index=out.index)

        have_h = {'beta_h', 'beta_h_CI_low'}.issubset(out.columns)
        have_c = {'beta_c', 'beta_c_CI_low'}.issubset(out.columns)
        have_load = 'mean_load' in out.columns

        if have_h and have_load:
            contrib_h = out['beta_h'].astype(float) * float(bootstrap_excursion_degC)
            h_active = (out['beta_h_CI_low'].astype(float) > 0) & (contrib_h > float(contrib_frac) * out['mean_load'].astype(float))
        elif 'beta_h' in out.columns:
            h_active = out['beta_h'].astype(float) > 0

        if have_c and have_load:
            contrib_c = out['beta_c'].astype(float) * float(bootstrap_excursion_degC)
            c_active = (out['beta_c_CI_low'].astype(float) > 0) & (contrib_c > float(contrib_frac) * out['mean_load'].astype(float))
        elif 'beta_c' in out.columns:
            c_active = out['beta_c'].astype(float) > 0

        return h_active, c_active


    results = results_df.copy()
    # Coerce numeric fields used in plots
    for c in ['T_bp', 'R2', 'beta_c_norm', 'beta_h_norm', 'T_bp_2016', 'T_bp_2017']:
        if c in results.columns:
            results[c] = pd.to_numeric(results[c], errors='coerce')

    active_df = active_subset(results)

    # -------------------------
    # F2: Model-type histogram (primary: stacked small multiples with density + KDE; appendix: overlay)
    tbp_by_type = {}
    for mt in ['Mc', 'Mh', 'Mhc']:
        vals = results_df.loc[results_df['model_type'] == mt, 'T_bp'].dropna().astype(float).values
        tbp_by_type[mt] = vals

    # Fixed x-range for fair comparison across panels (journal-ready)
    x_min, x_max = 0.0, 32.0
    bin_width = 2.0
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    x_grid = np.linspace(x_min, x_max, 400)

    colors = {'Mc': '#4C72B0', 'Mh': '#55A868', 'Mhc': '#C44E52'}  # muted, journal-friendly
    order = ['Mc', 'Mh', 'Mhc']

    # --- Main figure: stacked small multiples (density) ---
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(7.2, 8.2))
    fig.suptitle(r'$T_{bp}$ distribution by model type', y=0.98)

    for ax, mt in zip(axes, order):
        vals = tbp_by_type.get(mt, np.array([], dtype=float))
        vals_in = vals[(vals >= x_min) & (vals <= x_max)]

        # Histogram as probability density (not count) to handle unequal n
        ax.hist(vals_in, bins=bins, density=True, alpha=0.9, color=colors[mt], edgecolor='white', linewidth=0.5)

        # KDE overlay (optional but recommended): skip if too few points or near-zero variance
        if vals_in.size >= 8 and np.nanstd(vals_in) > 1e-6:
            try:
                kde = gaussian_kde(vals_in)
                ax.plot(x_grid, kde(x_grid), color=colors[mt], linewidth=2.0)
            except Exception:
                pass

        # Summary stats + median line
        n = int(vals.size)
        mu = float(np.nanmean(vals)) if n else float('nan')
        sigma = float(np.nanstd(vals, ddof=1)) if n > 1 else float('nan')
        med = float(np.nanmedian(vals)) if n else float('nan')

        if np.isfinite(med):
            ax.axvline(med, color=colors[mt], linestyle='--', linewidth=2.0)

        # Panel annotation (model, n, mean, std) — keep it compact
        ann = f"{mt} (n={n})\n$\\mu$={mu:.2f}°C, $\\sigma$={sigma:.2f}°C"
        ax.text(0.98, 0.92, ann, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.9))

        ax.set_ylabel('Density')
        format_ax(ax)

    axes[-1].set_xlabel(r'$T_{bp}$ (°C)')
    axes[-1].set_xlim(x_min, x_max)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, outdir / 'F2_Tbp_histogram', formats=fig_formats, dpi=fig_dpi)

    # --- Appendix figure: overlay density (kept for reference/comparison) ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mt in order:
        vals = tbp_by_type.get(mt, np.array([], dtype=float))
        vals_in = vals[(vals >= x_min) & (vals <= x_max)]
        ax.hist(vals_in, bins=bins, density=True, alpha=0.35, color=colors[mt], label=f"{mt} (n={len(vals)})")
        if vals_in.size >= 8 and np.nanstd(vals_in) > 1e-6:
            try:
                kde = gaussian_kde(vals_in)
                ax.plot(x_grid, kde(x_grid), color=colors[mt], linewidth=2.0)
            except Exception:
                pass
        if len(vals):
            ax.axvline(np.nanmedian(vals), color=colors[mt], linestyle='--', linewidth=2.0)

    ax.set_title(r'$T_{bp}$ distribution by model type (overlay; density)')
    ax.set_xlabel(r'$T_{bp}$ (°C)')
    ax.set_ylabel('Density')
    ax.set_xlim(x_min, x_max)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    format_ax(ax)
    save_figure(fig, outdir / 'F2_Tbp_histogram_overlay', formats=fig_formats, dpi=fig_dpi)

# F3: Violin + box overlay of T_bp by primaryspaceusage
    # -------------------------
    if 'primaryspaceusage' in results.columns:
        plot_df = results.copy()
        # Quality filter for usage-stratified TBP: avoid weak fits (journal requirement)
        plot_df = active_subset(plot_df, r2_min=plot_r2_min_usage)

        plot_df['primaryspaceusage'] = plot_df['primaryspaceusage'].fillna('Unknown')
        plot_df = plot_df[plot_df['primaryspaceusage'] != 'Unknown'].dropna(subset=['T_bp'])

        if not plot_df.empty:
            # Order by median
            order = plot_df.groupby('primaryspaceusage')['T_bp'].median().sort_values().index.tolist()

            data = []
            labels = []
            for cat in order:
                vals = plot_df.loc[plot_df['primaryspaceusage'] == cat, 'T_bp'].dropna().values
                if len(vals) < int(plot_usage_min_n):
                    continue
                data.append(vals)
                labels.append(f'{cat}\n(n={len(vals)})')

            if data:
                fig, ax = plt.subplots(figsize=(12.5, 5.0))
                parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
                for b in parts['bodies']:
                    b.set_facecolor('0.85')
                    b.set_edgecolor('0.35')
                    b.set_alpha(1.0)
                if 'cmedians' in parts:
                    parts['cmedians'].set_color('0.15')
                    parts['cmedians'].set_linewidth(1.5)

                ax.boxplot(
                    data,
                    widths=0.22,
                    showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor='0.2', linewidth=1.0),
                    whiskerprops=dict(color='0.2', linewidth=1.0),
                    capprops=dict(color='0.2', linewidth=1.0),
                    medianprops=dict(color='0.15', linewidth=1.5),
                )

                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(r'$T_{bp}$ (°C)')
                ax.set_title(r'$T_{bp}$ by primary space usage')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=min(12, len(labels))))
                save_figure(fig, outdir / 'F3_Tbp_boxplot', formats=fig_formats, dpi=fig_dpi)
        else:
            logging.warning("F3: no usable primaryspaceusage data to plot.")

    # Helper for trimmed scatter
    def scatter_trimmed(x, y, q, xlabel, ylabel, title, basepath, logy=False):
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        if df.empty:
            return
        # raw
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.scatter(df['x'], df['y'], s=14, alpha=0.55, edgecolors='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title + ' (raw)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        save_figure(fig, basepath.with_name(basepath.name + '_raw'), formats=fig_formats, dpi=fig_dpi)

        # trimmed
        df_t = df
        note = ''
        if q is not None and q < 1.0:
            yq = df['y'].quantile(q)
            df_t = df[df['y'] <= yq]
            note = f' (trim q={q}, n={len(df_t)}/{len(df)})'
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.scatter(df_t['x'], df_t['y'], s=14, alpha=0.55, edgecolors='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title + note)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        save_figure(fig, basepath, formats=fig_formats, dpi=fig_dpi)

        # log-y diagnostic
        if logy:
            df_log = df[df['y'] > 0]
            if not df_log.empty:
                fig, ax = plt.subplots(figsize=(4.8, 3.4))
                ax.scatter(df_log['x'], df_log['y'], s=14, alpha=0.55, edgecolors='none')
                ax.set_yscale('log')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel + ' [log]')
                ax.set_title(title + ' (log-y, raw positive)')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                save_figure(fig, basepath.with_name(basepath.name + '_logy'), formats=fig_formats, dpi=fig_dpi)

    # -------------------------
    # F4: Cooling sensitivity vs T_bp (normalized)
    # -------------------------
    if 'beta_c_norm' in results.columns and 'T_bp' in results.columns:
        h_active, c_active = compute_slope_active_flags(results)
        df4 = results.loc[c_active, ['T_bp', 'beta_c_norm']].dropna().copy()
        scatter_trimmed(
            df4['T_bp'].values,
            df4['beta_c_norm'].values,
            plot_trim_quantile,
            xlabel=r'$T_{bp}$ (°C)',
            ylabel=r'$\beta_{c,\mathrm{norm}}$ (kWh m$^{-2}$ day$^{-1}$ °C$^{-1}$)',
            title='Cooling sensitivity vs change-point temperature',
            basepath=outdir / 'F4_betac_vs_Tbp',
            logy=plot_logy
        )

    # -------------------------
    # F5: Heating sensitivity vs T_bp (normalized)
    # -------------------------
    if 'beta_h_norm' in results.columns and 'T_bp' in results.columns:
        h_active, c_active = compute_slope_active_flags(results)
        df5 = results.loc[h_active, ['T_bp', 'beta_h_norm']].dropna().copy()
        scatter_trimmed(
            df5['T_bp'].values,
            df5['beta_h_norm'].values,
            plot_trim_quantile,
            xlabel=r'$T_{bp}$ (°C)',
            ylabel=r'$\beta_{h,\mathrm{norm}}$ (kWh m$^{-2}$ day$^{-1}$ °C$^{-1}$)',
            title='Heating sensitivity vs change-point temperature',
            basepath=outdir / 'F5_betah_vs_Tbp',
            logy=plot_logy
        )

    # -------------------------
    # F6: Stability (2016 vs 2017) for Active buildings with fit quality
    # -------------------------
    if {'T_bp_2016', 'T_bp_2017'}.issubset(results.columns):
        stab_df = active_subset(results, r2_min=0.2).dropna(subset=['T_bp_2016', 'T_bp_2017']).copy()
        if not stab_df.empty:
            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            ax.scatter(stab_df['T_bp_2016'], stab_df['T_bp_2017'], s=16, alpha=0.6, edgecolors='none')

            lo = float(np.nanmin([stab_df['T_bp_2016'].min(), stab_df['T_bp_2017'].min()]))
            hi = float(np.nanmax([stab_df['T_bp_2016'].max(), stab_df['T_bp_2017'].max()]))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2, color='0.2')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal', adjustable='box')

            # Summary metrics annotation
            corr = stab_df[['T_bp_2016', 'T_bp_2017']].corr().iloc[0, 1]
            abs_delta = (stab_df['T_bp_2017'] - stab_df['T_bp_2016']).abs()
            txt = (f'n={len(stab_df)}\n'
                   f'corr={corr:.2f}\n'
                   f'median |Δ|={abs_delta.median():.2f}°C\n'
                   f'mean |Δ|={abs_delta.mean():.2f}°C')
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7'))

            ax.set_xlabel(r'$T_{bp,2016}$ (°C)')
            ax.set_ylabel(r'$T_{bp,2017}$ (°C)')
            ax.set_title(r'Stability of $T_{bp}$ (Active, $R^2 \geq 0.2$)')
            save_figure(fig, outdir / 'F6_stability', formats=fig_formats, dpi=fig_dpi)
        else:
            logging.warning("F6: No stability points after filtering (need both years + active + R² threshold).")


    # -------------------------
    # F1: Example buildings (1 cooling, 1 heating, 1 base), chosen by best R2 within class
    # -------------------------
    examples = []
    if 'typology' in results.columns and 'building_id' in results.columns:

        def pick_example(typ, r2_min):
            sub = results[results['typology'] == typ].copy()
            if sub.empty:
                return None
            sub['R2'] = pd.to_numeric(sub.get('R2', np.nan), errors='coerce')
            good = sub[sub['R2'] >= float(r2_min)]
            if not good.empty:
                return good.sort_values('R2', ascending=False)['building_id'].iloc[0]
            # fallback: highest R2 even if weak (still representative)
            return sub.sort_values('R2', ascending=False)['building_id'].iloc[0]

        for t in ["Cooling-dominated", "Heating-dominated", "Base-dominated/Insensitive"]:
            b = pick_example(t, plot_r2_min_examples)
            if b is not None:
                examples.append(b)

    # ensure 3 unique examples
    examples = [e for i, e in enumerate(examples) if e not in examples[:i]]
    if len(examples) < 3 and 'building_id' in results.columns:
        for b in results.sort_values('R2', ascending=False, na_position='last')['building_id'].tolist():
            if b not in examples:
                examples.append(b)
            if len(examples) >= 3:
                break

    for b_id in examples[:3]:
        df_b = daily_df[daily_df['building_id'] == b_id].dropna(subset=['T_day', 'E_day'])
        if df_b.empty:
            continue

        row = results[results['building_id'] == b_id].iloc[0]

        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        ax.scatter(df_b['T_day'], df_b['E_day'], s=18, alpha=0.55, edgecolors='none', label='Daily data')

        # fitted curve
        T_grid = np.linspace(df_b['T_day'].min(), df_b['T_day'].max(), 200)
        mt = str(row.get('model_type', ''))
        tbp = row.get('T_bp', np.nan)
        b0 = float(row.get('beta_0', 0.0))
        bh = float(row.get('beta_h', 0.0))
        bc = float(row.get('beta_c', 0.0))

        if mt == 'M0' or (not np.isfinite(tbp)):
            y_pred = np.full_like(T_grid, b0, dtype=float)
            tbp_txt = '—'
        else:
            y_pred = b0 + bh * np.maximum(0.0, float(tbp) - T_grid) + bc * np.maximum(0.0, T_grid - float(tbp))
            tbp_txt = f'{float(tbp):.2f}'

        ax.plot(T_grid, y_pred, linewidth=2.2, color='0.15', label='Fitted model')
        if np.isfinite(tbp):
            ax.axvline(float(tbp), linestyle='--', linewidth=1.2, color='0.25')

        typ = str(row.get('typology', ''))
        r2 = row.get('R2', np.nan)
        r2_txt = f'{float(r2):.2f}' if np.isfinite(r2) else '—'

        if mt == 'M0':
            ann = (f'{mt} | $T_{{bp}}$={tbp_txt}°C\n'
                   f'$\\beta_0$={b0:.2f}, $\\beta_h$=0.00, $\\beta_c$=0.00\n'
                   f'$R^2$={r2_txt}')
        else:
            ann = (f'{mt} | $T_{{bp}}$={tbp_txt}°C\n'
                   f'$\\beta_0$={b0:.2f}, $\\beta_h$={bh:.2f}, $\\beta_c$={bc:.2f}\n'
                   f'$R^2$={r2_txt}')

        ax.text(0.02, 0.98, ann, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7'))

        ax.set_xlabel('Daily mean air temperature (°C)')
        ax.set_ylabel('Daily electricity use (kWh/day)')
        ax.set_title(f'{b_id} ({typ})')
        ax.legend(frameon=True, framealpha=0.9, edgecolor='0.7', loc='lower right')
        save_figure(fig, outdir / f'F1_example_{b_id}', formats=fig_formats, dpi=fig_dpi)


    logging.info("Plots generated (publication style).")

def write_paper_summary_tables(results_df: pd.DataFrame, outdir: Path) -> None:
    """Generate compact, paper-ready summary tables (CSV) with minimal assumptions.

    Outputs:
      - results_table_typology.csv: typology shares + medians of T_bp and normalized slopes
      - results_table_stability.csv: stability metrics for Active buildings with R² ≥ 0.2
      - paper_notes.txt: short interpretability notes for T_bp under different model structures
    """
    df = results_df.copy()

    # Coerce numeric fields if present
    for c in ['T_bp', 'beta_c_norm', 'beta_h_norm', 'beta_0_norm', 'R2', 'T_bp_2016', 'T_bp_2017']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # -------------------------
    # Table 1: Typology shares + medians
    # -------------------------
    if 'typology' in df.columns:
        n_total = df['building_id'].nunique() if 'building_id' in df.columns else len(df)

        g = (
            df.groupby('typology', dropna=False)
              .agg(
                  n=('building_id', 'count') if 'building_id' in df.columns else ('T_bp', 'size'),
                  T_bp_median=('T_bp', 'median'),
                  beta_c_norm_median=('beta_c_norm', 'median') if 'beta_c_norm' in df.columns else ('T_bp', 'median'),
                  beta_h_norm_median=('beta_h_norm', 'median') if 'beta_h_norm' in df.columns else ('T_bp', 'median'),
              )
              .reset_index()
        )
        g['pct'] = 100.0 * g['n'] / float(n_total) if n_total else np.nan
        g = g.sort_values(['pct', 'n'], ascending=False)

        g.to_csv(outdir / 'results_table_typology.csv', index=False)
        logging.info(f"Saved paper summary table: {outdir / 'results_table_typology.csv'}")
        # -------------------------
        # Sensitivity (optional, paper-strengthening):
        # Repeat typology table on subset with R² ≥ 0.1
        # -------------------------
        if 'R2' in df.columns:
            df_r2 = df.copy()
            df_r2 = df_r2[pd.to_numeric(df_r2['R2'], errors='coerce') >= 0.1]
            n_total_r2 = df_r2['building_id'].nunique() if 'building_id' in df_r2.columns else len(df_r2)

            if n_total_r2 and n_total_r2 > 0:
                g_r2 = (
                    df_r2.groupby('typology', dropna=False)
                         .agg(
                             n=('building_id', 'count') if 'building_id' in df_r2.columns else ('T_bp', 'size'),
                             T_bp_median=('T_bp', 'median'),
                             beta_c_norm_median=('beta_c_norm', 'median') if 'beta_c_norm' in df_r2.columns else ('T_bp', 'median'),
                             beta_h_norm_median=('beta_h_norm', 'median') if 'beta_h_norm' in df_r2.columns else ('T_bp', 'median'),
                         )
                         .reset_index()
                )
                g_r2['pct'] = 100.0 * g_r2['n'] / float(n_total_r2)
                g_r2 = g_r2.sort_values(['pct', 'n'], ascending=False)

                out_path_r2 = outdir / 'results_table_typology_r2_ge_0_1.csv'
                g_r2.to_csv(out_path_r2, index=False)
                logging.info(f"Saved sensitivity typology table (R² ≥ 0.1): {out_path_r2}")

                # Repeat typology table on subset with R² ≥ 0.2 (optional, Q1-friendly)
                df_r2_02 = df.copy()
                df_r2_02 = df_r2_02[pd.to_numeric(df_r2_02['R2'], errors='coerce') >= 0.2]
                n_total_r2_02 = df_r2_02['building_id'].nunique() if 'building_id' in df_r2_02.columns else len(df_r2_02)

                if n_total_r2_02 and n_total_r2_02 > 0:
                    g_r2_02 = (
                        df_r2_02.groupby('typology', dropna=False)
                                .agg(
                                    n=('building_id', 'count') if 'building_id' in df_r2_02.columns else ('T_bp', 'size'),
                                    T_bp_median=('T_bp', 'median'),
                                    beta_c_norm_median=('beta_c_norm', 'median') if 'beta_c_norm' in df_r2_02.columns else ('T_bp', 'median'),
                                    beta_h_norm_median=('beta_h_norm', 'median') if 'beta_h_norm' in df_r2_02.columns else ('T_bp', 'median'),
                                )
                                .reset_index()
                    )
                    g_r2_02['pct'] = 100.0 * g_r2_02['n'] / float(n_total_r2_02)
                    g_r2_02 = g_r2_02.sort_values(['pct', 'n'], ascending=False)

                    out_path_r2_02 = outdir / 'results_table_typology_r2_ge_0_2.csv'
                    g_r2_02.to_csv(out_path_r2_02, index=False)
                    logging.info(f"Saved sensitivity typology table (R² ≥ 0.2): {out_path_r2_02}")
                else:
                    logging.warning("R² ≥ 0.2 sensitivity typology table: no rows after filtering; skipping.")
            else:
                logging.warning("R² ≥ 0.1 sensitivity typology table: no rows after filtering; skipping.")
        else:
            logging.warning("R2 column missing; skipping R² ≥ 0.1 sensitivity typology table.")
    else:
        logging.warning("Typology column missing; skipping results_table_typology.csv")

    # -------------------------
    # Table 2: Stability metrics (Active + R² ≥ 0.2)
    # -------------------------
    stab = df.copy()
    if 'model_type' in stab.columns:
        stab = stab[stab['model_type'] != 'M0']
    if 'typology' in stab.columns:
        stab = stab[stab['typology'] != 'Base-dominated/Insensitive']
    if 'R2' in stab.columns:
        stab = stab[stab['R2'] >= 0.2]

    req_cols = {'T_bp_2016', 'T_bp_2017'}
    if req_cols.issubset(stab.columns):
        stab = stab.dropna(subset=['T_bp_2016', 'T_bp_2017'])
        if not stab.empty:
            stab = stab.copy()
            stab['abs_delta_T_bp'] = (stab['T_bp_2017'] - stab['T_bp_2016']).abs()

            corr = stab[['T_bp_2016', 'T_bp_2017']].corr().iloc[0, 1]
            metrics = pd.DataFrame([{
                'n_buildings': int(len(stab)),
                'pearson_corr_Tbp_2016_2017': float(corr),
                'median_abs_delta_Tbp_C': float(stab['abs_delta_T_bp'].median()),
                'mean_abs_delta_Tbp_C': float(stab['abs_delta_T_bp'].mean()),
                'p90_abs_delta_Tbp_C': float(stab['abs_delta_T_bp'].quantile(0.90)),
            }])
            metrics.to_csv(outdir / 'results_table_stability.csv', index=False)
            logging.info(f"Saved paper stability table: {outdir / 'results_table_stability.csv'}")
        else:
            logging.warning("No rows available for stability metrics after filtering; skipping results_table_stability.csv")
    else:
        logging.warning("Stability columns T_bp_2016/T_bp_2017 missing; skipping results_table_stability.csv")

    # -------------------------
    # Notes for manuscript wording (lightweight)
    # -------------------------
    notes = []
    notes.append("Interpretability note for T_bp:")
    notes.append("- For one-sided models (Mh or Mc), T_bp is best interpreted as an onset/change-point temperature (start of heating/cooling response), not a classical balance point.")
    notes.append("- For the two-sided model (Mhc), T_bp more closely corresponds to the conventional balance point temperature separating heating and cooling regimes.")
    notes.append("")
    notes.append("Outlier handling for area-normalized slopes:")
    notes.append("- Area-normalized slopes (beta_h_norm, beta_c_norm) can be inflated for small-area buildings. We therefore report trimmed scatterplots (q=0.99) alongside raw plots.")
    (outdir / 'paper_notes.txt').write_text("\n".join(notes), encoding='utf-8')
    logging.info(f"Saved paper notes: {outdir / 'paper_notes.txt'}")

def main():
    args = parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)
    logging.info(f"Starting analysis with arguments: {args}")
    
    rng = np.random.default_rng(args.seed)
    
    # Write manuscript helper snippets (equations / nomenclature) to reduce journal friction
    try:
        write_manuscript_snippets(outdir)
    except Exception as e:
        logging.warning(f"Could not write manuscript snippets: {e}")

    if args.demo:
        logging.info("Running in DEMO mode.")
        tbp_step = 1.0
    else:
        logging.info("Running in PRODUCTION mode.")
        tbp_step = 0.5

    try:
        electricity_file, weather_file, metadata_file = resolve_bdg2_data_files(args)
    except Exception as e:
        logging.error(f"Failed to resolve data files: {e}")
        sys.exit(1)

    for f in [electricity_file, weather_file, metadata_file]:
        if not Path(f).exists():
            logging.error(f"File not found: {f}")
            sys.exit(1)

    logging.info(f"Using electricity file: {electricity_file}")
    logging.info(f"Using weather file: {weather_file}")
    logging.info(f"Using metadata file: {metadata_file}")
            
    # Load & Preprocess
    elec_df, weather_df, meta_df = load_and_audit(electricity_file, weather_file, metadata_file)
    daily_df, qc_stats = preprocess_wide_and_join(elec_df, weather_df, meta_df, outdir)
    # Optional: filter daily_df by an external building list BEFORE screening
    if args.building_list is not None:
        keep_ids = load_building_list(args.building_list)
        before = daily_df['building_id'].nunique()
        daily_df = daily_df[daily_df['building_id'].isin(keep_ids)].copy()
        after = daily_df['building_id'].nunique()
        logging.info(f"Applied --building_list filter: {after}/{before} buildings remain.")
    else:
        logging.info("No --building_list filter applied.")

    # Screening
    valid_bldgs, screening_df = screen_buildings(daily_df, is_demo=args.demo, max_buildings=args.max_buildings)
    qc_stats['buildings_passing_screening'] = len(valid_bldgs)
    
    pd.DataFrame([qc_stats]).to_csv(outdir / 'data_join_qc_report.csv', index=False)
    screening_df.to_csv(outdir / 'screening_report.csv', index=False)
    
    if len(valid_bldgs) == 0:
        logging.warning("No valid buildings found. Exiting.")
        sys.exit(0)

    # Fitting
    logging.info(f"Fitting models for {len(valid_bldgs)} buildings...")
    results = []
    
    for i, b_id in enumerate(valid_bldgs):
        df_b = daily_df[daily_df['building_id'] == b_id].copy()
        try:
            # 1. Fit best model
            res = fit_model_candidates(df_b, tbp_step=tbp_step)
            res['building_id'] = b_id
            
            # 2. Metadata (Area, Site)
            res['site_id'] = df_b['site_id'].iloc[0]
            res['primaryspaceusage'] = df_b['primaryspaceusage'].iloc[0] if 'primaryspaceusage' in df_b.columns else 'Unknown'
            
            area = df_b['sqm'].iloc[0] if 'sqm' in df_b.columns else np.nan
            if pd.isna(area) and 'sqft' in df_b.columns:
                area = df_b['sqft'].iloc[0] * 0.092903
            res['area'] = area
            res['T_range'] = df_b['T_day'].max() - df_b['T_day'].min()
            
            # 3. Bootstrap Uncertainty
            ci_res = bootstrap_uncertainty(df_b, res['model_type'], tbp_step, n_iters=int(args.bootstrap_iters), rng=rng, block_size=int(args.bootstrap_block_size))
            res.update(ci_res)

            # Derived uncertainty summaries (handy for manuscript tables)
            try:
                res['T_bp_CI_width'] = float(res['T_bp_CI_high'] - res['T_bp_CI_low']) if (not pd.isna(res['T_bp_CI_low']) and not pd.isna(res['T_bp_CI_high'])) else np.nan
            except Exception:
                res['T_bp_CI_width'] = np.nan
            res['n_days_valid'] = int(df_b[['T_day','E_day']].dropna().shape[0])

            # 3b. Model-form stability under bootstrap (optional but recommended for journal robustness)
            if args.model_selection_bootstrap_iters is not None and int(args.model_selection_bootstrap_iters) > 0:
                rng_msel = np.random.default_rng(rng.integers(0, 2**32 - 1))
                msel = bootstrap_model_selection_stability(
                    df_b, tbp_step=tbp_step, n_iters=int(args.model_selection_bootstrap_iters),
                    rng=rng_msel, block_size=int(args.bootstrap_block_size)
                )
                res.update(msel)

            # 3c. Blocked time-series CV (optional; expensive)
            if args.blocked_cv_folds is not None and int(args.blocked_cv_folds) > 1:
                cv = blocked_time_series_cv(df_b, tbp_step=tbp_step, n_folds=int(args.blocked_cv_folds))
                res.update(cv)

            # 4. Typology
            mean_load = df_b['E_day'].mean()
            res['mean_load'] = float(mean_load)
            res['nrmse'] = float(res['rmse'] / mean_load) if (mean_load is not None and mean_load > 0 and not pd.isna(res.get('rmse', np.nan))) else np.nan
            res['typology'] = classify_typology(res, mean_load)
            
            # 5. Normalization
            if not pd.isna(area) and area > 0:
                res['beta_0_norm'] = res['beta_0'] / area
                res['beta_h_norm'] = res['beta_h'] / area
                res['beta_c_norm'] = res['beta_c'] / area
            else:
                res['beta_0_norm'] = np.nan
                res['beta_h_norm'] = np.nan
                res['beta_c_norm'] = np.nan
            
            # 6. Stability (Longitudinal)
            min_stab_days = 200 if not args.demo else 5
            stab_res = analyze_stability(df_b, res['model_type'], tbp_step, min_days=min_stab_days)
            res.update(stab_res)
            
            results.append(res)
            
            if (i+1) % 10 == 0:
                logging.info(f"Processed {i+1}/{len(valid_bldgs)} buildings...")
                
        except Exception as e:
            logging.error(f"Error processing building {b_id}: {e}")
            
    results_df = pd.DataFrame(results)
    out_csv = outdir / 'balance_point_parameters.csv'
    results_df.to_csv(out_csv, index=False)
    logging.info(f"Results saved to {out_csv}")

    # Paper-ready summary tables (typology shares + stability metrics)
    write_paper_summary_tables(results_df, outdir)

    # Plots
    fig_formats = tuple([s.strip() for s in str(args.fig_formats).split(',') if s.strip()])

    plot_figures(
        results_df, daily_df, outdir,
        plot_trim_quantile=float(args.plot_trim_quantile),
        plot_logy=bool(args.plot_logy),
        plot_r2_min_usage=float(args.plot_r2_min_usage),
        plot_usage_min_n=int(args.plot_usage_min_n),
        plot_r2_min_examples=float(args.plot_r2_min_examples),
        # typology rule parameters used in plotting filters
        bootstrap_excursion_degC=10.0,
        contrib_frac=0.05,
        fig_formats=fig_formats,
        fig_dpi=int(args.fig_dpi)
    )
    logging.info("Plots generated.")

if __name__ == "__main__":
    main()
