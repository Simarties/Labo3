# Peak detection on the uploaded CSV with a reusable algorithm (no external web access required)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# ---------- Generic peak finding (no SciPy required) ----------
def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    if window is None or window < 3 or window % 2 == 0:
        return a.copy()
    pad = window // 2
    padded = np.pad(a, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

def _prominence(y: np.ndarray, idx: int) -> float:
    """A simple (naïve) prominence estimate.
    Look left/right until the signal rises >= peak height;
    take the max of the two valley mins; prominence = peak - that baseline.
    """
    peak_h = y[idx]
    # Left search
    min_left = peak_h
    j = idx - 1
    while j >= 0 and y[j] < peak_h:
        if y[j] < min_left:
            min_left = y[j]
        j -= 1
    # Right search
    min_right = peak_h
    k = idx + 1
    n = len(y)
    while k < n and y[k] < peak_h:
        if y[k] < min_right:
            min_right = y[k]
        k += 1
    base = max(min_left, min_right)
    return float(peak_h - base)

def _enforce_min_distance(peaks: List[int], y: np.ndarray, min_distance: int) -> List[int]:
    """Keep the highest peak in any window of +/- min_distance indices."""
    if min_distance is None or min_distance <= 1:
        return peaks
    peaks_sorted = sorted(peaks, key=lambda i: y[i], reverse=True)
    accepted = []
    taken = np.zeros(len(y), dtype=bool)
    for i in peaks_sorted:
        lo = max(0, i - min_distance)
        hi = min(len(y), i + min_distance + 1)
        if not taken[lo:hi].any():
            accepted.append(i)
            taken[lo:hi] = True
    return sorted(accepted)

def find_local_maxima(
    x: np.ndarray,
    y: np.ndarray,
    smooth_window: Optional[int] = 9,
    min_prominence: Optional[float] = None,
    min_distance: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Returns indices of local maxima of y(x) and a diagnostics dict.
    - smooth_window: odd int for moving average smoothing (set None or <3 to disable)
    - min_prominence: discard peaks with smaller prominence (auto-estimated if None)
    - min_distance: minimum separation in index units between peaks
    """
    y_proc = moving_average(y, smooth_window) if smooth_window else y.copy()

    # Strict local maxima candidates
    cand = np.where((y_proc[1:-1] > y_proc[:-2]) & (y_proc[1:-1] >= y_proc[2:]))[0] + 1
    cand = cand.tolist()

    # Prominence filtering
    prom = np.array([_prominence(y_proc, i) for i in cand], dtype=float)
    if min_prominence is None:
        # Heuristic: keep peaks above median + 0.5*IQR of prom (robust to scaling)
        if len(prom) > 0:
            q1, q2, q3 = np.percentile(prom, [25, 50, 75])
            thresh = q2 + 0.5 * (q3 - q1)
            min_prominence = float(thresh)
        else:
            min_prominence = 0.0
    keep_mask = prom >= min_prominence
    cand = [i for i, keep in zip(cand, keep_mask) if keep]
    prom = prom[keep_mask]

    # Min distance enforcement
    cand = _enforce_min_distance(cand, y_proc, min_distance)

    # Recompute prominences on the final set (based on processed signal)
    final_prom = np.array([_prominence(y_proc, i) for i in cand], dtype=float)

    diag = {
        "smoothed": y_proc,
        "prominence_threshold": float(min_prominence),
        "prominences": final_prom,
    }
    return np.array(cand, dtype=int), diag

# ---------- Load your CSV and infer columns ----------
path = "LaboZeeman/donnees/labo3/ZEE_AR_NS/546nm/plot_calibration.csv"
df = pd.read_csv(path)

# Infer numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    # Try to coerce all columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Choose x and y heuristically:
# - x: first numeric column
# - y: last numeric column (different from x if possible)
if len(num_cols) == 1:
    x_col = num_cols[0]
    y_col = num_cols[0]
else:
    x_col = num_cols[0]
    y_col = num_cols[-1] if num_cols[-1] != x_col else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

x = df[x_col].to_numpy()
y = df[y_col].to_numpy()

# Run peak detection (defaults are reasonable; you can tweak below)
peaks_idx, diag = find_local_maxima(
    x, y,
    smooth_window= 9 if 'nine' in globals() else 9,  # ensure odd int
    min_prominence=None,     # auto threshold
    min_distance=max(1, len(y)//200)  # roughly avoid very close peaks (adjustable)
)

# Build peaks dataframe
peaks_df = pd.DataFrame({
    "index": peaks_idx,
    f"{x_col}": x[peaks_idx],
    f"{y_col}": y[peaks_idx],
    "prominence_est": diag["prominences"],
}).sort_values(by=f"{x_col}").reset_index(drop=True)

# Save peaks to CSV
out_path = "LaboZeeman/donnees/labo3/ZEE_AR_NS/546nm/test.csv"
peaks_df.to_csv(out_path, index=False)

# Show a quick glance of the data & detected peaks
'''import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Raw data (first 200 rows shown)", df.head(200))

caas_jupyter_tools.display_dataframe_to_user("Detected local maxima", peaks_df)
'''
# Plot
plt.figure()
plt.plot(x, y, linewidth=1)
plt.scatter(x[peaks_idx], y[peaks_idx], marker='o')
plt.title("Signal with detected local maxima")
plt.xlabel(str(x_col))
plt.ylabel(str(y_col))
plt.tight_layout()
plt.show()

out_path
