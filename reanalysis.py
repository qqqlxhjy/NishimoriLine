import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


def read_scan_csv(path: Path) -> Tuple[List[float], List[float]]:
    temps = []
    mags = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                t = float(row[0])
                m = float(row[2])
            except ValueError:
                continue
            temps.append(t)
            mags.append(m)
    return temps, mags


def read_full_scan_csv(path: Path):
    temps = []
    e_vals = []
    m_vals = []
    c_vals = []
    x_vals = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                t = float(row[0])
                e = float(row[1])
                m = float(row[2])
                c = float(row[3])
                x = float(row[4])
            except ValueError:
                continue
            temps.append(t)
            e_vals.append(e)
            m_vals.append(m)
            c_vals.append(c)
            x_vals.append(x)
    return temps, e_vals, m_vals, c_vals, x_vals


def two_peak_half_intervals(values: List[float], temps: List[float]) -> List[Optional[Tuple[float, float]]]:
    n = len(values)
    if n == 0:
        return [None, None]
    peaks: List[int] = []
    if n == 1:
        peaks.append(0)
    else:
        for i in range(n):
            v = values[i]
            left_ok = i == 0 or v >= values[i - 1]
            right_ok = i == n - 1 or v >= values[i + 1]
            if left_ok and right_ok and v > 0.0:
                if i < 5:
                    continue
                left_slice = values[:i]
                mean_left = sum(left_slice) / len(left_slice)
                strong_enough = True
                max_k = min(3, i)
                for k in range(max_k + 1):
                    idx = i - k
                    if values[idx] <= mean_left:
                        strong_enough = False
                        break
                if strong_enough:
                    peaks.append(i)
    peaks.sort(key=lambda idx: values[idx], reverse=True)
    dedup = []
    for idx in peaks:
        if idx not in dedup:
            dedup.append(idx)
    peaks = dedup

    result: List[Optional[Tuple[float, float]]] = [None, None]
    for slot, idx in enumerate(peaks[:2]):
        peak_val = values[idx]
        if peak_val <= 0.0:
            continue
        half = peak_val / 2.0
        left_idx = idx
        while left_idx > 0 and values[left_idx - 1] >= half:
            left_idx -= 1
        right_idx = idx
        last = n - 1
        while right_idx < last and values[right_idx + 1] >= half:
            right_idx += 1
        result[slot] = (temps[left_idx], temps[right_idx])
    return result


def build_window(intervals: List[Optional[Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    lefts: List[float] = []
    rights: List[float] = []
    for it in intervals:
        if it is not None:
            l, r = it
            lefts.append(l)
            rights.append(r)
    if not lefts or not rights:
        return None
    t_env_min = min(lefts)
    t_env_max = max(rights)
    tc_ov_min = max(lefts)
    tc_ov_max = min(rights)
    if not (tc_ov_min <= tc_ov_max):
        tc_ov_min = t_env_min
        tc_ov_max = t_env_max
    return t_env_min, t_env_max, tc_ov_min, tc_ov_max


def auto_windows_from_scan(c_vals: List[float], x_vals: List[float], temps: List[float]):
    c_ints = two_peak_half_intervals(c_vals, temps)
    chi_ints = two_peak_half_intervals(x_vals, temps)
    primary = build_window([c_ints[0], chi_ints[0]])
    secondary = None
    if c_ints[1] is not None or chi_ints[1] is not None:
        secondary = build_window([c_ints[1], chi_ints[1]])
    return primary, secondary


def run_loglog_tc_scan(
    temps: List[float],
    mags: List[float],
    t_min: float,
    t_max: float,
    tc_min: float,
    tc_max: float,
    tc_step: float,
):
    results = []
    n_steps = int(round((tc_max - tc_min) / tc_step))
    for i in range(n_steps + 1):
        tc = tc_min + i * tc_step
        if tc < tc_min or tc > tc_max:
            continue
        x_vals = []
        y_vals = []
        for t, m in zip(temps, mags):
            if t < tc and t >= t_min and t <= t_max and m > 0.0:
                x_vals.append(math.log(tc - t))
                y_vals.append(math.log(m))
        if len(x_vals) < 4:
            results.append((tc, 0.0, float("-inf"), 0.0, 0.0, len(x_vals), False))
            continue
        n = float(len(x_vals))
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_x2 = sum(x * x for x in x_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0.0:
            results.append((tc, 0.0, float("-inf"), 0.0, 0.0, len(x_vals), False))
            continue
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in y_vals)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))
        if ss_tot == 0.0:
            r2 = 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        is_valid = slope > 0.0 and r2 > 0.0 and r2 <= 1.0
        beta = slope
        results.append((tc, beta, r2, slope, intercept, len(x_vals), is_valid))
    best = None
    for r in results:
        tc, beta, r2, slope, intercept, n_pts, is_valid = r
        if not is_valid or not math.isfinite(r2) or r2 <= 0.0:
            continue
        if best is None or r2 > best[2]:
            best = r
    return results, best


def main():
    parser = argparse.ArgumentParser(description="Re-run Tc log-log analysis from saved data directory.")
    parser.add_argument("data_dir", type=str, help="Path to data/ising_results_YYYYMMDD_HHMMSS directory")
    parser.add_argument("--tmin", type=float, default=None, help="Manual T analysis min (override auto)")
    parser.add_argument("--tmax", type=float, default=None, help="Manual T analysis max (override auto)")
    parser.add_argument("--tcmin", type=float, default=None, help="Manual Tc min (override auto)")
    parser.add_argument("--tcmax", type=float, default=None, help="Manual Tc max (override auto)")
    parser.add_argument("--tcstep", type=float, default=0.0001, help="Tc scan step")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Directory not found: {base_dir}")

    scan_csv = base_dir / "ising_results_scan.csv"
    if not scan_csv.is_file():
        raise SystemExit(f"Scan CSV not found in {base_dir}: expected ising_results_scan.csv")

    temps, e_vals, m_vals, c_vals, x_vals = read_full_scan_csv(scan_csv)

    primary, secondary = auto_windows_from_scan(c_vals, x_vals, temps)
    if primary is None:
        raise SystemExit("Auto analysis failed to find any valid window from C and chi.")
    auto_tmin, auto_tmax, auto_tcmin, auto_tcmax = primary

    t_min = args.tmin if args.tmin is not None else auto_tmin
    t_max = args.tmax if args.tmax is not None else auto_tmax
    tc_min = args.tcmin if args.tcmin is not None else auto_tcmin
    tc_max = args.tcmax if args.tcmax is not None else auto_tcmax

    results, best = run_loglog_tc_scan(temps, m_vals, t_min, t_max, tc_min, tc_max, args.tcstep)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"reanalysis_loglog_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    csv_out = out_dir / "re_tc_scan.csv"
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tc", "beta", "r_squared", "slope", "intercept", "fit_points", "is_valid"])
        for r in results:
            tc, beta, r2, slope, intercept, n_pts, is_valid = r
            writer.writerow([f"{tc:.8f}", f"{beta:.8f}", f"{r2:.8f}", f"{slope:.8f}", f"{intercept:.8f}", n_pts, is_valid])

    summary = out_dir / "re_summary.txt"
    with summary.open("w") as f:
        f.write("Reanalysis summary\n")
        f.write(f"Source directory: {base_dir}\n\n")
        f.write("Auto analysis windows (from reanalysis):\n")
        f.write(f"T window (envelope) = [{auto_tmin:.6f}, {auto_tmax:.6f}]\n")
        f.write(f"Tc window (overlap) = [{auto_tcmin:.6f}, {auto_tcmax:.6f}]\n\n")
        f.write("Actual windows used for Tc scan:\n")
        f.write(f"T window = [{t_min:.6f}, {t_max:.6f}]\n")
        f.write(f"Tc window = [{tc_min:.6f}, {tc_max:.6f}]\n")
        f.write(f"Tc step = {args.tcstep:.6f}\n\n")
        if best is not None:
            tc, beta, r2, slope, intercept, n_pts, is_valid = best
            f.write("Best Tc from reanalysis\n")
            f.write(f"Tc_best    = {tc:.8f}\n")
            f.write(f"beta       = {beta:.8f}\n")
            f.write(f"R_squared  = {r2:.8f}\n")
            f.write(f"fit_points = {n_pts}\n")
        else:
            f.write("No valid Tc found (no positive-slope fits with R^2>0).\n")

    if best is not None:
        tc, beta, r2, slope, intercept, n_pts, is_valid = best
        print(f"Tc_best = {tc:.8f}, beta = {beta:.8f}, R^2 = {r2:.8f}, points = {n_pts}")
    else:
        print("No valid Tc found in reanalysis.")


if __name__ == "__main__":
    main()
