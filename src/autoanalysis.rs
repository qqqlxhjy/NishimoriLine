#[derive(Clone)]
pub struct AutoWindow {
    pub t_envelope_min: f64,
    pub t_envelope_max: f64,
    pub tc_overlap_min: f64,
    pub tc_overlap_max: f64,
}

#[derive(Clone)]
pub struct AutoAnalysisIntervals {
    pub primary: AutoWindow,
    pub secondary: Option<AutoWindow>,
    pub c_peak_t: Option<f64>,
    pub chi_peak_t: Option<f64>,
    pub m_slope_peak_t: Option<f64>,
}

pub fn compute_intervals(
    temps: &[f64],
    heat_caps: &[f64],
    suscepts: &[f64],
    mags: &[f64],
) -> Result<AutoAnalysisIntervals, String> {
    let n = temps.len();
    if n == 0 {
        return Err("No data points for auto analysis".into());
    }
    if heat_caps.len() != n || suscepts.len() != n || mags.len() != n {
        return Err("Input arrays must have the same length".into());
    }

    let c_intervals = two_peak_half_intervals(heat_caps, temps);
    let chi_intervals = two_peak_half_intervals(suscepts, temps);

    let c_peak_t = peak_location(heat_caps, temps);
    let chi_peak_t = peak_location(suscepts, temps);
    let m_slope_peak_t = slope_peak_location(mags, temps);

    let primary = build_window(&[c_intervals[0], chi_intervals[0]])?;
    let secondary_opt = {
        let any_secondary = c_intervals[1].is_some() || chi_intervals[1].is_some();
        if any_secondary {
            build_window(&[c_intervals[1], chi_intervals[1]]).ok()
        } else {
            None
        }
    };

    Ok(AutoAnalysisIntervals {
        primary,
        secondary: secondary_opt,
        c_peak_t,
        chi_peak_t,
        m_slope_peak_t,
    })
}

fn two_peak_half_intervals(values: &[f64], temps: &[f64]) -> [Option<(f64, f64)>; 2] {
    if values.is_empty() {
        return [None, None];
    }
    let mut peaks = Vec::new();
    let n = values.len();
    if n == 1 {
        peaks.push(0usize);
    } else {
        for i in 0..n {
            let v = values[i];
            let left_ok = i == 0 || v >= values[i - 1];
            let right_ok = i == n - 1 || v >= values[i + 1];
            if left_ok && right_ok && v > 0.0 {
                if i < 5 {
                    continue;
                }
                let left_slice_end = i - 1;
                let left_slice = &values[0..=left_slice_end];
                let mean_left =
                    left_slice.iter().copied().sum::<f64>() / left_slice.len() as f64;
                let mut strong_enough = true;
                let max_k = 3usize.min(i);
                for k in 0..=max_k {
                    let idx = i - k;
                    if values[idx] <= mean_left {
                        strong_enough = false;
                        break;
                    }
                }
                if strong_enough {
                    peaks.push(i);
                }
            }
        }
    }
    peaks.sort_by(|&i, &j| values[j].partial_cmp(&values[i]).unwrap_or(std::cmp::Ordering::Equal));
    peaks.dedup();

    let mut result = [None, None];
    for (slot, idx) in peaks.into_iter().take(2).enumerate() {
        let peak_val = values[idx];
        if peak_val <= 0.0 {
            continue;
        }
        let half = peak_val / 2.0;

        let mut left_idx = idx;
        while left_idx > 0 && values[left_idx - 1] >= half {
            left_idx -= 1;
        }

        let mut right_idx = idx;
        let last = values.len() - 1;
        while right_idx < last && values[right_idx + 1] >= half {
            right_idx += 1;
        }

        result[slot] = Some((temps[left_idx], temps[right_idx]));
    }
    result
}

fn two_slope_half_intervals(mags: &[f64], temps: &[f64]) -> [Option<(f64, f64)>; 2] {
    let n = mags.len();
    if n < 3 {
        return [None, None];
    }

    let mut slopes = vec![0.0f64; n];
    for i in 1..(n - 1) {
        let dt = temps[i + 1] - temps[i - 1];
        if dt != 0.0 {
            slopes[i] = (mags[i + 1] - mags[i - 1]) / dt;
        } else {
            slopes[i] = 0.0;
        }
    }

    let mut peaks = Vec::new();
    for i in 1..(n - 1) {
        let v = slopes[i].abs();
        let left_ok = i == 1 || v >= slopes[i - 1].abs();
        let right_ok = i == n - 2 || v >= slopes[i + 1].abs();
        if left_ok && right_ok && v > 0.0 {
            peaks.push(i);
        }
    }
    peaks.sort_by(|&i, &j| {
        slopes[j]
            .abs()
            .partial_cmp(&slopes[i].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    peaks.dedup();

    let mut result = [None, None];
    for (slot, idx) in peaks.into_iter().take(2).enumerate() {
        let peak_abs = slopes[idx].abs();
        if peak_abs <= 0.0 {
            continue;
        }

        let half = peak_abs / 2.0;
        let mut left_idx = idx;
        while left_idx > 1 && slopes[left_idx - 1].abs() >= half {
            left_idx -= 1;
        }
        let mut right_idx = idx;
        let last = n - 1;
        while right_idx < last - 1 && slopes[right_idx + 1].abs() >= half {
            right_idx += 1;
        }

        result[slot] = Some((temps[left_idx], temps[right_idx]));
    }
    result
}

fn build_window(intervals: &[Option<(f64, f64)>]) -> Result<AutoWindow, String> {
    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    for it in intervals {
        if let Some((l, r)) = it {
            lefts.push(*l);
            rights.push(*r);
        }
    }
    if lefts.is_empty() || rights.is_empty() {
        return Err("Failed to determine window from intervals".into());
    }

    let mut t_envelope_min = f64::INFINITY;
    let mut t_envelope_max = f64::NEG_INFINITY;
    for l in &lefts {
        if *l < t_envelope_min {
            t_envelope_min = *l;
        }
    }
    for r in &rights {
        if *r > t_envelope_max {
            t_envelope_max = *r;
        }
    }

    let mut tc_overlap_min = f64::NEG_INFINITY;
    let mut tc_overlap_max = f64::INFINITY;
    for l in &lefts {
        if *l > tc_overlap_min {
            tc_overlap_min = *l;
        }
    }
    for r in &rights {
        if *r < tc_overlap_max {
            tc_overlap_max = *r;
        }
    }

    if !(tc_overlap_min <= tc_overlap_max) {
        tc_overlap_min = t_envelope_min;
        tc_overlap_max = t_envelope_max;
    }

    Ok(AutoWindow {
        t_envelope_min,
        t_envelope_max,
        tc_overlap_min,
        tc_overlap_max,
    })
}

fn peak_location(values: &[f64], temps: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut idx = 0usize;
    let mut peak = values[0];
    for (i, v) in values.iter().enumerate().skip(1) {
        if *v > peak {
            peak = *v;
            idx = i;
        }
    }
    if peak <= 0.0 {
        None
    } else {
        Some(temps[idx])
    }
}

fn slope_peak_location(mags: &[f64], temps: &[f64]) -> Option<f64> {
    let n = mags.len();
    if n < 3 {
        return None;
    }
    let mut slopes = vec![0.0f64; n];
    for i in 1..(n - 1) {
        let dt = temps[i + 1] - temps[i - 1];
        if dt != 0.0 {
            slopes[i] = (mags[i + 1] - mags[i - 1]) / dt;
        }
    }
    let mut idx = 1usize;
    let mut peak_abs = slopes[1].abs();
    for i in 2..(n - 1) {
        let v = slopes[i].abs();
        if v > peak_abs {
            peak_abs = v;
            idx = i;
        }
    }
    if peak_abs <= 0.0 {
        None
    } else {
        Some(temps[idx])
    }
}
