use std::fs;
use std::io::Read;

use crate::{InitialState, SimParams};

pub fn load_params_from_summary_dir(dir: &str) -> Result<SimParams, String> {
    let path = format!("{}/summary.txt", dir);
    let mut file = fs::File::open(&path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let mut l: Option<usize> = None;
    let mut j: Option<f64> = None;
    let mut bond_p: Option<f64> = None;
    let mut h: Option<f64> = None;
    let mut initial_state: Option<InitialState> = None;
    let mut mc_steps: Option<usize> = None;
    let mut therm_steps: Option<usize> = None;
    let mut stride: Option<usize> = None;
    let mut t_start: Option<f64> = None;
    let mut t_end: Option<f64> = None;
    let mut t_step: Option<f64> = None;
    let mut tc_step: Option<f64> = None;

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let val_str = line[eq_pos + 1..].trim();
            match key {
                "L" => {
                    l = Some(
                        val_str
                            .parse::<usize>()
                            .map_err(|_| format!("Invalid L value in {}: '{}'", path, val_str))?,
                    );
                }
                "J" => {
                    j = Some(
                        val_str
                            .parse::<f64>()
                            .map_err(|_| format!("Invalid J value in {}: '{}'", path, val_str))?,
                    );
                }
                "p" => {
                    bond_p = Some(
                        val_str
                            .parse::<f64>()
                            .map_err(|_| format!("Invalid p value in {}: '{}'", path, val_str))?,
                    );
                }
                "H" => {
                    h = Some(
                        val_str
                            .parse::<f64>()
                            .map_err(|_| format!("Invalid H value in {}: '{}'", path, val_str))?,
                    );
                }
                "Initial state" => {
                    initial_state = InitialState::from_label(val_str);
                    if initial_state.is_none() {
                        return Err(format!(
                            "Invalid Initial state value in {}: '{}'",
                            path, val_str
                        ));
                    }
                }
                "MC steps" => {
                    mc_steps = Some(
                        val_str.parse::<usize>().map_err(|_| {
                            format!("Invalid MC steps value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "Therm steps" => {
                    therm_steps = Some(
                        val_str.parse::<usize>().map_err(|_| {
                            format!("Invalid Therm steps value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "Stride" => {
                    stride = Some(
                        val_str.parse::<usize>().map_err(|_| {
                            format!("Invalid Stride value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "T_start" => {
                    t_start = Some(
                        val_str.parse::<f64>().map_err(|_| {
                            format!("Invalid T_start value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "T_end" => {
                    t_end = Some(
                        val_str.parse::<f64>().map_err(|_| {
                            format!("Invalid T_end value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "T_step" => {
                    t_step = Some(
                        val_str.parse::<f64>().map_err(|_| {
                            format!("Invalid T_step value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                "Tc_step" => {
                    tc_step = Some(
                        val_str.parse::<f64>().map_err(|_| {
                            format!("Invalid Tc_step value in {}: '{}'", path, val_str)
                        })?,
                    );
                }
                _ => {}
            }
        }
    }

    let l = l.ok_or_else(|| format!("Missing L in {}", path))?;
    let j = j.ok_or_else(|| format!("Missing J in {}", path))?;
    let bond_p = bond_p.ok_or_else(|| format!("Missing p in {}", path))?;
    let h = h.ok_or_else(|| format!("Missing H in {}", path))?;
    let initial_state =
        initial_state.ok_or_else(|| format!("Missing Initial state in {}", path))?;
    let mc_steps = mc_steps.ok_or_else(|| format!("Missing MC steps in {}", path))?;
    let therm_steps = therm_steps.ok_or_else(|| format!("Missing Therm steps in {}", path))?;
    let stride = stride.ok_or_else(|| format!("Missing Stride in {}", path))?;
    let t_start = t_start.ok_or_else(|| format!("Missing T_start in {}", path))?;
    let t_end = t_end.ok_or_else(|| format!("Missing T_end in {}", path))?;
    let t_step = t_step.ok_or_else(|| format!("Missing T_step in {}", path))?;
    let tc_step = tc_step.ok_or_else(|| format!("Missing Tc_step in {}", path))?;

    let mut sample_count: usize = 1;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let val_str = line[eq_pos + 1..].trim();
            if key == "Disorder samples" {
                if let Ok(v) = val_str.parse::<usize>() {
                    if v >= 1 {
                        sample_count = v;
                    }
                }
            }
        }
    }

    Ok(SimParams {
        l,
        j,
        bond_p,
        sample_count,
        initial_state,
        t_start,
        t_end,
        t_step,
        t_analysis_min: t_start,
        t_analysis_max: t_end,
        tc_min: t_start,
        tc_max: t_end,
        tc_step,
        mc_steps,
        therm_steps,
        stride,
        h,
        use_outlier_filter: false,
    })
}
