mod autoanalysis;
mod load_params;

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use plotters::prelude::*;
use rand::{Rng, seq::SliceRandom};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color as TuiColor, Modifier, Style},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table},
    Terminal,
};
use chrono::Local;
use std::io::{self, Write};

// ─────────────────────────────────────────────
// Ising model core
// ─────────────────────────────────────────────

struct IsingModel {
    spins: Vec<Vec<i8>>,
    size: usize,
    j: f64,
    j_horiz: Vec<Vec<f64>>,
    j_vert: Vec<Vec<f64>>,
    h: f64,
    temperature: f64,
}

impl IsingModel {
    fn build_bonds(size: usize, j: f64, p: f64, rng: &mut impl Rng) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let total_bonds = 2 * size * size;
        let mut target_neg = (p * total_bonds as f64).round() as usize;
        if target_neg > total_bonds {
            target_neg = total_bonds;
        }
        let mut flags = vec![false; total_bonds];
        for k in 0..target_neg {
            flags[k] = true;
        }
        flags.shuffle(rng);
        let mut iter = flags.into_iter();
        let j_horiz = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| {
                        let is_neg = iter.next().unwrap_or(false);
                        if is_neg { -j } else { j }
                    })
                    .collect()
            })
            .collect();
        let j_vert = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| {
                        let is_neg = iter.next().unwrap_or(false);
                        if is_neg { -j } else { j }
                    })
                    .collect()
            })
            .collect();
        (j_horiz, j_vert)
    }

    fn new_random(size: usize, j: f64, p: f64, h: f64, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        let spins = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| if rng.gen::<f64>() < 0.5 { -1i8 } else { 1 })
                    .collect()
            })
            .collect();
        let (j_horiz, j_vert) = Self::build_bonds(size, j, p, &mut rng);
        Self { spins, size, j, j_horiz, j_vert, h, temperature }
    }

    fn new_all_up(size: usize, j: f64, p: f64, h: f64, temperature: f64) -> Self {
        let spins = vec![vec![1i8; size]; size];
        let mut rng = rand::thread_rng();
        let (j_horiz, j_vert) = Self::build_bonds(size, j, p, &mut rng);
        Self { spins, size, j, j_horiz, j_vert, h, temperature }
    }

    fn new_all_down(size: usize, j: f64, p: f64, h: f64, temperature: f64) -> Self {
        let spins = vec![vec![-1i8; size]; size];
        let mut rng = rand::thread_rng();
        let (j_horiz, j_vert) = Self::build_bonds(size, j, p, &mut rng);
        Self { spins, size, j, j_horiz, j_vert, h, temperature }
    }

    fn energy_at_site(&self, i: usize, jc: usize) -> f64 {
        let l = self.size;
        let spin = self.spins[i][jc] as f64;
        let top_i = (i + l - 1) % l;
        let bottom_i = (i + 1) % l;
        let left_j = (jc + l - 1) % l;
        let right_j = (jc + 1) % l;
        let top = self.spins[top_i][jc] as f64;
        let bottom = self.spins[bottom_i][jc] as f64;
        let left = self.spins[i][left_j] as f64;
        let right = self.spins[i][right_j] as f64;
        let j_top = self.j_vert[top_i][jc];
        let j_bottom = self.j_vert[i][jc];
        let j_left = self.j_horiz[i][left_j];
        let j_right = self.j_horiz[i][jc];
        -spin * (j_top * top + j_bottom * bottom + j_left * left + j_right * right) - self.h * spin
    }

    fn total_energy(&self) -> f64 {
        let l = self.size;
        let mut e = 0.0;
        for i in 0..l {
            for jc in 0..l {
                let spin  = self.spins[i][jc] as f64;
                let right  = self.spins[i][(jc + 1) % l] as f64;
                let bottom = self.spins[(i + 1) % l][jc] as f64;
                let j_right = self.j_horiz[i][jc];
                let j_bottom = self.j_vert[i][jc];
                e -= j_right * spin * right;
                e -= j_bottom * spin * bottom;
            }
        }
        e - self.h * self.total_magnetization() as f64
    }

    fn total_magnetization(&self) -> i64 {
        self.spins.iter().flat_map(|r| r.iter()).map(|&s| s as i64).sum()
    }

    fn metropolis_step(&mut self, rng: &mut impl Rng) {
        let i  = rng.gen_range(0..self.size);
        let jc = rng.gen_range(0..self.size);
        let old_e = self.energy_at_site(i, jc);
        self.spins[i][jc] = -self.spins[i][jc];
        let new_e = self.energy_at_site(i, jc);
        let delta_e = new_e - old_e;
        if delta_e > 0.0 && rng.gen::<f64>() >= (-delta_e / self.temperature).exp() {
            // reject: flip back
            self.spins[i][jc] = -self.spins[i][jc];
        }
    }
}

// ─────────────────────────────────────────────
// Simulation parameters & results
// ─────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum InitialState {
    Random,
    AllUp,
    AllDown,
}

impl InitialState {
    fn label(self) -> &'static str {
        match self {
            InitialState::Random  => "Random",
            InitialState::AllUp   => "All Up  (+1)",
            InitialState::AllDown => "All Down (-1)",
        }
    }
    fn from_label(s: &str) -> Option<Self> {
        match s.trim() {
            "Random" => Some(InitialState::Random),
            "All Up  (+1)" => Some(InitialState::AllUp),
            "All Down (-1)" => Some(InitialState::AllDown),
            _ => None,
        }
    }
    fn next(self) -> Self {
        match self {
            InitialState::Random  => InitialState::AllUp,
            InitialState::AllUp   => InitialState::AllDown,
            InitialState::AllDown => InitialState::Random,
        }
    }
    fn prev(self) -> Self {
        match self {
            InitialState::Random  => InitialState::AllDown,
            InitialState::AllUp   => InitialState::Random,
            InitialState::AllDown => InitialState::AllUp,
        }
    }
}

#[derive(Clone)]
struct SimParams {
    l: usize,
    j: f64,
    bond_p: f64,
    sample_count: usize,
    initial_state: InitialState,
    t_start: f64,
    t_end: f64,
    t_step: f64,
    t_analysis_min: f64,
    t_analysis_max: f64,
    tc_min: f64,
    tc_max: f64,
    tc_step: f64,
    mc_steps: usize,
    therm_steps: usize,
    stride: usize,
    h: f64,
    use_outlier_filter: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        let mc_steps = 10_000;
        Self {
            l: 32,
            j: 1.0,
            bond_p: 0.0,
            sample_count: 1,
            initial_state: InitialState::Random,
            t_start: 1.0,
            t_end: 4.0,
            t_step: 0.1,
            t_analysis_min: 2.0,
            t_analysis_max: 2.3,
            tc_min: 2.20,
            tc_max: 2.40,
            tc_step: 0.0001,
            mc_steps,
            therm_steps: mc_steps / 2,
            stride: 10,
            h: 0.0,
            use_outlier_filter: false,
        }
    }
}

#[derive(Clone)]
struct SimResult {
    temperature:   f64,
    mean_e:        f64, // <E>/N
    mean_m:        f64, // <|M|>/N
    heat_cap:      f64, // Var(E)/(T²·N)
    susceptibility: f64, // Var(M)/(T·N)
    is_outlier:    bool,
}

// ─────────────────────────────────────────────
// App / TUI state
// ─────────────────────────────────────────────

const FIELD_L:              usize = 0;
const FIELD_J:              usize = 1;
const FIELD_P:              usize = 2;
const FIELD_INIT:           usize = 3;
const FIELD_T_START:        usize = 4;
const FIELD_T_END:          usize = 5;
const FIELD_T_STEP:         usize = 6;
const FIELD_MC_STEPS:       usize = 7;
const FIELD_THERM:          usize = 8;
const FIELD_STRIDE:         usize = 9;
const FIELD_H:              usize = 10;
const FIELD_TC_STEP:        usize = 11;
const FIELD_SAMPLE_COUNT:   usize = 12;
const NUM_FIELDS:           usize = 13;

const FIELD_ORDER: [usize; NUM_FIELDS] = [
    FIELD_L,
    FIELD_J,
    FIELD_P,
    FIELD_INIT,
    FIELD_H,
    FIELD_T_START,
    FIELD_T_END,
    FIELD_T_STEP,
    FIELD_TC_STEP,
    FIELD_MC_STEPS,
    FIELD_THERM,
    FIELD_STRIDE,
    FIELD_SAMPLE_COUNT,
];

enum AppMode {
    Setup,
    LoadParams,
    RunningSweep { current_t: f64, t_end: f64, done: usize, total: usize },
    Step1Summary,
    ManualWindowEdit,
    RunningTcScan { done: usize, total: usize },
    Done,
}

#[derive(Clone)]
struct ManualWindowEditState {
    fields:   [String; 4],
    selected: usize,
}

struct App {
    mode:                 AppMode,
    field_buffers:        Vec<String>,
    selected_field:       usize,
    initial_state:        InitialState,
    outlier_filter:       bool,
    error_msg:            Option<String>,
    results:              Option<Vec<SimResult>>,
    auto_intervals:       Option<autoanalysis::AutoAnalysisIntervals>,
    sim_params:           Option<SimParams>,
    manual_window_state:  Option<ManualWindowEditState>,
    saved_runs:           Vec<(String, String)>,
    saved_run_selected:   usize,
}

impl App {
    fn new() -> Self {
        let d = SimParams::default();
        let mut b = vec![String::new(); NUM_FIELDS];
        b[FIELD_L]        = d.l.to_string();
        b[FIELD_J]        = format!("{}", d.j);
        b[FIELD_P]        = format!("{}", d.bond_p);
        b[FIELD_T_START]  = format!("{}", d.t_start);
        b[FIELD_T_END]    = format!("{}", d.t_end);
        b[FIELD_T_STEP]   = format!("{}", d.t_step);
        b[FIELD_MC_STEPS] = d.mc_steps.to_string();
        b[FIELD_THERM]    = d.therm_steps.to_string();
        b[FIELD_STRIDE]   = d.stride.to_string();
        b[FIELD_H]        = format!("{}", d.h);
        b[FIELD_TC_STEP]  = format!("{}", d.tc_step);
        b[FIELD_SAMPLE_COUNT] = d.sample_count.to_string();
        Self {
            mode: AppMode::Setup,
            field_buffers: b,
            selected_field: 0,
            initial_state: d.initial_state,
            outlier_filter: d.use_outlier_filter,
            error_msg: None,
            results: None,
            auto_intervals: None,
            sim_params: Some(d),
            manual_window_state: None,
            saved_runs: Vec::new(),
            saved_run_selected: 0,
        }
    }

    fn parse_params(&self) -> Result<SimParams, String> {
        let l = self.field_buffers[FIELD_L].trim().parse::<usize>()
            .map_err(|_| format!("L must be a positive integer, got '{}'", self.field_buffers[FIELD_L]))?;
        if l < 2 { return Err("Lattice size L must be >= 2".into()); }

        let j = self.field_buffers[FIELD_J].trim().parse::<f64>()
            .map_err(|_| format!("J must be a number, got '{}'", self.field_buffers[FIELD_J]))?;

        let bond_p = self.field_buffers[FIELD_P].trim().parse::<f64>()
            .map_err(|_| format!("p must be a number, got '{}'", self.field_buffers[FIELD_P]))?;
        if !(0.0..=1.0).contains(&bond_p) { return Err("p must be in [0, 1]".into()); }

        let t_start = self.field_buffers[FIELD_T_START].trim().parse::<f64>()
            .map_err(|_| format!("T start must be a number, got '{}'", self.field_buffers[FIELD_T_START]))?;
        if t_start <= 0.0 { return Err("T start must be > 0".into()); }

        let t_end = self.field_buffers[FIELD_T_END].trim().parse::<f64>()
            .map_err(|_| format!("T end must be a number, got '{}'", self.field_buffers[FIELD_T_END]))?;
        if t_end < t_start { return Err("T end must be >= T start".into()); }

        let t_step = self.field_buffers[FIELD_T_STEP].trim().parse::<f64>()
            .map_err(|_| format!("T step must be a number, got '{}'", self.field_buffers[FIELD_T_STEP]))?;
        if t_step <= 0.0 { return Err("T step must be > 0".into()); }

        let tc_step = self.field_buffers[FIELD_TC_STEP].trim().parse::<f64>()
            .map_err(|_| format!("Tc step must be a number, got '{}'", self.field_buffers[FIELD_TC_STEP]))?;
        if tc_step <= 0.0 {
            return Err("Tc step must be > 0".into());
        }

        let mc_steps = self.field_buffers[FIELD_MC_STEPS].trim().parse::<usize>()
            .map_err(|_| format!("MC Steps must be a positive integer, got '{}'", self.field_buffers[FIELD_MC_STEPS]))?;
        if mc_steps < 2 { return Err("MC Steps must be >= 2".into()); }

        let therm_steps = self.field_buffers[FIELD_THERM].trim().parse::<usize>()
            .map_err(|_| format!("Therm Steps must be a positive integer, got '{}'", self.field_buffers[FIELD_THERM]))?;
        if therm_steps == 0 { return Err("Therm Steps must be >= 1".into()); }

        let stride = self.field_buffers[FIELD_STRIDE].trim().parse::<usize>()
            .map_err(|_| format!("Stride must be a positive integer, got '{}'", self.field_buffers[FIELD_STRIDE]))?;
        if stride == 0 { return Err("Stride must be >= 1".into()); }

        let h = self.field_buffers[FIELD_H].trim().parse::<f64>()
            .map_err(|_| format!("H must be a number, got '{}'", self.field_buffers[FIELD_H]))?;

        let sample_count = self.field_buffers[FIELD_SAMPLE_COUNT].trim().parse::<usize>()
            .map_err(|_| format!("Disorder samples must be a positive integer, got '{}'", self.field_buffers[FIELD_SAMPLE_COUNT]))?;
        if sample_count == 0 { return Err("Disorder samples must be >= 1".into()); }

        Ok(SimParams {
            l,
            j,
            bond_p,
            sample_count,
            initial_state: self.initial_state,
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
            use_outlier_filter: self.outlier_filter,
        })
    }
}

// ─────────────────────────────────────────────
// Simulation engine
// ─────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

fn build_lattice(p: &SimParams, temperature: f64) -> IsingModel {
    match p.initial_state {
        InitialState::Random  => IsingModel::new_random(p.l, p.j, p.bond_p, p.h, temperature),
        InitialState::AllUp   => IsingModel::new_all_up(p.l, p.j, p.bond_p, p.h, temperature),
        InitialState::AllDown => IsingModel::new_all_down(p.l, p.j, p.bond_p, p.h, temperature),
    }
}

fn measure_at_temperature(p: &SimParams, temperature: f64, rng: &mut impl Rng) -> SimResult {
    let n = (p.l * p.l) as f64;
    let mut mean_e_acc = 0.0;
    let mut mean_m_acc = 0.0;
    let mut heat_cap_acc = 0.0;
    let mut chi_acc = 0.0;

    let samples = p.sample_count.max(1);

    for _ in 0..samples {
        let mut model = build_lattice(p, temperature);

        for _ in 0..p.therm_steps {
            for _ in 0..p.l * p.l {
                model.metropolis_step(rng);
            }
        }

        let mut e_samples: Vec<f64> = Vec::new();
        let mut m_samples: Vec<f64> = Vec::new();
        let mut m_abs_samples: Vec<f64> = Vec::new();
        for step in 0..p.mc_steps {
            for _ in 0..p.l * p.l {
                model.metropolis_step(rng);
            }
            if step % p.stride == 0 {
                e_samples.push(model.total_energy());
                let m = model.total_magnetization() as f64;
                m_samples.push(m);
                m_abs_samples.push(m.abs());
            }
        }

        let mean_e = mean(&e_samples) / n;
        let mean_m = mean(&m_abs_samples) / n;
        let heat_cap = variance(&e_samples) / (temperature * temperature * n);
        let chi = variance(&m_samples) / (temperature * n);

        mean_e_acc += mean_e;
        mean_m_acc += mean_m;
        heat_cap_acc += heat_cap;
        chi_acc += chi;
    }

    let inv_samples = 1.0 / samples as f64;
    let mean_e = mean_e_acc * inv_samples;
    let mean_m = mean_m_acc * inv_samples;
    let heat_cap = heat_cap_acc * inv_samples;
    let chi = chi_acc * inv_samples;

    SimResult { temperature, mean_e, mean_m, heat_cap, susceptibility: chi, is_outlier: false }
}

fn run_sweep(
    params: &SimParams,
    mut progress_cb: impl FnMut(f64, usize, usize),
) -> Vec<SimResult> {
    let total = {
        let n = ((params.t_end - params.t_start) / params.t_step).ceil() as usize + 1;
        n
    };
    let mut rng = rand::thread_rng();
    let mut results = Vec::with_capacity(total);
    for i in 0..total {
        let t = params.t_start + i as f64 * params.t_step;
        if t > params.t_end + 1e-9 { break; }
        progress_cb(t, i, total);
        results.push(measure_at_temperature(params, t, &mut rng));
    }
    results
}

struct TcScanResult {
    tc: f64,
    beta: f64,
    r_squared: f64,
    slope: f64,
    intercept: f64,
    fit_points: usize,
    is_valid: bool,
}

fn run_loglog_analysis(
    params: &SimParams,
    results: &[SimResult],
    output_root: &str,
    mut progress_cb: impl FnMut(usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    if results.is_empty() {
        return Ok(());
    }

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let dir = format!("{}/loglog_singleProfile_{}", output_root, timestamp);
    std::fs::create_dir_all(&dir)?;
    let prefix = format!("{}/loglog_singleProfile", dir);

    {
        let mut file = File::create(format!("{}_scan.csv", prefix))?;
        writeln!(file, "temperature,e_per_spin,m_abs_per_spin,c_per_spin,susceptibility")?;
        for r in results {
            writeln!(
                file,
                "{:.8},{:.8},{:.8},{:.8},{:.8}",
                r.temperature, r.mean_e, r.mean_m, r.heat_cap, r.susceptibility
            )?;
        }
    }

    let t_min = params.t_analysis_min;
    let t_max = params.t_analysis_max;

    let mut marked: Vec<SimResult> = results.to_vec();
    if params.use_outlier_filter {
        let mut max_c = f64::NEG_INFINITY;
        let mut max_chi = f64::NEG_INFINITY;
        for r in &marked {
            if r.temperature >= t_min && r.temperature <= t_max {
                if r.heat_cap > max_c {
                    max_c = r.heat_cap;
                }
                if r.susceptibility > max_chi {
                    max_chi = r.susceptibility;
                }
            }
        }
        if max_c.is_finite() && max_chi.is_finite() {
            for r in &mut marked {
                if (r.temperature < t_min || r.temperature > t_max)
                    && (r.heat_cap > max_c || r.susceptibility > max_chi)
                {
                    r.is_outlier = true;
                }
            }
        }
    }

    save_overview_to_path(&marked, &format!("{}_overview.png", prefix))?;

    let temps: Vec<f64> = marked.iter().map(|r| r.temperature).collect();
    let mags: Vec<f64> = marked.iter().map(|r| r.mean_m).collect();

    let mut tc_results = Vec::new();

    let n_steps = ((params.tc_max - params.tc_min) / params.tc_step).round() as usize;
    let total_steps = n_steps + 1;
    for i in 0..=n_steps {
        let tc = params.tc_min + i as f64 * params.tc_step;
        if tc < params.tc_min || tc > params.tc_max {
            continue;
        }

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        for (idx, (&t, &m)) in temps.iter().zip(mags.iter()).enumerate() {
            if marked[idx].is_outlier {
                continue;
            }
            if t < tc && t >= t_min && t <= t_max && m > 0.0 {
                let x = (tc - t).ln();
                let y = m.ln();
                x_vals.push(x);
                y_vals.push(y);
            }
        }

        if x_vals.len() < 4 {
            tc_results.push(TcScanResult {
                tc,
                beta: 0.0,
                r_squared: f64::NEG_INFINITY,
                slope: 0.0,
                intercept: 0.0,
                fit_points: x_vals.len(),
                is_valid: false,
            });
            continue;
        }

        let n = x_vals.len() as f64;
        let sum_x: f64 = x_vals.iter().sum();
        let sum_y: f64 = y_vals.iter().sum();
        let sum_x2: f64 = x_vals.iter().map(|x| x * x).sum();
        let sum_xy: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * y).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator == 0.0 {
            tc_results.push(TcScanResult {
                tc,
                beta: 0.0,
                r_squared: f64::NEG_INFINITY,
                slope: 0.0,
                intercept: 0.0,
                fit_points: x_vals.len(),
                is_valid: false,
            });
            continue;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        let mean_y = sum_y / n;
        let ss_tot = y_vals.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
        let ss_res = y_vals
            .iter()
            .zip(x_vals.iter())
            .map(|(y, x)| (y - (slope * x + intercept)).powi(2))
            .sum::<f64>();
        let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

        let is_valid = slope > 0.0 && r_squared > 0.0 && r_squared <= 1.0;

        tc_results.push(TcScanResult {
            tc,
            beta: slope,
            r_squared,
            slope,
            intercept,
            fit_points: x_vals.len(),
            is_valid,
        });
        progress_cb(i + 1, total_steps);
    }

    {
        let mut file = File::create(format!("{}_tc_scan.csv", prefix))?;
        writeln!(file, "tc,beta,r_squared,slope,intercept,fit_points,is_valid")?;
        for r in &tc_results {
            writeln!(
                file,
                "{:.8},{:.8},{:.8},{:.8},{:.8},{} ,{}",
                r.tc, r.beta, r.r_squared, r.slope, r.intercept, r.fit_points, r.is_valid
            )?;
        }
    }

    let best = tc_results
        .iter()
        .filter(|r| r.is_valid && r.r_squared.is_finite() && r.r_squared > 0.0)
        .max_by(|a, b| a.r_squared.partial_cmp(&b.r_squared).unwrap_or(std::cmp::Ordering::Equal));

    {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"><title>Tc log-log analysis</title></head>\n<body>\n");
        html.push_str("<h1>Tc log-log analysis</h1>\n");
        html.push_str(&format!(
            "<p>T analysis window: [{:.6}, {:.6}]</p>\n",
            t_min, t_max
        ));
        html.push_str(&format!(
            "<p>Tc scan range: [{:.6}, {:.6}] step {:.6}</p>\n",
            params.tc_min, params.tc_max, params.tc_step
        ));
        if let Some(b) = best {
            html.push_str(&format!(
                "<p>Best Tc: {:.8}, beta: {:.8}, R²: {:.8}, fit points: {}</p>\n",
                b.tc, b.beta, b.r_squared, b.fit_points
            ));
        } else {
            html.push_str("<p>No valid Tc found (no positive-slope fits with R²>0).</p>\n");
        }
        html.push_str("<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">\n");
        html.push_str("<tr><th>Tc</th><th>beta</th><th>R²</th><th>slope</th><th>intercept</th><th>fit_points</th><th>valid</th></tr>\n");
        for r in &tc_results {
            let highlight = if best.map_or(false, |b| (b.tc - r.tc).abs() < 1e-10) {
                " style=\"background-color:#ffffcc;\""
            } else {
                ""
            };
            html.push_str(&format!(
                "<tr{}><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{}</td><td>{}</td></tr>\n",
                highlight, r.tc, r.beta, r.r_squared, r.slope, r.intercept, r.fit_points, r.is_valid
            ));
        }
        html.push_str("</table>\n</body>\n</html>\n");

        let mut file = File::create(format!("{}_loglog_detailed.html", prefix))?;
        file.write_all(html.as_bytes())?;
    }

    if let Some(b) = best {
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        for (idx, (&t, &m)) in temps.iter().zip(mags.iter()).enumerate() {
            if marked[idx].is_outlier {
                continue;
            }
            if t < b.tc && t >= t_min && t <= t_max && m > 0.0 {
                let x = (b.tc - t).ln();
                let y = m.ln();
                x_vals.push(x);
                y_vals.push(y);
            }
        }
        if !x_vals.is_empty() {
            let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let x_pad = (x_max - x_min).abs() * 0.1 + 1e-10;
            let y_pad = (y_max - y_min).abs() * 0.1 + 1e-10;

            let path = format!("{}_loglog_plot.png", prefix);
            let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .caption("log(M) vs log(Tc - T)", ("sans-serif", 18).into_font())
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d((x_min - x_pad)..(x_max + x_pad), (y_min - y_pad)..(y_max + y_pad))?;

            chart.configure_mesh().x_desc("log(Tc - T)").y_desc("log(M)").draw()?;

            chart.draw_series(
                x_vals
                    .iter()
                    .zip(y_vals.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 4, BLUE.filled())),
            )?;

            let line_x0 = x_min - x_pad;
            let line_x1 = x_max + x_pad;
            let line_y0 = b.slope * line_x0 + b.intercept;
            let line_y1 = b.slope * line_x1 + b.intercept;

            chart.draw_series(LineSeries::new(
                vec![(line_x0, line_y0), (line_x1, line_y1)],
                &RED,
            ))?;

            root.present()?;
        }
    }

    {
        let mut file = File::create(format!("{}/summary.txt", dir))?;
        writeln!(file, "Simulation summary")?;
        writeln!(file, "Timestamp: {}", timestamp)?;
        writeln!(file, "Output directory: {}", dir)?;
        writeln!(file)?;
        writeln!(file, "Model parameters")?;
        writeln!(file, "L = {}", params.l)?;
        writeln!(file, "J = {}", params.j)?;
        writeln!(file, "p = {}", params.bond_p)?;
        writeln!(file, "H = {}", params.h)?;
        writeln!(file, "Initial state = {}", params.initial_state.label())?;
        let total_bonds = 2usize.saturating_mul(params.l).saturating_mul(params.l);
        let neg_target = (params.bond_p.max(0.0).min(1.0) * total_bonds as f64).round() as usize;
        writeln!(file, "Total bonds = {}", total_bonds)?;
        writeln!(file, "-J bonds (rigid) = {}", neg_target)?;
        writeln!(file)?;
        writeln!(file, "MC parameters")?;
        writeln!(file, "MC steps   = {}", params.mc_steps)?;
        writeln!(file, "Therm steps = {}", params.therm_steps)?;
        writeln!(file, "Stride      = {}", params.stride)?;
        writeln!(file, "Disorder samples = {}", params.sample_count)?;
        writeln!(file)?;
        writeln!(file, "Scan parameters")?;
        writeln!(file, "T_start = {}", params.t_start)?;
        writeln!(file, "T_end   = {}", params.t_end)?;
        writeln!(file, "T_step  = {}", params.t_step)?;
        writeln!(file, "Tc_step = {}", params.tc_step)?;
        writeln!(file)?;
        writeln!(file, "Auto analysis windows")?;
        writeln!(file, "T window (envelope) = [{:.6}, {:.6}]", params.t_analysis_min, params.t_analysis_max)?;
        writeln!(file, "Tc window (overlap)  = [{:.6}, {:.6}]", params.tc_min, params.tc_max)?;
        writeln!(file)?;
        writeln!(file, "Best Tc from log-log fit")?;
        if let Some(b) = best {
            writeln!(file, "Tc_best    = {:.8}", b.tc)?;
            writeln!(file, "beta       = {:.8}", b.beta)?;
            writeln!(file, "R_squared  = {:.8}", b.r_squared)?;
            writeln!(file, "fit_points = {}", b.fit_points)?;
        } else {
            writeln!(file, "No valid Tc found (no positive-slope fits with R^2>0).")?;
        }
    }

    Ok(())
}

fn run_headless_single(params: &SimParams) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_sweep_done: usize = 0;
    let results = run_sweep(params, |cur_t, done, total| {
        if total > 0 && done != last_sweep_done {
            last_sweep_done = done;
            println!("BATCH_PROGRESS SWEEP {} {} {:.8}", done, total, cur_t);
            let _ = io::stdout().flush();
        }
    });

    let mut params_for_tc = params.clone();
    let window_mode =
        std::env::var("BATCH_WINDOW_MODE").unwrap_or_else(|_| "fixed".to_string());
    if window_mode == "auto" {
        let temps: Vec<f64> = results.iter().map(|r| r.temperature).collect();
        let mags: Vec<f64> = results.iter().map(|r| r.mean_m).collect();
        let c_vals: Vec<f64> = results.iter().map(|r| r.heat_cap).collect();
        let chi_vals: Vec<f64> = results.iter().map(|r| r.susceptibility).collect();
        let intervals = autoanalysis::compute_intervals(&temps, &c_vals, &chi_vals, &mags)?;
        let primary = intervals.primary;
        params_for_tc.t_analysis_min = primary.t_envelope_min;
        params_for_tc.t_analysis_max = primary.t_envelope_max;
        params_for_tc.tc_min = primary.tc_overlap_min;
        params_for_tc.tc_max = primary.tc_overlap_max;
    } else {
        let t_min = std::env::var("BATCH_T_MIN")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.0);
        let t_max = std::env::var("BATCH_T_MAX")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.45);
        let tc_min = std::env::var("BATCH_TC_MIN")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.25);
        let tc_max = std::env::var("BATCH_TC_MAX")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.45);
        params_for_tc.t_analysis_min = t_min;
        params_for_tc.t_analysis_max = t_max;
        params_for_tc.tc_min = tc_min;
        params_for_tc.tc_max = tc_max;
    }

    let output_root =
        std::env::var("BATCH_OUTPUT_ROOT").unwrap_or_else(|_| "data_batch".to_string());

    let mut last_tc_done: usize = 0;
    run_loglog_analysis(&params_for_tc, &results, &output_root, |done, total| {
        if total > 0 && done != last_tc_done {
            last_tc_done = done;
            println!("BATCH_PROGRESS TC {} {}", done, total);
            let _ = io::stdout().flush();
        }
    })
}

fn run_batch_from_env() -> Result<(), Box<dyn std::error::Error>> {
    let l = std::env::var("BATCH_L")?.parse::<usize>()?;
    let j = std::env::var("BATCH_J")?.parse::<f64>()?;
    let p = std::env::var("BATCH_P")?.parse::<f64>()?;
    let t_start = std::env::var("BATCH_T_START")?.parse::<f64>()?;
    let t_end = std::env::var("BATCH_T_END")?.parse::<f64>()?;
    let t_step = std::env::var("BATCH_T_STEP")?.parse::<f64>()?;
    let mc_steps = std::env::var("BATCH_MC_STEPS")?.parse::<usize>()?;
    let therm_steps = std::env::var("BATCH_THERM_STEPS")?.parse::<usize>()?;
    let stride = std::env::var("BATCH_STRIDE")?.parse::<usize>()?;
    let h = std::env::var("BATCH_H")?.parse::<f64>()?;
    let sample_count = std::env::var("BATCH_SAMPLE_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let init = std::env::var("BATCH_INIT").unwrap_or_else(|_| "Random".into());
    let initial_state = match init.as_str() {
        "AllUp" => InitialState::AllUp,
        "AllDown" => InitialState::AllDown,
        _ => InitialState::Random,
    };

    let params = SimParams {
        l,
        j,
        bond_p: p,
        sample_count,
        initial_state,
        t_start,
        t_end,
        t_step,
        t_analysis_min: t_start,
        t_analysis_max: t_end,
        tc_min: t_start,
        tc_max: t_end,
        tc_step: 0.0001,
        mc_steps,
        therm_steps,
        stride,
        h,
        use_outlier_filter: std::env::var("BATCH_OUTLIER_FILTER").ok().as_deref() == Some("1"),
    };

    run_headless_single(&params)
}

// ─────────────────────────────────────────────
// Plot generation
// ─────────────────────────────────────────────

fn draw_subplot(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    title: &str,
    y_label: &str,
    temps: &[f64],
    values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let t_min = temps.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).abs() * 0.1 + 1e-10;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18).into_font())
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, (y_min - y_pad)..(y_max + y_pad))?;

    chart
        .configure_mesh()
        .x_desc("Temperature T")
        .y_desc(y_label)
        .draw()?;

    chart.draw_series(LineSeries::new(
        temps.iter().zip(values.iter()).map(|(&t, &v)| (t, v)),
        &BLUE,
    ))?;

    chart.draw_series(
        temps.iter().zip(values.iter()).map(|(&t, &v)| Circle::new((t, v), 4, BLUE.filled())),
    )?;

    Ok(())
}

fn save_overview_to_path(
    results: &[SimResult],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));

    let temps:  Vec<f64> = results.iter().map(|r| r.temperature).collect();
    let e_vals: Vec<f64> = results.iter().map(|r| r.mean_e).collect();
    let m_vals: Vec<f64> = results.iter().map(|r| r.mean_m).collect();
    let c_vals: Vec<f64> = results.iter().map(|r| r.heat_cap).collect();
    let x_vals: Vec<f64> = results.iter().map(|r| r.susceptibility).collect();

    draw_subplot(&areas[0], "Mean Energy per Spin",         "<E>/N",  &temps, &e_vals)?;
    draw_subplot(&areas[1], "Mean |Magnetization| per Spin","<|M|>/N",&temps, &m_vals)?;
    draw_subplot(&areas[2], "Heat Capacity per Spin",       "C",      &temps, &c_vals)?;
    draw_subplot(&areas[3], "Magnetic Susceptibility",      "chi",    &temps, &x_vals)?;

    root.present()?;
    Ok(())
}

fn save_bond_sample(params: &SimParams, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    if params.bond_p <= 0.0 {
        return Ok(());
    }
    let size = params.l;
    if size == 0 {
        return Ok(());
    }

    let samples = params.sample_count.max(1);
    let mut rng = rand::thread_rng();
    let mut neg_horiz = vec![vec![0usize; size]; size];
    let mut neg_vert = vec![vec![0usize; size]; size];

    for _ in 0..samples {
        let (j_h, j_v) = IsingModel::build_bonds(size, params.j, params.bond_p, &mut rng);
        for i in 0..size {
            for j in 0..size {
                if j_h[i][j] < 0.0 {
                    neg_horiz[i][j] += 1;
                }
                if j_v[i][j] < 0.0 {
                    neg_vert[i][j] += 1;
                }
            }
        }
    }

    let filename = format!("{}/bond_sample.png", dir);
    let root = BitMapBackend::new(&filename, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));

    let draw_heatmap = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                        title: &str,
                        counts: &Vec<Vec<usize>>|
     -> Result<(), Box<dyn std::error::Error>> {
        let n = size as i32;
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 18).into_font())
            .margin(10)
            .build_cartesian_2d(0..n, 0..n)?;

        chart.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

        chart.draw_series((0..size).flat_map(|i| (0..size).map(move |j| (i, j))).map(
            |(i, j)| {
                let frac = counts[i][j] as f64 / samples as f64;
                let intensity = (frac * 255.0).round().clamp(0.0, 255.0) as u8;
                let color = RGBColor(255, 255 - intensity, 255 - intensity);
                Rectangle::new(
                    [(j as i32, i as i32), (j as i32 + 1, i as i32 + 1)],
                    color.filled(),
                )
            },
        ))?;

        Ok(())
    };

    draw_heatmap(
        &areas[0],
        "Horizontal bonds: fraction of -J over samples",
        &neg_horiz,
    )?;
    draw_heatmap(
        &areas[1],
        "Vertical bonds: fraction of -J over samples",
        &neg_vert,
    )?;
    root.present()?;
    Ok(())
}

fn save_plots(params: &SimParams, results: &[SimResult]) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let dir = format!("data/ising_results_{}", timestamp);
    std::fs::create_dir_all(&dir)?;
    let filename = format!("{}/ising_results.png", dir);
    save_overview_to_path(results, &filename)?;

    save_bond_sample(params, &dir)?;

    let csv_path = format!("{}/ising_results_scan.csv", dir);
    {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(csv_path)?;
        writeln!(file, "temperature,e_per_spin,m_abs_per_spin,c_per_spin,susceptibility")?;
        for r in results {
            writeln!(
                file,
                "{:.8},{:.8},{:.8},{:.8},{:.8}",
                r.temperature, r.mean_e, r.mean_m, r.heat_cap, r.susceptibility
            )?;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────
// TUI drawing
// ─────────────────────────────────────────────

fn draw_setup(f: &mut ratatui::Frame<'_>, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(11),
            Constraint::Length(3),
        ])
        .split(f.area());

    // Header / controls
    let header = Paragraph::new(
        "2D Ising Model — Parameter Setup\n\
         \u{2191}/\u{2193} navigate fields   type to edit   Backspace delete   \u{2190}/\u{2192} cycle Initial State\n\
         Enter: run simulation    c: copy params from previous run    q: quit"
    )
    .block(Block::default().borders(Borders::ALL).title("Controls"))
    .style(Style::default().fg(TuiColor::Cyan));
    f.render_widget(header, outer[0]);

    let param_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(outer[1]);

    let bonds_info = if let (Ok(l), Ok(p)) = (
        app.field_buffers[FIELD_L].trim().parse::<usize>(),
        app.field_buffers[FIELD_P].trim().parse::<f64>(),
    ) {
        if l > 0 {
            let total_bonds = 2usize.saturating_mul(l).saturating_mul(l);
            let neg_target = (p.max(0.0).min(1.0) * total_bonds as f64).round() as usize;
            Some((total_bonds, neg_target))
        } else {
            None
        }
    } else {
        None
    };

    let model_fields = [
        (FIELD_L, "Lattice Size L"),
        (FIELD_J, "Interaction J"),
        (FIELD_P, "Bond disorder p"),
        (FIELD_INIT, "Initial State"),
        (FIELD_H, "External Field H"),
    ];

    let scan_fields = [
        (FIELD_T_START, "T start"),
        (FIELD_T_END, "T end"),
        (FIELD_T_STEP, "T step"),
        (FIELD_TC_STEP, "Tc_step"),
    ];

    let mc_fields = [
        (FIELD_MC_STEPS, "MC Steps"),
        (FIELD_THERM, "Therm Steps (default: MC/2)"),
        (FIELD_STRIDE, "Stride"),
        (FIELD_SAMPLE_COUNT, "Disorder samples (p>0)"),
    ];

    let build_rows = |fields: &[(usize, &str)], app: &App| {
        fields
            .iter()
            .map(|(idx, name)| {
                let selected = *idx == app.selected_field;
                let value = if *idx == FIELD_INIT {
                    if selected {
                        format!("[{}]  <- / ->", app.initial_state.label())
                    } else {
                        format!("[{}]", app.initial_state.label())
                    }
                } else if selected {
                    format!("{}_", app.field_buffers[*idx])
                } else {
                    app.field_buffers[*idx].clone()
                };

                let style = if selected {
                    Style::default().fg(TuiColor::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(TuiColor::White)
                };

                Row::new(vec![
                    Cell::from((*name).to_string()).style(style),
                    Cell::from(value).style(style),
                ])
            })
            .collect::<Vec<Row>>()
    };

    let mut model_rows = build_rows(&model_fields, app);
    let bonds_row_style = Style::default().fg(TuiColor::Magenta).add_modifier(Modifier::BOLD);
    let bonds_text = if let Some((total_bonds, neg_target)) = bonds_info {
        format!(
            "Total bonds = {},  -J bonds = {}",
            total_bonds, neg_target
        )
    } else {
        "Total bonds = N/A,  -J bonds = N/A".to_string()
    };
    model_rows.push(Row::new(vec![
        Cell::from("Bond summary").style(bonds_row_style),
        Cell::from(bonds_text).style(bonds_row_style),
    ]));
    let scan_rows = build_rows(&scan_fields, app);
    let mut mc_rows = build_rows(&mc_fields, app);
    let filter_style = Style::default()
        .fg(TuiColor::Magenta)
        .add_modifier(Modifier::BOLD);
    let filter_text = if app.outlier_filter { "open" } else { "off" };
    mc_rows.push(Row::new(vec![
        Cell::from("Outlier filter").style(filter_style),
        Cell::from(format!("{} (press 'o' to toggle)", filter_text)).style(filter_style),
    ]));

    let model_table = Table::new(model_rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("Model Parameters"))
        .column_spacing(2);
    f.render_widget(model_table, param_areas[0]);

    let scan_table = Table::new(scan_rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("Current Scan Parameters"))
        .column_spacing(2);
    f.render_widget(scan_table, param_areas[1]);

    let mc_table = Table::new(mc_rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("MC Parameters"))
        .column_spacing(2);
    f.render_widget(mc_table, param_areas[2]);

    let footer_text = app
        .error_msg
        .as_deref()
        .unwrap_or("Fill in parameters and press Enter to run the simulation.");
    let footer_style = if app.error_msg.is_some() {
        Style::default().fg(TuiColor::Red)
    } else {
        Style::default().fg(TuiColor::Gray)
    };
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL).title("Messages"))
        .style(footer_style);
    f.render_widget(footer, outer[2]);
}

fn draw_running_sweep(
    f: &mut ratatui::Frame<'_>,
    current_t: f64,
    t_end: f64,
    done: usize,
    total: usize,
) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(f.area());

    let pct = if total > 0 { (done * 100) / total } else { 0 };
    let text = format!(
        "Running simulation — please wait...\n\n\
         Current temperature : {:.4}\n\
         Target T_end        : {:.4}\n\
         Progress            : {}/{} temperatures  ({}%)",
        current_t, t_end, done, total, pct
    );
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Simulation Running"))
        .style(Style::default().fg(TuiColor::Green));
    f.render_widget(para, layout[0]);

    let ratio = if total > 0 { (done as f64 / total as f64).clamp(0.0, 1.0) } else { 0.0 };
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Progress"))
        .gauge_style(Style::default().fg(TuiColor::Green).bg(TuiColor::Black))
        .ratio(ratio);
    f.render_widget(gauge, layout[1]);
}

fn draw_step1_summary(f: &mut ratatui::Frame<'_>, app: &App) {
    let area = f.area();
    let text = if let (Some(intervals), Some(params)) = (&app.auto_intervals, &app.sim_params) {
        let p = &intervals.primary;
        let s = intervals.secondary.as_ref();
        let c_peak = intervals
            .c_peak_t
            .map(|t| format!("{:.6}", t))
            .unwrap_or_else(|| "N/A".to_string());
        let chi_peak = intervals
            .chi_peak_t
            .map(|t| format!("{:.6}", t))
            .unwrap_or_else(|| "N/A".to_string());
        let m_peak = intervals
            .m_slope_peak_t
            .map(|t| format!("{:.6}", t))
            .unwrap_or_else(|| "N/A".to_string());
        let filter_label = if params.use_outlier_filter { "open" } else { "off" };
        if let Some(sec) = s {
            format!(
                "Step 1 complete.\n\n\
                 Model parameters:\n\
                 L = {l}, J = {j}, p = {pval}, H = {h}, Init = {init}\n\
                 T scan: start = {ts:.6}, end = {te:.6}, step = {dt:.6}\n\
                 Tc step = {dtc:.6}\n\
                 Outlier filter: {filter}\n\n\
                 Critical points (from Step 1):\n\
                 C(T) peak at T = {c_peak}\n\
                 χ(T) peak at T = {chi_peak}\n\
                 |dm/dT| max at T = {m_peak}\n\n\
                 Candidate 1 (primary):\n\
                 T window:  [{p_ta_min:.6}, {p_ta_max:.6}]\n\
                 Tc window: [{p_tc_min:.6}, {p_tc_max:.6}]\n\n\
                 Candidate 2 (secondary):\n\
                 T window:  [{s_ta_min:.6}, {s_ta_max:.6}]\n\
                 Tc window: [{s_tc_min:.6}, {s_tc_max:.6}]\n\n\
                 Press '1' to use Candidate 1.\n\
                 Press '2' to use Candidate 2.\n\
                 Press '3' for manual edit (Candidate 3).\n\
                 Press 'q' to quit.",
                l = params.l,
                j = params.j,
                pval = params.bond_p,
                h = params.h,
                init = params.initial_state.label(),
                ts = params.t_start,
                te = params.t_end,
                dt = params.t_step,
                dtc = params.tc_step,
                filter = filter_label,
                c_peak = c_peak,
                chi_peak = chi_peak,
                m_peak = m_peak,
                p_ta_min = p.t_envelope_min,
                p_ta_max = p.t_envelope_max,
                p_tc_min = p.tc_overlap_min,
                p_tc_max = p.tc_overlap_max,
                s_ta_min = sec.t_envelope_min,
                s_ta_max = sec.t_envelope_max,
                s_tc_min = sec.tc_overlap_min,
                s_tc_max = sec.tc_overlap_max,
            )
        } else {
            format!(
                "Step 1 complete.\n\n\
                 Model parameters:\n\
                 L = {l}, J = {j}, p = {pval}, H = {h}, Init = {init}\n\
                 T scan: start = {ts:.6}, end = {te:.6}, step = {dt:.6}\n\
                 Tc step = {dtc:.6}\n\
                 Outlier filter: {filter}\n\n\
                 Critical points (from Step 1):\n\
                 C(T) peak at T = {c_peak}\n\
                 χ(T) peak at T = {chi_peak}\n\
                 |dm/dT| max at T = {m_peak}\n\n\
                 Candidate (primary):\n\
                 T window:  [{p_ta_min:.6}, {p_ta_max:.6}]\n\
                 Tc window: [{p_tc_min:.6}, {p_tc_max:.6}]\n\n\
                 Press '1' to use this candidate.\n\
                 Press '3' for manual edit (Candidate 3).\n\
                 Press 'q' to quit.",
                l = params.l,
                j = params.j,
                pval = params.bond_p,
                h = params.h,
                init = params.initial_state.label(),
                ts = params.t_start,
                te = params.t_end,
                dt = params.t_step,
                dtc = params.tc_step,
                filter = filter_label,
                c_peak = c_peak,
                chi_peak = chi_peak,
                m_peak = m_peak,
                p_ta_min = p.t_envelope_min,
                p_ta_max = p.t_envelope_max,
                p_tc_min = p.tc_overlap_min,
                p_tc_max = p.tc_overlap_max,
            )
        }
    } else {
        "Step 1 complete, but auto analysis intervals or parameters are not available.\n\nPress 'q' to quit."
            .to_string()
    };
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Auto Analysis Summary"))
        .style(Style::default().fg(TuiColor::Green));
    f.render_widget(para, area);
}

fn draw_load_params(f: &mut ratatui::Frame<'_>, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(3)])
        .split(f.area());

    let header = Paragraph::new(
        "Copy parameters from previous run\n\
         \u{2191}/\u{2193} select run   Enter: load   Esc: back   q: quit",
    )
    .block(Block::default().borders(Borders::ALL).title("Copy Parameters"))
    .style(Style::default().fg(TuiColor::Cyan));
    f.render_widget(header, layout[0]);

    let mut rows = Vec::new();
    if app.saved_runs.is_empty() {
        rows.push(Row::new(vec![Cell::from("No previous runs with summary.txt found under data/")]));
    } else {
        for (idx, (name, _path)) in app.saved_runs.iter().enumerate() {
            let selected = idx == app.saved_run_selected;
            let style = if selected {
                Style::default().fg(TuiColor::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TuiColor::White)
            };
            rows.push(Row::new(vec![Cell::from(name.clone()).style(style)]));
        }
    }

    let table = Table::new(rows, [Constraint::Percentage(100)])
        .block(Block::default().borders(Borders::ALL).title("Available runs"))
        .column_spacing(1);
    f.render_widget(table, layout[1]);

    let footer_text = app
        .error_msg
        .as_deref()
        .unwrap_or("Select a run and press Enter to load its parameters.");
    let footer_style = if app.error_msg.is_some() {
        Style::default().fg(TuiColor::Red)
    } else {
        Style::default().fg(TuiColor::Gray)
    };
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL).title("Messages"))
        .style(footer_style);
    f.render_widget(footer, layout[2]);
}

fn draw_running_tc_scan(
    f: &mut ratatui::Frame<'_>,
    done: usize,
    total: usize,
) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(f.area());

    let pct = if total > 0 { (done * 100) / total } else { 0 };
    let text = format!(
        "Running Tc log-log analysis — please wait...\n\n\
         Progress            : {}/{} Tc candidates  ({}%)",
        done, total, pct
    );
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Tc Scan Running"))
        .style(Style::default().fg(TuiColor::Green));
    f.render_widget(para, layout[0]);

    let ratio = if total > 0 { (done as f64 / total as f64).clamp(0.0, 1.0) } else { 0.0 };
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Progress"))
        .gauge_style(Style::default().fg(TuiColor::Green).bg(TuiColor::Black))
        .ratio(ratio);
    f.render_widget(gauge, layout[1]);
}

fn draw_manual_window_edit(f: &mut ratatui::Frame<'_>, app: &App) {
    let area = f.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(area);

    let state = if let Some(s) = &app.manual_window_state {
        s
    } else {
        return;
    };

    let labels = [
        "T_analysis_min",
        "T_analysis_max",
        "Tc_min",
        "Tc_max",
    ];

    let rows = labels
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let val = &state.fields[i];
            let style = if i == state.selected {
                Style::default().fg(TuiColor::Yellow)
            } else {
                Style::default()
            };
            Row::new(vec![Cell::from(*name), Cell::from(val.clone())]).style(style)
        });

    let table = Table::new(rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Manual Tc Window (Candidate 3)"),
        )
        .column_spacing(2);
    f.render_widget(table, layout[0]);

    let footer_text = app
        .error_msg
        .as_deref()
        .unwrap_or("Use Up/Down to select, type to edit, Enter to run, Esc to go back, 'q' to quit.");
    let footer_style = if app.error_msg.is_some() {
        Style::default().fg(TuiColor::Red)
    } else {
        Style::default().fg(TuiColor::Gray)
    };
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL).title("Manual Edit Help"))
        .style(footer_style);
    f.render_widget(footer, layout[1]);
}

fn draw_done(f: &mut ratatui::Frame<'_>, app: &App) {
    let results_slice: &[SimResult] = app
        .results
        .as_deref()
        .unwrap_or(&[]);
    let t0 = results_slice.first().map(|r| r.temperature).unwrap_or(0.0);
    let t1 = results_slice.last().map(|r| r.temperature).unwrap_or(0.0);
    let text = format!(
        "Simulation complete!\n\n\
         Temperatures computed : {}\n\
         T range               : {:.3} — {:.3}\n\n\
         Results saved to: ising_results_<timestamp>.png\n\n\
         Press 'q' to quit.",
        results_slice.len(),
        t0,
        t1
    );
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Done"))
        .style(Style::default().fg(TuiColor::Green));
    f.render_widget(para, f.area());
}

fn draw_frame(f: &mut ratatui::Frame<'_>, app: &App) {
    match &app.mode {
        AppMode::Setup => draw_setup(f, app),
        AppMode::LoadParams => draw_load_params(f, app),
        AppMode::RunningSweep { current_t, t_end, done, total } => {
            draw_running_sweep(f, *current_t, *t_end, *done, *total)
        }
        AppMode::Step1Summary => draw_step1_summary(f, app),
        AppMode::ManualWindowEdit => draw_manual_window_edit(f, app),
        AppMode::RunningTcScan { done, total } => {
            draw_running_tc_scan(f, *done, *total)
        }
        AppMode::Done => draw_done(f, app),
    }
}

// ─────────────────────────────────────────────
// Event handling
// ─────────────────────────────────────────────

/// Returns Err("quit") to signal a clean exit.
fn handle_key(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<(), String> {
    match &app.mode {
        AppMode::Done => {
            if key == KeyCode::Char('q') {
                return Err("quit".into());
            }
        }
        AppMode::LoadParams => {
            match key {
                KeyCode::Char('q') => return Err("quit".into()),
                KeyCode::Esc => {
                    app.mode = AppMode::Setup;
                    app.error_msg = None;
                }
                KeyCode::Up => {
                    if app.saved_run_selected > 0 {
                        app.saved_run_selected -= 1;
                    }
                    app.error_msg = None;
                }
                KeyCode::Down => {
                    if !app.saved_runs.is_empty() {
                        if app.saved_run_selected + 1 < app.saved_runs.len() {
                            app.saved_run_selected += 1;
                        }
                    }
                    app.error_msg = None;
                }
                KeyCode::Enter => {
                    if app.saved_runs.is_empty() {
                        app.error_msg =
                            Some("No previous runs with summary.txt found under data/".into());
                        return Ok(());
                    }
                    let (_name, path) = app.saved_runs[app.saved_run_selected].clone();
                    match load_params::load_params_from_summary_dir(&path) {
                        Ok(params) => {
                            app.outlier_filter = params.use_outlier_filter;
                            app.sim_params = Some(params.clone());
                            app.initial_state = params.initial_state;
                            app.field_buffers[FIELD_L] = params.l.to_string();
                            app.field_buffers[FIELD_J] = format!("{}", params.j);
                            app.field_buffers[FIELD_P] = format!("{}", params.bond_p);
                            app.field_buffers[FIELD_T_START] = format!("{}", params.t_start);
                            app.field_buffers[FIELD_T_END] = format!("{}", params.t_end);
                            app.field_buffers[FIELD_T_STEP] = format!("{}", params.t_step);
                            app.field_buffers[FIELD_MC_STEPS] = params.mc_steps.to_string();
                            app.field_buffers[FIELD_THERM] = params.therm_steps.to_string();
                            app.field_buffers[FIELD_STRIDE] = params.stride.to_string();
                            app.field_buffers[FIELD_H] = format!("{}", params.h);
                            app.field_buffers[FIELD_TC_STEP] = format!("{}", params.tc_step);
                            app.field_buffers[FIELD_SAMPLE_COUNT] = params.sample_count.to_string();
                            app.selected_field = 0;
                            app.error_msg = None;
                            app.mode = AppMode::Setup;
                        }
                        Err(e) => {
                            app.error_msg = Some(e);
                        }
                    }
                }
                _ => {}
            }
        }
        AppMode::RunningSweep { .. } | AppMode::RunningTcScan { .. } => {}
        AppMode::Step1Summary => {
            match key {
                KeyCode::Char('q') => return Err("quit".into()),
                KeyCode::Char('1') | KeyCode::Char('2') => {
                    let use_secondary = matches!(key, KeyCode::Char('2'));
                    let params = match app.sim_params.clone() {
                        Some(p) => p,
                        None => {
                            app.error_msg = Some("No simulation parameters available for Tc scan".into());
                            return Ok(());
                        }
                    };
                    let intervals = match app.auto_intervals.clone() {
                        Some(v) => v,
                        None => {
                            app.error_msg = Some("No auto analysis intervals available for Tc scan".into());
                            return Ok(());
                        }
                    };
                    let window = if use_secondary {
                        match intervals.secondary {
                            Some(w) => w,
                            None => intervals.primary,
                        }
                    } else {
                        intervals.primary
                    };
                    let results_slice: Vec<SimResult> = match app.results.clone() {
                        Some(v) => v,
                        None => {
                            app.error_msg = Some("No simulation results available for Tc scan".into());
                            return Ok(());
                        }
                    };
                    let mut params_for_tc = params;
                    params_for_tc.t_analysis_min = window.t_envelope_min;
                    params_for_tc.t_analysis_max = window.t_envelope_max;
                    params_for_tc.tc_min = window.tc_overlap_min;
                    params_for_tc.tc_max = window.tc_overlap_max;

                    let total_steps = {
                        let n_steps = ((params_for_tc.tc_max - params_for_tc.tc_min)
                            / params_for_tc.tc_step)
                            .round() as usize;
                        n_steps + 1
                    };

                    app.mode = AppMode::RunningTcScan { done: 0, total: total_steps };
                    let _ = terminal.draw(|f| draw_frame(f, app));

                    match run_loglog_analysis(&params_for_tc, &results_slice, "data", |done, total| {
                        app.mode = AppMode::RunningTcScan { done, total };
                        let _ = terminal.draw(|f| draw_frame(f, app));
                    }) {
                        Ok(()) => {
                            app.mode = AppMode::Done;
                            app.results = Some(results_slice);
                        }
                        Err(e) => {
                            app.mode = AppMode::Setup;
                            app.error_msg = Some(format!("Log-log analysis error: {}", e));
                        }
                    }
                }
                KeyCode::Char('3') => {
                    let intervals = match app.auto_intervals.clone() {
                        Some(v) => v,
                        None => {
                            app.error_msg = Some("No auto analysis intervals available for manual edit".into());
                            return Ok(());
                        }
                    };
                    let base = intervals.primary;
                    let fields = [
                        format!("{:.6}", base.t_envelope_min),
                        format!("{:.6}", base.t_envelope_max),
                        format!("{:.6}", base.tc_overlap_min),
                        format!("{:.6}", base.tc_overlap_max),
                    ];
                    app.manual_window_state = Some(ManualWindowEditState {
                        fields,
                        selected: 0,
                    });
                    app.error_msg = None;
                    app.mode = AppMode::ManualWindowEdit;
                }
                _ => {}
            }
        }
        AppMode::ManualWindowEdit => {
            match key {
                KeyCode::Char('q') => return Err("quit".into()),
                KeyCode::Esc => {
                    app.manual_window_state = None;
                    app.error_msg = None;
                    app.mode = AppMode::Step1Summary;
                }
                KeyCode::Up => {
                    if let Some(state) = &mut app.manual_window_state {
                        if state.selected > 0 {
                            state.selected -= 1;
                        }
                    }
                    app.error_msg = None;
                }
                KeyCode::Down => {
                    if let Some(state) = &mut app.manual_window_state {
                        if state.selected < 3 {
                            state.selected += 1;
                        }
                    }
                    app.error_msg = None;
                }
                KeyCode::Backspace => {
                    if let Some(state) = &mut app.manual_window_state {
                        let idx = state.selected;
                        state.fields[idx].pop();
                    }
                    app.error_msg = None;
                }
                KeyCode::Char(c) => {
                    if let Some(state) = &mut app.manual_window_state {
                        let idx = state.selected;
                        state.fields[idx].push(c);
                    }
                    app.error_msg = None;
                }
                KeyCode::Enter => {
                    let state = match &app.manual_window_state {
                        Some(s) => s.clone(),
                        None => {
                            app.error_msg = Some("No manual window state".into());
                            return Ok(());
                        }
                    };
                    let parse_f = |s: &str| -> Result<f64, String> {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| format!("Invalid number '{}'", s))
                    };
                    let t_min = parse_f(&state.fields[0])?;
                    let t_max = parse_f(&state.fields[1])?;
                    let tc_min = parse_f(&state.fields[2])?;
                    let tc_max = parse_f(&state.fields[3])?;
                    if t_min >= t_max {
                        app.error_msg = Some("T_analysis_min must be < T_analysis_max".into());
                        return Ok(());
                    }
                    if tc_min >= tc_max {
                        app.error_msg = Some("Tc_min must be < Tc_max".into());
                        return Ok(());
                    }

                    let params = match app.sim_params.clone() {
                        Some(p) => p,
                        None => {
                            app.error_msg =
                                Some("No simulation parameters available for Tc scan".into());
                            return Ok(());
                        }
                    };
                    let results_slice: Vec<SimResult> = match app.results.clone() {
                        Some(v) => v,
                        None => {
                            app.error_msg =
                                Some("No simulation results available for Tc scan".into());
                            return Ok(());
                        }
                    };

                    let mut params_for_tc = params;
                    params_for_tc.t_analysis_min = t_min;
                    params_for_tc.t_analysis_max = t_max;
                    params_for_tc.tc_min = tc_min;
                    params_for_tc.tc_max = tc_max;

                    let total_steps = {
                        let n_steps = ((params_for_tc.tc_max - params_for_tc.tc_min)
                            / params_for_tc.tc_step)
                            .round() as usize;
                        n_steps + 1
                    };

                    app.mode = AppMode::RunningTcScan { done: 0, total: total_steps };
                    let _ = terminal.draw(|f| draw_frame(f, app));

                    match run_loglog_analysis(&params_for_tc, &results_slice, "data", |done, total| {
                        app.mode = AppMode::RunningTcScan { done, total };
                        let _ = terminal.draw(|f| draw_frame(f, app));
                    }) {
                        Ok(()) => {
                            app.mode = AppMode::Done;
                            app.results = Some(results_slice);
                        }
                        Err(e) => {
                            app.mode = AppMode::Setup;
                            app.error_msg =
                                Some(format!("Log-log analysis error: {}", e));
                        }
                    }
                }
                _ => {}
            }
        }
        AppMode::Setup => {
            match key {
                KeyCode::Char('q') => return Err("quit".into()),

                KeyCode::Char('c') => {
                    let mut entries = Vec::new();
                    if let Ok(dir) = std::fs::read_dir("data") {
                        for e in dir.flatten() {
                            if let Ok(ft) = e.file_type() {
                                if ft.is_dir() {
                                    let path = e.path();
                                    let summary = path.join("summary.txt");
                                    if summary.is_file() {
                                        let name = path
                                            .file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or_default()
                                            .to_string();
                                        let path_str = path.to_string_lossy().to_string();
                                        entries.push((name, path_str));
                                    }
                                }
                            }
                        }
                    }
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    app.saved_run_selected = 0;
                    app.saved_runs = entries;
                    app.mode = AppMode::LoadParams;
                    app.error_msg = None;
                }

                KeyCode::Up => {
                    let pos = FIELD_ORDER
                        .iter()
                        .position(|&f| f == app.selected_field)
                        .unwrap_or(0);
                    let new_pos = pos.saturating_sub(1);
                    app.selected_field = FIELD_ORDER[new_pos];
                    app.error_msg = None;
                }
                KeyCode::Down => {
                    let pos = FIELD_ORDER
                        .iter()
                        .position(|&f| f == app.selected_field)
                        .unwrap_or(0);
                    let new_pos = (pos + 1).min(NUM_FIELDS - 1);
                    app.selected_field = FIELD_ORDER[new_pos];
                    app.error_msg = None;
                }

                KeyCode::Left => {
                    if app.selected_field == FIELD_INIT {
                        app.initial_state = app.initial_state.prev();
                    }
                }
                KeyCode::Right => {
                    if app.selected_field == FIELD_INIT {
                        app.initial_state = app.initial_state.next();
                    }
                }

                KeyCode::Char('o') => {
                    app.outlier_filter = !app.outlier_filter;
                    app.error_msg = None;
                }

                KeyCode::Char(c) if app.selected_field != FIELD_INIT => {
                    app.field_buffers[app.selected_field].push(c);
                    app.error_msg = None;
                }
                KeyCode::Backspace if app.selected_field != FIELD_INIT => {
                    app.field_buffers[app.selected_field].pop();
                    app.error_msg = None;
                }

                KeyCode::Enter => {
                    match app.parse_params() {
                        Err(msg) => {
                            app.error_msg = Some(msg);
                        }
                        Ok(params) => {
                            app.error_msg = None;
                            app.sim_params = Some(params.clone());
                            let t_end = params.t_end;
                            let results = run_sweep(&params, |cur_t, done, total| {
                                app.mode = AppMode::RunningSweep {
                                    current_t: cur_t,
                                    t_end,
                                    done,
                                    total,
                                };
                                let _ = terminal.draw(|f| draw_frame(f, app));
                            });
                            match save_plots(&params, &results) {
                                Ok(()) => {
                                    let temps: Vec<f64> =
                                        results.iter().map(|r| r.temperature).collect();
                                    let c_vals: Vec<f64> =
                                        results.iter().map(|r| r.heat_cap).collect();
                                    let x_vals: Vec<f64> =
                                        results.iter().map(|r| r.susceptibility).collect();
                                    let m_vals: Vec<f64> =
                                        results.iter().map(|r| r.mean_m).collect();
                                    match autoanalysis::compute_intervals(
                                        &temps,
                                        &c_vals,
                                        &x_vals,
                                        &m_vals,
                                    ) {
                                        Ok(intervals) => {
                                            app.auto_intervals = Some(intervals);
                                            app.results = Some(results);
                                            app.mode = AppMode::Step1Summary;
                                        }
                                        Err(e) => {
                                            app.mode = AppMode::Setup;
                                            app.error_msg =
                                                Some(format!("Auto analysis error: {}", e));
                                        }
                                    }
                                }
                                Err(e) => {
                                    app.mode = AppMode::Setup;
                                    app.error_msg = Some(format!("Plot error: {}", e));
                                }
                            }
                        }
                    }
                }

                _ => {}
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), String> {
    loop {
        terminal.draw(|f| draw_frame(f, app)).map_err(|e| e.to_string())?;

        if event::poll(std::time::Duration::from_millis(50)).map_err(|e| e.to_string())? {
            if let Ok(Event::Key(key)) = event::read() {
                handle_key(app, key.code, terminal)?;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("BATCH_MODE").ok().as_deref() == Some("1") {
        return run_batch_from_env();
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = &res {
        if e != "quit" {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
