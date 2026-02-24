use chrono::Local;
use plotters::prelude::*;
use rand::{Rng, seq::SliceRandom};
use std::cmp::Ordering;
use std::fs;
use std::io::{self, Write};
use std::thread;

#[derive(Clone, Copy)]
enum InitialState {
    Random,
    AllUp,
    AllDown,
}

impl InitialState {
    fn from_str(s: &str) -> Self {
        match s.trim() {
            "AllUp" | "up" | "U" => InitialState::AllUp,
            "AllDown" | "down" | "D" => InitialState::AllDown,
            _ => InitialState::Random,
        }
    }
    fn label(self) -> &'static str {
        match self {
            InitialState::Random => "Random",
            InitialState::AllUp => "AllUp",
            InitialState::AllDown => "AllDown",
        }
    }
}

#[derive(Clone)]
struct Params {
    l: usize,
    j: f64,
    h: f64,
    p_start: f64,
    p_end: f64,
    p_step: f64,
    t_start_curve2: f64,
    t_end_curve2: f64,
    t_step_curve2: f64,
    mc_steps: usize,
    therm_steps: usize,
    stride: usize,
    disorder_samples: usize,
    initial_state: InitialState,
}

fn read_line(prompt: &str, default: &str) -> String {
    print!("{} [{}]: ", prompt, default);
    let _ = io::stdout().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_err() {
        return default.to_string();
    }
    let s = line.trim();
    if s.is_empty() {
        default.to_string()
    } else {
        s.to_string()
    }
}

fn parse_usize(input: &str, default: usize) -> usize {
    input.trim().parse::<usize>().unwrap_or(default)
}

fn parse_f64(input: &str, default: f64) -> f64 {
    input.trim().parse::<f64>().unwrap_or(default)
}

fn read_params() -> Params {
    let l = parse_usize(&read_line("Lattice size L", "32"), 32);
    let j = parse_f64(&read_line("Interaction J", "1.0"), 1.0);
    let h = parse_f64(&read_line("External field h", "0.0"), 0.0);
    let init_str = read_line("Initial state (Random/AllUp/AllDown)", "Random");
    let initial_state = InitialState::from_str(&init_str);

    let p_start = parse_f64(&read_line("p start", "0.0"), 0.0);
    let p_end = parse_f64(&read_line("p end", "0.5"), 0.5);
    let p_step = parse_f64(&read_line("p step", "0.05"), 0.05);

    let t_start_curve2 = parse_f64(&read_line("Curve-2 T start", "0.01"), 0.01);
    let t_end_curve2 = parse_f64(&read_line("Curve-2 T end", "4.01"), 4.01);
    let t_step_curve2 = parse_f64(&read_line("Curve-2 T step", "0.2"), 0.2);

    let mc_steps = parse_usize(&read_line("MC steps", "10000"), 10000);
    let therm_steps = parse_usize(&read_line("Therm steps", "5000"), 5000);
    let stride = parse_usize(&read_line("Stride", "10"), 10);
    let disorder_samples = parse_usize(&read_line("Disorder samples", "8"), 8);

    Params {
        l,
        j,
        h,
        p_start,
        p_end,
        p_step,
        t_start_curve2,
        t_end_curve2,
        t_step_curve2,
        mc_steps,
        therm_steps,
        stride,
        disorder_samples,
        initial_state,
    }
}

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

    fn new_with_state(size: usize, j: f64, p: f64, h: f64, temperature: f64, initial_state: InitialState, rng: &mut impl Rng) -> Self {
        let spins = match initial_state {
            InitialState::Random => {
                (0..size)
                    .map(|_| {
                        (0..size)
                            .map(|_| if rng.gen::<f64>() < 0.5 { -1i8 } else { 1 })
                            .collect()
                    })
                    .collect()
            }
            InitialState::AllUp => vec![vec![1i8; size]; size],
            InitialState::AllDown => vec![vec![-1i8; size]; size],
        };
        let (j_horiz, j_vert) = Self::build_bonds(size, j, p, rng);
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
                let spin = self.spins[i][jc] as f64;
                let right = self.spins[i][(jc + 1) % l] as f64;
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
        let i = rng.gen_range(0..self.size);
        let jc = rng.gen_range(0..self.size);
        let old_e = self.energy_at_site(i, jc);
        self.spins[i][jc] = -self.spins[i][jc];
        let new_e = self.energy_at_site(i, jc);
        let delta_e = new_e - old_e;
        if delta_e > 0.0 && rng.gen::<f64>() >= (-delta_e / self.temperature).exp() {
            self.spins[i][jc] = -self.spins[i][jc];
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn ferromagnet_energy(j: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let beta = 1.0 / t;
    let z = 4.0;
    let x = beta * j;
    -(z * 0.5) * j * x.tanh()
}

fn measure_energy_per_spin(params: &Params, p: f64, t: f64, rng: &mut impl Rng) -> f64 {
    let n = (params.l * params.l) as f64;
    let samples = params.disorder_samples.max(1);
    let mut acc = 0.0;

    for _ in 0..samples {
        let mut model = IsingModel::new_with_state(params.l, params.j, p, params.h, t, params.initial_state, rng);

        for _ in 0..params.therm_steps {
            for _ in 0..params.l * params.l {
                model.metropolis_step(rng);
            }
        }

        let mut e_samples = Vec::new();
        for step in 0..params.mc_steps {
            for _ in 0..params.l * params.l {
                model.metropolis_step(rng);
            }
            if step % params.stride == 0 {
                e_samples.push(model.total_energy());
            }
        }

        let mean_e = mean(&e_samples) / n;
        acc += mean_e;
    }

    acc / samples as f64
}

fn compute_curve1(params: &Params) -> Vec<(f64, f64)> {
    let mut res = Vec::new();
    if params.p_step <= 0.0 {
        return res;
    }
    let mut p = params.p_start;
    while p <= params.p_end + 1e-9 {
        if p < 1.0 {
            let t = if p <= 0.0 {
                0.0
            } else {
                let ratio = (1.0 - p) / p;
                if ratio <= 0.0 {
                    p += params.p_step;
                    continue;
                }
                let denom = ratio.ln();
                if denom.abs() <= 1e-8 {
                    p += params.p_step;
                    continue;
                }
                2.0 * params.j / denom
            };
            if t.is_finite() && t >= 0.0 {
                res.push((p.max(0.0), t));
            }
        }
        p += params.p_step;
    }
    res
}

fn compute_curve2(params: &Params) -> Vec<(f64, f64)> {
    let mut res = Vec::new();
    if params.p_step <= 0.0 || params.t_step_curve2 <= 0.0 {
        return res;
    }
    let mut rng = rand::thread_rng();
    let mut debug_info: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();

    let mut total_points = 0usize;
    let mut p = params.p_start;
    while p <= params.p_end + 1e-9 {
        let mut t = params.t_start_curve2 + params.t_step_curve2;
        while t < params.t_end_curve2 - 1e-9 {
            total_points += 1;
            t += params.t_step_curve2;
        }
        p += params.p_step;
    }

    if total_points == 0 {
        return res;
    }

    println!(
        "Curve-2: starting MC scan over {} (p, T) points...",
        total_points
    );
    let mut done_points = 0usize;

    let mut p = params.p_start;
    while p <= params.p_end + 1e-9 {
        let mut best_t = None;
        let mut best_diff = f64::INFINITY;
        let mut candidates: Vec<(f64, f64)> = Vec::new();

        let mut t = params.t_start_curve2 + params.t_step_curve2;
        while t < params.t_end_curve2 - 1e-9 {
            let u_mc = measure_energy_per_spin(params, p, t, &mut rng);
            let u_th = ferromagnet_energy(params.j, t);
            let diff = (u_mc - u_th).abs();
            candidates.push((t, diff));
            match best_t {
                None => {
                    best_t = Some(t);
                    best_diff = diff;
                }
                Some(_) => {
                    if diff < best_diff {
                        best_diff = diff;
                        best_t = Some(t);
                    }
                }
            }

            done_points += 1;
            if done_points % 10 == 0 || done_points == total_points {
                let pct = 100.0 * done_points as f64 / total_points as f64;
                println!(
                    "Curve-2 progress: {:>6.2}% ({}/{})",
                    pct, done_points, total_points
                );
            }

            t += params.t_step_curve2;
        }

        if let Some(t_star) = best_t {
            res.push((p, t_star));
        }
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let top_k = candidates.into_iter().take(3).collect::<Vec<_>>();
        debug_info.push((p, top_k));

        p += params.p_step;
    }

    res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    if !debug_info.is_empty() {
        let out_root = "data3_nishimoriline";
        let _ = fs::create_dir_all(out_root);
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let path = format!("{}/nishimori_debug_{}.txt", out_root, timestamp);
        if let Ok(mut f) = fs::File::create(&path) {
            let _ = writeln!(f, "Debug candidates for Curve-2");
            for (p_val, list) in debug_info {
                let _ = writeln!(f, "p = {}", p_val);
                for (idx, (t, d)) in list.iter().enumerate() {
                    let _ = writeln!(f, "  candidate {}: T = {}, delta = {}", idx + 1, t, d);
                }
            }
        }
        println!("Curve-2 debug candidates written to {}", path);
    }
    res
}

fn draw_tp_plot(
    curve1: &[(f64, f64)],
    curve2: &[(f64, f64)],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if curve1.is_empty() && curve2.is_empty() {
        return Ok(());
    }

    let mut p_vals = Vec::new();
    let mut t_vals = Vec::new();

    for (p, t) in curve1 {
        p_vals.push(*p);
        t_vals.push(*t);
    }
    for (p, t) in curve2 {
        p_vals.push(*p);
        t_vals.push(*t);
    }

    let p_min = p_vals
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let p_max = p_vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_max = t_vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let p_pad = (p_max - p_min).abs() * 0.05 + 1e-6;
    let t_pad = t_max.abs() * 0.05 + 1e-6;

    let root = BitMapBackend::new(path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("T vs disorder p (Nishimori line)", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((p_min - p_pad)..(p_max + p_pad), 0.0..(t_max + t_pad))?;

    chart
        .configure_mesh()
        .x_desc("p")
        .y_desc("T")
        .draw()?;

    if !curve2.is_empty() {
        chart
            .draw_series(curve2.iter().map(|(p, t)| Circle::new((*p, *t), 4, BLUE.filled())))?
            .label("Curve-2 from u_MC(T,p)")
            .legend(|(x, y)| Circle::new((x + 10, y), 4, BLUE.filled()));
    }

    if !curve1.is_empty() {
        chart
            .draw_series(LineSeries::new(
                curve1.iter().map(|(p, t)| (*p, *t)),
                &RED,
            ))?
            .label("T(p) = 2J / ln((1-p)/p)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    }

    if !curve1.is_empty() || !curve2.is_empty() {
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

fn investigate_p_values(
    params: &Params,
    curve2: &[(f64, f64)],
    out_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if curve2.is_empty() {
        return Ok(());
    }

    println!();
    println!("Select p values to investigate.");
    println!("Use 'y' to mark a p for investigation, 'n' to skip.");
    println!("For each p, you will be asked once.");

    let mut selected = Vec::new();
    for (p, t_star) in curve2 {
        loop {
            print!("Investigate p = {:.6} (best T = {:.6}) ? [y/n]: ", p, t_star);
            let _ = io::stdout().flush();
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                break;
            }
            let s = line.trim().to_lowercase();
            if s == "y" {
                selected.push(*p);
                break;
            } else if s == "n" {
                break;
            } else {
                println!("Please enter 'y' or 'n'.");
            }
        }
    }

    if selected.is_empty() {
        return Ok(());
    }

    println!();
    println!(
        "Launching investigations for {} p values in parallel...",
        selected.len()
    );

    let mut handles = Vec::new();
    for p_val in selected {
        let params_cloned = params.clone();
        let out_dir_owned = out_dir.to_string();
        let handle = thread::spawn(move || {
            run_investigation_for_p(params_cloned, p_val, out_dir_owned);
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    Ok(())
}

fn run_investigation_for_p(params: Params, p_val: f64, out_dir: String) {
    println!("Starting investigation for p = {:.6}", p_val);
    let mut rng = rand::thread_rng();
    let mut temps = Vec::new();
    let mut e_vals = Vec::new();
    let mut m_vals = Vec::new();
    let mut c_vals = Vec::new();
    let mut x_vals = Vec::new();

    let mut t = params.t_start_curve2;
    while t <= params.t_end_curve2 + 1e-9 {
        let n = (params.l * params.l) as f64;
        let samples = params.disorder_samples.max(1);
        let mut e_acc = 0.0;
        let mut m_acc = 0.0;
        let mut c_acc = 0.0;
        let mut x_acc = 0.0;

        for _ in 0..samples {
            let mut model = IsingModel::new_with_state(
                params.l,
                params.j,
                p_val,
                params.h,
                t,
                params.initial_state,
                &mut rng,
            );

            for _ in 0..params.therm_steps {
                for _ in 0..params.l * params.l {
                    model.metropolis_step(&mut rng);
                }
            }

            let mut e_samples = Vec::new();
            let mut m_samples = Vec::new();
            let mut m_abs_samples = Vec::new();
            for step in 0..params.mc_steps {
                for _ in 0..params.l * params.l {
                    model.metropolis_step(&mut rng);
                }
                if step % params.stride == 0 {
                    let e = model.total_energy();
                    let m = model.total_magnetization() as f64;
                    e_samples.push(e);
                    m_samples.push(m);
                    m_abs_samples.push(m.abs());
                }
            }

            let mean_e = mean(&e_samples) / n;
            let mean_m = mean(&m_abs_samples) / n;
            let var_e = if e_samples.len() < 2 {
                0.0
            } else {
                let m_e = mean(&e_samples);
                e_samples
                    .iter()
                    .map(|x| (x - m_e) * (x - m_e))
                    .sum::<f64>()
                    / e_samples.len() as f64
            };
            let var_m = if m_samples.len() < 2 {
                0.0
            } else {
                let m_m = mean(&m_samples);
                m_samples
                    .iter()
                    .map(|x| (x - m_m) * (x - m_m))
                    .sum::<f64>()
                    / m_samples.len() as f64
            };

            let heat_cap = var_e / (t * t * n);
            let chi = var_m / (t * n);

            e_acc += mean_e;
            m_acc += mean_m;
            c_acc += heat_cap;
            x_acc += chi;
        }

        let inv = 1.0 / params.disorder_samples.max(1) as f64;
        temps.push(t);
        e_vals.push(e_acc * inv);
        m_vals.push(m_acc * inv);
        c_vals.push(c_acc * inv);
        x_vals.push(x_acc * inv);

        t += params.t_step_curve2;
    }

    let inv_dir = format!("{}/investigation_p_{:.6}", out_dir, p_val);
    if let Err(e) = fs::create_dir_all(&inv_dir) {
        println!(
            "Failed to create investigation dir for p = {:.6}: {}",
            p_val, e
        );
        return;
    }
    let png_path = format!("{}/overview.png", inv_dir);

    let root = BitMapBackend::new(&png_path, (1200, 900)).into_drawing_area();
    if root.fill(&WHITE).is_err() {
        println!(
            "Failed to init drawing area for p = {:.6} (overview).",
            p_val
        );
        return;
    }
    let areas = root.split_evenly((2, 2));

    let draw_subplot = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                        title: &str,
                        y_label: &str,
                        values: &[f64]|
     -> Result<(), Box<dyn std::error::Error>> {
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
            temps.iter().zip(values.iter()).map(|(&tt, &v)| (tt, v)),
            &BLUE,
        ))?;

        chart.draw_series(
            temps
                .iter()
                .zip(values.iter())
                .map(|(&tt, &v)| Circle::new((tt, v), 4, BLUE.filled())),
        )?;

        Ok(())
    };

    if draw_subplot(&areas[0], "Mean Energy per Spin", "<E>/N", &e_vals).is_err()
        || draw_subplot(&areas[1], "Mean |Magnetization| per Spin", "<|M|>/N", &m_vals).is_err()
        || draw_subplot(&areas[2], "Heat Capacity per Spin", "C", &c_vals).is_err()
        || draw_subplot(&areas[3], "Magnetic Susceptibility", "chi", &x_vals).is_err()
    {
        println!("Failed to draw overview for p = {:.6}", p_val);
        return;
    }

    if root.present().is_err() {
        println!("Failed to present overview for p = {:.6}", p_val);
        return;
    }

    let summary_path = format!("{}/summary.txt", inv_dir);
    match fs::File::create(&summary_path) {
        Ok(mut f) => {
            let _ = writeln!(f, "Investigation summary for p = {:.6}", p_val);
            let _ = writeln!(f, "Lattice size L = {}", params.l);
            let _ = writeln!(f, "Interaction J = {}", params.j);
            let _ = writeln!(f, "External field h = {}", params.h);
            let _ = writeln!(f, "Initial state = {}", params.initial_state.label());
            let _ = writeln!(
                f,
                "T range: start = {}, end = {}, step = {}",
                params.t_start_curve2, params.t_end_curve2, params.t_step_curve2
            );
            let _ = writeln!(f, "MC steps = {}", params.mc_steps);
            let _ = writeln!(f, "Therm steps = {}", params.therm_steps);
            let _ = writeln!(f, "Stride = {}", params.stride);
            let _ = writeln!(f, "Disorder samples = {}", params.disorder_samples);
        }
        Err(e) => {
            println!(
                "Failed to write summary for p = {:.6}: {}",
                p_val, e
            );
        }
    }

    println!("Investigation for p = {:.6} finished.", p_val);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Nishimori line analysis: Curve-1 theory vs Curve-2 from fresh MC u_MC(T,p)");
    let params = read_params();

    let curve1 = compute_curve1(&params);
    if curve1.is_empty() {
        println!("No valid Curve-1 points generated. Check p range and step.");
    }

    let curve2 = compute_curve2(&params);

    let out_root = "data3_nishimoriline";
    fs::create_dir_all(out_root)?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let out_dir = format!("{}/nishimori_line_{}", out_root, timestamp);
    fs::create_dir_all(&out_dir)?;

    let png_path = format!("{}/tp_nishimori.png", out_dir);
    draw_tp_plot(&curve1, &curve2, &png_path)?;

    let summary_path = format!("{}/nishimori_summary.txt", out_dir);
    let mut f = fs::File::create(&summary_path)?;
    writeln!(f, "Nishimori line analysis summary")?;
    writeln!(f, "Output directory: {}", out_dir)?;
    writeln!(f)?;
    writeln!(f, "Input parameters:")?;
    writeln!(f, "Lattice size L = {}", params.l)?;
    writeln!(f, "Interaction J = {}", params.j)?;
    writeln!(f, "Initial state = {}", params.initial_state.label())?;
    writeln!(f, "External field h = {}", params.h)?;
    writeln!(f, "p range: start = {}, end = {}, step = {}", params.p_start, params.p_end, params.p_step)?;
    writeln!(
        f,
        "Curve-2 T range: start = {}, end = {}, step = {}",
        params.t_start_curve2, params.t_end_curve2, params.t_step_curve2
    )?;
    writeln!(f, "MC steps = {}", params.mc_steps)?;
    writeln!(f, "Therm steps = {}", params.therm_steps)?;
    writeln!(f, "Stride = {}", params.stride)?;
    writeln!(f, "Disorder samples = {}", params.disorder_samples)?;
    writeln!(f)?;
    writeln!(f, "Curve-1 points: {}", curve1.len())?;
    writeln!(f, "Curve-2 points: {}", curve2.len())?;

    investigate_p_values(&params, &curve2, &out_dir)?;

    println!();
    println!("Analysis written to directory: {}", out_dir);
    println!("T-P plot: {}", png_path);
    println!("Summary: {}", summary_path);

    Ok(())
}
