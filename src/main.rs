use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use plotters::prelude::*;
use rand::Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color as TuiColor, Modifier, Style},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table},
    Terminal,
};
use std::io;

// ─────────────────────────────────────────────
// Ising model core
// ─────────────────────────────────────────────

struct IsingModel {
    spins: Vec<Vec<i8>>,
    size: usize,
    j: f64,
    h: f64,
    temperature: f64,
}

impl IsingModel {
    fn new_random(size: usize, j: f64, h: f64, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        let spins = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| if rng.gen::<f64>() < 0.5 { -1i8 } else { 1 })
                    .collect()
            })
            .collect();
        Self { spins, size, j, h, temperature }
    }

    fn new_all_up(size: usize, j: f64, h: f64, temperature: f64) -> Self {
        Self { spins: vec![vec![1i8; size]; size], size, j, h, temperature }
    }

    fn new_all_down(size: usize, j: f64, h: f64, temperature: f64) -> Self {
        Self { spins: vec![vec![-1i8; size]; size], size, j, h, temperature }
    }

    fn energy_at_site(&self, i: usize, jc: usize) -> f64 {
        let l = self.size;
        let spin = self.spins[i][jc] as f64;
        let top    = self.spins[(i + l - 1) % l][jc] as f64;
        let bottom = self.spins[(i + 1) % l][jc] as f64;
        let left   = self.spins[i][(jc + l - 1) % l] as f64;
        let right  = self.spins[i][(jc + 1) % l] as f64;
        -self.j * spin * (top + bottom + left + right) - self.h * spin
    }

    fn total_energy(&self) -> f64 {
        let l = self.size;
        let mut e = 0.0;
        for i in 0..l {
            for jc in 0..l {
                let spin  = self.spins[i][jc] as f64;
                let right  = self.spins[i][(jc + 1) % l] as f64;
                let bottom = self.spins[(i + 1) % l][jc] as f64;
                e -= self.j * spin * (right + bottom);
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
    initial_state: InitialState,
    t_start: f64,
    t_end: f64,
    t_step: f64,
    mc_steps: usize,
    therm_steps: usize,
    stride: usize,
    h: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        let mc_steps = 10_000;
        Self {
            l: 32,
            j: 1.0,
            initial_state: InitialState::Random,
            t_start: 1.0,
            t_end: 4.0,
            t_step: 0.1,
            mc_steps,
            therm_steps: mc_steps / 2,
            stride: 10,
            h: 0.0,
        }
    }
}

struct SimResult {
    temperature:   f64,
    mean_e:        f64, // <E>/N
    mean_m:        f64, // <|M|>/N
    heat_cap:      f64, // Var(E)/(T²·N)
    susceptibility: f64, // Var(M)/(T·N)
}

// ─────────────────────────────────────────────
// App / TUI state
// ─────────────────────────────────────────────

const FIELD_L:          usize = 0;
const FIELD_J:          usize = 1;
const FIELD_INIT:       usize = 2;
const FIELD_T_START:    usize = 3;
const FIELD_T_END:      usize = 4;
const FIELD_T_STEP:     usize = 5;
const FIELD_MC_STEPS:   usize = 6;
const FIELD_THERM:      usize = 7;
const FIELD_STRIDE:     usize = 8;
const FIELD_H:          usize = 9;
const NUM_FIELDS:       usize = 10;

enum AppMode {
    Setup,
    Running { current_t: f64, t_end: f64, done: usize, total: usize },
    Done(Vec<SimResult>),
}

struct App {
    mode:           AppMode,
    field_buffers:  Vec<String>,
    selected_field: usize,
    initial_state:  InitialState,
    error_msg:      Option<String>,
}

impl App {
    fn new() -> Self {
        let d = SimParams::default();
        let mut b = vec![String::new(); NUM_FIELDS];
        b[FIELD_L]        = d.l.to_string();
        b[FIELD_J]        = format!("{}", d.j);
        b[FIELD_T_START]  = format!("{}", d.t_start);
        b[FIELD_T_END]    = format!("{}", d.t_end);
        b[FIELD_T_STEP]   = format!("{}", d.t_step);
        b[FIELD_MC_STEPS] = d.mc_steps.to_string();
        b[FIELD_THERM]    = d.therm_steps.to_string();
        b[FIELD_STRIDE]   = d.stride.to_string();
        b[FIELD_H]        = format!("{}", d.h);
        // FIELD_INIT uses initial_state directly
        Self {
            mode: AppMode::Setup,
            field_buffers: b,
            selected_field: 0,
            initial_state: d.initial_state,
            error_msg: None,
        }
    }

    fn parse_params(&self) -> Result<SimParams, String> {
        let l = self.field_buffers[FIELD_L].trim().parse::<usize>()
            .map_err(|_| format!("L must be a positive integer, got '{}'", self.field_buffers[FIELD_L]))?;
        if l < 2 { return Err("Lattice size L must be >= 2".into()); }

        let j = self.field_buffers[FIELD_J].trim().parse::<f64>()
            .map_err(|_| format!("J must be a number, got '{}'", self.field_buffers[FIELD_J]))?;

        let t_start = self.field_buffers[FIELD_T_START].trim().parse::<f64>()
            .map_err(|_| format!("T start must be a number, got '{}'", self.field_buffers[FIELD_T_START]))?;
        if t_start <= 0.0 { return Err("T start must be > 0".into()); }

        let t_end = self.field_buffers[FIELD_T_END].trim().parse::<f64>()
            .map_err(|_| format!("T end must be a number, got '{}'", self.field_buffers[FIELD_T_END]))?;
        if t_end < t_start { return Err("T end must be >= T start".into()); }

        let t_step = self.field_buffers[FIELD_T_STEP].trim().parse::<f64>()
            .map_err(|_| format!("T step must be a number, got '{}'", self.field_buffers[FIELD_T_STEP]))?;
        if t_step <= 0.0 { return Err("T step must be > 0".into()); }

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

        Ok(SimParams { l, j, initial_state: self.initial_state, t_start, t_end, t_step, mc_steps, therm_steps, stride, h })
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
        InitialState::Random  => IsingModel::new_random(p.l, p.j, p.h, temperature),
        InitialState::AllUp   => IsingModel::new_all_up(p.l, p.j, p.h, temperature),
        InitialState::AllDown => IsingModel::new_all_down(p.l, p.j, p.h, temperature),
    }
}

fn measure_at_temperature(p: &SimParams, temperature: f64, rng: &mut impl Rng) -> SimResult {
    let n = (p.l * p.l) as f64;
    let mut model = build_lattice(p, temperature);

    // Thermalization
    for _ in 0..p.therm_steps {
        model.metropolis_step(rng);
    }

    // Measurement
    let mut e_samples: Vec<f64> = Vec::new();
    let mut m_samples: Vec<f64> = Vec::new();
    for step in 0..p.mc_steps {
        model.metropolis_step(rng);
        if step % p.stride == 0 {
            e_samples.push(model.total_energy());
            m_samples.push(model.total_magnetization().unsigned_abs() as f64);
        }
    }

    let mean_e   = mean(&e_samples) / n;
    let mean_m   = mean(&m_samples) / n;
    let heat_cap = variance(&e_samples) / (temperature * temperature * n);
    let chi      = variance(&m_samples) / (temperature * n);

    SimResult { temperature, mean_e, mean_m, heat_cap, susceptibility: chi }
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

fn save_plots(results: &[SimResult]) -> Result<(), Box<dyn std::error::Error>> {
    let path = "ising_results.png";
    let root = BitMapBackend::new(path, (1200, 900)).into_drawing_area();
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
         Enter: run simulation    q: quit"
    )
    .block(Block::default().borders(Borders::ALL).title("Controls"))
    .style(Style::default().fg(TuiColor::Cyan));
    f.render_widget(header, outer[0]);

    // Parameters table
    let field_names = [
        "Lattice Size L",
        "Interaction J",
        "Initial State",
        "T start",
        "T end",
        "T step",
        "MC Steps",
        "Therm Steps (default: MC/2)",
        "Stride",
        "External Field H",
    ];

    let rows: Vec<Row> = field_names
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let selected = i == app.selected_field;
            let value = if i == FIELD_INIT {
                if selected {
                    format!("[{}]  <- / ->", app.initial_state.label())
                } else {
                    format!("[{}]", app.initial_state.label())
                }
            } else if selected {
                format!("{}_", app.field_buffers[i])
            } else {
                app.field_buffers[i].clone()
            };

            let style = if selected {
                Style::default().fg(TuiColor::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TuiColor::White)
            };

            Row::new(vec![
                Cell::from(name).style(style),
                Cell::from(value).style(style),
            ])
        })
        .collect();

    let table = Table::new(rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("Parameters"))
        .column_spacing(2);
    f.render_widget(table, outer[1]);

    // Footer: error or hint
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

fn draw_running(
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

fn draw_done(f: &mut ratatui::Frame<'_>, results: &[SimResult]) {
    let t0 = results.first().map(|r| r.temperature).unwrap_or(0.0);
    let t1 = results.last().map(|r| r.temperature).unwrap_or(0.0);
    let text = format!(
        "Simulation complete!\n\n\
         Temperatures computed : {}\n\
         T range               : {:.3} — {:.3}\n\n\
         Results saved to: ising_results.png\n\n\
         Press 'q' to quit.",
        results.len(), t0, t1
    );
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Done"))
        .style(Style::default().fg(TuiColor::Green));
    f.render_widget(para, f.area());
}

fn draw_frame(f: &mut ratatui::Frame<'_>, app: &App) {
    match &app.mode {
        AppMode::Setup => draw_setup(f, app),
        AppMode::Running { current_t, t_end, done, total } => {
            draw_running(f, *current_t, *t_end, *done, *total)
        }
        AppMode::Done(results) => draw_done(f, results),
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
        AppMode::Done(_) => {
            if key == KeyCode::Char('q') {
                return Err("quit".into());
            }
        }
        AppMode::Running { .. } => {} // should not be reached
        AppMode::Setup => {
            match key {
                KeyCode::Char('q') => return Err("quit".into()),

                KeyCode::Up => {
                    app.selected_field = app.selected_field.saturating_sub(1);
                    app.error_msg = None;
                }
                KeyCode::Down => {
                    app.selected_field = (app.selected_field + 1).min(NUM_FIELDS - 1);
                    app.error_msg = None;
                }

                KeyCode::Left if app.selected_field == FIELD_INIT => {
                    app.initial_state = app.initial_state.prev();
                }
                KeyCode::Right if app.selected_field == FIELD_INIT => {
                    app.initial_state = app.initial_state.next();
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
                            // Run the sweep, redrawing the progress TUI at each temperature
                            let t_end = params.t_end;
                            let results = run_sweep(&params, |cur_t, done, total| {
                                // Temporarily update mode to show progress
                                app.mode = AppMode::Running {
                                    current_t: cur_t,
                                    t_end,
                                    done,
                                    total,
                                };
                                let _ = terminal.draw(|f| draw_frame(f, app));
                            });
                            // Generate plots
                            match save_plots(&results) {
                                Ok(()) => app.mode = AppMode::Done(results),
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
