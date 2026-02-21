use std::io::{self, Write, BufRead, BufReader};
use std::process::{Command, Stdio};
use std::fs;
use chrono::Local;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color as TuiColor, Modifier, Style},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Terminal,
};

struct BatchParams {
    l: usize,
    j: f64,
    h: f64,
    mc_steps: usize,
    therm_steps: usize,
    stride: usize,
    sample_count: usize,
    t_start: f64,
    t_end: f64,
    t_step: f64,
    p_start: f64,
    p_end: f64,
    p_step: f64,
    use_outlier: bool,
    use_auto_window: bool,
    t_win_min: f64,
    t_win_max: f64,
    tc_win_min: f64,
    tc_win_max: f64,
}

impl Default for BatchParams {
    fn default() -> Self {
        let mc_steps = 10_000;
        Self {
            l: 32,
            j: 1.0,
            h: 0.0,
            mc_steps,
            therm_steps: mc_steps / 2,
            stride: 10,
            sample_count: 1,
            t_start: 1.0,
            t_end: 4.0,
            t_step: 0.1,
            p_start: 0.0,
            p_end: 0.1,
            p_step: 0.01,
            use_outlier: false,
            use_auto_window: false,
            t_win_min: 2.0,
            t_win_max: 2.45,
            tc_win_min: 2.25,
            tc_win_max: 2.45,
        }
    }
}

const FIELD_L: usize = 0;
const FIELD_J: usize = 1;
const FIELD_H: usize = 2;
const FIELD_T_START: usize = 3;
const FIELD_T_END: usize = 4;
const FIELD_T_STEP: usize = 5;
const FIELD_MC_STEPS: usize = 6;
const FIELD_THERM: usize = 7;
const FIELD_STRIDE: usize = 8;
const FIELD_SAMPLE_COUNT: usize = 9;
const FIELD_P_START: usize = 10;
const FIELD_P_END: usize = 11;
const FIELD_P_STEP: usize = 12;
const FIELD_T_WIN_MIN: usize = 13;
const FIELD_T_WIN_MAX: usize = 14;
const FIELD_TC_WIN_MIN: usize = 15;
const FIELD_TC_WIN_MAX: usize = 16;
const NUM_FIELDS: usize = 17;

const FIELD_ORDER: [usize; NUM_FIELDS] = [
    FIELD_L,
    FIELD_J,
    FIELD_H,
    FIELD_T_START,
    FIELD_T_END,
    FIELD_T_STEP,
    FIELD_MC_STEPS,
    FIELD_THERM,
    FIELD_STRIDE,
    FIELD_SAMPLE_COUNT,
    FIELD_P_START,
    FIELD_P_END,
    FIELD_P_STEP,
    FIELD_T_WIN_MIN,
    FIELD_T_WIN_MAX,
    FIELD_TC_WIN_MIN,
    FIELD_TC_WIN_MAX,
];

struct BatchApp {
    fields: Vec<String>,
    selected: usize,
    error_msg: Option<String>,
}

impl BatchApp {
    fn new() -> Self {
        let d = BatchParams::default();
        let mut f = vec![String::new(); NUM_FIELDS];
        f[FIELD_L] = d.l.to_string();
        f[FIELD_J] = format!("{}", d.j);
        f[FIELD_H] = format!("{}", d.h);
        f[FIELD_T_START] = format!("{}", d.t_start);
        f[FIELD_T_END] = format!("{}", d.t_end);
        f[FIELD_T_STEP] = format!("{}", d.t_step);
        f[FIELD_MC_STEPS] = d.mc_steps.to_string();
        f[FIELD_THERM] = d.therm_steps.to_string();
        f[FIELD_STRIDE] = d.stride.to_string();
        f[FIELD_SAMPLE_COUNT] = d.sample_count.to_string();
        f[FIELD_P_START] = format!("{}", d.p_start);
        f[FIELD_P_END] = format!("{}", d.p_end);
        f[FIELD_P_STEP] = format!("{}", d.p_step);
        f[FIELD_T_WIN_MIN] = format!("{}", d.t_win_min);
        f[FIELD_T_WIN_MAX] = format!("{}", d.t_win_max);
        f[FIELD_TC_WIN_MIN] = format!("{}", d.tc_win_min);
        f[FIELD_TC_WIN_MAX] = format!("{}", d.tc_win_max);
        Self {
            fields: f,
            selected: FIELD_L,
            error_msg: None,
        }
    }

    fn parse(&self, use_outlier: bool, use_auto_window: bool) -> Result<BatchParams, String> {
        let l = self.fields[FIELD_L].trim().parse::<usize>()
            .map_err(|_| format!("L must be a positive integer, got '{}'", self.fields[FIELD_L]))?;
        if l < 2 {
            return Err("Lattice size L must be >= 2".into());
        }
        let j = self.fields[FIELD_J].trim().parse::<f64>()
            .map_err(|_| format!("J must be a number, got '{}'", self.fields[FIELD_J]))?;
        let h = self.fields[FIELD_H].trim().parse::<f64>()
            .map_err(|_| format!("H must be a number, got '{}'", self.fields[FIELD_H]))?;
        let t_start = self.fields[FIELD_T_START].trim().parse::<f64>()
            .map_err(|_| format!("T start must be a number, got '{}'", self.fields[FIELD_T_START]))?;
        if t_start <= 0.0 {
            return Err("T start must be > 0".into());
        }
        let t_end = self.fields[FIELD_T_END].trim().parse::<f64>()
            .map_err(|_| format!("T end must be a number, got '{}'", self.fields[FIELD_T_END]))?;
        if t_end < t_start {
            return Err("T end must be >= T start".into());
        }
        let t_step = self.fields[FIELD_T_STEP].trim().parse::<f64>()
            .map_err(|_| format!("T step must be a number, got '{}'", self.fields[FIELD_T_STEP]))?;
        if t_step <= 0.0 {
            return Err("T step must be > 0".into());
        }
        let mc_steps = self.fields[FIELD_MC_STEPS].trim().parse::<usize>()
            .map_err(|_| format!("MC steps must be a positive integer, got '{}'", self.fields[FIELD_MC_STEPS]))?;
        if mc_steps < 2 {
            return Err("MC steps must be >= 2".into());
        }
        let therm_steps = self.fields[FIELD_THERM].trim().parse::<usize>()
            .map_err(|_| format!("Thermalization steps must be a positive integer, got '{}'", self.fields[FIELD_THERM]))?;
        if therm_steps == 0 {
            return Err("Thermalization steps must be >= 1".into());
        }
        let stride = self.fields[FIELD_STRIDE].trim().parse::<usize>()
            .map_err(|_| format!("Stride must be a positive integer, got '{}'", self.fields[FIELD_STRIDE]))?;
        if stride == 0 {
            return Err("Stride must be >= 1".into());
        }
        let sample_count = self.fields[FIELD_SAMPLE_COUNT].trim().parse::<usize>()
            .map_err(|_| format!("Disorder samples must be a positive integer, got '{}'", self.fields[FIELD_SAMPLE_COUNT]))?;
        if sample_count == 0 {
            return Err("Disorder samples must be >= 1".into());
        }
        let p_start = self.fields[FIELD_P_START].trim().parse::<f64>()
            .map_err(|_| format!("p start must be a number, got '{}'", self.fields[FIELD_P_START]))?;
        let p_end = self.fields[FIELD_P_END].trim().parse::<f64>()
            .map_err(|_| format!("p end must be a number, got '{}'", self.fields[FIELD_P_END]))?;
        let p_step = self.fields[FIELD_P_STEP].trim().parse::<f64>()
            .map_err(|_| format!("p step must be a number, got '{}'", self.fields[FIELD_P_STEP]))?;
        if p_step <= 0.0 {
            return Err("p step must be > 0".into());
        }
        if p_end < p_start {
            return Err("p end must be >= p start".into());
        }
        let t_win_min = self.fields[FIELD_T_WIN_MIN].trim().parse::<f64>()
            .map_err(|_| format!("T window min must be a number, got '{}'", self.fields[FIELD_T_WIN_MIN]))?;
        let t_win_max = self.fields[FIELD_T_WIN_MAX].trim().parse::<f64>()
            .map_err(|_| format!("T window max must be a number, got '{}'", self.fields[FIELD_T_WIN_MAX]))?;
        if t_win_max < t_win_min {
            return Err("T window max must be >= T window min".into());
        }
        let tc_win_min = self.fields[FIELD_TC_WIN_MIN].trim().parse::<f64>()
            .map_err(|_| format!("Tc window min must be a number, got '{}'", self.fields[FIELD_TC_WIN_MIN]))?;
        let tc_win_max = self.fields[FIELD_TC_WIN_MAX].trim().parse::<f64>()
            .map_err(|_| format!("Tc window max must be a number, got '{}'", self.fields[FIELD_TC_WIN_MAX]))?;
        if tc_win_max < tc_win_min {
            return Err("Tc window max must be >= Tc window min".into());
        }
        Ok(BatchParams {
            l,
            j,
            h,
            mc_steps,
            therm_steps,
            stride,
            sample_count,
            t_start,
            t_end,
            t_step,
            p_start,
            p_end,
            p_step,
            use_outlier,
            use_auto_window,
            t_win_min,
            t_win_max,
            tc_win_min,
            tc_win_max,
        })
    }
}

fn draw_batch_setup(
    f: &mut ratatui::Frame<'_>,
    app: &BatchApp,
    use_outlier: bool,
    use_auto_window: bool,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(11),
            Constraint::Length(3),
        ])
        .split(f.area());

    let filter_label = if use_outlier { "open" } else { "off" };
    let mode_label = if use_auto_window { "A: primary" } else { "B: fixed" };
    let header_text = format!(
        "mode={}  outlier={}  keys: \u{2191}\u{2193} move  Enter start  q quit  o outlier  w window mode",
        mode_label, filter_label
    );
    let header = Paragraph::new(header_text)
    .block(Block::default().borders(Borders::ALL).title("Controls"))
    .style(Style::default().fg(TuiColor::Cyan));
    f.render_widget(header, outer[0]);

    let param_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30),
            Constraint::Percentage(30),
            Constraint::Percentage(40),
        ])
        .split(outer[1]);

    let model_fields = [
        (FIELD_L, "Lattice Size L"),
        (FIELD_J, "Interaction J"),
        (FIELD_H, "External Field H"),
    ];

    let scan_fields = [
        (FIELD_T_START, "T start"),
        (FIELD_T_END, "T end"),
        (FIELD_T_STEP, "T step"),
    ];

    let mc_fields_full = [
        (FIELD_MC_STEPS, "MC Steps"),
        (FIELD_THERM, "Therm Steps"),
        (FIELD_STRIDE, "Stride"),
        (FIELD_SAMPLE_COUNT, "Disorder samples"),
        (FIELD_P_START, "p start"),
        (FIELD_P_END, "p end"),
        (FIELD_P_STEP, "p step"),
        (FIELD_T_WIN_MIN, "T win min"),
        (FIELD_T_WIN_MAX, "T win max"),
        (FIELD_TC_WIN_MIN, "Tc win min"),
        (FIELD_TC_WIN_MAX, "Tc win max"),
    ];
    let mc_fields_basic = [
        (FIELD_MC_STEPS, "MC Steps"),
        (FIELD_THERM, "Therm Steps"),
        (FIELD_STRIDE, "Stride"),
        (FIELD_SAMPLE_COUNT, "Disorder samples"),
        (FIELD_P_START, "p start"),
        (FIELD_P_END, "p end"),
        (FIELD_P_STEP, "p step"),
    ];

    let build_rows = |fields: &[(usize, &str)], app: &BatchApp| {
        fields
            .iter()
            .map(|(idx, name)| {
                let selected = *idx == app.selected;
                let value = if selected {
                    format!("{}_", app.fields[*idx])
                } else {
                    app.fields[*idx].clone()
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

    let model_rows = build_rows(&model_fields, app);
    let scan_rows = build_rows(&scan_fields, app);
    let mc_fields_current: &[(usize, &str)] = if use_auto_window {
        &mc_fields_basic
    } else {
        &mc_fields_full
    };
    let mc_rows = build_rows(mc_fields_current, app);

    let model_table = Table::new(model_rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("Model Parameters"))
        .column_spacing(2);
    f.render_widget(model_table, param_areas[0]);

    let scan_table = Table::new(scan_rows, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("Scan Parameters"))
        .column_spacing(2);
    f.render_widget(scan_table, param_areas[1]);

    // MC + Batch 参数区：根据高度做简单“翻页”，保证选中行尽量可见
    let mc_area = param_areas[2];
    let available_height = mc_area.height.saturating_sub(2).max(1) as usize;
    let total_mc_rows = mc_rows.len();
    let selected_pos_in_mc = mc_fields_current
        .iter()
        .position(|(idx, _)| *idx == app.selected);
    let start_row = if let Some(sel) = selected_pos_in_mc {
        if sel + 1 > available_height {
            sel + 1 - available_height
        } else {
            0
        }
    } else {
        0
    };
    let end_row = (start_row + available_height).min(total_mc_rows);
    let mc_visible = mc_rows[start_row..end_row].to_vec();

    let mc_table = Table::new(mc_visible, [Constraint::Percentage(40), Constraint::Percentage(60)])
        .block(Block::default().borders(Borders::ALL).title("MC + Batch Parameters"))
        .column_spacing(2);
    f.render_widget(mc_table, param_areas[2]);

    let footer_text = app
        .error_msg
        .as_deref()
        .unwrap_or("Adjust parameters and press Enter to start batch.");
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

fn run_tui() -> Result<BatchParams, String> {
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| e.to_string())?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;

    let mut app = BatchApp::new();
    let mut use_outlier = false;
    let mut use_auto_window = false;
    loop {
        terminal
            .draw(|f| draw_batch_setup(f, &app, use_outlier, use_auto_window))
            .map_err(|e| e.to_string())?;
        if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
            match key.code {
                KeyCode::Char('q') => {
                    disable_raw_mode().map_err(|e| e.to_string())?;
                    execute!(terminal.backend_mut(), LeaveAlternateScreen)
                        .map_err(|e| e.to_string())?;
                    return Err("quit".into());
                }
                KeyCode::Up => {
                    let pos = FIELD_ORDER
                        .iter()
                        .position(|&f| f == app.selected)
                        .unwrap_or(0);
                    let mut new_pos = pos;
                    while new_pos > 0 {
                        new_pos -= 1;
                        let candidate = FIELD_ORDER[new_pos];
                        if use_auto_window
                            && (candidate == FIELD_T_WIN_MIN
                                || candidate == FIELD_T_WIN_MAX
                                || candidate == FIELD_TC_WIN_MIN
                                || candidate == FIELD_TC_WIN_MAX)
                        {
                            if new_pos == 0 {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                    app.selected = FIELD_ORDER[new_pos];
                    app.error_msg = None;
                }
                KeyCode::Down => {
                    let pos = FIELD_ORDER
                        .iter()
                        .position(|&f| f == app.selected)
                        .unwrap_or(0);
                    let mut new_pos = pos;
                    while new_pos + 1 < NUM_FIELDS {
                        new_pos += 1;
                        let candidate = FIELD_ORDER[new_pos];
                        if use_auto_window
                            && (candidate == FIELD_T_WIN_MIN
                                || candidate == FIELD_T_WIN_MAX
                                || candidate == FIELD_TC_WIN_MIN
                                || candidate == FIELD_TC_WIN_MAX)
                        {
                            if new_pos + 1 >= NUM_FIELDS {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                    app.selected = FIELD_ORDER[new_pos];
                    app.error_msg = None;
                }
                KeyCode::Char('o') => {
                    use_outlier = !use_outlier;
                    app.error_msg = None;
                }
                KeyCode::Char('w') => {
                    use_auto_window = !use_auto_window;
                    if use_auto_window
                        && (app.selected == FIELD_T_WIN_MIN
                            || app.selected == FIELD_T_WIN_MAX
                            || app.selected == FIELD_TC_WIN_MIN
                            || app.selected == FIELD_TC_WIN_MAX)
                    {
                        app.selected = FIELD_P_STEP;
                    }
                    app.error_msg = None;
                }
                KeyCode::Char(c) => {
                    app.fields[app.selected].push(c);
                    app.error_msg = None;
                }
                KeyCode::Backspace => {
                    app.fields[app.selected].pop();
                    app.error_msg = None;
                }
                KeyCode::Enter => match app.parse(use_outlier, use_auto_window) {
                    Ok(params) => {
                        disable_raw_mode().map_err(|e| e.to_string())?;
                        execute!(terminal.backend_mut(), LeaveAlternateScreen)
                            .map_err(|e| e.to_string())?;
                        return Ok(params);
                    }
                    Err(msg) => {
                        app.error_msg = Some(msg);
                    }
                },
                _ => {}
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = match run_tui() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    let mut p_vals = Vec::new();
    let mut p = params.p_start;
    while p <= params.p_end + 1e-12 {
        p_vals.push(p);
        p += params.p_step;
    }

    if p_vals.is_empty() {
        println!("No p values generated.");
        return Ok(());
    }

    println!("Planned runs:");
    for (idx, val) in p_vals.iter().enumerate() {
        println!("  {}: p = {:.6}", idx + 1, val);
    }

    let total = p_vals.len();
    println!("Total runs: {}", total);
    println!("Starting batch runs...");

    let mut completed = 0usize;
    let batch_ts = Local::now().format("%Y%m%d_%H%M%S");
    let batch_root = format!("data_batch/batch_{}", batch_ts);
    fs::create_dir_all(&batch_root)?;
    for (idx, p_val) in p_vals.iter().enumerate() {
        println!();
        println!(
            "Starting run {}/{} with p = {:.6}",
            idx + 1,
            total,
            p_val
        );

        let mut cmd = Command::new("target/debug/ising-monte-carlo");
        cmd
            .env("BATCH_MODE", "1")
            .env("BATCH_L", params.l.to_string())
            .env("BATCH_J", params.j.to_string())
            .env("BATCH_P", format!("{:.8}", p_val))
            .env("BATCH_T_START", params.t_start.to_string())
            .env("BATCH_T_END", params.t_end.to_string())
            .env("BATCH_T_STEP", params.t_step.to_string())
            .env("BATCH_MC_STEPS", params.mc_steps.to_string())
            .env("BATCH_THERM_STEPS", params.therm_steps.to_string())
            .env("BATCH_STRIDE", params.stride.to_string())
            .env("BATCH_H", params.h.to_string())
            .env("BATCH_SAMPLE_COUNT", params.sample_count.to_string())
            .env("BATCH_INIT", "Random");
        if params.use_outlier {
            cmd.env("BATCH_OUTLIER_FILTER", "1");
        }
        if params.use_auto_window {
            cmd.env("BATCH_WINDOW_MODE", "auto");
        } else {
            cmd.env("BATCH_WINDOW_MODE", "fixed")
                .env("BATCH_T_MIN", params.t_win_min.to_string())
                .env("BATCH_T_MAX", params.t_win_max.to_string())
                .env("BATCH_TC_MIN", params.tc_win_min.to_string())
                .env("BATCH_TC_MAX", params.tc_win_max.to_string());
        }
        cmd.env("BATCH_OUTPUT_ROOT", &batch_root);
        cmd.stdout(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "Failed to start run {}/{} with p = {:.6}: {}",
                    idx + 1,
                    total,
                    p_val,
                    e
                );
                continue;
            }
        };

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut sweep_done = 0usize;
            let mut sweep_total = 0usize;
            let mut tc_done = 0usize;
            let mut tc_total = 0usize;
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                if let Some(rest) = line.strip_prefix("BATCH_PROGRESS ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if parts.len() >= 3 {
                        if parts[0] == "SWEEP" {
                            if let (Ok(d), Ok(t)) =
                                (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                            {
                                sweep_done = d;
                                sweep_total = t;
                            }
                        } else if parts[0] == "TC" {
                            if let (Ok(d), Ok(t)) =
                                (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                            {
                                tc_done = d;
                                tc_total = t;
                            }
                        }
                    }
                    let sweep_pct = if sweep_total > 0 {
                        100.0 * sweep_done as f64 / sweep_total as f64
                    } else {
                        0.0
                    };
                    let tc_pct = if tc_total > 0 {
                        100.0 * tc_done as f64 / tc_total as f64
                    } else {
                        0.0
                    };
                    let overall_frac = (completed as f64 + (tc_done > 0) as u8 as f64)
                        / total as f64;
                    let overall_pct = overall_frac * 100.0;
                    println!(
                        "Run {}/{} (p = {:.6})\n  Sweep: {:>4}/{:<4} ({:>5.1}%)\n  Tc scan: {:>4}/{:<4} ({:>5.1}%)\n  Overall batch: {:>5.1}%\n",
                        idx + 1,
                        total,
                        p_val,
                        sweep_done,
                        sweep_total,
                        sweep_pct,
                        tc_done,
                        tc_total,
                        tc_pct,
                        overall_pct
                    );
                } else if !line.trim().is_empty() {
                    println!("{}", line);
                }
            }
        }

        let status = child.wait();

        match status {
            Ok(s) if s.success() => {
                completed += 1;
                let frac = completed as f64 / total as f64;
                let percent = frac * 100.0;
                println!(
                    "Finished run {}/{} (p = {:.6}). Overall progress: {:.1}%",
                    idx + 1,
                    total,
                    p_val,
                    percent
                );
            }
            Ok(s) => {
                println!(
                    "Run {}/{} with p = {:.6} exited with status {:?}",
                    idx + 1,
                    total,
                    p_val,
                    s.code()
                );
            }
            Err(e) => {
                println!(
                    "Failed to start run {}/{} with p = {:.6}: {}",
                    idx + 1,
                    total,
                    p_val,
                    e
                );
            }
        }
    }

    println!();
    println!("Batch runs finished. Completed {}/{} runs.", completed, total);

    Ok(())
}
