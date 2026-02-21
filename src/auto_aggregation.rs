use chrono::Local;
use plotters::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

struct Sample {
    p: f64,
    tc: f64,
    dir: String,
}

fn parse_summary(path: &str) -> Option<(f64, f64)> {
    let mut file = fs::File::open(path).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;

    let mut p_opt: Option<f64> = None;
    let mut tc_opt: Option<f64> = None;

    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with("p =") {
            let val = line.split('=').nth(1)?.trim();
            if let Ok(v) = val.parse::<f64>() {
                p_opt = Some(v);
            }
        } else if line.starts_with("Tc_best") {
            let val = line.split('=').nth(1)?.trim();
            if let Ok(v) = val.parse::<f64>() {
                tc_opt = Some(v);
            }
        }
    }

    match (p_opt, tc_opt) {
        (Some(p), Some(tc)) => Some((p, tc)),
        _ => None,
    }
}

fn collect_recent_samples(limit: usize) -> io::Result<Vec<Sample>> {
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    if let Ok(dir) = fs::read_dir("candidate_data") {
        for e in dir.flatten() {
            if let Ok(ft) = e.file_type() {
                if ft.is_dir() {
                    let name = e
                        .file_name()
                        .into_string()
                        .unwrap_or_default();
                    if name.starts_with("loglog_singleProfile_") {
                        let path = e.path();
                        let summary = path.join("summary.txt");
                        if summary.is_file() {
                            entries.push((name, path));
                        }
                    }
                }
            }
        }
    }

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    entries.sort_by(|a, b| b.0.cmp(&a.0));
    let take_n = limit.min(entries.len());

    let mut samples = Vec::new();
    for (name, path) in entries.into_iter().take(take_n) {
        let summary_path = path.join("summary.txt");
        if let Some((p, tc)) = parse_summary(summary_path.to_string_lossy().as_ref()) {
            samples.push(Sample {
                p,
                tc,
                dir: name,
            });
        }
    }

    Ok(samples)
}

fn group_by_p(samples: &[Sample]) -> BTreeMap<i64, Vec<&Sample>> {
    let mut groups: BTreeMap<i64, Vec<&Sample>> = BTreeMap::new();
    for s in samples {
        let key = (s.p * 1e6).round() as i64;
        groups.entry(key).or_default().push(s);
    }
    groups
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn variance(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        0.0
    } else {
        xs.iter()
            .map(|x| (x - m) * (x - m))
            .sum::<f64>()
            / (xs.len() as f64 - 1.0)
    }
}

fn draw_tp_plot(samples: &[Sample], groups: &BTreeMap<i64, Vec<&Sample>>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if samples.is_empty() {
        return Ok(());
    }

    let p_vals: Vec<f64> = samples.iter().map(|s| s.p).collect();
    let t_vals: Vec<f64> = samples.iter().map(|s| s.tc).collect();

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
        .caption("Tc vs disorder p", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((p_min - p_pad)..(p_max + p_pad), 0.0..(t_max + t_pad))?;

    chart
        .configure_mesh()
        .x_desc("p")
        .y_desc("Tc_best")
        .draw()?;

    chart.draw_series(samples.iter().map(|s| {
        Circle::new((s.p, s.tc), 4, BLUE.filled())
    }))?;

    for (p_key, group) in groups {
        let tc_vals: Vec<f64> = group.iter().map(|s| s.tc).collect();
        let m_tc = mean(&tc_vals);
        let p_val = *p_key as f64 / 1e6;
        chart.draw_series(std::iter::once(Circle::new((p_val, m_tc), 6, RED.filled())))?;
    }

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut n: usize = 0;
    let mut interactive = true;

    if args.len() >= 2 {
        if let Ok(v) = args[1].parse::<usize>() {
            n = v;
            interactive = false;
        }
    }

    if n == 0 {
        print!("Enter number of recent runs N to aggregate: ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        n = line.trim().parse::<usize>().unwrap_or(0);
    }

    if n == 0 {
        println!("N must be > 0.");
        return Ok(());
    }

    let samples = collect_recent_samples(n)?;
    if samples.is_empty() {
        println!("No recent loglog_singleProfile_* entries with summary.txt found under candidate_data/.");
        return Ok(());
    }

    println!("Collected {} samples (p, Tc_best):", samples.len());
    for s in &samples {
        println!("dir = {}, p = {:.6}, Tc_best = {:.8}", s.dir, s.p, s.tc);
    }

    let groups = group_by_p(&samples);
    println!();
    println!("Grouped by p:");
    for (p_key, group) in &groups {
        let tc_vals: Vec<f64> = group.iter().map(|s| s.tc).collect();
        let m_tc = mean(&tc_vals);
        let var_tc = variance(&tc_vals, m_tc);
        println!(
            "p = {:.6}, count = {}, mean Tc = {:.8}, var Tc = {:.8}",
            *p_key as f64 / 1e6,
            group.len(),
            m_tc,
            var_tc
        );
    }

    if interactive {
        println!();
        println!("Press Enter to generate T-P plot and summary into data2/ (or Ctrl+C to abort)...");
        let mut tmp = String::new();
        let _ = io::stdin().read_line(&mut tmp);
    }

    let out_root = "data2";
    fs::create_dir_all(out_root)?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let out_dir = format!("{}/auto_aggregation_{}", out_root, timestamp);
    fs::create_dir_all(&out_dir)?;

    let png_path = format!("{}/tp_aggregation.png", out_dir);
    draw_tp_plot(&samples, &groups, &png_path)?;

    let summary_path = format!("{}/tp_aggregation_summary.txt", out_dir);
    let mut f = fs::File::create(&summary_path)?;
    writeln!(f, "Tc vs p aggregation summary")?;
    writeln!(f, "Output directory: {}", out_dir)?;
    writeln!(f)?;
    writeln!(f, "Per-sample points:")?;
    writeln!(f, "dir,p,Tc_best")?;
    for s in &samples {
        writeln!(f, "{},{:.6},{:.8}", s.dir, s.p, s.tc)?;
    }
    writeln!(f)?;
    writeln!(f, "Grouped statistics by p:")?;
    writeln!(f, "p,count,mean_Tc,var_Tc")?;
    for (p_key, group) in &groups {
        let tc_vals: Vec<f64> = group.iter().map(|s| s.tc).collect();
        let m_tc = mean(&tc_vals);
        let var_tc = variance(&tc_vals, m_tc);
        writeln!(
            f,
            "{:.6},{},{:.8},{:.8}",
            *p_key as f64 / 1e6,
            group.len(),
            m_tc,
            var_tc
        )?;
    }

    println!();
    println!("Aggregation written to directory: {}", out_dir);
    println!("T-P plot: {}", png_path);
    println!("Summary: {}", summary_path);

    Ok(())
}
