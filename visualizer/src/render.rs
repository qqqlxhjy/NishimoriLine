use macroquad::prelude::*;
use crate::ising::IsingModel;

pub fn draw_spins(model: &IsingModel, offset_x: f32, offset_y: f32, cell_size: f32) {
    let l = model.L;
    
    // Draw spins
    for y in 0..l {
        for x in 0..l {
            let idx = y * l + x;
            let color = if model.spins[idx] == 1 {
                RED
            } else {
                BLUE
            };
            
            draw_rectangle(
                offset_x + x as f32 * cell_size,
                offset_y + y as f32 * cell_size,
                cell_size,
                cell_size,
                color
            );
        }
    }

    // Draw recent flips (flash effect)
    let current_time = get_time();
    for &(fx, fy, t) in &model.recent_flips {
        let age = (current_time as f32 - t).max(0.0);
        if age < 0.5 {
            let alpha = 1.0 - (age / 0.5);
            let color = Color::new(1.0, 1.0, 1.0, alpha); // White flash
            draw_rectangle(
                offset_x + fx as f32 * cell_size,
                offset_y + fy as f32 * cell_size,
                cell_size,
                cell_size,
                color
            );
        }
    }
}

pub fn draw_bonds(model: &IsingModel, offset_x: f32, offset_y: f32, cell_size: f32) {
    let l = model.L;
    let thickness = 1.0;

    // Draw horizontal bonds
    for y in 0..l {
        for x in 0..l {
            let idx = y * l + x;
            let j_val = model.Jx[idx];
            let color = if j_val > 0 {
                Color::new(0.0, 1.0, 0.0, 0.3) // Faint green
            } else {
                Color::new(1.0, 1.0, 0.0, 0.8) // Bright yellow for frustration source
            };

            // Bond connects (x,y) to (x+1,y)
            let start_x = offset_x + (x as f32 + 0.5) * cell_size;
            let start_y = offset_y + (y as f32 + 0.5) * cell_size;
            let end_x = offset_x + ((x + 1) as f32 + 0.5) * cell_size;
            
            // Only draw if within bounds visually (or wrap around if needed, but usually just draw grid)
            if x < l - 1 {
                 draw_line(start_x, start_y, end_x, start_y, thickness, color);
            }
        }
    }

    // Draw vertical bonds
    for y in 0..l {
        for x in 0..l {
            let idx = y * l + x;
            let j_val = model.Jy[idx];
            let color = if j_val > 0 {
                Color::new(0.0, 1.0, 0.0, 0.3) // Faint green
            } else {
                Color::new(1.0, 1.0, 0.0, 0.8) // Bright yellow
            };

            // Bond connects (x,y) to (x,y+1)
            let start_x = offset_x + (x as f32 + 0.5) * cell_size;
            let start_y = offset_y + (y as f32 + 0.5) * cell_size;
            let end_y = offset_y + ((y + 1) as f32 + 0.5) * cell_size;
            
            if y < l - 1 {
                draw_line(start_x, start_y, start_x, end_y, thickness, color);
            }
        }
    }
}

pub fn draw_magnetization_plot(history: &[f32], x: f32, y: f32, w: f32, h: f32, color: Color) {
    draw_rectangle(x, y, w, h, Color::new(0.2, 0.2, 0.2, 0.5)); // Background
    
    if history.len() < 2 {
        return;
    }

    let max_points = 500;
    let start_idx = if history.len() > max_points {
        history.len() - max_points
    } else {
        0
    };

    let points = &history[start_idx..];
    let step_x = w / (points.len() as f32 - 1.0).max(1.0);
    
    // Map magnetization [-1, 1] to [y+h, y]
    // y_val = y + h/2 - (m * h/2)
    
    let map_y = |m: f32| -> f32 {
        y + h/2.0 - (m * h/2.0)
    };

    for i in 0..points.len()-1 {
        let m1 = points[i];
        let m2 = points[i+1];
        
        let px1 = x + i as f32 * step_x;
        let py1 = map_y(m1);
        let px2 = x + (i+1) as f32 * step_x;
        let py2 = map_y(m2);
        
        draw_line(px1, py1, px2, py2, 2.0, color);
    }
    
    // Draw zero line
    draw_line(x, y + h/2.0, x + w, y + h/2.0, 1.0, LIGHTGRAY);
}
