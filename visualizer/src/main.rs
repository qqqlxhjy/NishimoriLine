mod ising;
mod bonds;
mod physics;
mod render;

use macroquad::prelude::*;
use ising::IsingModel;
use bonds::{generate_bonds, calc_nish_temp};
use render::{draw_spins, draw_bonds, draw_magnetization_plot};

#[macroquad::main("Nishimori Visualizer")]
async fn main() {
    let l = 32;
    let p = 0.15;
    
    // Generate shared bonds
    let (jx, jy) = generate_bonds(l, p);
    
    // Temperatures
    let t_nish = calc_nish_temp(p);
    let t_custom = 0.3; // Low temperature phase usually
    
    // Create two models with shared disorder
    let mut model_nish = IsingModel::new(l, t_nish, jx.clone(), jy.clone());
    let mut model_custom = IsingModel::new(l, t_custom, jx.clone(), jy.clone());
    
    let mut step = 0;
    
    // Layout parameters
    let cell_size = 10.0;
    let offset_x = 50.0;
    let offset_y_nish = 50.0;
    let offset_y_custom = 400.0;
    
    loop {
        clear_background(BLACK);
        
        // Physics update
        // We can do multiple sweeps per frame to speed up simulation
        for _ in 0..5 {
            let current_time = get_time();
            model_nish.metropolis_sweep_with_time(current_time);
            model_custom.metropolis_sweep_with_time(current_time);
            step += 1;
        }
        
        // Draw Nishimori Model (Top)
        draw_text(&format!("Nishimori Line (T={:.2}, p={:.2})", t_nish, p), offset_x, offset_y_nish - 10.0, 20.0, WHITE);
        draw_spins(&model_nish, offset_x, offset_y_nish, cell_size);
        draw_bonds(&model_nish, offset_x, offset_y_nish, cell_size);
        
        // Draw Custom Model (Bottom)
        draw_text(&format!("Custom Temp (T={:.2})", t_custom), offset_x, offset_y_custom - 10.0, 20.0, WHITE);
        draw_spins(&model_custom, offset_x, offset_y_custom, cell_size);
        draw_bonds(&model_custom, offset_x, offset_y_custom, cell_size);
        
        // Draw Magnetization Plots (Right side)
        let plot_x = offset_x + (l as f32 * cell_size) + 50.0;
        let plot_w = 300.0;
        let plot_h = 150.0;
        
        draw_text("Magnetization History", plot_x, offset_y_nish - 10.0, 20.0, WHITE);
        
        // Nishimori plot
        draw_magnetization_plot(&model_nish.magnet_history, plot_x, offset_y_nish, plot_w, plot_h, RED);
        draw_text("Nishimori", plot_x + 5.0, offset_y_nish + 20.0, 15.0, RED);
        
        // Custom plot
        draw_magnetization_plot(&model_custom.magnet_history, plot_x, offset_y_custom, plot_w, plot_h, BLUE);
        draw_text("Custom", plot_x + 5.0, offset_y_custom + 20.0, 15.0, BLUE);

        // Global Info
        draw_text(&format!("MC Steps: {}", step), 20.0, 20.0, 30.0, YELLOW);
        
        next_frame().await
    }
}
