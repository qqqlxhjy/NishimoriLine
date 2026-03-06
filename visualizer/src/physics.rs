use crate::ising::IsingModel;
use rand::Rng;

impl IsingModel {
    pub fn metropolis_sweep(&mut self) {
        let mut rng = rand::thread_rng();
        let l = self.L;
        let n = l * l;
        
        // Clean up old flash effects
        // Keep only recent ones (e.g. last 0.5s equivalent, handled in render logic or simple limit)
        // For simplicity here, we clear or limit the vector size in render or main loop.
        // But the user suggested `recent_flips: Vec<(x,y,time)>`. 
        // We will just append here, and let the render system filter/clear based on time.
        // To avoid infinite growth, we might want to clear it periodically or use a ring buffer.
        // For now, let's just assume main loop clears it or we add a cleanup method.

        for _ in 0..n {
            let x = rng.gen_range(0..l);
            let y = rng.gen_range(0..l);

            let idx = self.index(x, y);
            let de = self.delta_energy(x, y);

            if de <= 0.0 || rng.gen::<f32>() < (-de / self.T).exp() {
                self.spins[idx] *= -1;
                // Record flip for visual effect. Time is managed by macroquad::time::get_time() in main loop
                // But we don't have access to macroquad time here easily without dependency.
                // We'll store a placeholder or passed time if needed. 
                // Let's change the signature or just store raw frames? 
                // The user's pseudo code: `recent_flips: Vec<(x,y,time)>`
                // We'll add a method `metropolis_sweep_with_time`
            }
        }

        self.record_observables();
    }

    pub fn metropolis_sweep_with_time(&mut self, current_time: f64) {
        let mut rng = rand::thread_rng();
        let l = self.L;
        let n = l * l;

        // Optional: Remove very old flips to save memory
        self.recent_flips.retain(|&(_, _, t)| (current_time as f32 - t) < 1.0); 

        for _ in 0..n {
            let x = rng.gen_range(0..l);
            let y = rng.gen_range(0..l);

            let idx = self.index(x, y);
            let de = self.delta_energy(x, y);

            if de <= 0.0 || rng.gen::<f32>() < (-de / self.T).exp() {
                self.spins[idx] *= -1;
                self.recent_flips.push((x, y, current_time as f32));
            }
        }
        self.record_observables();
    }

    pub fn delta_energy(&self, x: usize, y: usize) -> f32 {
        let l = self.L;
        let s = self.spins[self.index(x, y)] as f32;

        let xp = (x + 1) % l;
        let xm = (x + l - 1) % l;
        let yp = (y + 1) % l;
        let ym = (y + l - 1) % l;

        let mut sum = 0.0;

        // Right neighbor (x+1, y)
        // Bond is Jx[index(x,y)] connecting (x,y) and (x+1,y)
        sum += self.Jx[self.index(x, y)] as f32 * self.spins[self.index(xp, y)] as f32;

        // Left neighbor (x-1, y)
        // Bond is Jx[index(x-1,y)] connecting (x-1,y) and (x,y)
        sum += self.Jx[self.index(xm, y)] as f32 * self.spins[self.index(xm, y)] as f32;

        // Top neighbor (y+1, y) - Note: y increases downwards usually in images, 
        // but physically it doesn't matter as long as consistent.
        // Bond is Jy[index(x,y)] connecting (x,y) and (x,y+1)
        sum += self.Jy[self.index(x, y)] as f32 * self.spins[self.index(x, yp)] as f32;

        // Bottom neighbor (y-1, y)
        // Bond is Jy[index(x,y-1)] connecting (x,y-1) and (x,y)
        sum += self.Jy[self.index(x, ym)] as f32 * self.spins[self.index(x, ym)] as f32;

        // Delta E = E_new - E_old
        // E = - sum J * s_i * s_j
        // The term involving s_i is - s_i * sum (J * s_neighbor)
        // If s_i flips to -s_i, change is:
        // E_new = - (-s_i) * sum = s_i * sum
        // E_old = - s_i * sum
        // Delta E = 2 * s_i * sum
        // But wait, the formula in user input is `2.0 * s * sum`.
        // Let's verify signs.
        // Hamiltonian H = - \sum J_{ij} S_i S_j
        // Contribution of S_i is H_i = - S_i * \sum_{j \in neighbors} J_{ij} S_j
        // If S_i -> -S_i, then H_i' = - (-S_i) * ... = S_i * ...
        // Delta E = H_i' - H_i = 2 * S_i * \sum J_{ij} S_j
        // So `2.0 * s * sum` is correct.
        
        2.0 * s * sum
    }

    pub fn record_observables(&mut self) {
        let mut m = 0.0;
        for s in &self.spins {
            m += *s as f32;
        }
        m /= (self.L * self.L) as f32;
        self.magnet_history.push(m);
        
        // Optional: limit history size
        if self.magnet_history.len() > 1000 {
            self.magnet_history.remove(0);
        }
    }
}
