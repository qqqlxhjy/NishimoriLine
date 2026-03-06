pub struct IsingModel {
    pub L: usize,
    pub T: f32,

    // spins
    pub spins: Vec<i8>,  // +1 or -1

    // bonds (shared reference)
    // Using Rc<Vec<i8>> or just cloning the Vec<i8> since i8 is small.
    // However, the user asked for "shared reference" concept.
    // For simplicity and avoiding lifetime issues in this context, 
    // we can keep Vec<i8> and clone it during construction as shown in user's example.
    // If strict sharing is needed, we could use Rc or Arc.
    // The user's example shows:
    // pub Jx: Vec<i8>,
    // pub Jy: Vec<i8>,
    // and passing clones. So we will follow that.
    pub Jx: Vec<i8>,
    pub Jy: Vec<i8>,

    // observables
    pub energy_history: Vec<f32>,
    pub magnet_history: Vec<f32>,
    
    // For flip flash effect
    pub recent_flips: Vec<(usize, usize, f32)>, // x, y, time
}

impl IsingModel {
    pub fn new(L: usize, T: f32, Jx: Vec<i8>, Jy: Vec<i8>) -> Self {
        let spins = vec![1; L * L];

        Self {
            L,
            T,
            spins,
            Jx,
            Jy,
            energy_history: Vec::new(),
            magnet_history: Vec::new(),
            recent_flips: Vec::new(),
        }
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.L + x
    }
}
