use rand::Rng;

pub fn generate_bonds(L: usize, p: f32) -> (Vec<i8>, Vec<i8>) {
    let mut rng = rand::thread_rng();
    let mut jx = Vec::with_capacity(L * L);
    let mut jy = Vec::with_capacity(L * L);

    for _ in 0..L * L {
        let valx = if rng.gen::<f32>() < p { -1 } else { 1 };
        let valy = if rng.gen::<f32>() < p { -1 } else { 1 };

        jx.push(valx);
        jy.push(valy);
    }

    (jx, jy)
}

pub fn calc_nish_temp(p: f32) -> f32 {
    // T_Nishimori satisfies: 2*beta*J = ln((1-p)/p)
    // J=1. So 2/T = ln((1-p)/p) => T = 2 / ln((1-p)/p)
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    let ratio = (1.0 - p) / p;
    if ratio <= 0.0 {
        return 0.0;
    }
    2.0 / ratio.ln()
}
