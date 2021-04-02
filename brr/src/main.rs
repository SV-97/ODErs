#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![allow(dead_code)]

mod mat;
mod ode;
use mat::*;
use ode::*;

/// crude birth rate value for 2015-2020 from wikipedia https://en.wikipedia.org/wiki/Birth_rate
const CBR: f64 = 18.5;
/// crude death rate estimate for 2020 from wikipedia https://en.wikipedia.org/wiki/Mortality_rate
const CDR: f64 = 7.7;
const BIRTH_RATE: f64 = CBR * 1.0e-3;
const DEATH_RATE: f64 = CDR * 1.0e-3;
const POPULATION: f64 = 7.0;
const STANDARD_OF_LIVING: f64 = 5.0;
const POLLUTION: f64 = 3.0;
const POLLUTION_RATE: f64 = 0.005;
const RECOVERY_RATE: f64 = 0.002;
/// Standard of living growth rate
const SOL_GROWTH_RATE: f64 = 0.005;
const SOL_DECREASE_RATE: f64 = 0.002;
//const B_MAX: f64 = 10.0e9;
const L_TIP: f64 = 100.0;
/// Death rate for deaths caused by pollution
const ENVIRONMENTAL_DEATH_RATE: f64 = DEATH_RATE * 0.2;

fn world_model(x: Vector<3>, _t: f64, _params: Option<&Params>) -> Vector<3> {
    let b = x[0]; // population
    let l = x[1]; // standard of living
    let u = x[2]; // environment

    let db = ((BIRTH_RATE - DEATH_RATE) * l - ENVIRONMENTAL_DEATH_RATE * u) * b;
    let dl = SOL_GROWTH_RATE * b - SOL_DECREASE_RATE * u;
    let du = -u * (RECOVERY_RATE - POLLUTION_RATE * b) + l * (-(l - L_TIP).powi(2)).exp();

    Vector::new_vec([db, dl, du])
}

fn main() {
    let (ts, ys) = solve_runge_kutta_4(
        world_model,
        0.0,
        Vector::new_vec([POPULATION, STANDARD_OF_LIVING, POLLUTION]),
        10.0e-6,
        100.0,
        None,
    );
    let f = |n| {
        ts.iter()
            .map(|t| *t as f32)
            .zip(ys.iter().map(|y| {
                let y_1 = y[n] as f32;
                if y_1.is_nan() {
                    panic!("Encountered NaN in computation.");
                } else if y_1.is_infinite() {
                    panic!("Encountered Infinite in computation.");
                } else {
                    y_1
                }
            }))
            .collect::<Vec<_>>()
    };
    let ts_n_ys_0 = f(0);
    let ts_n_ys_1 = f(1);
    let ts_n_ys_2 = f(2);
    plot_solution(ts_n_ys_0, "population.png").expect("Failed to draw image");
    plot_solution(ts_n_ys_1, "standard_of_living.png").expect("Failed to draw image");
    plot_solution(ts_n_ys_2, "environment.png").expect("Failed to draw image");
}
