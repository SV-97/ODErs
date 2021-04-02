use super::mat::*;
use plotters::prelude::*;
use std::collections::HashMap;

pub type Params = HashMap<&'static str, f64>;
pub type OdeSystem<const DIM: usize> = fn(Vector<DIM>, f64, Option<&Params>) -> Vector<DIM>;

pub fn solve_euler<const D: usize>(
    f: OdeSystem<D>,
    t_0: f64,
    y_0: Vector<D>,
    step_size: f64,
    t_end: f64,
    p: Option<&Params>,
) -> (Vec<f64>, Vec<Vector<D>>) {
    assert!(step_size > 0.0);
    let steps = ((t_end - t_0) / step_size) as usize;
    let mut ts = Vec::with_capacity(steps);
    let mut ys = Vec::with_capacity(steps);
    ys.push(y_0);
    for n in 0.. {
        let t_k = t_0 + (n as f64) * step_size;
        if t_k > t_end {
            break;
        } else {
            ts.push(t_k);
            let y_k = ys.last().unwrap();
            let y_k1 = *y_k + step_size * f(*y_k, t_k, p);
            ys.push(y_k1);
        }
    }
    (ts, ys)
}

pub fn solve_runge_kutta_4<const D: usize>(
    f: OdeSystem<D>,
    t_0: f64,
    y_0: Vector<D>,
    step_size: f64,
    t_end: f64,
    p: Option<&Params>,
) -> (Vec<f64>, Vec<Vector<D>>) {
    assert!(step_size > 0.0);
    let steps = ((t_end - t_0) / step_size) as usize;
    let mut ts = Vec::with_capacity(steps);
    let mut ys = Vec::with_capacity(steps);
    ys.push(y_0);
    for n in 0.. {
        let t_k = t_0 + (n as f64) * step_size;
        if t_k > t_end {
            break;
        } else {
            ts.push(t_k);
            let y_k = ys.last().unwrap();
            let k_1 = f(*y_k, t_k, p);
            let k_2 = f(*y_k + step_size / 2. * k_1, t_k + step_size / 2., p);
            let k_3 = f(*y_k + step_size / 2. * k_2, t_k + step_size / 2., p);
            let k_4 = f(*y_k + step_size * k_3, t_k + step_size, p);
            let y_k1 = *y_k + step_size * 1. / 6. * (k_1 + 2. * k_2 + 2.0 * k_3 + k_4);
            ys.push(y_k1);
        }
    }
    (ts, ys)
}

pub fn plot_solution(
    ts_n_ys: Vec<(f32, f32)>,
    name: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    // find window borders
    let mut t_min = ts_n_ys[0].0;
    let mut t_max = ts_n_ys[0].0;
    let mut y_min = ts_n_ys[0].1;
    let mut y_max = ts_n_ys[0].1;
    for &(t, y) in ts_n_ys.iter() {
        if t < t_min {
            t_min = t;
        } else if t > t_max {
            t_max = t;
        }
        if y < y_min {
            y_min = y;
        } else if y > y_max {
            y_max = y;
        }
    }
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d((t_min - 0.1)..(t_max + 0.1), (y_min - 0.1)..(y_max + 0.1))?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(ts_n_ys, &RED))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_ode(x: Vector<1>, _t: f64, _params: Option<&Params>) -> Vector<1> {
        return x;
    }

    fn test_simple_solve() {
        let (ts, ys) = solve_euler(simple_ode, 0.0, Vector::new_vec([1.0]), 10.0e-9, 3.0, None);
        let ts_n_ys = ts
            .iter()
            .map(|t| *t as f32)
            .zip(ys.iter().map(|y| y[0] as f32))
            .collect::<Vec<_>>();
        plot_solution(ts_n_ys, "simple_ODE.png").expect("Failed to draw image");
    }
}
