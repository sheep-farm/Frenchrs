use frenchrs::{Carhart4Factor, FamaFrench3Factor, FamaFrench5Factor, CAPM};
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO COMPLETA: CAPM vs FF3 vs Carhart vs FF5");
    println!("{}", "=".repeat(80));

    // Data simulados
    let fund = array![
        0.058, 0.042, -0.018, 0.082, 0.045, -0.032, 0.068, 0.052, -0.015, 0.062, 0.041, 0.075
    ];
    let market = array![
        0.042, 0.030, -0.015, 0.060, 0.033, -0.025, 0.050, 0.038, -0.012, 0.045, 0.032, 0.055
    ];
    let smb = array![
        0.008, -0.003, 0.012, 0.005, -0.007, 0.015, 0.002, -0.005, 0.010, 0.003, -0.004, 0.006
    ];
    let hml = array![
        0.005, 0.008, -0.010, 0.012, 0.003, -0.006, 0.009, 0.004, -0.008, 0.011, 0.002, 0.007
    ];
    let mom = array![
        0.012, 0.008, -0.015, 0.018, 0.010, -0.012, 0.015, 0.009, -0.010, 0.013, 0.007, 0.014
    ];
    let rmw = array![
        0.006, 0.004, -0.005, 0.008, 0.003, -0.004, 0.007, 0.005, -0.003, 0.006, 0.004, 0.007
    ];
    let cma = array![
        0.003, -0.002, 0.005, 0.002, -0.003, 0.004, 0.003, -0.002, 0.004, 0.003, -0.001, 0.003
    ];

    let rf = 0.03 / 12.0;

    println!("\nEstimatesndthe models...\n");
    let capm = CAPM::fit(&fund, &market, rf, CovarianceType::HC3)?;
    let ff3 = FamaFrench3Factor::fit(&fund, &market, &smb, &hml, rf, CovarianceType::HC3)?;
    let carhart = Carhart4Factor::fit(&fund, &market, &smb, &hml, &mom, rf, CovarianceType::HC3)?;
    let ff5 = FamaFrench5Factor::fit(
        &fund,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        rf,
        CovarianceType::HC3,
    )?;

    // Tabela comparativa
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12}",
        "Métrica", "CAPM", "FF3", "Carhart", "FF5"
    );
    println!("{}", "-".repeat(80));
    println!(
        "{:<25} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "Alpha", capm.alpha, ff3.alpha, carhart.alpha, ff5.alpha
    );
    println!(
        "{:<25} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "Beta Market", capm.beta, ff3.beta_market, carhart.beta_market, ff5.beta_market
    );
    println!(
        "{:<25} {:>12} {:>12.6} {:>12.6} {:>12.6}",
        "Beta SMB", "-", ff3.beta_smb, carhart.beta_smb, ff5.beta_smb
    );
    println!(
        "{:<25} {:>12} {:>12.6} {:>12.6} {:>12.6}",
        "Beta HML", "-", ff3.beta_hml, carhart.beta_hml, ff5.beta_hml
    );
    println!(
        "{:<25} {:>12} {:>12} {:>12.6} {:>12}",
        "Beta MOM", "-", "-", carhart.beta_mom, "-"
    );
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12.6}",
        "Beta RMW", "-", "-", "-", ff5.beta_rmw
    );
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12.6}",
        "Beta CMA", "-", "-", "-", ff5.beta_cma
    );
    println!("{}", "-".repeat(80));
    println!(
        "{:<25} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        "R²", capm.r_squared, ff3.r_squared, carhart.r_squared, ff5.r_squared
    );
    println!(
        "{:<25} {:>11.4}% {:>11.4}% {:>11.4}% {:>11.4}%",
        "Tracking Error",
        capm.tracking_error * 100.0,
        ff3.tracking_error * 100.0,
        carhart.tracking_error * 100.0,
        ff5.tracking_error * 100.0
    );

    println!("\n{}", "-".repeat(80));
    println!("EVOLUÇÃO DO PODER EXPLICATIVO");
    println!("{}", "-".repeat(80));

    println!(
        "\nCAPM (1964):        R² = {:.4} ({:.2}%)",
        capm.r_squared,
        capm.r_squared * 100.0
    );
    println!(
        "FF3 (1993):         R² = {:.4} ({:.2}%)  [+{:.4}]",
        ff3.r_squared,
        ff3.r_squared * 100.0,
        ff3.r_squared - capm.r_squared
    );
    println!(
        "Carhart (1997):     R² = {:.4} ({:.2}%)  [+{:.4}]",
        carhart.r_squared,
        carhart.r_squared * 100.0,
        carhart.r_squared - ff3.r_squared
    );
    println!(
        "FF5 (2015):         R² = {:.4} ({:.2}%)  [+{:.4}]",
        ff5.r_squared,
        ff5.r_squared * 100.0,
        ff5.r_squared - carhart.r_squared
    );

    println!(
        "\nMelhoria Total: {:.4} ({:.2}% → {:.2}%)",
        ff5.r_squared - capm.r_squared,
        capm.r_squared * 100.0,
        ff5.r_squared * 100.0
    );

    println!("\n{}", "=".repeat(80));
    println!("RESULTADO DETALHADO: FAMA-FRENCH 5 FACTOR");
    println!("{}", "=".repeat(80));
    println!("{}", ff5);

    Ok(())
}
