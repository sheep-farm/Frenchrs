use frenchrs::APT;
use greeners::CovarianceType;
use ndarray::{Array2, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("ARBITRAGE PRICING THEORY (APT) - DEMONSTRAÇÃO");
    println!("{}", "=".repeat(80));

    // Returns mensais simulados of um fundo
    let fund_returns = array![
        0.058, 0.042, -0.018, 0.082, 0.045, -0.032, 0.068, 0.052, -0.015, 0.062, 0.041, 0.075
    ];

    let rf = 0.03 / 12.0; // Risk-free rate monthly

    // ========================================================================
    // EXEMPLO 1: APT with 2 factors (Market e Momentum)
    // ========================================================================
    println!("\n{}", "-".repeat(80));
    println!("EXEMPLO 1: APT with 2 Factors (Market + Momentum)");
    println!("{}", "-".repeat(80));

    let factors_2 = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Market, Momentum
            0.042, 0.012, 0.030, 0.008, -0.015, -0.015, 0.060, 0.018, 0.033, 0.010, -0.025, -0.012,
            0.050, 0.015, 0.038, 0.009, -0.012, -0.010, 0.045, 0.013, 0.032, 0.007, 0.055, 0.014,
        ],
    )?;

    let factor_names_2 = Some(vec!["Market".to_string(), "Momentum".to_string()]);

    let apt2 = APT::fit(
        &fund_returns,
        &factors_2,
        rf,
        CovarianceType::HC3,
        factor_names_2,
    )?;

    println!("{}", apt2);

    // ========================================================================
    // EXEMPLO 2: APT with 4 factors (Market, Size, Value, Momentum)
    // ========================================================================
    println!("\n{}", "-".repeat(80));
    println!("EXEMPLO 2: APT with 4 Factors (Market + SMB + HML + Momentum)");
    println!("{}", "-".repeat(80));

    let factors_4 = Array2::from_shape_vec(
        (12, 4),
        vec![
            // Market, SMB, HML, Momentum
            0.042, 0.008, 0.005, 0.012, 0.030, -0.003, 0.008, 0.008, -0.015, 0.012, -0.010, -0.015,
            0.060, 0.005, 0.012, 0.018, 0.033, -0.007, 0.003, 0.010, -0.025, 0.015, -0.006, -0.012,
            0.050, 0.002, 0.009, 0.015, 0.038, -0.005, 0.004, 0.009, -0.012, 0.010, -0.008, -0.010,
            0.045, 0.003, 0.011, 0.013, 0.032, -0.004, 0.002, 0.007, 0.055, 0.006, 0.007, 0.014,
        ],
    )?;

    let factor_names_4 = Some(vec![
        "Market".to_string(),
        "SMB".to_string(),
        "HML".to_string(),
        "Momentum".to_string(),
    ]);

    let apt4 = APT::fit(
        &fund_returns,
        &factors_4,
        rf,
        CovarianceType::HC3,
        factor_names_4,
    )?;

    println!("{}", apt4);

    // ========================================================================
    // EXEMPLO 3: APT with 6 factors customizados
    // ========================================================================
    println!("\n{}", "-".repeat(80));
    println!("EXEMPLO 3: APT with 6 Factors Customizados");
    println!("{}", "-".repeat(80));

    let factors_6 = Array2::from_shape_vec(
        (12, 6),
        vec![
            // Market, SMB, HML, RMW, CMA, Momentum
            0.042, 0.008, 0.005, 0.006, 0.003, 0.012, 0.030, -0.003, 0.008, 0.004, -0.002, 0.008,
            -0.015, 0.012, -0.010, -0.005, 0.005, -0.015, 0.060, 0.005, 0.012, 0.008, 0.002, 0.018,
            0.033, -0.007, 0.003, 0.003, -0.003, 0.010, -0.025, 0.015, -0.006, -0.004, 0.004,
            -0.012, 0.050, 0.002, 0.009, 0.007, 0.003, 0.015, 0.038, -0.005, 0.004, 0.005, -0.002,
            0.009, -0.012, 0.010, -0.008, -0.003, 0.004, -0.010, 0.045, 0.003, 0.011, 0.006, 0.003,
            0.013, 0.032, -0.004, 0.002, 0.004, -0.001, 0.007, 0.055, 0.006, 0.007, 0.007, 0.003,
            0.014,
        ],
    )?;

    let factor_names_6 = Some(vec![
        "Market".to_string(),
        "SMB".to_string(),
        "HML".to_string(),
        "RMW".to_string(),
        "CMA".to_string(),
        "Momentum".to_string(),
    ]);

    let apt6 = APT::fit(
        &fund_returns,
        &factors_6,
        rf,
        CovarianceType::HC3,
        factor_names_6,
    )?;

    println!("{}", apt6);

    // ========================================================================
    // COMPARAÇÃO: Poder Explicativo
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO: EVOLUÇÃO DO PODER EXPLICATIVO");
    println!("{}", "=".repeat(80));
    println!("\n{:<30} {:>12} {:>12}", "Model", "R²", "R² Adjusted");
    println!("{}", "-".repeat(80));
    println!(
        "{:<30} {:>12.4} {:>12.4}",
        "APT-2 (Market + Mom)", apt2.r_squared, apt2.adj_r_squared
    );
    println!(
        "{:<30} {:>12.4} {:>12.4}  [+{:.4}]",
        "APT-4 (+ SMB + HML)",
        apt4.r_squared,
        apt4.adj_r_squared,
        apt4.r_squared - apt2.r_squared
    );
    println!(
        "{:<30} {:>12.4} {:>12.4}  [+{:.4}]",
        "APT-6 (+ RMW + CMA)",
        apt6.r_squared,
        apt6.adj_r_squared,
        apt6.r_squared - apt4.r_squared
    );

    println!(
        "\nGanho Total: {:.4} pontos of R²",
        apt6.r_squared - apt2.r_squared
    );

    // ========================================================================
    // EXEMPLO 4: Previare with APT
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("PREVISÃO DE RETORNO ESPERADO");
    println!("{}", "=".repeat(80));

    // Cenário: expectativas for os próximos factors
    let expected_factors = array![0.08, 0.01, 0.02, 0.01, 0.005, 0.015];

    let expected_return = apt6.expected_return(&expected_factors);

    println!("\nExpectativas of Factors:");
    println!("  Market:   {:.2}%", expected_factors[0] * 100.0);
    println!("  SMB:       {:.2}%", expected_factors[1] * 100.0);
    println!("  HML:       {:.2}%", expected_factors[2] * 100.0);
    println!("  RMW:       {:.2}%", expected_factors[3] * 100.0);
    println!("  CMA:       {:.2}%", expected_factors[4] * 100.0);
    println!("  Momentum:  {:.2}%", expected_factors[5] * 100.0);
    println!(
        "\nReturn Esperado of the Fundo: {:.2}%",
        expected_return * 100.0
    );

    // ========================================================================
    // ANALYSIS OF SIGNIFICÂNCIA
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS OF SIGNIFICÂNCIA DOS FATORES (α = 0.05)");
    println!("{}", "=".repeat(80));

    println!("\nAPT-6 Factors:");
    let factor_labels = ["Market", "SMB", "HML", "RMW", "CMA", "Momentum"];
    for (i, label) in factor_labels.iter().enumerate() {
        let is_sig = apt6.factor_is_significant(i, 0.05);
        let marker = if is_sig {
            "✓ Significante"
        } else {
            "✗ Não significante"
        };
        println!("  {:12} (β = {:>7.4}): {}", label, apt6.betas[i], marker);
    }

    if apt6.is_significantly_outperforming(0.05) {
        println!("\n>>> O fundo is gerandthe alpha positivo significant!");
    } else if apt6.alpha > 0.0 {
        println!("\n>>> O fundo tem alpha positivo, mas not é thististicamente significante.");
    } else {
        println!("\n>>> O fundo not is gerandthe alpha positivo.");
    }

    println!("\n{}", "=".repeat(80));

    Ok(())
}
