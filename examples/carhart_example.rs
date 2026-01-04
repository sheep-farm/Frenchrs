use frenchrs::{CAPM, Carhart4Factor, FamaFrench3Factor};
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: Comparison CAPM vs FF3 vs Carhart 4 Factor");
    println!("{}", "=".repeat(80));

    // Data simulados of um fundo momentum
    let fund_returns = array![
        0.058, 0.042, -0.018, 0.082, 0.045, -0.032, 0.068, 0.052, -0.015, 0.062, 0.041, 0.075,
    ];

    let market_returns = array![
        0.042, 0.030, -0.015, 0.060, 0.033, -0.025, 0.050, 0.038, -0.012, 0.045, 0.032, 0.055,
    ];

    let smb_returns = array![
        0.008, -0.003, 0.012, 0.005, -0.007, 0.015, 0.002, -0.005, 0.010, 0.003, -0.004, 0.006,
    ];

    let hml_returns = array![
        0.005, 0.008, -0.010, 0.012, 0.003, -0.006, 0.009, 0.004, -0.008, 0.011, 0.002, 0.007,
    ];

    // Factor MOM (Momentum) - winners minus lobes
    let mom_returns = array![
        0.012, 0.008, -0.015, 0.018, 0.010, -0.012, 0.015, 0.009, -0.010, 0.013, 0.007, 0.014,
    ];

    let risk_free = 0.03 / 12.0; // 3% ao ano

    // Estimates os 3 models
    println!("\nEstimatesndo CAPM...");
    let capm = CAPM::fit(
        &fund_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )?;

    println!("Estimatesndo Fama-French 3 Factor...");
    let ff3 = FamaFrench3Factor::fit(
        &fund_returns,
        &market_returns,
        &smb_returns,
        &hml_returns,
        risk_free,
        CovarianceType::HC3,
    )?;

    println!("Estimatesndo Carhart 4 Factor...\n");
    let carhart = Carhart4Factor::fit(
        &fund_returns,
        &market_returns,
        &smb_returns,
        &hml_returns,
        &mom_returns,
        risk_free,
        CovarianceType::HC3,
    )?;

    // Exibir result completo of the Carhart
    println!("{}", carhart);

    // Tabela comparativa
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO DOS TRÊS MODELOS");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<25} {:>15} {:>15} {:>15}",
        "Métrica", "CAPM", "FF3", "Carhart 4F"
    );
    println!("{}", "-".repeat(75));

    println!(
        "{:<25} {:>14.6} {:>14.6} {:>14.6}",
        "Alpha", capm.alpha, ff3.alpha, carhart.alpha
    );
    println!(
        "{:<25} {:>14.6} {:>14.6} {:>14.6}",
        "Beta Market", capm.beta, ff3.beta_market, carhart.beta_market
    );
    println!(
        "{:<25} {:>15} {:>14.6} {:>14.6}",
        "Beta SMB", "-", ff3.beta_smb, carhart.beta_smb
    );
    println!(
        "{:<25} {:>15} {:>14.6} {:>14.6}",
        "Beta HML", "-", ff3.beta_hml, carhart.beta_hml
    );
    println!(
        "{:<25} {:>15} {:>15} {:>14.6}",
        "Beta MOM", "-", "-", carhart.beta_mom
    );

    println!("{}", "-".repeat(75));

    println!(
        "{:<25} {:>14.4} {:>14.4} {:>14.4}",
        "R²", capm.r_squared, ff3.r_squared, carhart.r_squared
    );
    println!(
        "{:<25} {:>13.4}% {:>13.4}% {:>13.4}%",
        "Tracking Error",
        capm.tracking_error * 100.0,
        ff3.tracking_error * 100.0,
        carhart.tracking_error * 100.0
    );
    println!(
        "{:<25} {:>14.4} {:>14.4} {:>14.4}",
        "Information Ratio",
        capm.information_ratio,
        ff3.information_ratio,
        carhart.information_ratio
    );

    println!("\n{}", "-".repeat(80));
    println!("EVOLUÇÃO DO PODER EXPLICATIVO");
    println!("{}", "-".repeat(80));

    let r2_improvement_ff3 = ff3.r_squared - capm.r_squared;
    let r2_improvement_carhart = carhart.r_squared - ff3.r_squared;
    let te_reduction_ff3 = (capm.tracking_error - ff3.tracking_error) / capm.tracking_error;
    let te_reduction_carhart = (ff3.tracking_error - carhart.tracking_error) / ff3.tracking_error;

    println!("\nCAPM → FF3:");
    println!(
        "  Aumento of R²: {:.4} ({:.2}% → {:.2}%)",
        r2_improvement_ff3,
        capm.r_squared * 100.0,
        ff3.r_squared * 100.0
    );
    println!(
        "  Redução of Tracking Error: {:.2}%",
        te_reduction_ff3 * 100.0
    );

    println!("\nFF3 → Carhart 4F:");
    println!(
        "  Aumento of R²: {:.4} ({:.2}% → {:.2}%)",
        r2_improvement_carhart,
        ff3.r_squared * 100.0,
        carhart.r_squared * 100.0
    );
    println!(
        "  Redução of Tracking Error: {:.2}%",
        te_reduction_carhart * 100.0
    );

    println!("\nCAPM → Carhart 4F (total):");
    let r2_total = carhart.r_squared - capm.r_squared;
    let te_total = (capm.tracking_error - carhart.tracking_error) / capm.tracking_error;
    println!(
        "  Aumento of R²: {:.4} ({:.2}% → {:.2}%)",
        r2_total,
        capm.r_squared * 100.0,
        carhart.r_squared * 100.0
    );
    println!("  Redução of Tracking Error: {:.2}%", te_total * 100.0);

    println!("\n{}", "-".repeat(80));
    println!("IMPORTÂNCIA DO FATOR MOMENTUM");
    println!("{}", "-".repeat(80));

    if carhart.is_mom_significant(0.05) {
        println!("\n✓ MOMENTUM É SIGNIFICATIVO (p < 0.05)");
        println!(
            "  Beta MOM: {:.4} (p-value: {:.4})",
            carhart.beta_mom, carhart.beta_mom_pvalue
        );

        if carhart.beta_mom > 0.0 {
            println!("\n  → Fundo exibe MOMENTUM POSITIVO");
            println!("  → Estrup togia: TREND FOLLOWING (follows tendências)");
            println!(
                "  → Quando winners underem 1%, fundo undere {:.2}% adicional",
                carhart.beta_mom * 100.0
            );
        } else {
            println!("\n  → Fundo exibe REVERSÃO (contrarian)");
            println!("  → Estrup togia: compra perdedores, vende vencedores");
        }

        let (_, _, _, mom_contrib) = carhart.factor_contributions();
        println!(
            "\n  Contribuição for return: {:.4}% por period",
            mom_contrib * 100.0
        );
    } else {
        println!("\n○ Momentum NÃO é significant");
        println!("  Fama-French 3 Factor beia suficiente for this asset");
    }

    println!("\n{}", "=".repeat(80));
    Ok(())
}
