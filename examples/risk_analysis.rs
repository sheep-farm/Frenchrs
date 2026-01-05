use frenchrs::{
    Carhart4Factor, FamaFrench3Factor, FamaFrench5Factor, IVOLAnalysis, TrackingErrorAnalysis, CAPM,
};
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS OF RISCO: IVOL & TRACKING ERROR");
    println!("{}", "=".repeat(80));

    // ========================================================================
    // DADOS SIMULADOS
    // ========================================================================
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

    // ========================================================================
    // ESTIMAÇÃO DOS MODELOS
    // ========================================================================
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

    // ========================================================================
    // ANALYSIS OF IVOL
    // ========================================================================
    println!("{}", "=".repeat(80));
    println!("COMPARAÇÃO DE IVOL (IDIOSYNCRATIC VOLATILITY)");
    println!("{}", "=".repeat(80));

    let ivol_capm = IVOLAnalysis::from_residuals(&capm.residuals)?;
    let ivol_ff3 = IVOLAnalysis::from_residuals(&ff3.residuals)?;
    let ivol_carhart = IVOLAnalysis::from_residuals(&carhart.residuals)?;
    let ivol_ff5 = IVOLAnalysis::from_residuals(&ff5.residuals)?;

    println!(
        "\n{:<20} {:>15} {:>20} {:>20}",
        "Model", "IVOL", "IVOL Anual (Monthly)", "Classification"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:>14.4}% {:>19.2}% {:>20}",
        "CAPM",
        ivol_capm.ivol * 100.0,
        ivol_capm.ivol_annualized_monthly * 100.0,
        ivol_capm.ivol_classification()
    );

    println!(
        "{:<20} {:>14.4}% {:>19.2}% {:>20}",
        "FF3",
        ivol_ff3.ivol * 100.0,
        ivol_ff3.ivol_annualized_monthly * 100.0,
        ivol_ff3.ivol_classification()
    );

    println!(
        "{:<20} {:>14.4}% {:>19.2}% {:>20}",
        "Carhart",
        ivol_carhart.ivol * 100.0,
        ivol_carhart.ivol_annualized_monthly * 100.0,
        ivol_carhart.ivol_classification()
    );

    println!(
        "{:<20} {:>14.4}% {:>19.2}% {:>20}",
        "FF5",
        ivol_ff5.ivol * 100.0,
        ivol_ff5.ivol_annualized_monthly * 100.0,
        ivol_ff5.ivol_classification()
    );

    println!("\n{}", "-".repeat(80));
    println!("INTERPRETATION:");
    println!("• IVOL diminui à medida que adicionamos factors to the model");
    println!("• Menor IVOL indica que the model explica better the returns");
    println!("• IVOL representa the risk diversificável/específico of the asset");

    // ========================================================================
    // ANALYSIS OF TRACKING ERROR
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO DE TRACKING ERROR");
    println!("{}", "=".repeat(80));

    let te_capm =
        TrackingErrorAnalysis::new(&fund, &capm.fitted_values, capm.alpha, capm.r_squared)?;
    let te_ff3 = TrackingErrorAnalysis::new(&fund, &ff3.fitted_values, ff3.alpha, ff3.r_squared)?;
    let te_carhart = TrackingErrorAnalysis::new(
        &fund,
        &carhart.fitted_values,
        carhart.alpha,
        carhart.r_squared,
    )?;
    let te_ff5 = TrackingErrorAnalysis::new(&fund, &ff5.fitted_values, ff5.alpha, ff5.r_squared)?;

    println!(
        "\n{:<20} {:>12} {:>18} {:>18} {:>15}",
        "Model", "TE", "TE Anual (Monthly)", "Info Ratio", "Classification"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:>11.4}% {:>17.2}% {:>18.4} {:>15}",
        "CAPM",
        te_capm.tracking_error * 100.0,
        te_capm.tracking_error_annualized_monthly * 100.0,
        te_capm.information_ratio,
        te_capm.te_classification()
    );

    println!(
        "{:<20} {:>11.4}% {:>17.2}% {:>18.4} {:>15}",
        "FF3",
        te_ff3.tracking_error * 100.0,
        te_ff3.tracking_error_annualized_monthly * 100.0,
        te_ff3.information_ratio,
        te_ff3.te_classification()
    );

    println!(
        "{:<20} {:>11.4}% {:>17.2}% {:>18.4} {:>15}",
        "Carhart",
        te_carhart.tracking_error * 100.0,
        te_carhart.tracking_error_annualized_monthly * 100.0,
        te_carhart.information_ratio,
        te_carhart.te_classification()
    );

    println!(
        "{:<20} {:>11.4}% {:>17.2}% {:>18.4} {:>15}",
        "FF5",
        te_ff5.tracking_error * 100.0,
        te_ff5.tracking_error_annualized_monthly * 100.0,
        te_ff5.information_ratio,
        te_ff5.te_classification()
    );

    println!("\n{}", "-".repeat(80));
    println!("INTERPRETATION:");
    println!("• Tracking Error measures o how much o fundo desvia of the model");
    println!("• Information Ratio = Alpha / Tracking Error");
    println!("• IR > 0.5 é considerado bom, > 1.0 é excelente");

    // ========================================================================
    // ANALYSIS OFTALHADA: CAPM
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS OFTALHADA: CAPM");
    println!("{}", "=".repeat(80));

    println!("{}", ivol_capm);
    println!("{}", te_capm);

    // ========================================================================
    // ANALYSIS OFTALHADA: FAMA-FRENCH 5 FACTOR
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS OFTALHADA: FAMA-FRENCH 5 FACTOR");
    println!("{}", "=".repeat(80));

    println!("{}", ivol_ff5);
    println!("{}", te_ff5);

    // ========================================================================
    // STATISTICS COMPARATIVAS Dthe residuals
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("RESIDUAL STATISTICS");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Model", "Mean", "Skewness", "Kurtosis", "Normal?"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:>12.6} {:>12.4} {:>12.4} {:>12}",
        "CAPM",
        ivol_capm.residual_mean,
        ivol_capm.residual_skewness,
        ivol_capm.residual_kurtosis,
        if ivol_capm.is_residuals_normal(0.05) {
            "SIM"
        } else {
            "NÃO"
        }
    );

    println!(
        "{:<20} {:>12.6} {:>12.4} {:>12.4} {:>12}",
        "FF3",
        ivol_ff3.residual_mean,
        ivol_ff3.residual_skewness,
        ivol_ff3.residual_kurtosis,
        if ivol_ff3.is_residuals_normal(0.05) {
            "SIM"
        } else {
            "NÃO"
        }
    );

    println!(
        "{:<20} {:>12.6} {:>12.4} {:>12.4} {:>12}",
        "Carhart",
        ivol_carhart.residual_mean,
        ivol_carhart.residual_skewness,
        ivol_carhart.residual_kurtosis,
        if ivol_carhart.is_residuals_normal(0.05) {
            "SIM"
        } else {
            "NÃO"
        }
    );

    println!(
        "{:<20} {:>12.6} {:>12.4} {:>12.4} {:>12}",
        "FF5",
        ivol_ff5.residual_mean,
        ivol_ff5.residual_skewness,
        ivol_ff5.residual_kurtosis,
        if ivol_ff5.is_residuals_normal(0.05) {
            "SIM"
        } else {
            "NÃO"
        }
    );

    println!("\n{}", "-".repeat(80));
    println!("INTERPRETATION:");
    println!("• Mean of the residuals should be próxima of zero");
    println!("• Skewness measures assimetria (0 = simétrico)");
    println!("• Kurtosis measures caudas pesadas (0 = normal)");
    println!("• Test of normalidade: Jarque-Bera a 5%");

    // ========================================================================
    // METRICS DE FIT QUALITY
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("FIT QUALITY");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Model", "R²", "Correlação", "RMSE", "MAE"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:>12.4} {:>12.4} {:>12.6} {:>12.6}",
        "CAPM", te_capm.r_squared, te_capm.correlation, te_capm.rmse, te_capm.mae
    );

    println!(
        "{:<20} {:>12.4} {:>12.4} {:>12.6} {:>12.6}",
        "FF3", te_ff3.r_squared, te_ff3.correlation, te_ff3.rmse, te_ff3.mae
    );

    println!(
        "{:<20} {:>12.4} {:>12.4} {:>12.6} {:>12.6}",
        "Carhart", te_carhart.r_squared, te_carhart.correlation, te_carhart.rmse, te_carhart.mae
    );

    println!(
        "{:<20} {:>12.4} {:>12.4} {:>12.6} {:>12.6}",
        "FF5", te_ff5.r_squared, te_ff5.correlation, te_ff5.rmse, te_ff5.mae
    );

    println!("\n{}", "-".repeat(80));
    println!("INTERPRETATION:");
    println!("• R² = proportion of the variesnce explieach");
    println!("• Correlação = relação linear between obbevado e prevthis");
    println!("• RMSE = erro quadrático médio");
    println!("• MAE = erro absoluto médio");

    // ========================================================================
    // CONCLUSÕES
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CONCLUSÕES");
    println!("{}", "=".repeat(80));

    println!("\n1. EVOLUÇÃO DO IVOL:");
    println!(
        "   • CAPM:    {:.2}% (annualized)",
        ivol_capm.ivol_annualized_monthly * 100.0
    );
    println!(
        "   • FF5:     {:.2}% (annualized)",
        ivol_ff5.ivol_annualized_monthly * 100.0
    );
    println!(
        "   • Redução: {:.2} pontos percentuais",
        (ivol_capm.ivol_annualized_monthly - ivol_ff5.ivol_annualized_monthly) * 100.0
    );

    println!("\n2. INFORMATION RATIO:");
    println!("   • CAPM:    {:.4}", te_capm.information_ratio);
    println!("   • FF5:     {:.4}", te_ff5.information_ratio);
    println!(
        "   • Melhor model por IR: {}",
        if te_ff5.information_ratio.abs() > te_capm.information_ratio.abs() {
            "FF5"
        } else {
            "CAPM"
        }
    );

    println!("\n3. DISTRIBUIÇÃO Dthe residuals:");
    println!(
        "   • Models with residuals normore: {}",
        [
            ("CAPM", ivol_capm.is_residuals_normal(0.05)),
            ("FF3", ivol_ff3.is_residuals_normal(0.05)),
            ("Carhart", ivol_carhart.is_residuals_normal(0.05)),
            ("FF5", ivol_ff5.is_residuals_normal(0.05))
        ]
        .iter()
        .filter(|(_, is_normal)| *is_normal)
        .map(|(name, _)| *name)
        .collect::<Vec<&str>>()
        .join(", ")
    );

    println!("\n4. RECOMENDAÇÃO:");
    if ivol_ff5.ivol < ivol_capm.ivol * 0.8 {
        println!("   • FF5 reduz significantly o IVOL (>20%)");
        println!("   • Recomenda-if usar FF5 for better explicação of the returns");
    } else {
        println!("   • Ganho marginal with models multi-fatoriais");
        println!("   • CAPM can be suficiente for this asset");
    }

    println!("\n{}", "=".repeat(80));

    Ok(())
}
