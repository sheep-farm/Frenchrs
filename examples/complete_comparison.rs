use frenchrs::{
    APT, CAPM, Carhart4Factor, FamaFrench3Factor, FamaFrench5Factor, FamaFrench6Factor,
};
use greeners::CovarianceType;
use ndarray::{Array2, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("FRENCHRS - COMPARAÇÃO COMPLETA DE TODOS OS MODELOS");
    println!("{}", "=".repeat(80));
    println!("\nModels: CAPM → FF3 → Carhart → FF5 → FF6 → APT");

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
    let umd = array![
        0.012, 0.008, -0.015, 0.018, 0.010, -0.012, 0.015, 0.009, -0.010, 0.013, 0.007, 0.014
    ];

    let rf = 0.03 / 12.0;

    // ========================================================================
    // ESTIMAÇÃO DE TODOS OS MODELOS
    // ========================================================================
    println!("\n{}", "-".repeat(80));
    println!("Estimatesndthe models...");
    println!("{}", "-".repeat(80));

    let capm = CAPM::fit(&fund, &market, rf, CovarianceType::HC3)?;
    println!("✓ CAPM (1964)");

    let ff3 = FamaFrench3Factor::fit(&fund, &market, &smb, &hml, rf, CovarianceType::HC3)?;
    println!("✓ Fama-French 3 Factor (1993)");

    let carhart = Carhart4Factor::fit(&fund, &market, &smb, &hml, &mom, rf, CovarianceType::HC3)?;
    println!("✓ Carhart 4 Factor (1997)");

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
    println!("✓ Fama-French 5 Factor (2015)");

    let ff6 = FamaFrench6Factor::fit(
        &fund,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        rf,
        CovarianceType::HC3,
    )?;
    println!("✓ Fama-French 6 Factor (2023)");

    // APT with 6 factors (equivalente ao FF6, mas with framework genérico)
    let apt_factors = Array2::from_shape_vec(
        (12, 6),
        vec![
            // MKT-RF, SMB, HML, RMW, CMA, UMD
            market[0] - rf,
            smb[0],
            hml[0],
            rmw[0],
            cma[0],
            umd[0],
            market[1] - rf,
            smb[1],
            hml[1],
            rmw[1],
            cma[1],
            umd[1],
            market[2] - rf,
            smb[2],
            hml[2],
            rmw[2],
            cma[2],
            umd[2],
            market[3] - rf,
            smb[3],
            hml[3],
            rmw[3],
            cma[3],
            umd[3],
            market[4] - rf,
            smb[4],
            hml[4],
            rmw[4],
            cma[4],
            umd[4],
            market[5] - rf,
            smb[5],
            hml[5],
            rmw[5],
            cma[5],
            umd[5],
            market[6] - rf,
            smb[6],
            hml[6],
            rmw[6],
            cma[6],
            umd[6],
            market[7] - rf,
            smb[7],
            hml[7],
            rmw[7],
            cma[7],
            umd[7],
            market[8] - rf,
            smb[8],
            hml[8],
            rmw[8],
            cma[8],
            umd[8],
            market[9] - rf,
            smb[9],
            hml[9],
            rmw[9],
            cma[9],
            umd[9],
            market[10] - rf,
            smb[10],
            hml[10],
            rmw[10],
            cma[10],
            umd[10],
            market[11] - rf,
            smb[11],
            hml[11],
            rmw[11],
            cma[11],
            umd[11],
        ],
    )?;

    let apt_names = Some(vec![
        "MKT-RF".to_string(),
        "SMB".to_string(),
        "HML".to_string(),
        "RMW".to_string(),
        "CMA".to_string(),
        "UMD".to_string(),
    ]);

    let apt = APT::fit(&fund, &apt_factors, rf, CovarianceType::HC3, apt_names)?;
    println!("✓ APT (1976) - 6 factors customizados");

    // ========================================================================
    // TABELA COMPARATIVA DE PARAMETERS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO DE ESTIMATED PARAMETERS");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Parameter", "CAPM", "FF3", "Carhart", "FF5", "FF6", "APT-6"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<25} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Alpha (α)", capm.alpha, ff3.alpha, carhart.alpha, ff5.alpha, ff6.alpha, apt.alpha
    );

    println!(
        "{:<25} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Beta Market",
        capm.beta,
        ff3.beta_market,
        carhart.beta_market,
        ff5.beta_market,
        ff6.beta_market,
        apt.betas[0]
    );

    println!(
        "{:<25} {:>10} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Beta SMB", "-", ff3.beta_smb, carhart.beta_smb, ff5.beta_smb, ff6.beta_smb, apt.betas[1]
    );

    println!(
        "{:<25} {:>10} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Beta HML", "-", ff3.beta_hml, carhart.beta_hml, ff5.beta_hml, ff6.beta_hml, apt.betas[2]
    );

    println!(
        "{:<25} {:>10} {:>10} {:>10.6} {:>10} {:>10} {:>10}",
        "Beta MOM", "-", "-", carhart.beta_mom, "-", "-", "-"
    );

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10.6} {:>10.6} {:>10.6}",
        "Beta RMW", "-", "-", "-", ff5.beta_rmw, ff6.beta_rmw, apt.betas[3]
    );

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10.6} {:>10.6} {:>10.6}",
        "Beta CMA", "-", "-", "-", ff5.beta_cma, ff6.beta_cma, apt.betas[4]
    );

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10.6} {:>10.6}",
        "Beta UMD", "-", "-", "-", "-", ff6.beta_umd, apt.betas[5]
    );

    // ========================================================================
    // FIT QUALITY
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("EVOLUÇÃO DO PODER EXPLICATIVO");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<30} {:>12} {:>15} {:>12}",
        "Model", "R²", "R² Adjusted", "Track Error"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%",
        "CAPM (1 factor)",
        capm.r_squared,
        capm.adj_r_squared,
        capm.tracking_error * 100.0
    );

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%  [+{:.4}]",
        "FF3 (3 factors)",
        ff3.r_squared,
        ff3.adj_r_squared,
        ff3.tracking_error * 100.0,
        ff3.r_squared - capm.r_squared
    );

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%  [+{:.4}]",
        "Carhart (4 factors)",
        carhart.r_squared,
        carhart.adj_r_squared,
        carhart.tracking_error * 100.0,
        carhart.r_squared - ff3.r_squared
    );

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%  [+{:.4}]",
        "FF5 (5 factors)",
        ff5.r_squared,
        ff5.adj_r_squared,
        ff5.tracking_error * 100.0,
        ff5.r_squared - carhart.r_squared
    );

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%  [+{:.4}]",
        "FF6 (6 factors)",
        ff6.r_squared,
        ff6.adj_r_squared,
        ff6.tracking_error * 100.0,
        ff6.r_squared - ff5.r_squared
    );

    println!(
        "{:<30} {:>12.4} {:>15.4} {:>11.4}%",
        "APT-6 (6 factors)",
        apt.r_squared,
        apt.adj_r_squared,
        apt.tracking_error * 100.0
    );

    println!("\n{}", "-".repeat(80));
    println!(
        "Melhoria Total (CAPM → FF6): {:.4} ({:.2}% → {:.2}%)",
        ff6.r_squared - capm.r_squared,
        capm.r_squared * 100.0,
        ff6.r_squared * 100.0
    );

    // ========================================================================
    // SIGNIFICÂNCIA Dthe alpha
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS OF ALPHA (α = 0.05)");
    println!("{}", "=".repeat(80));

    let models = vec![
        ("CAPM", capm.alpha, capm.alpha_pvalue),
        ("FF3", ff3.alpha, ff3.alpha_pvalue),
        ("Carhart", carhart.alpha, carhart.alpha_pvalue),
        ("FF5", ff5.alpha, ff5.alpha_pvalue),
        ("FF6", ff6.alpha, ff6.alpha_pvalue),
        ("APT-6", apt.alpha, apt.alpha_pvalue),
    ];

    println!(
        "\n{:<15} {:>12} {:>12} {:>20}",
        "Model", "Alpha", "P-value", "Significance"
    );
    println!("{}", "-".repeat(80));

    for (name, alpha, pvalue) in models {
        let sig = if pvalue < 0.001 {
            "*** (p < 0.001)"
        } else if pvalue < 0.01 {
            "**  (p < 0.01)"
        } else if pvalue < 0.05 {
            "*   (p < 0.05)"
        } else {
            "    (not sig.)"
        };

        println!("{:<15} {:>12.6} {:>12.4} {:>20}", name, alpha, pvalue, sig);
    }

    // ========================================================================
    // CLASSIFICATIONS (FF6)
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CLASSIFICAÇÃO DO FUNDO (Baseado em FF6)");
    println!("{}", "=".repeat(80));

    println!("\nPerformance:     {}", ff6.performance_classification());
    println!("Size:         {}", ff6.size_classification());
    println!("Value:           {}", ff6.value_classification());
    println!("Profitability:   {}", ff6.profitability_classification());
    println!("Investment:    {}", ff6.investment_classification());
    println!("Momentum:        {}", ff6.momentum_classification());

    // ========================================================================
    // CONCLUSÃO
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CONCLUSÕES");
    println!("{}", "=".repeat(80));

    println!("\n1. EVOLUÇÃO HISTÓRICA DOS MODELOS:");
    println!("   • 1964: CAPM introduz o conceito of beta of market");
    println!("   • 1993: FF3 adiciona factors size (SMB) e value (HML)");
    println!("   • 1997: Carhart adiciona factor momentum");
    println!("   • 2015: FF5 adiciona profitability (RMW) e investment (CMA)");
    println!("   • 2023: FF6 combina FF5 with momentum (UMD)");
    println!("   • 1976: APT oferece framework flexível for N factors");

    println!("\n2. PODER EXPLICATIVO:");
    println!(
        "   • CAPM captura {:.1}% of the variestion",
        capm.r_squared * 100.0
    );
    println!(
        "   • FF6 captura {:.1}% of the variestion",
        ff6.r_squared * 100.0
    );
    println!(
        "   • Ganho of {:.1} pontos percentuais",
        (ff6.r_squared - capm.r_squared) * 100.0
    );

    println!("\n3. FRAMEWORK APT:");
    println!("   • Permite escolha flexível of factors");
    println!("   • Não restringe a número específico of factors");
    println!("   • Ideal for models customizados e pesquisa");

    println!("\n{}", "=".repeat(80));

    Ok(())
}
