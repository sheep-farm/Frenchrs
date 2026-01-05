use frenchrs::IVOLTrackingMulti;
use greeners::CovarianceType;
use ndarray::{array, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("IVOL & TRACKING ERROR - MÚLTIPLOS ATIVOS");
    println!("{}", "=".repeat(80));
    println!("\nAnálise similar ao Python: procthat multiple assets simultaneamente");

    // ========================================================================
    // DADOS SIMULADOS - 3 assets, 24 meses, 3 factors (Market, SMB, HML)
    // ========================================================================

    // Matriz of returns excedentes: 24 obbevations × 3 assets
    #[rustfmt::skip]
    let returns_excess = Array2::from_shape_vec((24, 3), vec![
        // Tech Fund, Value Fund, Growth Fund
        0.058, 0.045, 0.052,  // mês 1
        0.042, 0.035, 0.038,  // mês 2
        -0.018, -0.015, -0.012, // mês 3
        0.082, 0.070, 0.075,  // mês 4
        0.045, 0.040, 0.042,  // mês 5
        -0.032, -0.025, -0.028, // mês 6
        0.068, 0.055, 0.060,  // mês 7
        0.052, 0.045, 0.048,  // mês 8
        -0.015, -0.012, -0.010, // mês 9
        0.062, 0.050, 0.055,  // mês 10
        0.041, 0.038, 0.040,  // mês 11
        0.075, 0.065, 0.070,  // mês 12
        0.055, 0.048, 0.050,  // mês 13
        0.038, 0.032, 0.035,  // mês 14
        -0.012, -0.010, -0.008, // mês 15
        0.078, 0.068, 0.072,  // mês 16
        0.048, 0.042, 0.045,  // mês 17
        -0.028, -0.022, -0.025, // mês 18
        0.065, 0.058, 0.062,  // mês 19
        0.050, 0.045, 0.048,  // mês 20
        -0.018, -0.015, -0.012, // mês 21
        0.060, 0.055, 0.058,  // mês 22
        0.043, 0.040, 0.042,  // mês 23
        0.072, 0.065, 0.068,  // mês 24
    ])?;

    // Matriz of factors: 24 obbevations × 3 factors (MKT, SMB, HML)
    #[rustfmt::skip]
    let factors = Array2::from_shape_vec((24, 3), vec![
        // MKT,    SMB,    HML
        0.042,  0.008,  0.005,   // mês 1
        0.030, -0.003,  0.008,   // mês 2
        -0.015,  0.012, -0.010,  // mês 3
        0.060,  0.005,  0.012,   // mês 4
        0.033, -0.007,  0.003,   // mês 5
        -0.025,  0.015, -0.006,  // mês 6
        0.050,  0.002,  0.009,   // mês 7
        0.038, -0.005,  0.004,   // mês 8
        -0.012,  0.010, -0.008,  // mês 9
        0.045,  0.003,  0.011,   // mês 10
        0.032, -0.004,  0.002,   // mês 11
        0.055,  0.006,  0.007,   // mês 12
        0.040,  0.007,  0.006,   // mês 13
        0.028, -0.002,  0.007,   // mês 14
        -0.010,  0.011, -0.009,  // mês 15
        0.058,  0.004,  0.010,   // mês 16
        0.035, -0.006,  0.004,   // mês 17
        -0.022,  0.014, -0.005,  // mês 18
        0.048,  0.003,  0.008,   // mês 19
        0.036, -0.004,  0.005,   // mês 20
        -0.013,  0.009, -0.007,  // mês 21
        0.043,  0.002,  0.009,   // mês 22
        0.030, -0.003,  0.003,   // mês 23
        0.052,  0.005,  0.006,   // mês 24
    ])?;

    // Benchmark opcional (índice of market)
    #[rustfmt::skip]
    let benchmark = array![
        0.044, 0.031, -0.014, 0.062, 0.034, -0.024,
        0.051, 0.039, -0.011, 0.046, 0.033, 0.056,
        0.041, 0.029, -0.009, 0.059, 0.036, -0.021,
        0.049, 0.037, -0.012, 0.044, 0.031, 0.053,
    ];

    let asset_names = vec![
        "Tech Fund".to_string(),
        "Value Fund".to_string(),
        "Growth Fund".to_string(),
    ];

    // ========================================================================
    // ANÁLISE SEM BENCHMARK
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANÁLISE 1: IVOL SEM BENCHMARK");
    println!("{}", "=".repeat(80));

    let ivol_no_bench = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None, // without benchmark
        CovarianceType::HC3,
        Some(asset_names.clone()),
        12.0, // data mensais
    )?;

    println!(
        "\n{:<20} {:>15} {:>15} {:>15} {:>8}",
        "Asset", "Return Anual", "IVOL Monthly", "IVOL Anual", "R²"
    );
    println!("{}", "-".repeat(80));

    for asset_name in &asset_names {
        let asset = ivol_no_bench.get_asset(asset_name).unwrap();
        println!(
            "{:<20} {:>14.2}% {:>14.4}% {:>14.2}% {:>8.4}",
            asset_name,
            asset.mean_excess_annual * 100.0,
            asset.ivol_monthly * 100.0,
            asset.ivol_annual * 100.0,
            asset.r_squared
        );
    }

    // ========================================================================
    // ANÁLISE COM BENCHMARK
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANÁLISE 2: IVOL & TRACKING ERROR COM BENCHMARK");
    println!("{}", "=".repeat(80));

    let ivol_with_bench = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        Some(&benchmark), // with benchmark
        CovarianceType::HC3,
        Some(asset_names.clone()),
        12.0,
    )?;

    println!(
        "\n{:<15} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Asset", "Ret. Anual", "IVOL Anual", "TE Monthly", "TE Anual", "R²"
    );
    println!("{}", "-".repeat(80));

    for asset_name in &asset_names {
        let asset = ivol_with_bench.get_asset(asset_name).unwrap();
        println!(
            "{:<15} {:>11.2}% {:>11.2}% {:>11.4}% {:>11.2}% {:>8.4}",
            asset_name,
            asset.mean_excess_annual * 100.0,
            asset.ivol_annual * 100.0,
            asset.tracking_error_monthly.unwrap_or(0.0) * 100.0,
            asset.tracking_error_annual.unwrap_or(0.0) * 100.0,
            asset.r_squared
        );
    }

    // ========================================================================
    // CLASSIFICAÇÃO DE RISCO
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CLASSIFICAÇÃO DE RISCO (IVOL ANUAL)");
    println!("{}", "=".repeat(80));

    for asset_name in &asset_names {
        let asset = ivol_with_bench.get_asset(asset_name).unwrap();
        let ivol_pct = asset.ivol_annual * 100.0;

        let classification = if ivol_pct < 5.0 {
            "Muito Baixo"
        } else if ivol_pct < 10.0 {
            "Baixo"
        } else if ivol_pct < 15.0 {
            "Moderado"
        } else if ivol_pct < 20.0 {
            "Alto"
        } else {
            "Muito Alto"
        };

        println!(
            "  • {:<18} - IVOL: {:>6.2}% - Risk: {}",
            asset_name, ivol_pct, classification
        );
    }

    // ========================================================================
    // CLASSIFICAÇÃO DE TRACKING ERROR
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CLASSIFICAÇÃO DE TRACKING ERROR (ANUAL)");
    println!("{}", "=".repeat(80));

    for asset_name in &asset_names {
        let asset = ivol_with_bench.get_asset(asset_name).unwrap();
        if let Some(te_annual) = asset.tracking_error_annual {
            let te_pct = te_annual * 100.0;

            let classification = if te_pct < 1.0 {
                "Index Fund"
            } else if te_pct < 3.0 {
                "Enhanced Index"
            } else if te_pct < 6.0 {
                "Active"
            } else {
                "Highly Active"
            };

            println!(
                "  • {:<18} - TE: {:>6.2}% - Tipo: {}",
                asset_name, te_pct, classification
            );
        }
    }

    // ========================================================================
    // EXPORTAÇÃO TABULAR
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("FORMATO TABULAR (Compatível with Python/Pandas)");
    println!("{}", "=".repeat(80));

    let table = ivol_with_bench.to_table();
    println!("\nTotal of linhas: {}", table.len());
    println!("\nPrimeiras linhas:");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Asset", "Mean Annual", "IVOL Month", "IVOL Annual", "TE Month", "NObs"
    );
    println!("{}", "-".repeat(80));

    for row in table.iter().take(3) {
        println!(
            "{:<15} {:>11.4}% {:>11.4}% {:>11.4}% {:>11.4}% {:>8}",
            row.asset,
            row.mean_excess_annual * 100.0,
            row.ivol_monthly * 100.0,
            row.ivol_annual * 100.0,
            row.tracking_error_monthly.unwrap_or(0.0) * 100.0,
            row.nobs
        );
    }

    // ========================================================================
    // EXPORTAÇÃO CSV
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("EXPORTAÇÃO CSV");
    println!("{}", "=".repeat(80));

    let csv = ivol_with_bench.to_csv_string();
    let csv_lines: Vec<&str> = csv.lines().collect();

    println!("\nPrimeiras linhas of the CSV:");
    println!("{}", "-".repeat(80));
    for line in csv_lines.iter().take(5) {
        println!("{}", line);
    }

    // Salvar em arquivo (opcional)
    // std::fs::write("ivol_tracking.csv", csv)?;
    // println!("\n✓ Arquivo 'ivol_tracking.csv' salvo with sucesso!");

    // ========================================================================
    // COMPARAÇÃO ENTRE ATIVOS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO ENTRE ATIVOS");
    println!("{}", "=".repeat(80));

    // Encontrar asset with greater IVOL
    let max_ivol_asset = asset_names
        .iter()
        .max_by(|a, b| {
            let ivol_a = ivol_with_bench.get_asset(a).unwrap().ivol_annual;
            let ivol_b = ivol_with_bench.get_asset(b).unwrap().ivol_annual;
            ivol_a.partial_cmp(&ivol_b).unwrap()
        })
        .unwrap();

    println!("\n✓ Maior IVOL: {}", max_ivol_asset);
    println!(
        "  IVOL Anual: {:.2}%",
        ivol_with_bench
            .get_asset(max_ivol_asset)
            .unwrap()
            .ivol_annual
            * 100.0
    );

    // Encontrar asset with greater Tracking Error
    let max_te_asset = asset_names
        .iter()
        .max_by(|a, b| {
            let te_a = ivol_with_bench
                .get_asset(a)
                .unwrap()
                .tracking_error_annual
                .unwrap_or(0.0);
            let te_b = ivol_with_bench
                .get_asset(b)
                .unwrap()
                .tracking_error_annual
                .unwrap_or(0.0);
            te_a.partial_cmp(&te_b).unwrap()
        })
        .unwrap();

    println!("\n✓ Maior Tracking Error: {}", max_te_asset);
    println!(
        "  TE Anual: {:.2}%",
        ivol_with_bench
            .get_asset(max_te_asset)
            .unwrap()
            .tracking_error_annual
            .unwrap_or(0.0)
            * 100.0
    );

    // Encontrar asset with better R²
    let best_r2_asset = asset_names
        .iter()
        .max_by(|a, b| {
            let r2_a = ivol_with_bench.get_asset(a).unwrap().r_squared;
            let r2_b = ivol_with_bench.get_asset(b).unwrap().r_squared;
            r2_a.partial_cmp(&r2_b).unwrap()
        })
        .unwrap();

    println!("\n✓ Melhor fit aos factors: {}", best_r2_asset);
    println!(
        "  R²: {:.4}",
        ivol_with_bench.get_asset(best_r2_asset).unwrap().r_squared
    );

    // ========================================================================
    // INTERPRETATION
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION DOS RESULTS");
    println!("{}", "=".repeat(80));

    println!("\n1. IDIOSYNCRATIC VOLATILITY (IVOL):");
    println!("   • Mede a volatility not explieach by the factors of risk");
    println!("   • IVOL alto = greater risk específico of the asset");
    println!("   • IVOL baixo = returns bem explicados by the factors");

    println!("\n2. TRACKING ERROR (TE):");
    println!("   • Mede o desvio relative to ao benchmark");
    println!("   • TE < 1%: Fundo passivo (index fund)");
    println!("   • TE 1-3%: Enhanced index");
    println!("   • TE 3-6%: Gestão ativa moderada");
    println!("   • TE > 6%: Gestão altamente ativa");

    println!("\n3. R² (COEFICIENTE DE DETERMINAÇÃO):");
    println!("   • Proporção of the variesnce explieach by the factors");
    println!("   • R² alto = returns bem modelados");
    println!("   • R² baixo = idiosyncratic risk elevado");

    println!("\n4. FORMATO DE DADOS:");
    println!("   • Tabela with 1 linha por asset");
    println!("   • Compatível with análises Python/Pandas");
    println!("   • Exportação CSV for análises posteriores");

    println!("\n{}", "=".repeat(80));

    Ok(())
}
