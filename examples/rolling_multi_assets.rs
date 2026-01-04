use frenchrs::RollingBetasMulti;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("ROLLING BETAS - MÚLTIPLOS ATIVOS");
    println!("{}", "=".repeat(80));
    println!("\nAnálise similar ao Python: processa múltiplos ativos simultaneamente");

    // ========================================================================
    // DADOS SIMULADOS - 3 ativos, 24 meses, 3 fatores (Mercado, SMB, HML)
    // ========================================================================

    // Matriz de retornos: 24 observações × 3 ativos
    #[rustfmt::skip]
    let returns = Array2::from_shape_vec((24, 3), vec![
        // Asset1, Asset2, Asset3
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

    // Matriz de fatores: 24 observações × 3 fatores (MKT, SMB, HML)
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

    let asset_names = vec![
        "Tech Fund".to_string(),
        "Value Fund".to_string(),
        "Growth Fund".to_string(),
    ];

    let factor_names = vec!["Market".to_string(), "SMB".to_string(), "HML".to_string()];

    // ========================================================================
    // CALCULAR ROLLING BETAS - Janela de 12 meses
    // ========================================================================
    println!("\n{}", "-".repeat(80));
    println!("Configuração:");
    println!("  • Ativos: {}", asset_names.len());
    println!("  • Fatores: {}", factor_names.len());
    println!("  • Observações: {}", returns.nrows());
    println!("  • Janela: 12 meses");
    println!("{}", "-".repeat(80));

    let rolling = RollingBetasMulti::fit(
        &returns,
        &factors,
        12, // janela de 12 meses
        CovarianceType::HC3,
        Some(asset_names.clone()),
        Some(factor_names.clone()),
    )?;

    println!("\n✓ Rolling Betas calculado com sucesso!");
    println!(
        "  • Número de janelas: {}",
        rolling.results.values().next().unwrap().n_windows
    );

    // ========================================================================
    // ANÁLISE POR ATIVO
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANÁLISE POR ATIVO");
    println!("{}", "=".repeat(80));

    for asset_name in rolling.asset_names() {
        let asset_result = rolling.get_asset(&asset_name).unwrap();

        println!("\n{}", "-".repeat(80));
        println!("ATIVO: {}", asset_name);
        println!("{}", "-".repeat(80));

        println!("\nEstatísticas dos Betas:");
        println!(
            "{:<20} {:>12} {:>12} {:>12}",
            "Fator", "Média", "Std Dev", "CV"
        );
        println!("{}", "-".repeat(60));

        for (i, factor_name) in factor_names.iter().enumerate() {
            println!(
                "{:<20} {:>12.4} {:>12.4} {:>12.4}",
                factor_name,
                asset_result.mean_beta(i),
                asset_result.std_beta(i),
                asset_result.cv_beta(i)
            );
        }

        // Análise de estabilidade avançada
        println!("\nAnálise de Estabilidade Avançada:");
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>25}",
            "Fator", "CV", "Tendência", "Autocorr", "Classificação"
        );
        println!("{}", "-".repeat(80));

        for (i, factor_name) in factor_names.iter().enumerate() {
            let stability = asset_result.beta_stability(i);
            println!(
                "{:<15} {:>12.4} {:>12.6} {:>12.4} {:>25}",
                factor_name,
                stability.coefficient_of_variation,
                stability.trend,
                stability.autocorrelation,
                stability.stability_classification()
            );
        }

        println!("\nTendências:");
        for (i, factor_name) in factor_names.iter().enumerate() {
            let stability = asset_result.beta_stability(i);
            println!(
                "  • {:<15} - {}",
                factor_name,
                stability.trend_classification()
            );
        }

        // Verificar se betas são estáveis (CV < 10%)
        println!("\nBetas Estáveis (CV < 10%):");
        for (i, factor_name) in factor_names.iter().enumerate() {
            let is_stable = asset_result.is_beta_stable(i, 0.1);
            let status = if is_stable { "✓ SIM" } else { "✗ NÃO" };
            println!("  • {:<15} - {}", factor_name, status);
        }
    }

    // ========================================================================
    // TABELA CONSOLIDADA
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("TABELA CONSOLIDADA (Primeiras 10 linhas)");
    println!("{}", "=".repeat(80));

    let table = rolling.to_table();

    println!(
        "\n{:<15} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Asset", "Date", "Alpha", "β_Market", "β_SMB", "β_HML", "R²"
    );
    println!("{}", "-".repeat(80));

    for row in table.iter().take(10) {
        println!(
            "{:<15} {:>8} {:>10.6} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            row.asset,
            row.date_idx,
            row.alpha,
            row.betas[0],
            row.betas[1],
            row.betas[2],
            row.r_squared
        );
    }

    if table.len() > 10 {
        println!("... ({} linhas restantes)", table.len() - 10);
    }

    // ========================================================================
    // EXPORTAÇÃO CSV
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("EXPORTAÇÃO CSV");
    println!("{}", "=".repeat(80));

    let csv = rolling.to_csv_string();
    let csv_lines: Vec<&str> = csv.lines().collect();

    println!("\nPrimeiras 10 linhas do CSV:");
    println!("{}", "-".repeat(80));
    for line in csv_lines.iter().take(10) {
        println!("{}", line);
    }

    if csv_lines.len() > 10 {
        println!("... ({} linhas restantes)", csv_lines.len() - 10);
    }

    // Salvar em arquivo (opcional)
    // std::fs::write("rolling_betas.csv", csv)?;
    // println!("\n✓ Arquivo 'rolling_betas.csv' salvo com sucesso!");

    // ========================================================================
    // COMPARAÇÃO ENTRE ATIVOS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO DE ESTABILIDADE ENTRE ATIVOS");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<20} {:>15} {:>15} {:>15}",
        "Métrica", "Tech Fund", "Value Fund", "Growth Fund"
    );
    println!("{}", "-".repeat(80));

    // Beta Market médio
    println!(
        "{:<20} {:>15.4} {:>15.4} {:>15.4}",
        "β_Market Médio",
        rolling.get_asset("Tech Fund").unwrap().mean_beta(0),
        rolling.get_asset("Value Fund").unwrap().mean_beta(0),
        rolling.get_asset("Growth Fund").unwrap().mean_beta(0)
    );

    // CV Beta Market
    println!(
        "{:<20} {:>15.4} {:>15.4} {:>15.4}",
        "β_Market CV",
        rolling.get_asset("Tech Fund").unwrap().cv_beta(0),
        rolling.get_asset("Value Fund").unwrap().cv_beta(0),
        rolling.get_asset("Growth Fund").unwrap().cv_beta(0)
    );

    // Beta SMB médio
    println!(
        "{:<20} {:>15.4} {:>15.4} {:>15.4}",
        "β_SMB Médio",
        rolling.get_asset("Tech Fund").unwrap().mean_beta(1),
        rolling.get_asset("Value Fund").unwrap().mean_beta(1),
        rolling.get_asset("Growth Fund").unwrap().mean_beta(1)
    );

    // Beta HML médio
    println!(
        "{:<20} {:>15.4} {:>15.4} {:>15.4}",
        "β_HML Médio",
        rolling.get_asset("Tech Fund").unwrap().mean_beta(2),
        rolling.get_asset("Value Fund").unwrap().mean_beta(2),
        rolling.get_asset("Growth Fund").unwrap().mean_beta(2)
    );

    // R² médio
    println!(
        "{:<20} {:>15.4} {:>15.4} {:>15.4}",
        "R² Médio",
        rolling
            .get_asset("Tech Fund")
            .unwrap()
            .r_squared
            .mean()
            .unwrap_or(0.0),
        rolling
            .get_asset("Value Fund")
            .unwrap()
            .r_squared
            .mean()
            .unwrap_or(0.0),
        rolling
            .get_asset("Growth Fund")
            .unwrap()
            .r_squared
            .mean()
            .unwrap_or(0.0)
    );

    // ========================================================================
    // CONCLUSÕES
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("CONCLUSÕES");
    println!("{}", "=".repeat(80));

    println!("\n1. ESTABILIDADE DOS BETAS:");
    for asset_name in &asset_names {
        let asset_result = rolling.get_asset(asset_name).unwrap();
        let market_cv = asset_result.cv_beta(0);

        if market_cv < 0.1 {
            println!(
                "   ✓ {}: Beta estável (CV = {:.2}%)",
                asset_name,
                market_cv * 100.0
            );
        } else {
            println!(
                "   ⚠ {}: Beta instável (CV = {:.2}%)",
                asset_name,
                market_cv * 100.0
            );
        }
    }

    println!("\n2. EXPOSIÇÃO AOS FATORES:");
    println!("   • Market: Todos os fundos têm forte exposição ao mercado");

    // Identificar qual fundo tem maior exposição SMB
    let smb_betas: Vec<f64> = asset_names
        .iter()
        .map(|name| rolling.get_asset(name).unwrap().mean_beta(1))
        .collect();

    let max_smb_idx = smb_betas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!(
        "   • SMB: {} tem maior exposição a small caps",
        asset_names[max_smb_idx]
    );

    println!("\n3. FORMATO DE DADOS:");
    println!("   • Total de linhas na tabela: {}", table.len());
    println!("   • Formato: (asset, date, alpha, betas..., r_squared)");
    println!("   • Compatível com análises em Python/Pandas");

    println!("\n{}", "=".repeat(80));

    Ok(())
}
