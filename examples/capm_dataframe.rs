use frenchrs::CAPM;
use greeners::{CovarianceType, DataFrame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: CAPM com DataFrame");
    println!("{}", "=".repeat(80));

    // ==========================================================================
    // Criar DataFrame com retornos sintéticos
    // ==========================================================================
    println!("\nCriando DataFrame com retornos mensais de 2023...\n");

    let df = DataFrame::builder()
        .add_column(
            "month",
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .add_column(
            "tesla_returns",
            vec![
                0.0825,  // Jan: +8.25%
                0.0650,  // Fev: +6.50%
                -0.0280, // Mar: -2.80%
                0.1120,  // Abr: +11.20%
                0.0385,  // Mai: +3.85%
                -0.0520, // Jun: -5.20%
                0.0920,  // Jul: +9.20%
                0.0640,  // Ago: +6.40%
                -0.0220, // Set: -2.20%
                0.0750,  // Out: +7.50%
                0.0510,  // Nov: +5.10%
                0.0980,  // Dez: +9.80%
            ],
        )
        .add_column(
            "sp500_returns",
            vec![
                0.0630,  // Jan
                0.0450,  // Fev
                -0.0180, // Mar
                0.0920,  // Abr
                0.0320,  // Mai
                -0.0380, // Jun
                0.0720,  // Jul
                0.0510,  // Ago
                -0.0140, // Set
                0.0590,  // Out
                0.0410,  // Nov
                0.0790,  // Dez
            ],
        )
        .build()?;

    println!("DataFrame criado com {} observações\n", df.n_rows());

    // ==========================================================================
    // Estimar CAPM
    // ==========================================================================
    println!("Estimando CAPM para Tesla vs S&P 500...\n");

    // Taxa livre de risco: ~4% ao ano = 0.04/12 ao mês
    let risk_free_monthly = 0.04 / 12.0;

    let capm = CAPM::from_dataframe(
        &df,
        "tesla_returns",
        "sp500_returns",
        risk_free_monthly,
        CovarianceType::HC3, // Erros padrão robustos
    )?;

    // Exibir resultados
    println!("{}", capm);

    // ==========================================================================
    // Análise específica
    // ==========================================================================
    println!("\n{}", "=".repeat(80));
    println!("ANÁLISE ESPECÍFICA DA TESLA");
    println!("{}", "=".repeat(80));

    println!(
        "\nBeta: {:.4} → Tesla é {:.1}x mais volátil que o S&P 500",
        capm.beta, capm.beta
    );

    let annual_alpha = capm.alpha * 12.0 * 100.0;
    println!(
        "Alpha: {:.4}% ao mês (~{:.2}% ao ano)",
        capm.alpha * 100.0,
        annual_alpha
    );

    if capm.is_significantly_outperforming(0.05) {
        println!("✓ OUTPERFORMANCE SIGNIFICATIVA!");
        println!(
            "  Tesla superou o S&P 500 em ~{:.2}% ao ano após ajuste por risco",
            annual_alpha
        );
    }

    println!(
        "\nPoder Explicativo: {:.1}% da variação da Tesla é explicada pelo mercado",
        capm.r_squared * 100.0
    );

    println!(
        "Risco Idiossincrático: {:.1}% (específico da Tesla, não do mercado)",
        (1.0 - capm.r_squared) * 100.0
    );

    // ==========================================================================
    // Predições
    // ==========================================================================
    println!("\n{}", "-".repeat(80));
    println!("PREDIÇÕES DE RETORNO");
    println!("{}", "-".repeat(80));

    println!(
        "\nSe o S&P 500 subir 10% em 2024, retorno esperado da Tesla: {:.2}%",
        capm.expected_return(0.10) * 100.0
    );

    println!(
        "Se o S&P 500 cair 10% em 2024, retorno esperado da Tesla: {:.2}%",
        capm.expected_return(-0.10) * 100.0
    );

    // ==========================================================================
    // Métricas de desempenho ajustado por risco
    // ==========================================================================
    println!("\n{}", "-".repeat(80));
    println!("MÉTRICAS AJUSTADAS POR RISCO");
    println!("{}", "-".repeat(80));

    println!("\nSharpe Ratio:");
    println!("  Tesla:     {:.4}", capm.sharpe_ratio);
    println!("  S&P 500:   {:.4}", capm.market_sharpe);

    if capm.sharpe_ratio > capm.market_sharpe {
        println!("  → Tesla tem melhor retorno por unidade de risco total");
    } else {
        println!("  → S&P 500 tem melhor retorno por unidade de risco");
    }

    println!("\nTreynor Ratio: {:.4}", capm.treynor_ratio);
    println!("  → Retorno por unidade de risco sistemático (beta)");

    println!("\nInformation Ratio: {:.4}", capm.information_ratio);
    if capm.information_ratio > 0.5 {
        println!("  → EXCELENTE: alpha consistente e tracking error baixo");
    } else if capm.information_ratio > 0.0 {
        println!("  → BOM: algum alpha positivo");
    } else {
        println!("  → FRACO: sem alpha positivo consistente");
    }

    println!("\n{}", "=".repeat(80));

    Ok(())
}
