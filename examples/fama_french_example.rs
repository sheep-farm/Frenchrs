use frenchrs::{CAPM, FamaFrench3Factor};
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: Fama-French 3 Factor Model");
    println!("{}", "=".repeat(80));

    // ==========================================================================
    // EXEMPLO 1: Estimação básica do FF3
    // ==========================================================================
    println!("\n[EXEMPLO 1] Estimação básica do Fama-French 3 Factor");
    println!("{}", "-".repeat(80));

    // Retornos mensais simulados
    // Exemplo: Fundo de investimento vs mercado + fatores FF
    let fund_returns = array![
        0.052,  // +5.2%
        0.038,  // +3.8%
        -0.022, // -2.2%
        0.075,  // +7.5%
        0.041,  // +4.1%
        -0.035, // -3.5%
        0.062,  // +6.2%
        0.048,  // +4.8%
        -0.018, // -1.8%
        0.055,  // +5.5%
        0.039,  // +3.9%
        0.068,  // +6.8%
    ];

    let market_returns = array![
        0.042,  // +4.2%
        0.030,  // +3.0%
        -0.015, // -1.5%
        0.060,  // +6.0%
        0.033,  // +3.3%
        -0.025, // -2.5%
        0.050,  // +5.0%
        0.038,  // +3.8%
        -0.012, // -1.2%
        0.045,  // +4.5%
        0.032,  // +3.2%
        0.055,  // +5.5%
    ];

    // Fator SMB (Small Minus Big)
    // Positivo indica que small caps superaram large caps
    let smb_returns = array![
        0.008,  // Small caps +0.8% melhor
        -0.003, // Large caps +0.3% melhor
        0.012,  // Small caps +1.2% melhor
        0.005,  // Small caps +0.5% melhor
        -0.007, // Large caps +0.7% melhor
        0.015,  // Small caps +1.5% melhor
        0.002,  // Small caps +0.2% melhor
        -0.005, // Large caps +0.5% melhor
        0.010,  // Small caps +1.0% melhor
        0.003,  // Small caps +0.3% melhor
        -0.004, // Large caps +0.4% melhor
        0.006,  // Small caps +0.6% melhor
    ];

    // Fator HML (High Minus Low)
    // Positivo indica que value stocks superaram growth stocks
    let hml_returns = array![
        0.005,  // Value +0.5% melhor
        0.008,  // Value +0.8% melhor
        -0.010, // Growth +1.0% melhor
        0.012,  // Value +1.2% melhor
        0.003,  // Value +0.3% melhor
        -0.006, // Growth +0.6% melhor
        0.009,  // Value +0.9% melhor
        0.004,  // Value +0.4% melhor
        -0.008, // Growth +0.8% melhor
        0.011,  // Value +1.1% melhor
        0.002,  // Value +0.2% melhor
        0.007,  // Value +0.7% melhor
    ];

    // Taxa livre de risco mensal (~3% ao ano = 0.03/12 ao mês)
    let risk_free_monthly = 0.03 / 12.0;

    println!("\nEstimando Fama-French 3 Factor para o fundo...\n");
    let ff3 = FamaFrench3Factor::fit(
        &fund_returns,
        &market_returns,
        &smb_returns,
        &hml_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    // Exibir resultados completos
    println!("{}", ff3);

    // ==========================================================================
    // EXEMPLO 2: Comparação com CAPM
    // ==========================================================================
    println!("\n\n[EXEMPLO 2] Comparação: CAPM vs Fama-French 3 Factor");
    println!("{}", "-".repeat(80));

    // Estimar CAPM para comparação
    let capm = CAPM::fit(
        &fund_returns,
        &market_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    println!("\nComparação de Modelos:");
    println!("{:<25} {:>15} {:>15}", "Métrica", "CAPM", "FF3");
    println!("{}", "-".repeat(60));
    println!("{:<25} {:>14.6} {:>14.6}", "Alpha", capm.alpha, ff3.alpha);
    println!(
        "{:<25} {:>14.6} {:>14.6}",
        "Beta Mercado", capm.beta, ff3.beta_market
    );
    println!("{:<25} {:>15} {:>14.6}", "Beta SMB", "-", ff3.beta_smb);
    println!("{:<25} {:>15} {:>14.6}", "Beta HML", "-", ff3.beta_hml);
    println!("{}", "-".repeat(60));
    println!(
        "{:<25} {:>14.4} {:>14.4}",
        "R²", capm.r_squared, ff3.r_squared
    );
    println!(
        "{:<25} {:>13.4}% {:>13.4}%",
        "Tracking Error",
        capm.tracking_error * 100.0,
        ff3.tracking_error * 100.0
    );
    println!(
        "{:<25} {:>14.4} {:>14.4}",
        "Information Ratio", capm.information_ratio, ff3.information_ratio
    );

    println!("\nMelhoria do FF3 sobre CAPM:");
    let r2_improvement = ff3.r_squared - capm.r_squared;
    let te_improvement = (capm.tracking_error - ff3.tracking_error) / capm.tracking_error;

    println!(
        "  → Aumento de R²: {:.4} ({:.2}% → {:.2}%)",
        r2_improvement,
        capm.r_squared * 100.0,
        ff3.r_squared * 100.0
    );
    println!(
        "  → Redução de Tracking Error: {:.2}%",
        te_improvement * 100.0
    );

    if r2_improvement > 0.05 {
        println!("  ✓ SIGNIFICATIVA: FF3 explica substancialmente mais variância");
    } else {
        println!("  ○ MODERADA: CAPM já captura a maior parte da variação");
    }

    // ==========================================================================
    // EXEMPLO 3: Interpretação dos fatores
    // ==========================================================================
    println!("\n\n[EXEMPLO 3] Interpretação dos fatores Fama-French");
    println!("{}", "-".repeat(80));

    println!("\n1. FATOR MERCADO (β_MKT = {:.4}):", ff3.beta_market);
    if ff3.beta_market > 1.2 {
        println!(
            "   → Fundo MUITO AGRESSIVO: varia {:.1}x mais que o mercado",
            ff3.beta_market
        );
    } else if ff3.beta_market > 1.0 {
        println!(
            "   → Fundo AGRESSIVO: varia {:.1}x mais que o mercado",
            ff3.beta_market
        );
    } else if ff3.beta_market > 0.8 {
        println!("   → Fundo NEUTRO: varia similar ao mercado");
    } else {
        println!("   → Fundo DEFENSIVO: varia menos que o mercado");
    }

    println!("\n2. FATOR TAMANHO - SMB (β_SMB = {:.4}):", ff3.beta_smb);
    println!("   Classificação: {}", ff3.size_classification());
    if ff3.is_smb_significant(0.05) {
        if ff3.beta_smb > 0.0 {
            println!("   → Exposição significativa a SMALL CAPS");
            println!(
                "   → Quando small caps sobem 1%, fundo sobe {:.2}% adicional",
                ff3.beta_smb * 100.0
            );
        } else {
            println!("   → Exposição significativa a LARGE CAPS");
            println!("   → Comportamento similar a empresas de grande capitalização");
        }
    } else {
        println!("   → Sem exposição significativa ao fator tamanho");
    }

    println!("\n3. FATOR VALOR - HML (β_HML = {:.4}):", ff3.beta_hml);
    println!("   Classificação: {}", ff3.value_classification());
    if ff3.is_hml_significant(0.05) {
        if ff3.beta_hml > 0.0 {
            println!("   → Exposição significativa a VALUE STOCKS");
            println!(
                "   → Quando value stocks sobem 1%, fundo sobe {:.2}% adicional",
                ff3.beta_hml * 100.0
            );
            println!("   → Características: P/L baixo, P/VPA baixo, dividendos altos");
        } else {
            println!("   → Exposição significativa a GROWTH STOCKS");
            println!("   → Características: alta expectativa de crescimento");
        }
    } else {
        println!("   → Sem exposição significativa ao fator valor");
    }

    println!(
        "\n4. ALPHA (α = {:.6} ou {:.4}% ao mês):",
        ff3.alpha,
        ff3.alpha * 100.0
    );
    println!("   Classificação: {}", ff3.performance_classification());
    if ff3.is_significantly_outperforming(0.05) {
        let annual_alpha = ff3.alpha * 12.0 * 100.0;
        println!("   ✓ SKILL DO GESTOR DETECTADO!");
        println!("   → Alpha anualizado: ~{:.2}%", annual_alpha);
        println!("   → Retorno acima do explicado pelos 3 fatores");
    } else if ff3.is_significantly_underperforming(0.05) {
        println!("   ✗ Underperformance após controlar por fatores");
    } else {
        println!("   ○ Retorno consistente com exposições aos fatores");
    }

    // ==========================================================================
    // EXEMPLO 4: Decomposição do retorno
    // ==========================================================================
    println!("\n\n[EXEMPLO 4] Decomposição do retorno do fundo");
    println!("{}", "-".repeat(80));

    let (market_contrib, smb_contrib, hml_contrib) = ff3.factor_contributions();
    let total_contrib = market_contrib + smb_contrib + hml_contrib;

    println!(
        "\nRetorno Médio do Fundo: {:.4}%",
        ff3.mean_asset_return * 100.0
    );
    println!("\nFontes do Retorno:");
    println!(
        "  Taxa Livre de Risco:      {:>10.4}% ({:>5.1}%)",
        risk_free_monthly * 100.0,
        (risk_free_monthly / ff3.mean_asset_return) * 100.0
    );
    println!(
        "  Prêmio de Mercado:        {:>10.4}% ({:>5.1}%)",
        market_contrib * 100.0,
        if ff3.mean_asset_return != 0.0 {
            (market_contrib / ff3.mean_asset_return) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  Prêmio SMB:               {:>10.4}% ({:>5.1}%)",
        smb_contrib * 100.0,
        if ff3.mean_asset_return != 0.0 {
            (smb_contrib / ff3.mean_asset_return) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  Prêmio HML:               {:>10.4}% ({:>5.1}%)",
        hml_contrib * 100.0,
        if ff3.mean_asset_return != 0.0 {
            (hml_contrib / ff3.mean_asset_return) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  Alpha (skill/sorte):      {:>10.4}% ({:>5.1}%)",
        ff3.alpha * 100.0,
        if ff3.mean_asset_return != 0.0 {
            (ff3.alpha / ff3.mean_asset_return) * 100.0
        } else {
            0.0
        }
    );
    println!("  {}", "-".repeat(50));
    println!(
        "  Total (aprox.):           {:>10.4}%",
        (risk_free_monthly + total_contrib + ff3.alpha) * 100.0
    );

    // ==========================================================================
    // EXEMPLO 5: Predições
    // ==========================================================================
    println!("\n\n[EXEMPLO 5] Predições de retorno");
    println!("{}", "-".repeat(80));

    println!("\nCenários para o próximo período:");
    println!("{:<40} {:>15}", "Cenário", "Retorno Esperado");
    println!("{}", "-".repeat(60));

    let scenarios = [
        ("Mercado +5%, SMB +0.5%, HML +0.3%", 0.05, 0.005, 0.003),
        ("Mercado +10%, SMB +1%, HML +0.5%", 0.10, 0.01, 0.005),
        ("Mercado -5%, SMB -0.5%, HML -0.3%", -0.05, -0.005, -0.003),
        ("Mercado 0%, SMB +2%, HML 0%", 0.0, 0.02, 0.0),
        ("Mercado 0%, SMB 0%, HML +2%", 0.0, 0.0, 0.02),
    ];

    for (scenario, mkt, smb, hml) in scenarios.iter() {
        let expected = ff3.expected_return(*mkt, *smb, *hml);
        println!("{:<40} {:>14.2}%", scenario, expected * 100.0);
    }

    // ==========================================================================
    // EXEMPLO 6: Análise de estilo (Style Analysis)
    // ==========================================================================
    println!("\n\n[EXEMPLO 6] Análise de estilo do fundo");
    println!("{}", "-".repeat(80));

    println!("\nCaracterização do Fundo:");

    // Estilo de risco
    if ff3.beta_market > 1.2 {
        println!("  Risco:     ALTO (beta de mercado > 1.2)");
    } else if ff3.beta_market < 0.8 {
        println!("  Risco:     BAIXO (beta de mercado < 0.8)");
    } else {
        println!("  Risco:     MODERADO (beta próximo de 1)");
    }

    // Estilo de tamanho
    if ff3.is_smb_significant(0.05) {
        if ff3.beta_smb > 0.3 {
            println!("  Tamanho:   SMALL CAP BIAS");
        } else if ff3.beta_smb < -0.3 {
            println!("  Tamanho:   LARGE CAP BIAS");
        } else {
            println!("  Tamanho:   MID CAP / DIVERSIFICADO");
        }
    } else {
        println!("  Tamanho:   NEUTRO (sem viés claro)");
    }

    // Estilo de valor/crescimento
    if ff3.is_hml_significant(0.05) {
        if ff3.beta_hml > 0.3 {
            println!("  Valor:     VALUE INVESTOR");
        } else if ff3.beta_hml < -0.3 {
            println!("  Valor:     GROWTH INVESTOR");
        } else {
            println!("  Valor:     BLEND (mistura value/growth)");
        }
    } else {
        println!("  Valor:     NEUTRO (sem viés claro)");
    }

    // Skill do gestor
    if ff3.is_significantly_outperforming(0.05) {
        println!("  Gestor:    ALPHA POSITIVO (skill detectado)");
    } else if ff3.is_significantly_underperforming(0.05) {
        println!("  Gestor:    ALPHA NEGATIVO (underperformance)");
    } else {
        println!("  Gestor:    SEM ALPHA SIGNIFICATIVO (retorno = fatores)");
    }

    println!("\n{}", "=".repeat(80));
    println!("FIM DOS EXEMPLOS");
    println!("{}", "=".repeat(80));

    Ok(())
}
