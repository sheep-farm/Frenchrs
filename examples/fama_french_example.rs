use frenchrs::{FamaFrench3Factor, CAPM};
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: Fama-French 3 Factor Model");
    println!("{}", "=".repeat(80));

    // ==========================================================================
    // EXEMPLO 1: Estimatestion básica of the FF3
    // ==========================================================================
    println!("\n[EXEMPLO 1] Estimatestion básica of the Fama-French 3 Factor");
    println!("{}", "-".repeat(80));

    // Returns mensais simulados
    // Exemplo: Fundo of investment vs market + factors FF
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

    // Factor SMB (Small Minus Big)
    // Positivo indica que small caps superaram large caps
    let smb_returns = array![
        0.008,  // Small caps +0.8% better
        -0.003, // Large caps +0.3% better
        0.012,  // Small caps +1.2% better
        0.005,  // Small caps +0.5% better
        -0.007, // Large caps +0.7% better
        0.015,  // Small caps +1.5% better
        0.002,  // Small caps +0.2% better
        -0.005, // Large caps +0.5% better
        0.010,  // Small caps +1.0% better
        0.003,  // Small caps +0.3% better
        -0.004, // Large caps +0.4% better
        0.006,  // Small caps +0.6% better
    ];

    // Factor HML (High Minus Low)
    // Positivo indica que value stocks superaram growth stocks
    let hml_returns = array![
        0.005,  // Value +0.5% better
        0.008,  // Value +0.8% better
        -0.010, // Growth +1.0% better
        0.012,  // Value +1.2% better
        0.003,  // Value +0.3% better
        -0.006, // Growth +0.6% better
        0.009,  // Value +0.9% better
        0.004,  // Value +0.4% better
        -0.008, // Growth +0.8% better
        0.011,  // Value +1.1% better
        0.002,  // Value +0.2% better
        0.007,  // Value +0.7% better
    ];

    // Risk-free rate monthly (~3% ao ano = 0.03/12 ao mês)
    let risk_free_monthly = 0.03 / 12.0;

    println!("\nEstimatesndo Fama-French 3 Factor for o fundo...\n");
    let ff3 = FamaFrench3Factor::fit(
        &fund_returns,
        &market_returns,
        &smb_returns,
        &hml_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    // Exibir results completos
    println!("{}", ff3);

    // ==========================================================================
    // EXEMPLO 2: Comparison with CAPM
    // ==========================================================================
    println!("\n\n[EXEMPLO 2] Comparison: CAPM vs Fama-French 3 Factor");
    println!("{}", "-".repeat(80));

    // Estimates CAPM for comparison
    let capm = CAPM::fit(
        &fund_returns,
        &market_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    println!("\nComparison of Models:");
    println!("{:<25} {:>15} {:>15}", "Métrica", "CAPM", "FF3");
    println!("{}", "-".repeat(60));
    println!("{:<25} {:>14.6} {:>14.6}", "Alpha", capm.alpha, ff3.alpha);
    println!(
        "{:<25} {:>14.6} {:>14.6}",
        "Beta Market", capm.beta, ff3.beta_market
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

    println!("\nMelhoria of the FF3 about CAPM:");
    let r2_improvement = ff3.r_squared - capm.r_squared;
    let te_improvement = (capm.tracking_error - ff3.tracking_error) / capm.tracking_error;

    println!(
        "  → Aumento of R²: {:.4} ({:.2}% → {:.2}%)",
        r2_improvement,
        capm.r_squared * 100.0,
        ff3.r_squared * 100.0
    );
    println!(
        "  → Redução of Tracking Error: {:.2}%",
        te_improvement * 100.0
    );

    if r2_improvement > 0.05 {
        println!("  ✓ SIGNIFICATIVA: FF3 explica substancialmente more variesnce");
    } else {
        println!("  ○ MODERADA: CAPM já captura a greater parte of the variestion");
    }

    // ==========================================================================
    // EXEMPLO 3: Interpretation of the factors
    // ==========================================================================
    println!("\n\n[EXEMPLO 3] Interpretation of the factors Fama-French");
    println!("{}", "-".repeat(80));

    println!("\n1. FATOR MERCADO (β_MKT = {:.4}):", ff3.beta_market);
    if ff3.beta_market > 1.2 {
        println!(
            "   → Fundo MUITO AGRESSIVO: varies {:.1}x more que the market",
            ff3.beta_market
        );
    } else if ff3.beta_market > 1.0 {
        println!(
            "   → Fundo AGRESSIVO: varies {:.1}x more que the market",
            ff3.beta_market
        );
    } else if ff3.beta_market > 0.8 {
        println!("   → Fundo NEUTRO: varies similar to the market");
    } else {
        println!("   → Fundo DEFENSIVO: varies less que the market");
    }

    println!("\n2. FATOR TAMANHO - SMB (β_SMB = {:.4}):", ff3.beta_smb);
    println!("   Classification: {}", ff3.size_classification());
    if ff3.is_smb_significant(0.05) {
        if ff3.beta_smb > 0.0 {
            println!("   → Exposição significativa a SMALL CAPS");
            println!(
                "   → Quando small caps underem 1%, fundo undere {:.2}% adicional",
                ff3.beta_smb * 100.0
            );
        } else {
            println!("   → Exposição significativa a LARGE CAPS");
            println!("   → Comportamento similar a empresas of grande capitalização");
        }
    } else {
        println!("   → Sem exposição significativa ao factor size");
    }

    println!("\n3. FATOR VALOR - HML (β_HML = {:.4}):", ff3.beta_hml);
    println!("   Classification: {}", ff3.value_classification());
    if ff3.is_hml_significant(0.05) {
        if ff3.beta_hml > 0.0 {
            println!("   → Exposição significativa a VALUE STOCKS");
            println!(
                "   → Quando value stocks underem 1%, fundo undere {:.2}% adicional",
                ff3.beta_hml * 100.0
            );
            println!("   → Características: P/L baixo, P/VPA baixo, dividendos altos");
        } else {
            println!("   → Exposição significativa a GROWTH STOCKS");
            println!("   → Características: alta expectativa of crescimento");
        }
    } else {
        println!("   → Sem exposição significativa ao factor value");
    }

    println!(
        "\n4. ALPHA (α = {:.6} ou {:.4}% ao mês):",
        ff3.alpha,
        ff3.alpha * 100.0
    );
    println!("   Classification: {}", ff3.performance_classification());
    if ff3.is_significantly_outperforming(0.05) {
        let annual_alpha = ff3.alpha * 12.0 * 100.0;
        println!("   ✓ SKILL DO GESTOR DETECTADO!");
        println!("   → Alpha annualized: ~{:.2}%", annual_alpha);
        println!("   → Return acima of the explained by the 3 factors");
    } else if ff3.is_significantly_underperforming(0.05) {
        println!("   ✗ Underperformance after controlar por factors");
    } else {
        println!("   ○ Return consistente with exposições aos factors");
    }

    // ==========================================================================
    // EXEMPLO 4: Decomposition of the return
    // ==========================================================================
    println!("\n\n[EXEMPLO 4] Decomposition of the return of the fundo");
    println!("{}", "-".repeat(80));

    let (market_contrib, smb_contrib, hml_contrib) = ff3.factor_contributions();
    let total_contrib = market_contrib + smb_contrib + hml_contrib;

    println!(
        "\nReturn Médio of the Fundo: {:.4}%",
        ff3.mean_asset_return * 100.0
    );
    println!("\nFontes of the return:");
    println!(
        "  Risk-Free Rate:      {:>10.4}% ({:>5.1}%)",
        risk_free_monthly * 100.0,
        (risk_free_monthly / ff3.mean_asset_return) * 100.0
    );
    println!(
        "  Prêmio of Market:        {:>10.4}% ({:>5.1}%)",
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
    println!("\n\n[EXEMPLO 5] Predições of return");
    println!("{}", "-".repeat(80));

    println!("\nCenários for o próximo period:");
    println!("{:<40} {:>15}", "Cenário", "Return Esperado");
    println!("{}", "-".repeat(60));

    let scenarios = [
        ("Market +5%, SMB +0.5%, HML +0.3%", 0.05, 0.005, 0.003),
        ("Market +10%, SMB +1%, HML +0.5%", 0.10, 0.01, 0.005),
        ("Market -5%, SMB -0.5%, HML -0.3%", -0.05, -0.005, -0.003),
        ("Market 0%, SMB +2%, HML 0%", 0.0, 0.02, 0.0),
        ("Market 0%, SMB 0%, HML +2%", 0.0, 0.0, 0.02),
    ];

    for (scenario, mkt, smb, hml) in scenarios.iter() {
        let expected = ff3.expected_return(*mkt, *smb, *hml);
        println!("{:<40} {:>14.2}%", scenario, expected * 100.0);
    }

    // ==========================================================================
    // EXEMPLO 6: Analysis of estilo (Style Analysis)
    // ==========================================================================
    println!("\n\n[EXEMPLO 6] Analysis of estilo of the fundo");
    println!("{}", "-".repeat(80));

    println!("\nCaracterização of the Fundo:");

    // Estilo of risk
    if ff3.beta_market > 1.2 {
        println!("  Risk:     ALTO (beta of market > 1.2)");
    } else if ff3.beta_market < 0.8 {
        println!("  Risk:     BAIXO (beta of market < 0.8)");
    } else {
        println!("  Risk:     MODERADO (beta próximo of 1)");
    }

    // Estilo of size
    if ff3.is_smb_significant(0.05) {
        if ff3.beta_smb > 0.3 {
            println!("  Size:   SMALL CAP BIAS");
        } else if ff3.beta_smb < -0.3 {
            println!("  Size:   LARGE CAP BIAS");
        } else {
            println!("  Size:   MID CAP / DIVERSIFICADO");
        }
    } else {
        println!("  Size:   NEUTRO (without viés claro)");
    }

    // Estilo of value/crescimento
    if ff3.is_hml_significant(0.05) {
        if ff3.beta_hml > 0.3 {
            println!("  Value:     VALUE INVESTOR");
        } else if ff3.beta_hml < -0.3 {
            println!("  Value:     GROWTH INVESTOR");
        } else {
            println!("  Value:     BLEND (mistura value/growth)");
        }
    } else {
        println!("  Value:     NEUTRO (without viés claro)");
    }

    // Skill of the gestor
    if ff3.is_significantly_outperforming(0.05) {
        println!("  Gestor:    ALPHA POSITIVO (skill detectado)");
    } else if ff3.is_significantly_underperforming(0.05) {
        println!("  Gestor:    ALPHA NEGATIVO (underperformance)");
    } else {
        println!("  Gestor:    SEM ALPHA SIGNIFICATIVO (return = factors)");
    }

    println!("\n{}", "=".repeat(80));
    println!("FIM DOS EXEMPLOS");
    println!("{}", "=".repeat(80));

    Ok(())
}
