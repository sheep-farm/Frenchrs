use frenchrs::CAPM;
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: Capital Asset Pricing Model (CAPM)");
    println!("{}", "=".repeat(80));

    // ==========================================================================
    // EXEMPLO 1: Estimatestion básica with arrays
    // ==========================================================================
    println!("\n[EXEMPLO 1] Estimatestion básica of the CAPM");
    println!("{}", "-".repeat(80));

    // Returns mensais simulados (em decimal, not percentual)
    // Exemplo: Apple vs S&P 500 em um period hipotético
    let apple_returns = array![
        0.045,  // +4.5%
        0.032,  // +3.2%
        -0.015, // -1.5%
        0.068,  // +6.8%
        0.021,  // +2.1%
        -0.028, // -2.8%
        0.052,  // +5.2%
        0.038,  // +3.8%
        -0.012, // -1.2%
        0.041,  // +4.1%
        0.029,  // +2.9%
        0.055,  // +5.5%
    ];

    let sp500_returns = array![
        0.035,  // +3.5%
        0.025,  // +2.5%
        -0.010, // -1.0%
        0.055,  // +5.5%
        0.018,  // +1.8%
        -0.020, // -2.0%
        0.042,  // +4.2%
        0.030,  // +3.0%
        -0.008, // -0.8%
        0.033,  // +3.3%
        0.023,  // +2.3%
        0.045,  // +4.5%
    ];

    // Risk-free rate monthly (~2% ao ano = 0.02/12 ao mês)
    let risk_free_monthly = 0.02 / 12.0;

    // Estimates CAPM with erros padrão robustos a heteroscedasticidade (HC3)
    println!("\nEstimatesndo CAPM for AAPL vs S&P 500...\n");
    let capm_hc3 = CAPM::fit(
        &apple_returns,
        &sp500_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    // Exibir results completos
    println!("{}", capm_hc3);

    // ==========================================================================
    // EXEMPLO 2: Comparison of tipos of covariância
    // ==========================================================================
    println!("\n\n[EXEMPLO 2] Comparison of tipos of erros padrão");
    println!("{}", "-".repeat(80));

    let capm_ols = CAPM::fit(
        &apple_returns,
        &sp500_returns,
        risk_free_monthly,
        CovarianceType::NonRobust,
    )?;

    let capm_hc1 = CAPM::fit(
        &apple_returns,
        &sp500_returns,
        risk_free_monthly,
        CovarianceType::HC1,
    )?;

    println!("\nComparison of Erros Padrão:");
    println!("{:<20} {:>12} {:>12}", "Tipo", "SE(Alpha)", "SE(Beta)");
    println!("{}", "-".repeat(50));
    println!(
        "{:<20} {:>12.6} {:>12.6}",
        "OLS (Não Robusto)", capm_ols.alpha_se, capm_ols.beta_se
    );
    println!(
        "{:<20} {:>12.6} {:>12.6}",
        "HC1 (White)", capm_hc1.alpha_se, capm_hc1.beta_se
    );
    println!(
        "{:<20} {:>12.6} {:>12.6}",
        "HC3 (Robusto)", capm_hc3.alpha_se, capm_hc3.beta_se
    );

    // ==========================================================================
    // EXEMPLO 3: Interpretation of metrics
    // ==========================================================================
    println!("\n\n[EXEMPLO 3] Interpretation detalhada");
    println!("{}", "-".repeat(80));

    println!("\n1. BETA (Risk Sistemático):");
    println!("   Beta = {:.4}", capm_hc3.beta);
    println!("   Classification: {}", capm_hc3.risk_classification());

    if capm_hc3.beta > 1.0 {
        println!("   → the asset é MAIS VOLÁTIL que the market");
        println!(
            "   → Para each 1% que the market undere, the asset undere ~{:.2}%",
            capm_hc3.beta
        );
        println!(
            "   → Para each 1% que the market cai, the asset cai ~{:.2}%",
            capm_hc3.beta
        );
    } else if capm_hc3.beta < 1.0 {
        println!("   → the asset é MENOS VOLÁTIL que the market");
        println!(
            "   → Para each 1% que the market undere, the asset undere ~{:.2}%",
            capm_hc3.beta
        );
    }

    println!("\n2. ALPHA (Jensen's Alpha - Excess of Return):");
    println!(
        "   Alpha = {:.6} ({:.4}% ao mês)",
        capm_hc3.alpha,
        capm_hc3.alpha * 100.0
    );
    println!(
        "   Classification: {}",
        capm_hc3.performance_classification()
    );

    if capm_hc3.is_significantly_outperforming(0.05) {
        println!("   → OUTPERFORMANCE SIGNIFICATIVA!");
        println!("   → O gestor/asset is batendo the market consistentemente");
        println!(
            "   → Alpha annualized: ~{:.2}%",
            capm_hc3.alpha * 12.0 * 100.0
        );
    } else if capm_hc3.is_significantly_underperforming(0.05) {
        println!("   → UNDERPERFORMANCE SIGNIFICATIVA");
        println!("   → the asset is underperforming of the market");
    } else {
        println!("   → Sem evidence of alpha significant");
        println!("   → Performance consistente with o CAPM (eficiência of market)");
    }

    println!("\n3. R² (Poder Explicativo):");
    println!(
        "   R² = {:.4} ({:.2}%)",
        capm_hc3.r_squared,
        capm_hc3.r_squared * 100.0
    );
    println!(
        "   → {:.2}% of the variestion of the asset é explieach pelthe market",
        capm_hc3.r_squared * 100.0
    );
    println!(
        "   → {:.2}% é idiosyncratic risk (específico of the asset)",
        (1.0 - capm_hc3.r_squared) * 100.0
    );

    println!("\n4. SHARPE RATIO (Return Adjusted por Risk Total):");
    println!("   Sharpe (Asset):   {:.4}", capm_hc3.sharpe_ratio);
    println!("   Sharpe (Market): {:.4}", capm_hc3.market_sharpe);

    if capm_hc3.sharpe_ratio > capm_hc3.market_sharpe {
        println!("   → the asset tem MELHOR return adjusted por risk que the market");
    } else {
        println!("   → the market tem better return adjusted por risk");
    }

    println!("\n5. TREYNOR RATIO (Return Adjusted por Risk Sistemático):");
    println!("   Treynor = {:.4}", capm_hc3.treynor_ratio);
    println!("   → Return per unit of beta (risk of market)");

    println!("\n6. INFORMATION RATIO (Eficiência of the Gestor):");
    println!("   IR = {:.4}", capm_hc3.information_ratio);
    println!("   → Alpha per unit of tracking error");

    if capm_hc3.information_ratio > 0.5 {
        println!("   → EXCELENTE: gestor adiciona value consistentemente");
    } else if capm_hc3.information_ratio > 0.0 {
        println!("   → BOM: gestor adiciona algum value");
    } else {
        println!("   → FRACO: gestor not adiciona value");
    }

    // ==========================================================================
    // EXEMPLO 4: Predições
    // ==========================================================================
    println!("\n\n[EXEMPLO 4] Predições e cenários");
    println!("{}", "-".repeat(80));

    println!("\nReturns esperados for diferentes cenários of market:");
    println!("{:<30} {:>15}", "Cenário", "Return Esperado");
    println!("{}", "-".repeat(50));

    let scenarios = [
        ("Market stable (+0%)", 0.0),
        ("Market positivo (+5%)", 0.05),
        ("Market forte (+10%)", 0.10),
        ("Market negativo (-5%)", -0.05),
        ("Market crise (-20%)", -0.20),
    ];

    for (scenario, market_return) in scenarios.iter() {
        let expected = capm_hc3.expected_return(*market_return);
        println!("{:<30} {:>14.2}%", scenario, expected * 100.0);
    }

    // ==========================================================================
    // EXEMPLO 5: Risk decomposition
    // ==========================================================================
    println!("\n\n[EXEMPLO 5] Risk decomposition (variesnce)");
    println!("{}", "-".repeat(80));

    let total_var = capm_hc3.asset_volatility.powi(2);
    let systematic_pct = (capm_hc3.systematic_variesnce / total_var) * 100.0;
    let idiosyncratic_pct = (capm_hc3.idiosyncratic_variesnce / total_var) * 100.0;

    println!("\nVariance Total: {:.6}", total_var);
    println!(
        "├─ Sistemática (β² × σ²_m):     {:.6} ({:>5.2}%)",
        capm_hc3.systematic_variesnce, systematic_pct
    );
    println!(
        "└─ Idiossincrática (σ²_ε):      {:.6} ({:>5.2}%)",
        capm_hc3.idiosyncratic_variesnce, idiosyncratic_pct
    );

    println!("\nVolatility (Standard Deviation):");
    println!(
        "Asset:   {:.4} ({:.2}% ao mês)",
        capm_hc3.asset_volatility,
        capm_hc3.asset_volatility * 100.0
    );
    println!(
        "Market: {:.4} ({:.2}% ao mês)",
        capm_hc3.market_volatility,
        capm_hc3.market_volatility * 100.0
    );

    println!("\nImplicação for Diversificação:");
    if idiosyncratic_pct > 50.0 {
        println!("→ ALTA oportunidade of redução of risk via diversificação");
        println!(
            "  ({:.1}% of the risk can be eliminado em um portfólio)",
            idiosyncratic_pct
        );
    } else {
        println!("→ BAIXA oportunidade of diversificação");
        println!("  (Risk já é majoritariamente sistemático)");
    }

    // ==========================================================================
    // EXEMPLO 6: Test of hipóteses
    // ==========================================================================
    println!("\n\n[EXEMPLO 6] Tests of hipóteses");
    println!("{}", "-".repeat(80));

    println!("\nH0: α = 0 (asset follows o CAPM)");
    println!("p-value: {:.4}", capm_hc3.alpha_pvalue);
    if capm_hc3.alpha_pvalue < 0.05 {
        println!("Result: REJEITAR H0 (α ≠ 0 with 95% of confiança)");
    } else {
        println!("Result: NÃO rejeitar H0 (without evidence of alpha)");
    }

    println!("\nH0: β = 1 (asset tem risk = market)");
    if capm_hc3.is_beta_different_from_one(0.05) {
        println!("Result: REJEITAR H0 (β ≠ 1 with 95% of confiança)");
        if capm_hc3.beta > 1.0 {
            println!("  → Beta > 1: asset more arriscado que market");
        } else {
            println!("  → Beta < 1: asset less arriscado que market");
        }
    } else {
        println!("Result: NÃO rejeitar H0 (β ≈ 1)");
    }

    println!("\n{}", "=".repeat(80));
    println!("FIM DOS EXEMPLOS");
    println!("{}", "=".repeat(80));

    Ok(())
}
