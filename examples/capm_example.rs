use frenchrs::CAPM;
use greeners::CovarianceType;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("EXEMPLO: Capital Asset Pricing Model (CAPM)");
    println!("{}", "=".repeat(80));

    // ==========================================================================
    // EXEMPLO 1: Estimação básica com arrays
    // ==========================================================================
    println!("\n[EXEMPLO 1] Estimação básica do CAPM");
    println!("{}", "-".repeat(80));

    // Retornos mensais simulados (em decimal, não percentual)
    // Exemplo: Apple vs S&P 500 em um período hipotético
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

    // Taxa livre de risco mensal (~2% ao ano = 0.02/12 ao mês)
    let risk_free_monthly = 0.02 / 12.0;

    // Estimar CAPM com erros padrão robustos a heteroscedasticidade (HC3)
    println!("\nEstimando CAPM para AAPL vs S&P 500...\n");
    let capm_hc3 = CAPM::fit(
        &apple_returns,
        &sp500_returns,
        risk_free_monthly,
        CovarianceType::HC3,
    )?;

    // Exibir resultados completos
    println!("{}", capm_hc3);

    // ==========================================================================
    // EXEMPLO 2: Comparação de tipos de covariância
    // ==========================================================================
    println!("\n\n[EXEMPLO 2] Comparação de tipos de erros padrão");
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

    println!("\nComparação de Erros Padrão:");
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
    // EXEMPLO 3: Interpretação de métricas
    // ==========================================================================
    println!("\n\n[EXEMPLO 3] Interpretação detalhada");
    println!("{}", "-".repeat(80));

    println!("\n1. BETA (Risco Sistemático):");
    println!("   Beta = {:.4}", capm_hc3.beta);
    println!("   Classificação: {}", capm_hc3.risk_classification());

    if capm_hc3.beta > 1.0 {
        println!("   → O ativo é MAIS VOLÁTIL que o mercado");
        println!(
            "   → Para cada 1% que o mercado sobe, o ativo sobe ~{:.2}%",
            capm_hc3.beta
        );
        println!(
            "   → Para cada 1% que o mercado cai, o ativo cai ~{:.2}%",
            capm_hc3.beta
        );
    } else if capm_hc3.beta < 1.0 {
        println!("   → O ativo é MENOS VOLÁTIL que o mercado");
        println!(
            "   → Para cada 1% que o mercado sobe, o ativo sobe ~{:.2}%",
            capm_hc3.beta
        );
    }

    println!("\n2. ALPHA (Jensen's Alpha - Excesso de Retorno):");
    println!(
        "   Alpha = {:.6} ({:.4}% ao mês)",
        capm_hc3.alpha,
        capm_hc3.alpha * 100.0
    );
    println!(
        "   Classificação: {}",
        capm_hc3.performance_classification()
    );

    if capm_hc3.is_significantly_outperforming(0.05) {
        println!("   → OUTPERFORMANCE SIGNIFICATIVA!");
        println!("   → O gestor/ativo está batendo o mercado consistentemente");
        println!(
            "   → Alpha anualizado: ~{:.2}%",
            capm_hc3.alpha * 12.0 * 100.0
        );
    } else if capm_hc3.is_significantly_underperforming(0.05) {
        println!("   → UNDERPERFORMANCE SIGNIFICATIVA");
        println!("   → O ativo está ficando atrás do mercado");
    } else {
        println!("   → Sem evidência de alpha significativo");
        println!("   → Desempenho consistente com o CAPM (eficiência de mercado)");
    }

    println!("\n3. R² (Poder Explicativo):");
    println!(
        "   R² = {:.4} ({:.2}%)",
        capm_hc3.r_squared,
        capm_hc3.r_squared * 100.0
    );
    println!(
        "   → {:.2}% da variação do ativo é explicada pelo mercado",
        capm_hc3.r_squared * 100.0
    );
    println!(
        "   → {:.2}% é risco idiossincrático (específico do ativo)",
        (1.0 - capm_hc3.r_squared) * 100.0
    );

    println!("\n4. SHARPE RATIO (Retorno Ajustado por Risco Total):");
    println!("   Sharpe (Ativo):   {:.4}", capm_hc3.sharpe_ratio);
    println!("   Sharpe (Mercado): {:.4}", capm_hc3.market_sharpe);

    if capm_hc3.sharpe_ratio > capm_hc3.market_sharpe {
        println!("   → O ativo tem MELHOR retorno ajustado por risco que o mercado");
    } else {
        println!("   → O mercado tem melhor retorno ajustado por risco");
    }

    println!("\n5. TREYNOR RATIO (Retorno Ajustado por Risco Sistemático):");
    println!("   Treynor = {:.4}", capm_hc3.treynor_ratio);
    println!("   → Retorno por unidade de beta (risco de mercado)");

    println!("\n6. INFORMATION RATIO (Eficiência do Gestor):");
    println!("   IR = {:.4}", capm_hc3.information_ratio);
    println!("   → Alpha por unidade de tracking error");

    if capm_hc3.information_ratio > 0.5 {
        println!("   → EXCELENTE: gestor adiciona valor consistentemente");
    } else if capm_hc3.information_ratio > 0.0 {
        println!("   → BOM: gestor adiciona algum valor");
    } else {
        println!("   → FRACO: gestor não adiciona valor");
    }

    // ==========================================================================
    // EXEMPLO 4: Predições
    // ==========================================================================
    println!("\n\n[EXEMPLO 4] Predições e cenários");
    println!("{}", "-".repeat(80));

    println!("\nRetornos esperados para diferentes cenários de mercado:");
    println!("{:<30} {:>15}", "Cenário", "Retorno Esperado");
    println!("{}", "-".repeat(50));

    let scenarios = [
        ("Mercado estável (+0%)", 0.0),
        ("Mercado positivo (+5%)", 0.05),
        ("Mercado forte (+10%)", 0.10),
        ("Mercado negativo (-5%)", -0.05),
        ("Mercado crise (-20%)", -0.20),
    ];

    for (scenario, market_return) in scenarios.iter() {
        let expected = capm_hc3.expected_return(*market_return);
        println!("{:<30} {:>14.2}%", scenario, expected * 100.0);
    }

    // ==========================================================================
    // EXEMPLO 5: Decomposição de risco
    // ==========================================================================
    println!("\n\n[EXEMPLO 5] Decomposição de risco (variância)");
    println!("{}", "-".repeat(80));

    let total_var = capm_hc3.asset_volatility.powi(2);
    let systematic_pct = (capm_hc3.systematic_variance / total_var) * 100.0;
    let idiosyncratic_pct = (capm_hc3.idiosyncratic_variance / total_var) * 100.0;

    println!("\nVariância Total: {:.6}", total_var);
    println!(
        "├─ Sistemática (β² × σ²_m):     {:.6} ({:>5.2}%)",
        capm_hc3.systematic_variance, systematic_pct
    );
    println!(
        "└─ Idiossincrática (σ²_ε):      {:.6} ({:>5.2}%)",
        capm_hc3.idiosyncratic_variance, idiosyncratic_pct
    );

    println!("\nVolatilidade (Desvio Padrão):");
    println!(
        "Ativo:   {:.4} ({:.2}% ao mês)",
        capm_hc3.asset_volatility,
        capm_hc3.asset_volatility * 100.0
    );
    println!(
        "Mercado: {:.4} ({:.2}% ao mês)",
        capm_hc3.market_volatility,
        capm_hc3.market_volatility * 100.0
    );

    println!("\nImplicação para Diversificação:");
    if idiosyncratic_pct > 50.0 {
        println!("→ ALTA oportunidade de redução de risco via diversificação");
        println!(
            "  ({:.1}% do risco pode ser eliminado em um portfólio)",
            idiosyncratic_pct
        );
    } else {
        println!("→ BAIXA oportunidade de diversificação");
        println!("  (Risco já é majoritariamente sistemático)");
    }

    // ==========================================================================
    // EXEMPLO 6: Teste de hipóteses
    // ==========================================================================
    println!("\n\n[EXEMPLO 6] Testes de hipóteses");
    println!("{}", "-".repeat(80));

    println!("\nH0: α = 0 (ativo segue o CAPM)");
    println!("p-value: {:.4}", capm_hc3.alpha_pvalue);
    if capm_hc3.alpha_pvalue < 0.05 {
        println!("Resultado: REJEITAR H0 (α ≠ 0 com 95% de confiança)");
    } else {
        println!("Resultado: NÃO rejeitar H0 (sem evidência de alpha)");
    }

    println!("\nH0: β = 1 (ativo tem risco = mercado)");
    if capm_hc3.is_beta_different_from_one(0.05) {
        println!("Resultado: REJEITAR H0 (β ≠ 1 com 95% de confiança)");
        if capm_hc3.beta > 1.0 {
            println!("  → Beta > 1: ativo mais arriscado que mercado");
        } else {
            println!("  → Beta < 1: ativo menos arriscado que mercado");
        }
    } else {
        println!("Resultado: NÃO rejeitar H0 (β ≈ 1)");
    }

    println!("\n{}", "=".repeat(80));
    println!("FIM DOS EXEMPLOS");
    println!("{}", "=".repeat(80));

    Ok(())
}
