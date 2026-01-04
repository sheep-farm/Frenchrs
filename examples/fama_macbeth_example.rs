use frenchrs::FamaMacBeth;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("FAMA-MACBETH TWO-PASS REGRESSION");
    println!("{}", "=".repeat(80));
    println!("\nEstimatestion of prêmios of risk usando metodologia of Fama-MacBeth (1973)");
    println!("with correção of Shanken for standard errors.");

    // ========================================================================
    // DADOS SIMULADOS - 25 portfolios, 60 meses, 3 factors
    // ========================================================================

    let t = 60; // meses
    let n_portfolios = 25;
    let n_factors = 3;

    println!("\n{}", "-".repeat(80));
    println!("Configuração:");
    println!("  • Portfolios: {}", n_portfolios);
    println!("  • Factors: {}", n_factors);
    println!("  • Periods: {} meses", t);
    println!("{}", "-".repeat(80));

    // Gerador of números pseudo-aleatórios simples
    let mut rng = 12345u64;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    };

    // ========================================================================
    // SIMULAR FATORES: Market, SMB, HML
    // ========================================================================

    let factor_names = vec!["Market".to_string(), "SMB".to_string(), "HML".to_string()];

    let factors = Array2::from_shape_fn((t, n_factors), |(_i, j)| {
        let base = match j {
            0 => 0.008, // Market: mean ~0.8% ao mês
            1 => 0.002, // SMB: mean ~0.2% ao mês
            2 => 0.003, // HML: mean ~0.3% ao mês
            _ => 0.0,
        };
        base + rand() * 0.04
    });

    println!("\nFactors gerados:");
    println!("{:<15} {:>12} {:>12}", "Factor", "Mean", "Std Dev");
    println!("{}", "-".repeat(40));
    for (i, name) in factor_names.iter().enumerate() {
        let col = factors.column(i);
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(1.0);
        println!(
            "{:<15} {:>11.2}% {:>11.2}%",
            name,
            mean * 100.0,
            std * 100.0
        );
    }

    // ========================================================================
    // SIMULAR RETORNOS DOS PORTFOLIOS
    // ========================================================================
    // Cada portfolio tem diferentes exposições aos factors (betas)
    // Organizados em quintis por size e value (5×5 = 25 portfolios)

    let mut portfolio_names = Vec::new();
    for size in &["Small", "2", "3", "4", "Big"] {
        for value in &["Growth", "2", "3", "4", "Value"] {
            portfolio_names.push(format!("{}/{}", size, value));
        }
    }

    let mut returns = Array2::<f64>::zeros((t, n_portfolios));

    // Definir betas "verdadeiros" for each portfolio
    let mut true_betas = Vec::new();

    for p in 0..n_portfolios {
        let size_quintile = p / 5; // 0-4
        let value_quintile = p % 5; // 0-4

        // Betas variesm por características:
        // - Small caps têm beta_market greater
        // - Small caps têm beta_smb positivo greater
        // - Value stocks têm beta_hml positivo greater

        let beta_market = 0.8 + (size_quintile as f64) * 0.1;
        let beta_smb = -0.5 + (4 - size_quintile) as f64 * 0.25; // small = high
        let beta_hml = -0.4 + (value_quintile as f64) * 0.2; // value = high

        true_betas.push(vec![beta_market, beta_smb, beta_hml]);

        // Gerar returns baseados nos factors + alpha + ruído idiossincrático
        let alpha = 0.001 + rand() * 0.002; // alpha pequeno

        for time in 0..t {
            let mut ret = alpha;

            // Exposição aos factors
            ret += beta_market * factors[[time, 0]];
            ret += beta_smb * factors[[time, 1]];
            ret += beta_hml * factors[[time, 2]];

            // Ruído idiossincrático
            ret += rand() * 0.015;

            returns[[time, p]] = ret;
        }
    }

    println!("\nReturns gerados for {} portfolios", n_portfolios);

    // ========================================================================
    // ESTIMAR FAMA-MACBETH
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("ESTIMANDthe modelO FAMA-MACBETH...");
    println!("{}", "=".repeat(80));

    let result = FamaMacBeth::fit(
        &returns,
        &factors,
        CovarianceType::HC3,
        Some(portfolio_names.clone()),
        Some(factor_names.clone()),
    )?;

    // ========================================================================
    // RESULTS
    // ========================================================================

    println!("{}", result);

    // ========================================================================
    // ANALYSIS OFTALHADA DOS LAMBDAS
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION DOS PRÊMIOS DE RISCO");
    println!("{}", "=".repeat(80));

    println!("\n1. CONSTANTE (λ_0):");
    println!("   Value: {:.4}% ao mês", result.lambda_const() * 100.0);
    println!("   Interpretation: Taxa of return for asset with beta zero");
    println!(
        "   Significance (Shanken): {}",
        if result.pval_shanken[0] < 0.05 {
            "Significativo a 5%"
        } else {
            "Não significant a 5%"
        }
    );

    for (i, name) in factor_names.iter().enumerate() {
        println!("\n{}. {} (λ_{}):", i + 2, name, i + 1);
        println!("   Prêmio of risk: {:.4}% ao mês", result.lambda(i) * 100.0);
        println!(
            "   Annualized: {:.2}% ao ano",
            result.lambda(i) * 12.0 * 100.0
        );

        if result.is_significant_shanken(i, 0.05) {
            println!("   ✓ Significativo a 5% (Shanken)");
            println!("   Interpretation: Factor é precificado pelthe market");
        } else {
            println!("   ✗ Não significant a 5% (Shanken)");
            println!("   Interpretation: Factor can not be relevante for pricing");
        }

        let t_shanken = result.tstat_shanken[i + 1];
        println!("   t-statistic (Shanken): {:.2}", t_shanken);
    }

    // ========================================================================
    // COMPARAÇÃO: BETAS VERDADEIROS vs ESTIMATED
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("COMPARAÇÃO: BETAS VERDADEIROS vs ESTIMATED (Primeiros 5 portfolios)");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Portfolio",
        "True β_Mkt",
        "Est β_Mkt",
        "True β_SMB",
        "Est β_SMB",
        "True β_HML",
        "Est β_HML"
    );
    println!("{}", "-".repeat(100));

    for p in 0..5.min(n_portfolios) {
        let true_beta = &true_betas[p];
        let est_beta = result.betas.row(p);

        println!(
            "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            &portfolio_names[p],
            true_beta[0],
            est_beta[0],
            true_beta[1],
            est_beta[1],
            true_beta[2],
            est_beta[2]
        );
    }

    println!("\n... ({} portfolios restbefore)", n_portfolios - 5);

    // ========================================================================
    // PRICING ERRORS
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("PRICING ERRORS (Primeiros 10 portfolios)");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<20} {:>15} {:>15} {:>15}",
        "Portfolio", "Mean Return", "Model Return", "Pricing Error"
    );
    println!("{}", "-".repeat(70));

    for (p, name) in portfolio_names
        .iter()
        .enumerate()
        .take(10.min(n_portfolios))
    {
        println!(
            "{:<20} {:>14.2}% {:>14.2}% {:>14.4}%",
            name,
            result.mean_returns[p] * 100.0,
            result.model_returns[p] * 100.0,
            result.pricing_errors[p] * 100.0
        );
    }

    if n_portfolios > 10 {
        println!("\n... ({} portfolios restbefore)", n_portfolios - 10);
    }

    // ========================================================================
    // METRICS DE AJUSTE
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("METRICS DE FIT QUALITY");
    println!("{}", "=".repeat(80));

    println!("\nCross-Sectional:");
    println!(
        "  • R² médio: {:.2}%",
        result.r2_cross_sectional_mean * 100.0
    );
    println!("  • Periods efetivos: {}", result.t_eff);

    println!("\nPricing:");
    println!("  • R² pricing: {:.2}%", result.r2_pricing() * 100.0);
    println!("  • RMSE: {:.4}%", result.rmse_pricing() * 100.0);
    println!("  • MAE: {:.4}%", result.mae_pricing() * 100.0);

    // ========================================================================
    // CONCLUSÕES
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSÕES");
    println!("{}", "=".repeat(80));

    println!("\n1. METODOLOGIA:");
    println!("   • Primeira passagem: Estimates betas via regresare time-beies");
    println!("   • Segunda passagem: Estimates prêmios of risk via regresare cross-section");
    println!("   • Shanken correction: Ajusta erros padrão for beta estimated");

    println!("\n2. PRÊMIOS DE RISCO:");
    let significant_factors: Vec<String> = (0..n_factors)
        .filter(|&i| result.is_significant_shanken(i, 0.05))
        .map(|i| factor_names[i].clone())
        .collect();

    if significant_factors.is_empty() {
        println!("   • Nenhum factor é thististicamente significant a 5%");
    } else {
        let factors_str: Vec<&str> = significant_factors.iter().map(|s| s.as_str()).collect();
        println!(
            "   • Factors significativos (5%): {}",
            factors_str.join(", ")
        );
    }

    println!("\n3. QUALIDADE Dthe model:");
    if result.r2_cross_sectional_mean > 0.7 {
        println!("   ✓ Bom fit cross-sectional (R² > 70%)");
    } else if result.r2_cross_sectional_mean > 0.4 {
        println!("   ~ Fit moderado cross-sectional");
    } else {
        println!("   ✗ Fit fraco cross-sectional");
    }

    if result.rmse_pricing() < 0.01 {
        println!("   ✓ Pricing errors pequenos (RMSE < 1%)");
    } else {
        println!("   ~ Pricing errors moderados");
    }

    println!("\n{}", "=".repeat(80));

    Ok(())
}
