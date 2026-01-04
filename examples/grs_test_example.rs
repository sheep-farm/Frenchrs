use frenchrs::GRSTest;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("GRS TEST (Gibbons, Ross & Shanken, 1989) - EXAMPLE");
    println!("{}\n", "=".repeat(80));

    // Generate synthetic data for demonstration
    let t = 120; // 120 months of data
    let n = 20; // 20 assets
    let k = 3; // 3 factors

    println!("Dataset: {} months, {} assets, {} factors\n", t, n, k);

    // Simulate factor returns
    let mut rng = 42u64;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    };

    let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.03);

    // Simulate asset returns with factor exposure
    // First 10 assets: well-explained by factors (small alphas)
    // Last 10 assets: significant mispricing (large alphas)
    let mut returns = Array2::from_shape_fn((t, n), |_| rand() * 0.02);

    for i in 0..t {
        for j in 0..n {
            let mut exposure = 0.0;
            for f in 0..k {
                let beta = 0.5 + (j as f64 / n as f64) * 0.8;
                exposure += factors[[i, f]] * beta;
            }
            returns[[i, j]] += exposure;

            // Add mispricing to last 10 assets
            if j >= 10 {
                returns[[i, j]] += 0.003; // 0.3% monthly alpha (3.6% annually)
            }
        }
    }

    // Asset names
    let asset_names: Vec<String> = (1..=n)
        .map(|i| {
            if i <= 10 {
                format!("Well-Priced Stock {}", i)
            } else {
                format!("Mispriced Stock {}", i)
            }
        })
        .collect();

    // Perform GRS test
    println!("Running GRS test...\n");

    let result = GRSTest::fit(
        &returns,
        &factors,
        CovarianceType::HC3,
        Some(asset_names.clone()),
    )
    .expect("Failed to perform GRS test");

    // Display summary table
    print!("{}", result.summary_table());

    // Display individual alphas
    println!("\n{}", "=".repeat(80));
    println!("ASSET-LEVEL PRICING ERRORS (ALPHAS)");
    println!("{}", "=".repeat(80));
    println!("\n{:<30} {:>15}", "Asset", "Alpha (Monthly)");
    println!("{}", "-".repeat(50));

    for name in &result.asset_names {
        if let Some(alpha) = result.get_alpha(name) {
            println!("{:<30} {:>15.6}", name, alpha);
        }
    }

    // Identify assets with largest absolute alphas
    println!("\n{}", "=".repeat(80));
    println!("TOP 5 ASSETS WITH LARGEST ABSOLUTE ALPHAS");
    println!("{}", "=".repeat(80));
    println!();

    let mut alpha_pairs: Vec<(String, f64)> = result
        .asset_names
        .iter()
        .filter_map(|name| result.get_alpha(name).map(|alpha| (name.clone(), alpha)))
        .collect();

    alpha_pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    for (i, (name, alpha)) in alpha_pairs.iter().take(5).enumerate() {
        println!(
            "{}. {:<30} Alpha: {:>10.6} ({:>6.2}% per year)",
            i + 1,
            name,
            alpha,
            alpha * 12.0 * 100.0
        );
    }

    // Detailed test statistics
    println!("\n{}", "=".repeat(80));
    println!("DETAILED TEST STATISTICS");
    println!("{}", "=".repeat(80));
    println!();

    println!("GRS F-Statistic:          {:.4}", result.grs_f_stat);
    println!(
        "Degrees of Freedom:       F({}, {})",
        result.df1, result.df2
    );
    println!("P-value:                  {:.6}", result.p_value);
    println!();
    println!("Alpha Quadratic Form:     {:.6}", result.alpha_quad_form);
    println!("  (α' Σ_ε^{{-1}} α)");
    println!();
    println!("Denominator:              {:.6}", result.denominator);
    println!("  (1 + μ_f' Σ_f^{{-1}} μ_f)");
    println!();

    // Statistical decision at different significance levels
    println!("{}", "=".repeat(80));
    println!("STATISTICAL DECISION");
    println!("{}", "=".repeat(80));
    println!();

    let alphas = [0.01, 0.05, 0.10];
    for &alpha in &alphas {
        let decision = if result.reject_model(alpha) {
            "REJECT H₀"
        } else {
            "DO NOT REJECT H₀"
        };
        println!("At α = {:.2}: {}", alpha, decision);
    }

    println!();
    if result.reject_model(0.05) {
        println!("✗ The factor model is REJECTED at 5% significance level");
        println!("  At least one asset has significant pricing error");
        println!("  Conclusion: Model does NOT adequately price all assets");
    } else {
        println!("✓ The factor model is NOT REJECTED at 5% significance level");
        println!("  All alphas are jointly not significantly different from zero");
        println!("  Conclusion: Model adequately prices the cross-section");
    }

    // Comparison with testing alphas individually
    println!("\n{}", "=".repeat(80));
    println!("GRS TEST vs. INDIVIDUAL T-TESTS");
    println!("{}", "=".repeat(80));
    println!();
    println!("Why use GRS test instead of testing each alpha individually?");
    println!();
    println!("1. JOINT TEST:");
    println!("   - GRS tests all alphas simultaneously");
    println!("   - Accounts for cross-correlation of residuals");
    println!("   - More powerful than multiple individual tests");
    println!();
    println!("2. MULTIPLE TESTING PROBLEM:");
    println!("   - Testing {} alphas individually at 5% level", n);
    println!(
        "   - Expected false rejections: {:.1} assets",
        n as f64 * 0.05
    );
    println!("   - GRS controls overall Type I error rate");
    println!();
    println!("3. CORRELATION STRUCTURE:");
    println!("   - Individual tests ignore Σ_ε (residual covariance)");
    println!("   - GRS explicitly accounts for correlation");
    println!("   - More accurate inference in presence of correlation");

    // Interpretation guide
    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION GUIDE");
    println!("{}", "=".repeat(80));
    println!();
    println!("GRS F-Statistic:");
    println!("  F = ((T-N-K)/N) × (α' Σ_ε^{{-1}} α) / (1 + μ_f' Σ_f^{{-1}} μ_f)");
    println!();
    println!("Under H₀ (all alphas = 0): F ~ F(N, T-N-K)");
    println!();
    println!("Components:");
    println!("  - α: Vector of pricing errors (alphas)");
    println!("  - Σ_ε: Covariance matrix of residuals");
    println!("  - μ_f: Mean vector of factors");
    println!("  - Σ_f: Covariance matrix of factors");
    println!();
    println!("Decision Rule:");
    println!("  - Large F-stat, small p-value → Reject H₀");
    println!("  - Small F-stat, large p-value → Do not reject H₀");
    println!();
    println!("Practical Implications:");
    println!("  - Rejection: Model has systematic pricing errors");
    println!("  - Non-rejection: Model adequately prices assets");
    println!("  - Consider model specification, factor choice, sample period");

    println!("\n{}", "=".repeat(80));
}
