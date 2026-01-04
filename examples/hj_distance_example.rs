use frenchrs::HJDistance;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("HANSEN-JAGANNATHAN DISTANCE TEST - EXAMPLE");
    println!("{}\n", "=".repeat(80));

    // Generate synthetic data for demonstration
    let t = 120; // 120 months of data
    let n = 15; // 15 assets
    let k = 3; // 3 factors

    println!("Dataset: {} months, {} assets, {} factors\n", t, n, k);

    // Simulate factor returns
    let mut rng = 42u64;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    };

    let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.03);

    // Simulate asset returns with factor exposure + some mispricing (non-zero alphas)
    let mut returns = Array2::from_shape_fn((t, n), |_| rand() * 0.02);

    for i in 0..t {
        for j in 0..n {
            let mut exposure = 0.0;
            for f in 0..k {
                let beta = 0.5 + (j as f64 / n as f64) * 0.8;
                exposure += factors[[i, f]] * beta;
            }
            // Add factor exposure
            returns[[i, j]] += exposure;

            // Add some mispricing (non-zero alpha) for some assets
            if j < 5 {
                returns[[i, j]] += 0.002; // Positive alpha for first 5 assets
            }
        }
    }

    // Asset names
    let asset_names: Vec<String> = (1..=n)
        .map(|i| {
            if i <= 5 {
                format!("High Alpha Stock {}", i)
            } else {
                format!("Stock {}", i)
            }
        })
        .collect();

    // Perform Hansen-Jagannathan distance test
    println!("Running Hansen-Jagannathan distance test...\n");

    let result = HJDistance::fit(
        &returns,
        &factors,
        CovarianceType::HC3,
        Some(asset_names.clone()),
    )
    .expect("Failed to compute HJ distance");

    // Display summary table
    print!("{}", result.summary_table());

    // Display individual alphas
    println!("\n{}", "=".repeat(80));
    println!("ASSET-LEVEL PRICING ERRORS (ALPHAS)");
    println!("{}", "=".repeat(80));
    println!("\n{:<25} {:>15}", "Asset", "Alpha (Monthly)");
    println!("{}", "-".repeat(45));

    for name in &result.asset_names {
        if let Some(alpha) = result.get_alpha(name) {
            println!("{:<25} {:>15.6}", name, alpha);
        }
    }

    // Identify assets with largest pricing errors
    println!("\n{}", "=".repeat(80));
    println!("TOP 5 ASSETS WITH LARGEST ABSOLUTE PRICING ERRORS");
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
            "{}. {:<25} Alpha: {:>10.6} ({} per year)",
            i + 1,
            name,
            alpha,
            format!("{:.2}%", alpha * 12.0 * 100.0)
        );
    }

    // Statistical inference
    println!("\n{}", "=".repeat(80));
    println!("STATISTICAL INFERENCE");
    println!("{}", "=".repeat(80));
    println!();

    println!("Test: H₀: Model correctly prices all assets (d = 0)");
    println!("      Hₐ: Model has pricing errors (d > 0)\n");

    println!("Test Statistic (T × d²): {:.4}", result.chi2_stat);
    println!("Degrees of Freedom:      {}", result.n_assets);
    println!("P-value:                 {:.4}\n", result.p_value);

    println!("Decision at α = 0.05:");
    if result.reject_model(0.05) {
        println!("  ✗ REJECT H₀");
        println!("  The factor model has statistically significant pricing errors.");
        println!("  Consider:");
        println!("    - Adding additional factors");
        println!("    - Revising model specification");
        println!("    - Examining which assets are mispriced");
    } else {
        println!("  ✓ DO NOT REJECT H₀");
        println!("  The factor model adequately prices the assets.");
        println!("  The observed pricing errors are not statistically significant.");
    }

    // Model quality classification
    println!("\n{}", "=".repeat(80));
    println!("MODEL QUALITY ASSESSMENT");
    println!("{}", "=".repeat(80));
    println!();

    println!("Classification: {}", result.model_quality_classification());
    println!();

    if result.hj_distance < 0.05 {
        println!("✓ Very low HJ distance - Excellent model fit");
    } else if result.hj_distance < 0.10 {
        println!("✓ Low HJ distance - Good model fit");
    } else if result.hj_distance < 0.20 {
        println!("⚠ Moderate HJ distance - Acceptable fit with some pricing errors");
    } else {
        println!("✗ High HJ distance - Poor model fit with significant pricing errors");
    }

    // Interpretation guide
    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION GUIDE");
    println!("{}", "=".repeat(80));
    println!();
    println!("Hansen-Jagannathan Distance:");
    println!("  - Measures distance between model SDF and set of valid SDFs");
    println!("  - In practice: weighted measure of pricing errors (alphas)");
    println!("  - d = sqrt(α' Σ⁻¹ α) where:");
    println!("    α = vector of pricing errors (alphas)");
    println!("    Σ = covariance matrix of residuals");
    println!();
    println!("Interpretation:");
    println!("  d = 0:     Perfect pricing (all alphas = 0)");
    println!("  d < 0.10:  Good model fit");
    println!("  d < 0.20:  Acceptable fit");
    println!("  d ≥ 0.20:  Poor fit, consider model revision");
    println!();
    println!("Chi-squared Test:");
    println!("  - Under H₀ (d = 0): T × d² ~ χ²(N) asymptotically");
    println!("  - Reject H₀ if p-value < significance level");
    println!("  - Rejection indicates model misspecification");

    // Export results
    println!("\n{}", "=".repeat(80));
    println!("CSV EXPORT");
    println!("{}", "=".repeat(80));
    println!("Results can be exported to CSV using to_csv_string()");

    println!("\n{}", "=".repeat(80));
}
