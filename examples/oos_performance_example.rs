use frenchrs::OOSPerformance;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("OUT-OF-SAMPLE PERFORMANCE ANALYSIS - EXAMPLE");
    println!("{}\n", "=".repeat(80));

    // Generate synthetic data for demonstration
    // In practice, you would load real asset returns and factor data
    let t = 120; // 120 months of data
    let n = 5; // 5 assets
    let k = 3; // 3 factors

    println!("Dataset: {} months, {} assets, {} factors", t, n, k);
    println!("Split: 70% in-sample (84 months), 30% out-of-sample (36 months)\n");

    // Simulate returns with some factor exposure
    let mut rng = 42u64;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    };

    let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.03);
    let mut returns = Array2::from_shape_fn((t, n), |_| rand() * 0.02);

    // Add factor exposure to returns (different for each asset)
    for i in 0..t {
        for j in 0..n {
            let mut exposure = 0.0;
            for f in 0..k {
                // Each asset has different sensitivities to factors
                let beta = 0.3 + (j as f64 / n as f64) * 0.8 + (f as f64 / k as f64) * 0.4;
                exposure += factors[[i, f]] * beta;
            }
            returns[[i, j]] += exposure;
        }
    }

    // Asset names
    let asset_names = vec![
        "Tech Stock".to_string(),
        "Financial Stock".to_string(),
        "Energy Stock".to_string(),
        "Healthcare Stock".to_string(),
        "Consumer Stock".to_string(),
    ];

    // Perform out-of-sample analysis
    println!("Running out-of-sample performance analysis...\n");

    let result = OOSPerformance::fit(
        &returns,
        &factors,
        0.7, // 70/30 split
        CovarianceType::HC3,
        Some(asset_names.clone()),
    )
    .expect("Failed to perform OOS analysis");

    // Display results
    println!("{}", "=".repeat(80));
    println!("INDIVIDUAL ASSET RESULTS");
    println!("{}", "=".repeat(80));
    println!(
        "\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>25}",
        "Asset", "R²_in", "R²_out", "R²_OOS_CT", "RMSE_out", "Predictive Power"
    );
    println!("{}", "-".repeat(100));

    for row in result.to_table() {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>25}",
            row.asset,
            row.r2_in,
            row.r2_out,
            row.r2_oos_ct,
            row.rmse_out,
            result
                .get(&row.asset)
                .unwrap()
                .predictive_power_classification()
        );
    }

    // Summary statistics
    let stats = result.summary_stats();
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY STATISTICS");
    println!("{}", "=".repeat(80));
    println!("Total Assets Analyzed:           {}", stats.total_assets);
    println!(
        "Assets Beating Benchmark:        {} ({:.1}%)",
        stats.assets_beating_benchmark, stats.pct_beating_benchmark
    );
    println!(
        "Mean Campbell-Thompson R²_OOS:   {:.4}",
        stats.mean_r2_oos_ct
    );
    println!(
        "Median Campbell-Thompson R²_OOS: {:.4}",
        stats.median_r2_oos_ct
    );

    // Detailed analysis for assets beating benchmark
    let beating_benchmark = result.assets_beating_benchmark();
    if !beating_benchmark.is_empty() {
        println!("\n{}", "=".repeat(80));
        println!("ASSETS WITH POSITIVE PREDICTIVE POWER (R²_OOS_CT > 0)");
        println!("{}", "=".repeat(80));

        for asset in beating_benchmark {
            println!("\n{}", asset.asset);
            println!(
                "  In-sample:  R² = {:.4}, RMSE = {:.4}",
                asset.r2_in, asset.rmse_in
            );
            println!(
                "  Out-of-sample: R² = {:.4}, RMSE = {:.4}",
                asset.r2_out, asset.rmse_out
            );
            println!("  Campbell-Thompson R²_OOS: {:.4}", asset.r2_oos_ct);
            println!(
                "  Beats benchmark by {:.2}% in terms of MSE reduction",
                asset.r2_oos_ct * 100.0
            );

            if asset.no_overfitting() {
                println!("  ✓ No overfitting detected");
            } else {
                println!("  ⚠ Possible overfitting (OOS R² much lower than in-sample)");
            }
        }
    } else {
        println!("\n⚠ No assets beat the historical mean benchmark");
        println!("  This suggests the factor model has limited predictive power");
        println!("  for these assets in the out-of-sample period.");
    }

    // Interpretation guide
    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION GUIDE");
    println!("{}", "=".repeat(80));
    println!("R²_in:       In-sample fit quality (how well model explains training data)");
    println!("R²_out:      Out-of-sample fit quality (how well model explains test data)");
    println!("R²_OOS_CT:   Campbell-Thompson R² (predictive power vs. historical mean)");
    println!("             - Positive: Model beats naive historical mean forecast");
    println!("             - Negative: Historical mean is better predictor");
    println!("             - Close to 0: Model adds little predictive value");
    println!("RMSE_out:    Root Mean Squared Error out-of-sample (lower is better)");
    println!("\nNote: Out-of-sample performance is the true test of a model's");
    println!("      predictive power, as it uses data the model has never seen.");

    println!("\n{}", "=".repeat(80));
}
