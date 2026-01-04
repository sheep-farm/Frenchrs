use frenchrs::ResidualCorrelation;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("RESIDUAL CORRELATION ANALYSIS - EXAMPLE");
    println!("{}\n", "=".repeat(80));

    // Generate synthetic data for demonstration
    let t = 120; // 120 months of data
    let n = 5; // 5 assets
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
    let mut returns = Array2::from_shape_fn((t, n), |_| rand() * 0.02);

    // Add common factor exposure
    for i in 0..t {
        for j in 0..n {
            let mut exposure = 0.0;
            for f in 0..k {
                let beta = 0.5 + (j as f64 / n as f64) * 0.8;
                exposure += factors[[i, f]] * beta;
            }
            returns[[i, j]] += exposure;
        }
    }

    // Add some correlated residual component between assets 0 and 1 (simulate missing factor)
    let missing_factor: Vec<f64> = (0..t).map(|_| rand() * 0.01).collect();
    for i in 0..t {
        returns[[i, 0]] += missing_factor[i] * 0.8;
        returns[[i, 1]] += missing_factor[i] * 0.8;
    }

    // Asset names
    let asset_names = vec![
        "Tech Stock A".to_string(),
        "Tech Stock B".to_string(),
        "Energy Stock".to_string(),
        "Healthcare Stock".to_string(),
        "Consumer Stock".to_string(),
    ];

    // Perform residual correlation analysis
    println!("Running residual correlation analysis...\n");

    let result = ResidualCorrelation::fit(
        &returns,
        &factors,
        CovarianceType::HC3,
        Some(asset_names.clone()),
    )
    .expect("Failed to perform residual correlation analysis");

    // Display summary statistics
    println!("{}", "=".repeat(80));
    println!("SUMMARY STATISTICS");
    println!("{}", "=".repeat(80));

    let summary = result.summary_stats();
    println!("Number of Assets:               {}", summary.n_assets);
    println!(
        "Average Off-Diagonal Correlation: {:.4}",
        summary.avg_off_diag_corr
    );
    println!(
        "Min Off-Diagonal Correlation:     {:.4}",
        summary.min_off_diag_corr
    );
    println!(
        "Max Off-Diagonal Correlation:     {:.4}",
        summary.max_off_diag_corr
    );
    println!(
        "Maximum Eigenvalue:              {:.4}",
        summary.max_eigenvalue
    );
    println!("\nClassification: {}", summary.correlation_classification());

    // Display correlation matrix
    println!("\n{}", "=".repeat(80));
    println!("CORRELATION MATRIX OF RESIDUALS");
    println!("{}", "=".repeat(80));
    println!(
        "\n{:<20} {:>15} {:>15} {:>15} {:>15} {:>15}",
        "", "Tech A", "Tech B", "Energy", "Healthcare", "Consumer"
    );
    println!("{}", "-".repeat(105));

    for (i, row_name) in asset_names.iter().enumerate() {
        print!("{:<20}", row_name);
        for j in 0..n {
            print!(" {:>15.4}", result.correlation_matrix[[i, j]]);
        }
        println!();
    }

    // Highlight high correlations
    println!("\n{}", "=".repeat(80));
    println!("HIGH RESIDUAL CORRELATIONS (> 0.3)");
    println!("{}", "=".repeat(80));
    println!();

    let mut found_high = false;
    for i in 0..n {
        for j in (i + 1)..n {
            let corr = result.correlation_matrix[[i, j]];
            if corr.abs() > 0.3 {
                println!(
                    "{:<20} <-> {:<20}  Correlation: {:.4}",
                    asset_names[i], asset_names[j], corr
                );
                found_high = true;
            }
        }
    }

    if !found_high {
        println!("No high residual correlations detected.");
    }

    // Most correlated assets for each asset
    println!("\n{}", "=".repeat(80));
    println!("TOP 2 MOST CORRELATED ASSETS FOR EACH ASSET");
    println!("{}", "=".repeat(80));
    println!();

    for asset in &asset_names {
        println!("{}:", asset);
        let top_correlated = result.most_correlated_assets(asset, 2);
        for (name, corr) in top_correlated {
            println!("  - {:<25} Correlation: {:.4}", name, corr);
        }
        println!();
    }

    // Interpretation guide
    println!("{}", "=".repeat(80));
    println!("INTERPRETATION GUIDE");
    println!("{}", "=".repeat(80));
    println!("Residual Correlation Matrix: Measures correlation between model residuals");
    println!();
    println!("Low correlation (< 0.1):");
    println!("  ✓ Indicates well-specified model");
    println!("  ✓ Factors capture most common variation");
    println!("  ✓ Remaining risk is mostly idiosyncratic");
    println!();
    println!("Moderate correlation (0.1 - 0.3):");
    println!("  ⚠ Some common variation not captured by factors");
    println!("  ⚠ May benefit from additional factors");
    println!();
    println!("High correlation (> 0.3):");
    println!("  ✗ Significant missing common factors");
    println!("  ✗ Model may be misspecified");
    println!("  ✗ Consider adding industry/sector factors");
    println!();
    println!("Maximum Eigenvalue:");
    println!("  - Measures concentration of correlation structure");
    println!("  - Higher values suggest stronger common component in residuals");
    println!("  - Compare to theoretical maximum (n_assets) for context");

    // Export to CSV
    println!("\n{}", "=".repeat(80));
    println!("CSV EXPORT");
    println!("{}", "=".repeat(80));
    println!("Correlation matrix and residuals can be exported to CSV:");
    println!("  - correlation_to_csv_string()");
    println!("  - residuals_to_csv_string()");

    println!("\n{}", "=".repeat(80));
}
