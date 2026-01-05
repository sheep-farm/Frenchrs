use frenchrs::ResidualDiagnostics;
use greeners::CovarianceType;
use ndarray::Array2;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("RESIDUAL DIAGNOSTICS - COMPREHENSIVE EXAMPLE");
    println!("{}\n", "=".repeat(80));

    // Generate synthetic data for demonstration
    let t = 120; // 120 months of data
    let n = 10; // 10 assets
    let k = 3; // 3 factors

    println!("Dataset: {} months, {} assets, {} factors\n", t, n, k);

    // Simulate factor returns
    let mut rng = 42u64;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    };

    let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.03);

    // Simulate asset returns with different properties:
    // - Assets 1-3: Well-behaved (iid normal residuals)
    // - Assets 4-5: Heteroscedasticity
    // - Assets 6-7: Autocorrelation
    // - Assets 8-9: Non-normal residuals
    // - Asset 10: Multiple issues
    let mut returns = Array2::zeros((t, n));

    for i in 0..t {
        for j in 0..n {
            let mut exposure = 0.0;
            for f in 0..k {
                let beta = 0.5 + (j as f64 / n as f64) * 0.8;
                exposure += factors[[i, f]] * beta;
            }
            returns[[i, j]] = exposure;

            // Add different types of residuals
            if j < 3 {
                // Assets 1-3: Well-behaved iid normal
                returns[[i, j]] += rand() * 0.015;
            } else if j < 5 {
                // Assets 4-5: Heteroscedasticity (variance increases over time)
                let het_factor = 1.0 + (i as f64 / t as f64);
                returns[[i, j]] += rand() * 0.015 * het_factor;
            } else if j < 7 {
                // Assets 6-7: Autocorrelation
                if i > 0 {
                    let prev_resid = returns[[i - 1, j]]
                        - (0..k)
                            .map(|f| factors[[i - 1, f]] * (0.5 + (j as f64 / n as f64) * 0.8))
                            .sum::<f64>();
                    returns[[i, j]] += 0.5 * prev_resid + rand() * 0.010;
                } else {
                    returns[[i, j]] += rand() * 0.010;
                }
            } else if j < 9 {
                // Assets 8-9: Non-normal (skewed)
                let skewed = if rand() > 0.0 {
                    rand().abs() * 0.025
                } else {
                    rand() * 0.008
                };
                returns[[i, j]] += skewed;
            } else {
                // Asset 10: Multiple issues (heteroscedasticity + autocorrelation)
                let het_factor = 1.0 + (i as f64 / t as f64);
                if i > 0 {
                    let prev_resid = returns[[i - 1, j]]
                        - (0..k)
                            .map(|f| factors[[i - 1, f]] * (0.5 + (j as f64 / n as f64) * 0.8))
                            .sum::<f64>();
                    returns[[i, j]] += 0.6 * prev_resid + rand() * 0.012 * het_factor;
                } else {
                    returns[[i, j]] += rand() * 0.012 * het_factor;
                }
            }
        }
    }

    // Asset names
    let asset_names: Vec<String> = vec![
        "Well-Behaved Asset 1".to_string(),
        "Well-Behaved Asset 2".to_string(),
        "Well-Behaved Asset 3".to_string(),
        "Heteroscedastic Asset 1".to_string(),
        "Heteroscedastic Asset 2".to_string(),
        "Autocorrelated Asset 1".to_string(),
        "Autocorrelated Asset 2".to_string(),
        "Non-Normal Asset 1".to_string(),
        "Non-Normal Asset 2".to_string(),
        "Multiple Issues Asset".to_string(),
    ];

    // Perform residual diagnostics
    println!("Running comprehensive residual diagnostics...\n");

    let result = ResidualDiagnostics::fit(
        &returns,
        &factors,
        CovarianceType::HC3,
        Some(asset_names.clone()),
    )
    .expect("Failed to perform residual diagnostics");

    // Display detailed diagnostics for each asset
    println!("{}", "=".repeat(80));
    println!("DETAILED DIAGNOSTICS BY ASSET");
    println!("{}", "=".repeat(80));

    for name in &asset_names {
        let diag = result.diagnostics.get(name).unwrap();

        println!("\n{}", "-".repeat(80));
        println!("Asset: {}", name);
        println!("{}", "-".repeat(80));

        println!("\n1. AUTOCORRELATION TESTS:");
        println!("   Durbin-Watson:     {:.4}", diag.durbin_watson);
        if diag.has_positive_autocorr() {
            println!("   └─ WARNING: Positive autocorrelation detected (DW < 1.5)");
        } else if diag.has_negative_autocorr() {
            println!("   └─ WARNING: Negative autocorrelation detected (DW > 2.5)");
        } else {
            println!("   └─ OK: No strong autocorrelation");
        }

        println!(
            "   Ljung-Box:         {:.4} (p = {:.4})",
            diag.lb_stat, diag.lb_p_value
        );
        if diag.lb_p_value < 0.05 {
            println!("   └─ WARNING: Significant autocorrelation at multiple lags");
        } else {
            println!("   └─ OK: No significant autocorrelation");
        }

        println!("\n2. HETEROSCEDASTICITY TESTS:");
        println!(
            "   Breusch-Pagan:     {:.4} (p = {:.4})",
            diag.bp_stat, diag.bp_p_value
        );
        if diag.has_heteroscedasticity() {
            println!("   └─ WARNING: Heteroscedasticity detected");
        } else {
            println!("   └─ OK: Homoscedastic residuals");
        }

        println!(
            "   White Test:        {:.4} (p = {:.4})",
            diag.white_stat, diag.white_p_value
        );
        if diag.white_rejects() {
            println!("   └─ WARNING: Heteroscedasticity detected (White test)");
        } else {
            println!("   └─ OK: Homoscedastic residuals");
        }

        println!(
            "   ARCH Test:         {:.4} (p = {:.4})",
            diag.arch_stat, diag.arch_p_value
        );
        if diag.has_arch_effects() {
            println!("   └─ WARNING: Conditional heteroscedasticity (ARCH effects)");
        } else {
            println!("   └─ OK: No ARCH effects");
        }

        println!("\n3. SPECIFICATION TESTS:");
        println!(
            "   RESET Test:        {:.4} (p = {:.4})",
            diag.reset_f, diag.reset_p_value
        );
        if diag.has_misspecification() {
            println!("   └─ WARNING: Functional form misspecification");
        } else {
            println!("   └─ OK: Functional form appears adequate");
        }

        println!(
            "   Chow Test:         {:.4} (p = {:.4})",
            diag.chow_f, diag.chow_p_value
        );
        if diag.has_structural_break() {
            println!("   └─ WARNING: Structural break detected");
        } else {
            println!("   └─ OK: No structural break");
        }

        println!("\n4. NORMALITY TEST:");
        println!(
            "   Jarque-Bera:       {:.4} (p = {:.4})",
            diag.jb_stat, diag.jb_p_value
        );
        if diag.non_normal_residuals() {
            println!("   └─ WARNING: Non-normal residuals");
        } else {
            println!("   └─ OK: Residuals appear normally distributed");
        }

        let issue_count = diag.count_issues();
        println!("\n   Total Issues Detected: {}", issue_count);
        if issue_count == 0 {
            println!("   ✓ All diagnostic tests passed");
        } else {
            println!("   ✗ Some diagnostic tests failed - model assumptions violated");
        }
    }

    // Summary table
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY TABLE - ISSUE COUNT BY ASSET");
    println!("{}", "=".repeat(80));
    println!();
    println!("{:<35} {:>10}", "Asset", "Issues");
    println!("{}", "-".repeat(50));

    for name in &asset_names {
        let diag = result.diagnostics.get(name).unwrap();
        let issues = diag.count_issues();
        let status = if issues == 0 { "✓" } else { "✗" };
        println!("{:<35} {:>9} {}", name, issues, status);
    }

    // Export to CSV
    println!("\n{}", "=".repeat(80));
    println!("CSV EXPORT");
    println!("{}", "=".repeat(80));
    println!();
    println!("CSV output (first 5 lines):");
    println!();

    let csv = result.to_csv_string();
    for (i, line) in csv.lines().take(5).enumerate() {
        if i == 0 {
            println!("Header: {}", line);
        } else {
            println!("Row {}: {}", i, line);
        }
    }
    println!("...");

    // Interpretation guide
    println!("\n{}", "=".repeat(80));
    println!("INTERPRETATION GUIDE");
    println!("{}", "=".repeat(80));
    println!();

    println!("1. DURBIN-WATSON STATISTIC:");
    println!("   - Range: [0, 4]");
    println!("   - DW ≈ 2: No autocorrelation");
    println!("   - DW < 1.5: Positive autocorrelation (current residual correlated with previous)");
    println!("   - DW > 2.5: Negative autocorrelation (alternating pattern)");
    println!();

    println!("2. LJUNG-BOX TEST:");
    println!("   - Tests for autocorrelation at multiple lags (default: 12)");
    println!("   - H₀: No autocorrelation at any lag");
    println!("   - p < 0.05: Reject H₀, autocorrelation present");
    println!();

    println!("3. BREUSCH-PAGAN TEST:");
    println!("   - Tests if variance depends on factor values");
    println!("   - H₀: Homoscedasticity (constant variance)");
    println!("   - p < 0.05: Reject H₀, heteroscedasticity present");
    println!();

    println!("4. WHITE TEST:");
    println!("   - More general heteroscedasticity test (includes squares and cross-products)");
    println!("   - H₀: Homoscedasticity");
    println!("   - p < 0.05: Reject H₀, heteroscedasticity present");
    println!();

    println!("5. RESET TEST:");
    println!("   - Ramsey's RESET test for functional form misspecification");
    println!("   - Tests if model is missing non-linear terms");
    println!("   - H₀: Linear specification is correct");
    println!("   - p < 0.05: Reject H₀, model misspecified");
    println!();

    println!("6. CHOW TEST:");
    println!("   - Tests for structural break at midpoint");
    println!("   - H₀: No structural break");
    println!("   - p < 0.05: Reject H₀, structural break present");
    println!();

    println!("7. ARCH TEST:");
    println!("   - Engle's test for conditional heteroscedasticity");
    println!("   - Tests if volatility clusters over time");
    println!("   - H₀: No ARCH effects");
    println!("   - p < 0.05: Reject H₀, ARCH effects present");
    println!();

    println!("8. JARQUE-BERA TEST:");
    println!("   - Tests if residuals are normally distributed");
    println!("   - Based on skewness and kurtosis");
    println!("   - H₀: Residuals are normal");
    println!("   - p < 0.05: Reject H₀, non-normal residuals");
    println!();

    println!("{}", "=".repeat(80));
    println!("PRACTICAL IMPLICATIONS");
    println!("{}", "=".repeat(80));
    println!();

    println!("When diagnostic tests fail:");
    println!();
    println!("• AUTOCORRELATION:");
    println!("  - Use Newey-West standard errors (HAC)");
    println!("  - Consider AR/MA models for residuals");
    println!("  - Check for omitted variables");
    println!();

    println!("• HETEROSCEDASTICITY:");
    println!("  - Use robust standard errors (HC3, HC4)");
    println!("  - Consider WLS (Weighted Least Squares)");
    println!("  - Transform variables (log, square root)");
    println!();

    println!("• FUNCTIONAL FORM:");
    println!("  - Add polynomial terms");
    println!("  - Include interaction effects");
    println!("  - Consider non-linear models");
    println!();

    println!("• STRUCTURAL BREAKS:");
    println!("  - Split sample at break point");
    println!("  - Include dummy variables");
    println!("  - Use regime-switching models");
    println!();

    println!("• ARCH EFFECTS:");
    println!("  - Use GARCH models");
    println!("  - Model time-varying volatility");
    println!("  - Consider stochastic volatility models");
    println!();

    println!("• NON-NORMALITY:");
    println!("  - Use robust inference");
    println!("  - Bootstrap standard errors");
    println!("  - Consider robust estimators (LAD, M-estimators)");
    println!();

    println!("\n{}", "=".repeat(80));
}
