//! # GRS Test (Gibbons, Ross & Shanken, 1989)
//!
//! This module implements the GRS test, a joint test of whether all pricing errors
//! (alphas) are simultaneously zero. This is one of the most widely used tests for
//! evaluating factor pricing models.
//!
//! ## Null Hypothesis
//!
//! H₀: α₁ = α₂ = ... = αₙ = 0 (all alphas are zero)
//! Hₐ: At least one alpha ≠ 0
//!
//! ## Test Statistic
//!
//! The GRS F-statistic is:
//!
//! F = ((T - N - K)/N) × (α' Σ_ε^{-1} α) / (1 + μ_f' Σ_f^{-1} μ_f)
//!
//! Under H₀, F ~ F(N, T-N-K)
//!
//! where:
//! - T = number of time periods
//! - N = number of assets
//! - K = number of factors
//! - α = vector of alphas
//! - Σ_ε = covariance matrix of residuals
//! - μ_f = mean vector of factors
//! - Σ_f = covariance matrix of factors
//!
//! ## Interpretation
//!
//! - **Low F-statistic, high p-value**: Model adequately prices the assets
//! - **High F-statistic, low p-value**: Reject H₀, model has pricing errors
//!
//! ## Example
//!
//! ```rust
//! use frenchrs::GRSTest;
//! use greeners::CovarianceType;
//! use ndarray::Array2;
//!
//! let t = 120; // 120 observations
//! let n = 10; // 10 assets
//! let k = 3; // 3 factors
//!
//! // Create sample data with random variation
//! let mut rng = 42u64;
//! let mut rand = || {
//!     rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
//!     ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
//! };
//!
//! let returns_excess = Array2::from_shape_fn((t, n), |_| rand() * 0.03);
//! let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
//!
//! let result = GRSTest::fit(
//!     &returns_excess,
//!     &factors,
//!     CovarianceType::HC3,
//!     None,
//! ).unwrap();
//!
//! println!("GRS F-statistic: {:.4}", result.grs_f_stat);
//! println!("p-value: {:.4}", result.p_value);
//! println!("Reject model: {}", result.reject_model(0.05));
//! ```

use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::collections::HashMap;

/// Result of GRS (Gibbons, Ross & Shanken) test
#[derive(Debug, Clone)]
pub struct GRSTest {
    /// GRS F-statistic
    pub grs_f_stat: f64,
    /// P-value for F-test
    pub p_value: f64,
    /// Degrees of freedom (numerator) = N
    pub df1: usize,
    /// Degrees of freedom (denominator) = T - N - K
    pub df2: usize,
    /// Number of effective observations
    pub t_eff: usize,
    /// Number of assets
    pub n_assets: usize,
    /// Number of factors
    pub k_factors: usize,
    /// Quadratic form: α' Σ_ε^{-1} α
    pub alpha_quad_form: f64,
    /// Denominator: 1 + μ_f' Σ_f^{-1} μ_f
    pub denominator: f64,
    /// Estimated alphas for each asset
    pub alphas: HashMap<String, f64>,
    /// Covariance matrix of residuals (N x N)
    pub sigma_eps: Array2<f64>,
    /// Asset names
    pub asset_names: Vec<String>,
}

impl GRSTest {
    /// Perform GRS test for a factor model
    ///
    /// The GRS test is a joint test of whether all alphas are zero. It accounts for
    /// both the cross-sectional correlation of residuals and the sampling variability
    /// of the factor means.
    ///
    /// # Arguments
    ///
    /// * `returns_excess` - T x N matrix of excess returns (time x assets)
    /// * `factors` - T x K matrix of factor returns (time x factors)
    /// * `cov_type` - Type of covariance matrix estimator (HC0-HC4, Newey-West, etc.)
    /// * `asset_names` - Optional vector of asset names
    ///
    /// # Returns
    ///
    /// * `GRSTest` - Structure containing F-statistic, p-value, and diagnostics
    ///
    /// # Null Hypothesis
    ///
    /// H₀: All alphas are jointly zero (α₁ = α₂ = ... = αₙ = 0)
    ///
    /// Under H₀, the F-statistic follows F(N, T-N-K) distribution
    ///
    /// # Important Notes
    ///
    /// - Requires balanced panel (no missing data)
    /// - T must be > N + K + 1 for valid test
    /// - More powerful than testing alphas individually
    /// - Accounts for cross-correlation of residuals
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input matrices have incompatible dimensions
    /// - Insufficient observations (T ≤ N + K + 1)
    /// - Covariance matrices are singular
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
    ) -> Result<Self, GreenersError> {
        let (t, n) = returns_excess.dim();
        let (t_factors, k) = factors.dim();

        // Validate dimensions
        if t != t_factors {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns has {} observations but factors has {}",
                t, t_factors
            )));
        }

        if t <= n + k + 1 {
            return Err(GreenersError::ShapeMismatch(format!(
                "Need T > N + K + 1 for valid GRS test. Got T={}, N={}, K={}",
                t, n, k
            )));
        }

        // Generate asset names if not provided
        let asset_names = asset_names.unwrap_or_else(|| {
            (0..n)
                .map(|i| format!("Asset{}", i + 1))
                .collect::<Vec<String>>()
        });

        if asset_names.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "Expected {} asset names but got {}",
                n,
                asset_names.len()
            )));
        }

        // Prepare X matrix with intercept
        let mut x_with_intercept = Array2::zeros((t, k + 1));
        x_with_intercept.column_mut(0).fill(1.0); // Intercept
        for i in 0..k {
            x_with_intercept
                .column_mut(i + 1)
                .assign(&factors.column(i));
        }

        // Estimate alphas and collect residuals for each asset
        let mut alphas = HashMap::new();
        let mut residuals_list = Vec::new();
        let mut valid_assets = Vec::new();

        for (j, asset_name) in asset_names.iter().enumerate() {
            let y = returns_excess.column(j).to_owned();

            // Run OLS regression
            match OLS::fit(&y, &x_with_intercept, cov_type.clone()) {
                Ok(model) => {
                    let alpha = model.params[0]; // Intercept is alpha
                    alphas.insert(asset_name.clone(), alpha);

                    let resid = model.residuals(&y, &x_with_intercept);
                    residuals_list.push(resid);
                    valid_assets.push(asset_name.clone());
                }
                Err(_) => {
                    // Skip assets that fail to estimate
                    continue;
                }
            }
        }

        if valid_assets.is_empty() {
            return Err(GreenersError::ShapeMismatch(
                "No assets could be successfully estimated".to_string(),
            ));
        }

        let n_eff = valid_assets.len();

        // Construct residuals matrix (T x N_eff)
        let mut residuals = Array2::zeros((t, n_eff));
        for (j, resid) in residuals_list.iter().enumerate() {
            residuals.column_mut(j).assign(resid);
        }

        // Compute covariance matrix of residuals
        let sigma_eps = compute_covariance_matrix(&residuals);

        // Extract alpha vector for valid assets
        let mut alpha_vec = Array1::zeros(n_eff);
        for (j, name) in valid_assets.iter().enumerate() {
            alpha_vec[j] = alphas[name];
        }

        // Compute factor statistics
        let mu_f = compute_mean_vector(factors);
        let sigma_f = compute_covariance_matrix(factors);

        // Compute GRS F-statistic
        let (grs_f_stat, p_value, alpha_quad_form, denominator, df1, df2) =
            compute_grs_statistic(&alpha_vec, &sigma_eps, &mu_f, &sigma_f, t, n_eff, k)?;

        Ok(GRSTest {
            grs_f_stat,
            p_value,
            df1,
            df2,
            t_eff: t,
            n_assets: n_eff,
            k_factors: k,
            alpha_quad_form,
            denominator,
            alphas,
            sigma_eps,
            asset_names: valid_assets,
        })
    }

    /// Returns true if the model is rejected at the given significance level
    pub fn reject_model(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Get alpha (pricing error) for a specific asset
    pub fn get_alpha(&self, asset: &str) -> Option<f64> {
        self.alphas.get(asset).copied()
    }

    /// Classify model fit quality based on p-value
    pub fn model_quality_classification(&self) -> &str {
        if self.p_value >= 0.10 {
            "Good Fit (all alphas jointly zero)"
        } else if self.p_value >= 0.05 {
            "Marginal Fit (borderline rejection)"
        } else if self.p_value >= 0.01 {
            "Poor Fit (rejected at 5%)"
        } else {
            "Very Poor Fit (rejected at 1%)"
        }
    }

    /// Export results to CSV string
    pub fn to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Summary statistics
        csv.push_str("Metric,Value\n");
        csv.push_str(&format!("GRS F-Statistic,{:.6}\n", self.grs_f_stat));
        csv.push_str(&format!("P-value,{:.6}\n", self.p_value));
        csv.push_str(&format!("DF Numerator (N),{}\n", self.df1));
        csv.push_str(&format!("DF Denominator (T-N-K),{}\n", self.df2));
        csv.push_str(&format!("N Assets,{}\n", self.n_assets));
        csv.push_str(&format!("K Factors,{}\n", self.k_factors));
        csv.push_str(&format!("T Observations,{}\n", self.t_eff));
        csv.push_str(&format!("Alpha Quad Form,{:.8}\n", self.alpha_quad_form));
        csv.push_str(&format!("Denominator,{:.8}\n", self.denominator));
        csv.push_str(&format!(
            "Model Quality,{}\n",
            self.model_quality_classification()
        ));

        // Alphas
        csv.push_str("\nAsset,Alpha\n");
        for name in &self.asset_names {
            if let Some(alpha) = self.alphas.get(name) {
                csv.push_str(&format!("{},{:.8}\n", name, alpha));
            }
        }

        csv
    }

    /// Get summary table as formatted string
    pub fn summary_table(&self) -> String {
        let mut table = String::new();

        table.push_str(&format!("{}\n", "=".repeat(70)));
        table.push_str("GRS TEST (Gibbons, Ross & Shanken, 1989)\n");
        table.push_str(&format!("{}\n\n", "=".repeat(70)));

        table.push_str(&format!(
            "GRS F-Statistic:          {:.4}\n",
            self.grs_f_stat
        ));
        table.push_str(&format!("P-value:                  {:.4}\n", self.p_value));
        table.push_str(&format!(
            "Degrees of Freedom:       F({}, {})\n",
            self.df1, self.df2
        ));
        table.push_str(&format!("N Assets:                 {}\n", self.n_assets));
        table.push_str(&format!("K Factors:                {}\n", self.k_factors));
        table.push_str(&format!("T Observations:           {}\n\n", self.t_eff));

        table.push_str(&format!(
            "Model Quality: {}\n\n",
            self.model_quality_classification()
        ));

        table.push_str(&format!("{}\n", "=".repeat(70)));
        table.push_str("INTERPRETATION\n");
        table.push_str(&format!("{}\n\n", "=".repeat(70)));

        table.push_str("H₀: All alphas are jointly zero (α₁ = α₂ = ... = αₙ = 0)\n");
        table.push_str("Hₐ: At least one alpha ≠ 0\n\n");

        if self.reject_model(0.05) {
            table.push_str("✗ Reject H₀ at 5% significance level\n");
            table.push_str("  At least one asset has significant pricing error\n");
            table.push_str("  Model fails to jointly price all assets\n");
        } else {
            table.push_str("✓ Do not reject H₀ at 5% significance level\n");
            table.push_str("  All alphas are jointly not significantly different from zero\n");
            table.push_str("  Model adequately prices the cross-section of assets\n");
        }

        table.push_str(&format!("\n{}\n", "=".repeat(70)));

        table
    }
}

/// Compute mean vector from matrix (column means)
fn compute_mean_vector(data: &Array2<f64>) -> Array1<f64> {
    let (_t, k) = data.dim();
    let mut means = Array1::zeros(k);
    for j in 0..k {
        means[j] = data.column(j).mean().unwrap();
    }
    means
}

/// Compute covariance matrix from data matrix
fn compute_covariance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let (t, n) = data.dim();
    let mut cov_matrix = Array2::zeros((n, n));

    // Compute means
    let means: Vec<f64> = (0..n).map(|j| data.column(j).mean().unwrap()).collect();

    // Compute covariances
    for i in 0..n {
        for j in 0..n {
            let col_i = data.column(i);
            let col_j = data.column(j);

            let cov = col_i
                .iter()
                .zip(col_j.iter())
                .map(|(xi, xj)| (xi - means[i]) * (xj - means[j]))
                .sum::<f64>()
                / (t - 1) as f64;

            cov_matrix[[i, j]] = cov;
        }
    }

    cov_matrix
}

/// Compute GRS F-statistic and related quantities
fn compute_grs_statistic(
    alpha: &Array1<f64>,
    sigma_eps: &Array2<f64>,
    mu_f: &Array1<f64>,
    sigma_f: &Array2<f64>,
    t: usize,
    n: usize,
    k: usize,
) -> Result<(f64, f64, f64, f64, usize, usize), GreenersError> {
    // Invert covariance matrices
    let sigma_eps_inv = match sigma_eps.inv() {
        Ok(inv) => inv,
        Err(_) => {
            // Try with regularization
            let mut regularized = sigma_eps.clone();
            for i in 0..n {
                regularized[[i, i]] += 1e-8;
            }
            regularized
                .inv()
                .map_err(|_| GreenersError::SingularMatrix)?
        }
    };

    let sigma_f_inv = match sigma_f.inv() {
        Ok(inv) => inv,
        Err(_) => {
            // Try with regularization
            let mut regularized = sigma_f.clone();
            for i in 0..k {
                regularized[[i, i]] += 1e-8;
            }
            regularized
                .inv()
                .map_err(|_| GreenersError::SingularMatrix)?
        }
    };

    // Compute quadratic forms
    // Numerator: α' Σ_ε^{-1} α
    let temp = sigma_eps_inv.dot(alpha);
    let alpha_quad_form: f64 = alpha.dot(&temp);

    // Denominator term: μ_f' Σ_f^{-1} μ_f
    let temp_f = sigma_f_inv.dot(mu_f);
    let factor_term: f64 = mu_f.dot(&temp_f);
    let denominator = 1.0 + factor_term;

    if denominator <= 0.0 {
        return Err(GreenersError::ShapeMismatch(
            "GRS denominator is non-positive - numerical instability".to_string(),
        ));
    }

    // GRS F-statistic
    // F = ((T - N - K) / N) * (α' Σ_ε^{-1} α) / (1 + μ_f' Σ_f^{-1} μ_f)
    let df1 = n;
    let df2 = t - n - k;

    let f_stat = ((t - n - k) as f64 / n as f64) * (alpha_quad_form / denominator);

    // Compute p-value from F-distribution
    let p_value = f_cdf_complement(f_stat, df1, df2);

    Ok((f_stat, p_value, alpha_quad_form, denominator, df1, df2))
}

/// Compute complement of F-distribution CDF (1 - F(x))
/// This gives the p-value for the test
fn f_cdf_complement(x: f64, df1: usize, df2: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    // F(d1, d2) can be related to Beta distribution
    // If X ~ F(d1, d2), then Y = d1*X/(d1*X + d2) ~ Beta(d1/2, d2/2)
    // P(X > x) = P(Y > d1*x/(d1*x + d2)) = I_{1-y}(d2/2, d1/2)
    // where I is the regularized incomplete beta function

    let d1 = df1 as f64;
    let d2 = df2 as f64;
    let y = d1 * x / (d1 * x + d2);

    // P(F > x) = I_{1-y}(d2/2, d1/2) = I_{1-y}(b, a) where a = d1/2, b = d2/2
    // This equals the regularized upper incomplete beta function
    incomplete_beta_complement(d2 / 2.0, d1 / 2.0, y)
}

/// Regularized incomplete beta function complement: I_{1-x}(a, b)
fn incomplete_beta_complement(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }

    // For better numerical stability, use the complement relation
    // I_x(a, b) + I_{1-x}(b, a) = 1
    // So I_{1-x}(a, b) = 1 - I_x(a, b) if we compute I_x(a,b)
    // But we want I_{1-x}(a, b) directly, which equals 1 - I_x(a, b)

    1.0 - incomplete_beta(a, b, x)
}

/// Regularized incomplete beta function I_x(a, b)
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use continued fraction representation
    // This is a simplified approximation
    let ln_beta_ab = ln_beta(a, b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta_ab).exp();

    // Continued fraction for incomplete beta
    let cf = beta_continued_fraction(a, b, x);

    front * cf / a
}

/// Natural logarithm of beta function: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Continued fraction for incomplete beta function
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let epsilon = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    h
}

/// Natural logarithm of gamma function (reuse from hj_distance)
fn ln_gamma(x: f64) -> f64 {
    // Stirling's approximation for large x
    if x > 10.0 {
        return (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x)
            - 1.0 / (360.0 * x.powi(3));
    }

    // Lanczos approximation
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;

    for &coef in coefficients.iter() {
        y += 1.0;
        ser += coef / y;
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use greeners::CovarianceType;
    use ndarray::Array2;

    #[test]
    fn test_grs_basic() {
        let t = 120;
        let n = 10;
        let k = 3;

        let mut rng = 12345u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = GRSTest::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        // Check basic properties
        assert!(result.grs_f_stat >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.n_assets, n);
        assert_eq!(result.k_factors, k);
        assert_eq!(result.t_eff, t);
        assert_eq!(result.df1, n);
        assert_eq!(result.df2, t - n - k);
    }

    #[test]
    fn test_grs_well_specified_model() {
        // Create data where returns are well-explained by factors (small alphas)
        let t = 150;
        let n = 8;
        let k = 2;

        let mut rng = 42u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);

        // Generate returns as pure factor exposure (no alpha, small noise)
        let mut returns = Array2::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                let beta1 = 0.8 + j as f64 * 0.1;
                let beta2 = 0.5 + j as f64 * 0.05;
                returns[[i, j]] =
                    factors[[i, 0]] * beta1 + factors[[i, 1]] * beta2 + rand() * 0.001;
            }
        }

        let result = GRSTest::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        // Should not reject well-specified model
        assert!(
            result.p_value > 0.05,
            "Should not reject well-specified model: p-value = {}",
            result.p_value
        );
    }

    #[test]
    fn test_grs_asset_names() {
        let t = 100;
        let n = 5;
        let k = 2;

        let mut rng = 9999u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let asset_names = vec![
            "Stock A".to_string(),
            "Stock B".to_string(),
            "Stock C".to_string(),
            "Stock D".to_string(),
            "Stock E".to_string(),
        ];

        let result = GRSTest::fit(
            &returns,
            &factors,
            CovarianceType::HC3,
            Some(asset_names.clone()),
        )
        .unwrap();

        // Check that all assets have alphas
        for name in &asset_names {
            assert!(result.get_alpha(name).is_some());
        }
    }

    #[test]
    fn test_reject_model() {
        let t = 120;
        let n = 12;
        let k = 2;

        let mut rng = 7777u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = GRSTest::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        // Test reject_model method
        let reject_at_10pct = result.reject_model(0.10);
        let reject_at_5pct = result.reject_model(0.05);
        let reject_at_1pct = result.reject_model(0.01);

        // If rejected at 1%, should also be rejected at 5% and 10%
        if reject_at_1pct {
            assert!(reject_at_5pct);
            assert!(reject_at_10pct);
        }
    }

    #[test]
    fn test_csv_export() {
        let t = 80;
        let n = 6;
        let k = 2;

        let mut rng = 5555u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = GRSTest::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let csv = result.to_csv_string();
        assert!(csv.contains("GRS F-Statistic"));
        assert!(csv.contains("P-value"));
        assert!(csv.contains("Asset,Alpha"));
    }

    #[test]
    fn test_summary_table() {
        let t = 100;
        let n = 8;
        let k = 3;

        let mut rng = 3333u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = GRSTest::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let summary = result.summary_table();
        assert!(summary.contains("GRS TEST"));
        assert!(summary.contains("GRS F-Statistic"));
        assert!(summary.contains("INTERPRETATION"));
    }
}
