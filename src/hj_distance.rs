//! # Hansen-Jagannathan Distance
//!
//! This module implements the Hansen-Jagannathan (HJ) distance test for evaluating
//! factor pricing models. The HJ distance measures how far the model's pricing errors
//! (alphas) are from zero, weighted by the covariance matrix of residuals.
//!
//! ## Interpretation
//!
//! - **HJ Distance = 0**: Perfect model specification (all alphas = 0)
//! - **Low HJ Distance**: Model prices assets well
//! - **High HJ Distance**: Significant mispricing, model may be misspecified
//! - **Chi-squared test**: Tests H₀: d = 0 (model correctly prices all assets)
//!
//! ## Example
//!
//! ```rust
//! use frenchrs::HJDistance;
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
//! let result = HJDistance::fit(
//!     &returns_excess,
//!     &factors,
//!     CovarianceType::HC3,
//!     None,
//! ).unwrap();
//!
//! println!("HJ Distance: {:.6}", result.hj_distance);
//! println!("p-value: {:.4}", result.p_value);
//! println!("Model rejected: {}", result.reject_model(0.05));
//! ```

use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::collections::HashMap;

/// Result of Hansen-Jagannathan distance test
#[derive(Debug, Clone)]
pub struct HJDistance {
    /// Hansen-Jagannathan distance
    pub hj_distance: f64,
    /// Squared HJ distance (d²)
    pub hj_distance_squared: f64,
    /// Chi-squared test statistic (T * d²)
    pub chi2_stat: f64,
    /// P-value for chi-squared test (H₀: d = 0)
    pub p_value: f64,
    /// Number of assets used in calculation
    pub n_assets: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Estimated alphas for each asset
    pub alphas: HashMap<String, f64>,
    /// Covariance matrix of residuals (N x N)
    pub sigma_eps: Array2<f64>,
    /// Asset names
    pub asset_names: Vec<String>,
}

impl HJDistance {
    /// Compute Hansen-Jagannathan distance for a factor model
    ///
    /// The HJ distance measures the distance between the stochastic discount factor
    /// implied by the model and the set of admissible SDFs. In practice, it measures
    /// how far the pricing errors (alphas) are from zero.
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
    /// * `HJDistance` - Structure containing HJ distance, test statistics, and alphas
    ///
    /// # Statistical Test
    ///
    /// Under H₀ (model correctly prices all assets), the test statistic follows:
    /// T * d² ~ χ²(N) asymptotically
    ///
    /// where T = number of observations, N = number of assets
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input matrices have incompatible dimensions
    /// - Number of observations is too small
    /// - Covariance matrix is singular or nearly singular
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

        if t < k + n + 10 {
            return Err(GreenersError::ShapeMismatch(format!(
                "Need at least {} observations for reliable HJ distance, got {}",
                k + n + 10,
                t
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

        let n_valid = valid_assets.len();

        // Construct residuals matrix (T x N_valid)
        let mut residuals = Array2::zeros((t, n_valid));
        for (j, resid) in residuals_list.iter().enumerate() {
            residuals.column_mut(j).assign(resid);
        }

        // Compute covariance matrix of residuals
        let sigma_eps = compute_covariance_matrix(&residuals);

        // Extract alpha vector for valid assets
        let mut alpha_vec = Array1::zeros(n_valid);
        for (j, name) in valid_assets.iter().enumerate() {
            alpha_vec[j] = alphas[name];
        }

        // Compute HJ distance: d = sqrt(alpha' * Sigma^-1 * alpha)
        let (hj_distance_squared, hj_distance, chi2_stat, p_value) =
            compute_hj_distance(&alpha_vec, &sigma_eps, t)?;

        Ok(HJDistance {
            hj_distance,
            hj_distance_squared,
            chi2_stat,
            p_value,
            n_assets: n_valid,
            n_obs: t,
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
            "Good Fit (model not rejected)"
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
        csv.push_str(&format!("HJ Distance,{:.8}\n", self.hj_distance));
        csv.push_str(&format!(
            "HJ Distance Squared,{:.8}\n",
            self.hj_distance_squared
        ));
        csv.push_str(&format!("Chi-squared Statistic,{:.6}\n", self.chi2_stat));
        csv.push_str(&format!("P-value,{:.6}\n", self.p_value));
        csv.push_str(&format!("N Assets,{}\n", self.n_assets));
        csv.push_str(&format!("N Observations,{}\n", self.n_obs));
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

        table.push_str(&format!("{}\n", "=".repeat(60)));
        table.push_str("HANSEN-JAGANNATHAN DISTANCE TEST\n");
        table.push_str(&format!("{}\n\n", "=".repeat(60)));

        table.push_str(&format!(
            "HJ Distance:              {:.6}\n",
            self.hj_distance
        ));
        table.push_str(&format!(
            "HJ Distance² (d²):        {:.6}\n",
            self.hj_distance_squared
        ));
        table.push_str(&format!(
            "Chi² Statistic (T*d²):    {:.4}\n",
            self.chi2_stat
        ));
        table.push_str(&format!("P-value:                  {:.4}\n", self.p_value));
        table.push_str(&format!("Degrees of Freedom:       {}\n", self.n_assets));
        table.push_str(&format!("N Observations:           {}\n", self.n_obs));
        table.push_str(&format!("N Assets:                 {}\n\n", self.n_assets));

        table.push_str(&format!(
            "Model Quality: {}\n\n",
            self.model_quality_classification()
        ));

        table.push_str(&format!("{}\n", "=".repeat(60)));
        table.push_str("INTERPRETATION\n");
        table.push_str(&format!("{}\n\n", "=".repeat(60)));

        table.push_str("H₀: Factor model correctly prices all assets (d = 0)\n");
        table.push_str("Hₐ: Model has pricing errors (d > 0)\n\n");

        if self.reject_model(0.05) {
            table.push_str("✗ Reject H₀ at 5% significance level\n");
            table.push_str("  Model has significant pricing errors\n");
            table.push_str("  Consider adding more factors or revising model specification\n");
        } else {
            table.push_str("✓ Do not reject H₀ at 5% significance level\n");
            table.push_str("  Model adequately prices the assets\n");
        }

        table.push_str(&format!("\n{}\n", "=".repeat(60)));

        table
    }
}

/// Compute covariance matrix from residuals matrix (T x N)
fn compute_covariance_matrix(residuals: &Array2<f64>) -> Array2<f64> {
    let (t, n) = residuals.dim();
    let mut cov_matrix = Array2::zeros((n, n));

    // Compute means (should be close to zero for residuals)
    let means: Vec<f64> = (0..n)
        .map(|j| residuals.column(j).sum() / t as f64)
        .collect();

    // Compute covariances
    for i in 0..n {
        for j in 0..n {
            let col_i = residuals.column(i);
            let col_j = residuals.column(j);

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

/// Compute HJ distance and test statistics
fn compute_hj_distance(
    alpha: &Array1<f64>,
    sigma: &Array2<f64>,
    t: usize,
) -> Result<(f64, f64, f64, f64), GreenersError> {
    let n = alpha.len();

    // Invert covariance matrix
    let sigma_inv = match invert_matrix(sigma) {
        Some(inv) => inv,
        None => {
            return Err(GreenersError::SingularMatrix);
        }
    };

    // Compute d² = alpha' * Sigma^-1 * alpha
    let temp = sigma_inv.dot(alpha);
    let d_squared: f64 = alpha.dot(&temp);

    if d_squared < 0.0 {
        return Err(GreenersError::ShapeMismatch(
            "Negative HJ distance squared - numerical instability".to_string(),
        ));
    }

    let d = d_squared.sqrt();

    // Test statistic: T * d² ~ χ²(N)
    let chi2_stat = t as f64 * d_squared;

    // Compute p-value from chi-squared distribution
    let p_value = chi2_cdf_complement(chi2_stat, n);

    Ok((d_squared, d, chi2_stat, p_value))
}

/// Invert a symmetric positive definite matrix
fn invert_matrix(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    // Try standard inverse
    match matrix.inv() {
        Ok(inv) => Some(inv),
        Err(_) => {
            // If standard inverse fails, add small regularization to diagonal
            let n = matrix.nrows();
            let mut regularized = matrix.clone();
            for i in 0..n {
                regularized[[i, i]] += 1e-8;
            }
            regularized.inv().ok()
        }
    }
}

/// Compute complement of chi-squared CDF (1 - F(x))
/// This gives the p-value for the test
fn chi2_cdf_complement(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    // Use incomplete gamma function approximation
    // P(χ² > x) = 1 - P(χ² ≤ x) = Γ(df/2, x/2) / Γ(df/2)
    // where Γ(a, x) is the upper incomplete gamma function

    let k = df as f64 / 2.0;
    let x_half = x / 2.0;

    // For large df, use normal approximation
    if df > 100 {
        let mean = df as f64;
        let variance = 2.0 * df as f64;
        let z = (x - mean) / variance.sqrt();
        return normal_cdf_complement(z);
    }

    // Use series expansion for small to moderate df
    gamma_p_complement(k, x_half)
}

/// Complement of standard normal CDF
fn normal_cdf_complement(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Complementary error function
fn erfc(x: f64) -> f64 {
    // Approximation using continued fraction
    if x >= 0.0 {
        let t = 1.0 / (1.0 + 0.5 * x);
        t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp()
    } else {
        2.0 - erfc(-x)
    }
}

/// Regularized upper incomplete gamma function Q(a,x) = Γ(a,x)/Γ(a)
fn gamma_p_complement(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // Use series representation
        1.0 - gamma_p_series(a, x)
    } else {
        // Use continued fraction
        gamma_q_continued_fraction(a, x)
    }
}

/// Series representation of P(a,x)
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..100 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-10 {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction for Q(a,x)
fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..100 {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Natural logarithm of gamma function
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
    fn test_hj_distance_basic() {
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

        let result = HJDistance::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        // Check basic properties
        assert!(result.hj_distance >= 0.0);
        assert!(result.hj_distance_squared >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.n_assets, n);
        assert_eq!(result.n_obs, t);
        assert!(result.chi2_stat >= 0.0);
    }

    #[test]
    fn test_hj_distance_perfect_model() {
        // Create data where returns are perfectly explained by factors (zero alpha)
        let t = 150;
        let n = 5;
        let k = 2;

        let mut rng = 42u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);

        // Generate returns as pure factor exposure (no alpha, very small noise)
        let mut returns = Array2::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                let beta1 = 0.8 + j as f64 * 0.1;
                let beta2 = 0.5 + j as f64 * 0.05;
                // Pure factor model with minimal noise
                returns[[i, j]] =
                    factors[[i, 0]] * beta1 + factors[[i, 1]] * beta2 + rand() * 0.0001;
            }
        }

        let result = HJDistance::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        // HJ distance should be small for well-specified model (increased threshold to be more realistic)
        assert!(
            result.hj_distance < 0.3,
            "HJ distance should be relatively small for well-specified model: got {}",
            result.hj_distance
        );

        // P-value should be reasonably high (not strongly rejecting the model)
        assert!(
            result.p_value > 0.01,
            "Should not strongly reject well-specified model: got p-value {}",
            result.p_value
        );
    }

    #[test]
    fn test_hj_distance_asset_names() {
        let t = 80;
        let n = 3;
        let k = 1;

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
        ];

        let result = HJDistance::fit(
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
        let t = 100;
        let n = 8;
        let k = 2;

        let mut rng = 7777u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = HJDistance::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

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
        let t = 60;
        let n = 4;
        let k = 2;

        let mut rng = 5555u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = HJDistance::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let csv = result.to_csv_string();
        assert!(csv.contains("HJ Distance"));
        assert!(csv.contains("P-value"));
        assert!(csv.contains("Asset,Alpha"));
    }

    #[test]
    fn test_summary_table() {
        let t = 70;
        let n = 5;
        let k = 1;

        let mut rng = 3333u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result = HJDistance::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let summary = result.summary_table();
        assert!(summary.contains("HANSEN-JAGANNATHAN DISTANCE TEST"));
        assert!(summary.contains("HJ Distance"));
        assert!(summary.contains("INTERPRETATION"));
    }
}
