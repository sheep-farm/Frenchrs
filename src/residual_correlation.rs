//! # Residual Correlation Analysis
//!
//! This module provides functionality to compute the correlation matrix of residuals
//! from factor model regressions. This is useful for:
//! - Detecting common unmodeled factors
//! - Assessing model specification quality
//! - Identifying asset clusters with correlated idiosyncratic risk
//!
//! ## Example
//!
//! ```rust
//! use frenchrs::ResidualCorrelation;
//! use greeners::CovarianceType;
//! use ndarray::Array2;
//!
//! let t = 60; // 60 observations
//! let n = 3; // 3 assets
//! let k = 2; // 2 factors
//!
//! // Create sample data with random variation
//! let mut rng = 12345u64;
//! let mut rand = || {
//!     rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
//!     ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
//! };
//!
//! let returns_excess = Array2::from_shape_fn((t, n), |_| rand() * 0.03);
//! let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
//!
//! let asset_names = vec!["Stock A".to_string(), "Stock B".to_string(), "Stock C".to_string()];
//!
//! let result = ResidualCorrelation::fit(
//!     &returns_excess,
//!     &factors,
//!     CovarianceType::HC3,
//!     Some(asset_names),
//! ).unwrap();
//!
//! let summary = result.summary_stats();
//! println!("Avg off-diagonal correlation: {:.4}", summary.avg_off_diag_corr);
//! println!("Max eigenvalue: {:.4}", summary.max_eigenvalue);
//! ```

use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};

/// Summary statistics for residual correlation matrix
#[derive(Debug, Clone)]
pub struct ResidualCorrSummary {
    /// Average off-diagonal correlation
    pub avg_off_diag_corr: f64,
    /// Minimum off-diagonal correlation
    pub min_off_diag_corr: f64,
    /// Maximum off-diagonal correlation
    pub max_off_diag_corr: f64,
    /// Maximum eigenvalue of correlation matrix
    pub max_eigenvalue: f64,
    /// Number of assets analyzed
    pub n_assets: usize,
}

impl ResidualCorrSummary {
    /// Returns true if average off-diagonal correlation is low (< 0.1)
    pub fn low_correlation(&self) -> bool {
        self.avg_off_diag_corr < 0.1
    }

    /// Returns true if average off-diagonal correlation is moderate (0.1-0.3)
    pub fn moderate_correlation(&self) -> bool {
        self.avg_off_diag_corr >= 0.1 && self.avg_off_diag_corr < 0.3
    }

    /// Returns true if average off-diagonal correlation is high (>= 0.3)
    pub fn high_correlation(&self) -> bool {
        self.avg_off_diag_corr >= 0.3
    }

    /// Classification of residual correlation level
    pub fn correlation_classification(&self) -> &str {
        if self.low_correlation() {
            "Low Residual Correlation (Well-Specified Model)"
        } else if self.moderate_correlation() {
            "Moderate Residual Correlation"
        } else {
            "High Residual Correlation (Missing Common Factors)"
        }
    }
}

/// Result of residual correlation analysis
#[derive(Debug, Clone)]
pub struct ResidualCorrelation {
    /// Residuals matrix (T x N) - each column is an asset's residuals
    pub residuals: Array2<f64>,
    /// Correlation matrix of residuals (N x N)
    pub correlation_matrix: Array2<f64>,
    /// Asset names (length N)
    pub asset_names: Vec<String>,
    /// Summary statistics
    summary: ResidualCorrSummary,
}

impl ResidualCorrelation {
    /// Compute residual correlation matrix from factor model regressions
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
    /// * `ResidualCorrelation` - Structure containing residuals, correlation matrix, and summary statistics
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input matrices have incompatible dimensions
    /// - Number of observations is too small
    /// - Asset names length doesn't match number of assets
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
    ) -> Result<Self, GreenersError> {
        let (t, n) = returns_excess.dim();
        let (t_factors, _k) = factors.dim();

        // Validate dimensions
        if t != t_factors {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns has {} observations but factors has {}",
                t, t_factors
            )));
        }

        if t < 30 {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 30 observations for reliable correlation analysis".to_string(),
            ));
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

        // Collect residuals for each asset
        let mut residuals_vec = Vec::with_capacity(n);

        // Prepare X matrix with intercept
        let k = factors.ncols();
        let mut x_with_intercept = Array2::zeros((t, k + 1));
        x_with_intercept.column_mut(0).fill(1.0); // Intercept
        for i in 0..k {
            x_with_intercept
                .column_mut(i + 1)
                .assign(&factors.column(i));
        }

        for j in 0..n {
            let y = returns_excess.column(j).to_owned();

            // Run OLS regression
            let model = OLS::fit(&y, &x_with_intercept, cov_type.clone())?;

            // Store residuals
            let resid = model.residuals(&y, &x_with_intercept);
            residuals_vec.push(resid);
        }

        // Construct residuals matrix (T x N)
        let mut residuals = Array2::zeros((t, n));
        for (j, resid) in residuals_vec.iter().enumerate() {
            residuals.column_mut(j).assign(resid);
        }

        // Compute correlation matrix
        let correlation_matrix = compute_correlation_matrix(&residuals);

        // Compute summary statistics
        let summary = compute_summary_stats(&correlation_matrix);

        Ok(ResidualCorrelation {
            residuals,
            correlation_matrix,
            asset_names,
            summary,
        })
    }

    /// Get summary statistics
    pub fn summary_stats(&self) -> &ResidualCorrSummary {
        &self.summary
    }

    /// Get residual for a specific asset by name
    pub fn get_residuals(&self, asset: &str) -> Option<Array1<f64>> {
        self.asset_names
            .iter()
            .position(|name| name == asset)
            .map(|idx| self.residuals.column(idx).to_owned())
    }

    /// Get correlation between two assets
    pub fn get_correlation(&self, asset1: &str, asset2: &str) -> Option<f64> {
        let idx1 = self.asset_names.iter().position(|name| name == asset1)?;
        let idx2 = self.asset_names.iter().position(|name| name == asset2)?;
        Some(self.correlation_matrix[[idx1, idx2]])
    }

    /// Export correlation matrix to CSV string
    pub fn correlation_to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Asset");
        for name in &self.asset_names {
            csv.push(',');
            csv.push_str(name);
        }
        csv.push('\n');

        // Data rows
        for (i, row_name) in self.asset_names.iter().enumerate() {
            csv.push_str(row_name);
            for j in 0..self.asset_names.len() {
                csv.push(',');
                csv.push_str(&format!("{:.6}", self.correlation_matrix[[i, j]]));
            }
            csv.push('\n');
        }

        csv
    }

    /// Export residuals to CSV string
    pub fn residuals_to_csv_string(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Period");
        for name in &self.asset_names {
            csv.push(',');
            csv.push_str(name);
        }
        csv.push('\n');

        // Data rows
        let (t, n) = self.residuals.dim();
        for i in 0..t {
            csv.push_str(&format!("{}", i + 1));
            for j in 0..n {
                csv.push(',');
                csv.push_str(&format!("{:.6}", self.residuals[[i, j]]));
            }
            csv.push('\n');
        }

        csv
    }

    /// Get assets with highest correlation to a given asset
    pub fn most_correlated_assets(&self, asset: &str, top_n: usize) -> Vec<(String, f64)> {
        let idx = match self.asset_names.iter().position(|name| name == asset) {
            Some(i) => i,
            None => return Vec::new(),
        };

        let mut correlations: Vec<(String, f64)> = self
            .asset_names
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(i, name)| (name.clone(), self.correlation_matrix[[idx, i]]))
            .collect();

        // Sort by absolute correlation (descending)
        correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        correlations.into_iter().take(top_n).collect()
    }
}

/// Compute correlation matrix from a T x N matrix
fn compute_correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
    let (_t, n) = data.dim();
    let mut corr_matrix = Array2::zeros((n, n));

    // Compute means
    let means: Vec<f64> = (0..n).map(|j| data.column(j).mean().unwrap()).collect();

    // Compute standard deviations
    let stds: Vec<f64> = (0..n)
        .map(|j| {
            let col = data.column(j);
            let mean = means[j];
            let variance = col.mapv(|x| (x - mean).powi(2)).sum() / (col.len() - 1) as f64;
            variance.sqrt()
        })
        .collect();

    // Compute correlations
    for i in 0..n {
        for j in 0..n {
            if i == j {
                corr_matrix[[i, j]] = 1.0;
            } else {
                let col_i = data.column(i);
                let col_j = data.column(j);
                let mean_i = means[i];
                let mean_j = means[j];

                let cov = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(xi, xj)| (xi - mean_i) * (xj - mean_j))
                    .sum::<f64>()
                    / (col_i.len() - 1) as f64;

                let corr = cov / (stds[i] * stds[j]);
                corr_matrix[[i, j]] = corr;
            }
        }
    }

    corr_matrix
}

/// Compute summary statistics from correlation matrix
fn compute_summary_stats(corr_matrix: &Array2<f64>) -> ResidualCorrSummary {
    let n = corr_matrix.nrows();

    // Collect off-diagonal elements
    let mut off_diag_values = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag_values.push(corr_matrix[[i, j]]);
            }
        }
    }

    // Compute statistics
    let avg_off_diag_corr = if off_diag_values.is_empty() {
        0.0
    } else {
        off_diag_values.iter().sum::<f64>() / off_diag_values.len() as f64
    };

    let min_off_diag_corr = off_diag_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let max_off_diag_corr = off_diag_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute eigenvalues
    let max_eigenvalue = compute_max_eigenvalue(corr_matrix);

    ResidualCorrSummary {
        avg_off_diag_corr,
        min_off_diag_corr,
        max_off_diag_corr,
        max_eigenvalue,
        n_assets: n,
    }
}

/// Compute maximum eigenvalue of a symmetric matrix using power iteration
fn compute_max_eigenvalue(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut v = Array1::from_vec(vec![1.0; n]);
    let norm: f64 = v.dot(&v);
    v /= norm.sqrt();

    let max_iterations = 1000;
    let tolerance = 1e-10;

    for _ in 0..max_iterations {
        let v_new = matrix.dot(&v);
        let eigenvalue = v_new.dot(&v);
        let v_new_norm = (v_new.dot(&v_new)).sqrt();
        let v_new_normalized = &v_new / v_new_norm;

        // Check convergence
        let diff = (&v_new_normalized - &v).mapv(|x| x.abs()).sum();
        if diff < tolerance {
            return eigenvalue;
        }

        v = v_new_normalized;
    }

    // Return final estimate
    let v_new = matrix.dot(&v);
    v_new.dot(&v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use greeners::CovarianceType;
    use ndarray::Array2;

    #[test]
    fn test_residual_correlation_basic() {
        // Create synthetic data
        let t = 100;
        let n = 3;
        let k = 2;

        let mut rng = 12345u64;
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

        let result = ResidualCorrelation::fit(
            &returns,
            &factors,
            CovarianceType::HC3,
            Some(asset_names.clone()),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.residuals.dim(), (t, n));
        assert_eq!(result.correlation_matrix.dim(), (n, n));
        assert_eq!(result.asset_names.len(), n);

        // Check diagonal is 1.0
        for i in 0..n {
            assert!((result.correlation_matrix[[i, i]] - 1.0).abs() < 1e-10);
        }

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (result.correlation_matrix[[i, j]] - result.correlation_matrix[[j, i]]).abs()
                        < 1e-10
                );
            }
        }

        // Check summary stats
        let summary = result.summary_stats();
        assert_eq!(summary.n_assets, n);
        assert!(summary.avg_off_diag_corr.abs() <= 1.0);
        assert!(summary.min_off_diag_corr >= -1.0);
        assert!(summary.max_off_diag_corr <= 1.0);
        assert!(summary.max_eigenvalue > 0.0);
    }

    #[test]
    fn test_residual_correlation_perfect() {
        // Create data with perfect factor exposure (zero residual correlation expected)
        let t = 100;
        let n = 2;
        let k = 1;

        let mut rng = 42u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);

        // Returns are pure factor exposure + independent noise
        let mut returns = Array2::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                let beta = 1.0 + j as f64 * 0.5;
                returns[[i, j]] = factors[[i, 0]] * beta + rand() * 0.001;
            }
        }

        let result =
            ResidualCorrelation::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let summary = result.summary_stats();

        // With independent residuals, correlation should be low
        assert!(summary.avg_off_diag_corr.abs() < 0.3);
    }

    #[test]
    fn test_csv_export() {
        let t = 50;
        let n = 2;
        let k = 1;

        let mut rng = 9999u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let asset_names = vec!["Asset1".to_string(), "Asset2".to_string()];

        let result = ResidualCorrelation::fit(
            &returns,
            &factors,
            CovarianceType::HC3,
            Some(asset_names.clone()),
        )
        .unwrap();

        let csv = result.correlation_to_csv_string();
        assert!(csv.contains("Asset1"));
        assert!(csv.contains("Asset2"));
        assert!(csv.contains("Asset,Asset1,Asset2"));

        let resid_csv = result.residuals_to_csv_string();
        assert!(resid_csv.contains("Period"));
        assert!(resid_csv.contains("Asset1"));
        assert!(resid_csv.contains("Asset2"));
    }

    #[test]
    fn test_get_correlation() {
        let t = 60;
        let n = 3;
        let k = 1;

        let mut rng = 7777u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let asset_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = ResidualCorrelation::fit(
            &returns,
            &factors,
            CovarianceType::HC3,
            Some(asset_names.clone()),
        )
        .unwrap();

        let corr_ab = result.get_correlation("A", "B").unwrap();
        assert!((-1.0..=1.0).contains(&corr_ab));

        let corr_aa = result.get_correlation("A", "A").unwrap();
        assert!((corr_aa - 1.0).abs() < 1e-10);

        let corr_invalid = result.get_correlation("A", "NonExistent");
        assert!(corr_invalid.is_none());
    }

    #[test]
    fn test_most_correlated_assets() {
        let t = 80;
        let n = 4;
        let k = 1;

        let mut rng = 5555u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let asset_names = vec![
            "Asset1".to_string(),
            "Asset2".to_string(),
            "Asset3".to_string(),
            "Asset4".to_string(),
        ];

        let result = ResidualCorrelation::fit(
            &returns,
            &factors,
            CovarianceType::HC3,
            Some(asset_names.clone()),
        )
        .unwrap();

        let top_correlated = result.most_correlated_assets("Asset1", 2);
        assert_eq!(top_correlated.len(), 2);

        // Check that Asset1 is not in the results
        for (name, _) in &top_correlated {
            assert_ne!(name, "Asset1");
        }
    }
}
