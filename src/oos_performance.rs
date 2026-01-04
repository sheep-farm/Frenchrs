use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

/// Result of out-of-sample performance analysis for a single asset
#[derive(Debug, Clone)]
pub struct OOSPerformanceAsset {
    /// Asset name
    pub asset: String,

    /// In-sample R²
    pub r2_in: f64,

    /// In-sample RMSE (Root Mean Squared Error)
    pub rmse_in: f64,

    /// In-sample MAE (Mean Absolute Error)
    pub mae_in: f64,

    /// Out-of-sample R²
    pub r2_out: f64,

    /// Out-of-sample RMSE
    pub rmse_out: f64,

    /// Out-of-sample MAE
    pub mae_out: f64,

    /// Campbell-Thompson out-of-sample R²
    ///
    /// Compares model predictions to historical mean benchmark.
    /// Positive values indicate the model beats the naive forecast.
    pub r2_oos_ct: f64,

    /// Number of in-sample observations
    pub nobs_in: usize,

    /// Number of out-of-sample observations
    pub nobs_out: usize,

    /// Split index used (in time series index, not date)
    pub split_index: usize,
}

impl OOSPerformanceAsset {
    /// Returns true if the model has positive out-of-sample predictive power
    /// (beats the historical mean benchmark)
    pub fn beats_benchmark(&self) -> bool {
        self.r2_oos_ct > 0.0
    }

    /// Returns true if out-of-sample performance is better than in-sample
    /// (no overfitting)
    pub fn no_overfitting(&self) -> bool {
        self.r2_out >= self.r2_in * 0.9 // Allow 10% degradation
    }

    /// Classification of out-of-sample predictive power based on Campbell-Thompson R²
    pub fn predictive_power_classification(&self) -> &str {
        if self.r2_oos_ct > 0.05 {
            "Strong Predictive Power"
        } else if self.r2_oos_ct > 0.01 {
            "Moderate Predictive Power"
        } else if self.r2_oos_ct > 0.0 {
            "Weak Predictive Power"
        } else {
            "No Predictive Power"
        }
    }
}

/// Out-of-sample performance analysis for multiple assets
pub struct OOSPerformance {
    /// Results by asset
    pub results: HashMap<String, OOSPerformanceAsset>,
}

impl OOSPerformance {
    /// Performs out-of-sample performance analysis for multiple assets
    ///
    /// # Arguments
    /// * `returns_excess` - Excess returns matrix (T x N)
    /// * `factors` - Factor returns matrix (T x K)
    /// * `split_ratio` - Fraction of data to use for in-sample (e.g., 0.7 for 70/30 split)
    /// * `cov_type` - Covariance type for OLS estimation
    /// * `asset_names` - Optional names for assets
    ///
    /// # Returns
    /// `OOSPerformance` with results for each asset
    ///
    /// # Example
    /// ```
    /// use frenchrs::OOSPerformance;
    /// use greeners::CovarianceType;
    /// use ndarray::Array2;
    ///
    /// let returns = Array2::from_shape_vec((100, 3), (0..300).map(|i| (i as f64) * 0.001).collect()).unwrap();
    /// let factors = Array2::from_shape_vec((100, 2), (0..200).map(|i| (i as f64) * 0.0005).collect()).unwrap();
    ///
    /// let result = OOSPerformance::fit(
    ///     &returns,
    ///     &factors,
    ///     0.7,
    ///     CovarianceType::NonRobust,
    ///     Some(vec!["Asset1".to_string(), "Asset2".to_string(), "Asset3".to_string()])
    /// ).unwrap();
    /// ```
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        split_ratio: f64,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
    ) -> Result<Self, GreenersError> {
        let (t, n) = returns_excess.dim();
        let (t_factors, k) = factors.dim();

        if t != t_factors {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns has {} observations but factors has {}",
                t, t_factors
            )));
        }

        if split_ratio <= 0.0 || split_ratio >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "split_ratio must be between 0 and 1".to_string(),
            ));
        }

        let split_index = (t as f64 * split_ratio) as usize;
        if split_index < k + 3 || t - split_index < 3 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for split: in-sample has {} obs, out-of-sample has {} obs",
                split_index,
                t - split_index
            )));
        }

        let names =
            asset_names.unwrap_or_else(|| (0..n).map(|i| format!("Asset{}", i + 1)).collect());

        if names.len() != n {
            return Err(GreenersError::InvalidOperation(format!(
                "Number of asset names ({}) does not match number of assets ({})",
                names.len(),
                n
            )));
        }

        let mut results = HashMap::new();

        for (i, asset_name) in names.iter().enumerate() {
            let asset_returns = returns_excess.column(i);

            // Split data
            let y_in = asset_returns.slice(s![..split_index]).to_owned();
            let y_out = asset_returns.slice(s![split_index..]).to_owned();
            let x_in = factors.slice(s![..split_index, ..]).to_owned();
            let x_out = factors.slice(s![split_index.., ..]).to_owned();

            // In-sample estimation
            let (r2_in, rmse_in, mae_in, beta) = if y_in.len() > k + 2 {
                match Self::estimate_in_sample(&y_in, &x_in, cov_type.clone()) {
                    Ok(metrics) => metrics,
                    Err(_) => continue, // Skip asset if estimation fails
                }
            } else {
                continue;
            };

            // Out-of-sample prediction
            let (r2_out, rmse_out, mae_out, r2_oos_ct) = if y_out.len() > 2 {
                Self::evaluate_out_of_sample(&y_out, &x_out, &beta, y_in.mean().unwrap_or(0.0))
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            };

            results.insert(
                asset_name.clone(),
                OOSPerformanceAsset {
                    asset: asset_name.clone(),
                    r2_in,
                    rmse_in,
                    mae_in,
                    r2_out,
                    rmse_out,
                    mae_out,
                    r2_oos_ct,
                    nobs_in: y_in.len(),
                    nobs_out: y_out.len(),
                    split_index,
                },
            );
        }

        Ok(OOSPerformance { results })
    }

    /// Estimates in-sample model and returns metrics
    fn estimate_in_sample(
        y: &Array1<f64>,
        x: &Array2<f64>,
        cov_type: CovarianceType,
    ) -> Result<(f64, f64, f64, Array1<f64>), GreenersError> {
        let ols = OLS::fit(y, x, cov_type)?;
        let fitted = ols.fitted_values(x);
        let residuals = y - &fitted;

        let t = y.len() as f64;
        let k = x.ncols() as f64;

        // R²
        let tss = y.mapv(|yi| (yi - y.mean().unwrap_or(0.0)).powi(2)).sum();
        let rss = residuals.mapv(|e| e.powi(2)).sum();
        let r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        // RMSE
        let rmse = if t > k {
            (rss / (t - k)).sqrt()
        } else {
            f64::NAN
        };

        // MAE
        let mae = residuals.mapv(|e| e.abs()).mean().unwrap_or(f64::NAN);

        Ok((r2, rmse, mae, ols.params))
    }

    /// Evaluates out-of-sample performance
    fn evaluate_out_of_sample(
        y: &Array1<f64>,
        x: &Array2<f64>,
        beta: &Array1<f64>,
        historical_mean: f64,
    ) -> (f64, f64, f64, f64) {
        // Predictions
        let y_pred = x.dot(beta);
        let errors = y - &y_pred;

        let t = y.len() as f64;

        // R²
        let tss = y.mapv(|yi| (yi - y.mean().unwrap_or(0.0)).powi(2)).sum();
        let rss = errors.mapv(|e| e.powi(2)).sum();
        let r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        // RMSE
        let rmse = if t > 0.0 { (rss / t).sqrt() } else { f64::NAN };

        // MAE
        let mae = errors.mapv(|e| e.abs()).mean().unwrap_or(f64::NAN);

        // Campbell-Thompson R²_OOS
        // Compares model MSE to benchmark MSE (historical mean)
        let mse_model = rss / t;
        let benchmark_errors = y.mapv(|yi| yi - historical_mean);
        let mse_benchmark = benchmark_errors.mapv(|e| e.powi(2)).sum() / t;
        let r2_oos_ct = if mse_benchmark > 0.0 {
            1.0 - mse_model / mse_benchmark
        } else {
            f64::NAN
        };

        (r2, rmse, mae, r2_oos_ct)
    }

    /// Gets result for a specific asset
    pub fn get(&self, asset: &str) -> Option<&OOSPerformanceAsset> {
        self.results.get(asset)
    }

    /// Returns assets that beat the benchmark (positive Campbell-Thompson R²)
    pub fn assets_beating_benchmark(&self) -> Vec<&OOSPerformanceAsset> {
        self.results
            .values()
            .filter(|r| r.beats_benchmark())
            .collect()
    }

    /// Returns assets with no overfitting
    pub fn assets_without_overfitting(&self) -> Vec<&OOSPerformanceAsset> {
        self.results
            .values()
            .filter(|r| r.no_overfitting())
            .collect()
    }

    /// Exports results to a table format (Vec of rows)
    pub fn to_table(&self) -> Vec<OOSPerformanceRow> {
        let mut rows: Vec<_> = self
            .results
            .values()
            .map(|r| OOSPerformanceRow {
                asset: r.asset.clone(),
                r2_in: r.r2_in,
                rmse_in: r.rmse_in,
                mae_in: r.mae_in,
                r2_out: r.r2_out,
                rmse_out: r.rmse_out,
                mae_out: r.mae_out,
                r2_oos_ct: r.r2_oos_ct,
                nobs_in: r.nobs_in,
                nobs_out: r.nobs_out,
            })
            .collect();

        rows.sort_by(|a, b| {
            b.r2_oos_ct
                .partial_cmp(&a.r2_oos_ct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        rows
    }

    /// Exports results to CSV string
    pub fn to_csv_string(&self) -> String {
        let mut csv = String::from(
            "asset,r2_in,rmse_in,mae_in,r2_out,rmse_out,mae_out,r2_oos_CT,nobs_in,nobs_out\n",
        );

        for row in self.to_table() {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                row.asset,
                row.r2_in,
                row.rmse_in,
                row.mae_in,
                row.r2_out,
                row.rmse_out,
                row.mae_out,
                row.r2_oos_ct,
                row.nobs_in,
                row.nobs_out
            ));
        }

        csv
    }

    /// Returns summary statistics across all assets
    pub fn summary_stats(&self) -> OOSSummaryStats {
        let beating_benchmark = self.assets_beating_benchmark().len();
        let total = self.results.len();

        let r2_oos_values: Vec<f64> = self
            .results
            .values()
            .filter_map(|r| {
                if r.r2_oos_ct.is_finite() {
                    Some(r.r2_oos_ct)
                } else {
                    None
                }
            })
            .collect();

        let mean_r2_oos = if !r2_oos_values.is_empty() {
            r2_oos_values.iter().sum::<f64>() / r2_oos_values.len() as f64
        } else {
            f64::NAN
        };

        let median_r2_oos = if !r2_oos_values.is_empty() {
            let mut sorted = r2_oos_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        } else {
            f64::NAN
        };

        OOSSummaryStats {
            total_assets: total,
            assets_beating_benchmark: beating_benchmark,
            pct_beating_benchmark: if total > 0 {
                beating_benchmark as f64 / total as f64 * 100.0
            } else {
                f64::NAN
            },
            mean_r2_oos_ct: mean_r2_oos,
            median_r2_oos_ct: median_r2_oos,
        }
    }
}

/// Row format for table export
#[derive(Debug, Clone)]
pub struct OOSPerformanceRow {
    pub asset: String,
    pub r2_in: f64,
    pub rmse_in: f64,
    pub mae_in: f64,
    pub r2_out: f64,
    pub rmse_out: f64,
    pub mae_out: f64,
    pub r2_oos_ct: f64,
    pub nobs_in: usize,
    pub nobs_out: usize,
}

/// Summary statistics for out-of-sample performance
#[derive(Debug, Clone)]
pub struct OOSSummaryStats {
    pub total_assets: usize,
    pub assets_beating_benchmark: usize,
    pub pct_beating_benchmark: f64,
    pub mean_r2_oos_ct: f64,
    pub median_r2_oos_ct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_oos_basic() {
        // Generate synthetic data
        let t = 100;
        let n = 3;
        let k = 2;

        let returns =
            Array2::from_shape_fn((t, n), |(i, j)| 0.01 * (i as f64 / 10.0) + 0.005 * j as f64);
        let factors = Array2::from_shape_fn((t, k), |(i, j)| {
            0.008 * (i as f64 / 10.0) + 0.003 * j as f64
        });

        let result = OOSPerformance::fit(
            &returns,
            &factors,
            0.7,
            CovarianceType::NonRobust,
            Some(vec![
                "Asset1".to_string(),
                "Asset2".to_string(),
                "Asset3".to_string(),
            ]),
        );

        assert!(result.is_ok());
        let oos = result.unwrap();
        assert_eq!(oos.results.len(), 3);

        for asset in &["Asset1", "Asset2", "Asset3"] {
            let res = oos.get(asset).unwrap();
            assert!(res.nobs_in == 70);
            assert!(res.nobs_out == 30);
            assert!(res.r2_in.is_finite());
            assert!(res.r2_out.is_finite());
        }
    }

    #[test]
    fn test_oos_summary() {
        let t = 100;
        let n = 5;
        let k = 2;

        let returns =
            Array2::from_shape_fn((t, n), |(i, j)| 0.01 * (i as f64 / 10.0) + 0.005 * j as f64);
        let factors = Array2::from_shape_fn((t, k), |(i, j)| {
            0.008 * (i as f64 / 10.0) + 0.003 * j as f64
        });

        let oos =
            OOSPerformance::fit(&returns, &factors, 0.7, CovarianceType::NonRobust, None).unwrap();

        let stats = oos.summary_stats();
        assert_eq!(stats.total_assets, 5);
        assert!(stats.mean_r2_oos_ct.is_finite() || stats.mean_r2_oos_ct.is_nan());
    }

    #[test]
    fn test_oos_csv_export() {
        let t = 100;
        let returns =
            Array2::from_shape_fn((t, 2), |(i, j)| 0.01 * (i as f64 / 10.0) + 0.005 * j as f64);
        let factors = Array2::from_shape_fn((t, 2), |(i, j)| {
            0.008 * (i as f64 / 10.0) + 0.003 * j as f64
        });

        let oos =
            OOSPerformance::fit(&returns, &factors, 0.7, CovarianceType::NonRobust, None).unwrap();

        let csv = oos.to_csv_string();
        assert!(csv.contains("asset,r2_in"));
        assert!(csv.contains("Asset1"));
    }
}
