use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Result of IVOL & Tracking Error for multiple assets
///
/// Equivalente ao DataFrame of the Python with índice por asset
#[derive(Debug, Clone)]
pub struct IVOLTrackingMulti {
    /// Results por asset
    pub results: HashMap<String, IVOLTrackingAsset>,
}

/// Result of IVOL & Tracking Error for um asset
#[derive(Debug, Clone)]
pub struct IVOLTrackingAsset {
    /// Nome of the asset
    pub asset: String,

    /// Mean of return in excess (annualized)
    pub mean_excess_annual: f64,

    /// IVOL monthly (standard deviation of the residuals)
    pub ivol_monthly: f64,

    /// IVOL annualized (monthly × √12)
    pub ivol_annual: f64,

    /// Tracking error monthly (vs benchmark, if fornecido)
    pub tracking_error_monthly: Option<f64>,

    /// Tracking error annualized (monthly × √12)
    pub tracking_error_annual: Option<f64>,

    /// Number of observations
    pub nobs: usize,

    /// R² of the model
    pub r_squared: f64,

    /// Alpha of the model
    pub alpha: f64,
}

impl IVOLTrackingMulti {
    /// Calculates IVOL and Tracking Error for multiple assets
    ///
    /// # Arguments
    /// * `returns_excess` - Matriz of returns in excess (n_obs × n_assets)
    /// * `factors` - Matriz of factors (n_obs × n_factors)
    /// * `benchmark` - Benchmark for tracking error (opcional, n_obs)
    /// * `cov_type` - Covariance type
    /// * `asset_names` - Nomes of the assets (opcional)
    /// * `periods_per_year` - Periods por ano (12 for monthly, 252 for daily)
    ///
    /// # Example
    /// ```
    /// use frenchrs::IVOLTrackingMulti;
    /// use greeners::CovarianceType;
    /// use ndarray::Array2;
    ///
    /// // 12 meses × 2 assets (returns in excess)
    /// let returns_excess = Array2::from_shape_vec((12, 2), vec![
    ///     0.01, 0.015,
    ///     0.02, 0.025,
    ///     -0.01, -0.005,
    ///     0.03, 0.035,
    ///     0.015, 0.020,
    ///     -0.005, 0.000,
    ///     0.025, 0.030,
    ///     0.01, 0.015,
    ///     0.02, 0.025,
    ///     -0.01, -0.005,
    ///     0.03, 0.035,
    ///     0.015, 0.020,
    /// ]).unwrap();
    ///
    /// // 12 meses × 1 factor
    /// let factors = Array2::from_shape_vec((12, 1), vec![
    ///     0.008, 0.015, -0.005, 0.025, 0.012, -0.003,
    ///     0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
    /// ]).unwrap();
    ///
    /// let ivol_tracking = IVOLTrackingMulti::fit(
    ///     &returns_excess,
    ///     &factors,
    ///     None, // without benchmark
    ///     CovarianceType::NonRobust,
    ///     Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
    ///     12.0, // monthly
    /// ).unwrap();
    ///
    /// assert_eq!(ivol_tracking.results.len(), 2);
    /// ```
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        benchmark: Option<&Array1<f64>>,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
        periods_per_year: f64,
    ) -> Result<Self, GreenersError> {
        let n_obs = returns_excess.nrows();
        let n_assets = returns_excess.ncols();
        let n_factors = factors.ncols();

        if factors.nrows() != n_obs {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns ({} obs) and factors ({} obs) must have same number of observations",
                n_obs,
                factors.nrows()
            )));
        }

        if let Some(bench) = benchmark
            && bench.len() != n_obs
        {
            return Err(GreenersError::ShapeMismatch(format!(
                "Benchmark ({} obs) must have same length as returns ({} obs)",
                bench.len(),
                n_obs
            )));
        }

        if n_obs <= n_factors + 5 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data: {} obs for {} factors (need at least {})",
                n_obs,
                n_factors,
                n_factors + 6
            )));
        }

        // Nomes padrão if not fornecidos
        let asset_names = asset_names
            .unwrap_or_else(|| (0..n_assets).map(|i| format!("Asset{}", i + 1)).collect());

        let mut results = HashMap::new();

        // Procthatr each asset
        for (asset_idx, asset_name) in asset_names.iter().enumerate() {
            let asset_returns = returns_excess.column(asset_idx);

            let asset_result = Self::fit_single_asset(
                &asset_returns.to_owned(),
                factors,
                benchmark,
                cov_type.clone(),
                asset_name.clone(),
                periods_per_year,
            )?;

            results.insert(asset_name.clone(), asset_result);
        }

        Ok(IVOLTrackingMulti { results })
    }

    /// Calculates IVOL and Tracking Error for um single asset
    fn fit_single_asset(
        asset_returns_excess: &Array1<f64>,
        factors: &Array2<f64>,
        benchmark: Option<&Array1<f64>>,
        cov_type: CovarianceType,
        asset_name: String,
        periods_per_year: f64,
    ) -> Result<IVOLTrackingAsset, GreenersError> {
        let n_obs = asset_returns_excess.len();
        let n_factors = factors.ncols();

        // Matriz of design: [1, factors]
        let mut x = Array2::<f64>::zeros((n_obs, n_factors + 1));
        x.column_mut(0).fill(1.0);
        for j in 0..n_factors {
            x.column_mut(j + 1).assign(&factors.column(j));
        }

        // Estimates OLS
        let ols = OLS::fit(asset_returns_excess, &x, cov_type)?;

        // Residuals
        let residuals = ols.residuals(asset_returns_excess, &x);

        // IVOL monthly
        let ivol_monthly = residuals.std(1.0);

        // IVOL annualized
        let ivol_annual = ivol_monthly * periods_per_year.sqrt();

        // Mean of return in excess annualized
        let mean_excess_monthly = asset_returns_excess.mean().unwrap_or(0.0);
        let mean_excess_annual = mean_excess_monthly * periods_per_year;

        // Tracking error (if benchmark fornecido)
        let (tracking_error_monthly, tracking_error_annual) = if let Some(bench) = benchmark {
            let diff = asset_returns_excess - bench;
            let te_monthly = diff.std(1.0);
            let te_annual = te_monthly * periods_per_year.sqrt();
            (Some(te_monthly), Some(te_annual))
        } else {
            (None, None)
        };

        Ok(IVOLTrackingAsset {
            asset: asset_name,
            mean_excess_annual,
            ivol_monthly,
            ivol_annual,
            tracking_error_monthly,
            tracking_error_annual,
            nobs: n_obs,
            r_squared: ols.r_squared,
            alpha: ols.params[0],
        })
    }

    /// Gets results for um asset específico
    pub fn get_asset(&self, asset_name: &str) -> Option<&IVOLTrackingAsset> {
        self.results.get(asset_name)
    }

    /// Lists all assets
    pub fn asset_names(&self) -> Vec<String> {
        self.results.keys().cloned().collect()
    }

    /// Converts to formato tabular (similar ao DataFrame of the Python)
    pub fn to_table(&self) -> Vec<IVOLTrackingRow> {
        self.results
            .values()
            .map(|asset| IVOLTrackingRow {
                asset: asset.asset.clone(),
                mean_excess_annual: asset.mean_excess_annual,
                ivol_monthly: asset.ivol_monthly,
                ivol_annual: asset.ivol_annual,
                tracking_error_monthly: asset.tracking_error_monthly,
                tracking_error_annual: asset.tracking_error_annual,
                nobs: asset.nobs,
                r_squared: asset.r_squared,
                alpha: asset.alpha,
            })
            .collect()
    }

    /// Exports to CSV-like string
    pub fn to_csv_string(&self) -> String {
        let mut result = String::new();

        // Header
        result.push_str(
            "asset,mean_excess_annual,ivol_monthly,ivol_annual,tracking_error_monthly,tracking_error_annual,r_squared,alpha,nobs\n",
        );

        // Data
        let table = self.to_table();
        for row in table {
            let te_monthly = row
                .tracking_error_monthly
                .map_or("".to_string(), |v| format!("{:.6}", v));
            let te_annual = row
                .tracking_error_annual
                .map_or("".to_string(), |v| format!("{:.6}", v));

            result.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{}\n",
                row.asset,
                row.mean_excess_annual,
                row.ivol_monthly,
                row.ivol_annual,
                te_monthly,
                te_annual,
                row.r_squared,
                row.alpha,
                row.nobs
            ));
        }

        result
    }
}

/// Linha of result IVOL/Tracking (formato tabular)
#[derive(Debug, Clone)]
pub struct IVOLTrackingRow {
    /// Nome of the asset
    pub asset: String,

    /// Mean of return in excess annualized
    pub mean_excess_annual: f64,

    /// IVOL monthly
    pub ivol_monthly: f64,

    /// IVOL annualized
    pub ivol_annual: f64,

    /// Tracking error monthly (opcional)
    pub tracking_error_monthly: Option<f64>,

    /// Tracking error annualized (opcional)
    pub tracking_error_annual: Option<f64>,

    /// Number of observations
    pub nobs: usize,

    /// R² of the model
    pub r_squared: f64,

    /// Alpha of the model
    pub alpha: f64,
}

impl IVOLTrackingAsset {
    /// Classifies the level of IVOL
    pub fn ivol_classification(&self) -> &str {
        if self.ivol_annual < 0.10 {
            "Baixo"
        } else if self.ivol_annual < 0.20 {
            "Moderado"
        } else if self.ivol_annual < 0.30 {
            "Alto"
        } else {
            "Muito Alto"
        }
    }

    /// Classifies the level of tracking error (if disponível)
    pub fn tracking_error_classification(&self) -> Option<&str> {
        self.tracking_error_annual.map(|te| {
            if te < 0.02 {
                "Muito Baixo (< 2%)"
            } else if te < 0.05 {
                "Baixo (2-5%)"
            } else if te < 0.10 {
                "Moderado (5-10%)"
            } else if te < 0.15 {
                "Alto (10-15%)"
            } else {
                "Muito Alto (> 15%)"
            }
        })
    }

    /// Information ratio (if tracking error disponível)
    pub fn information_ratio(&self) -> Option<f64> {
        self.tracking_error_annual
            .map(|te| if te > 1e-10 { self.alpha / te } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_ivol_tracking_multi_basic() {
        let returns_excess = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
                0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
            ],
        )
        .unwrap();

        let factors = Array2::from_shape_vec(
            (12, 1),
            vec![
                0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025,
                0.012,
            ],
        )
        .unwrap();

        let ivol_tracking = IVOLTrackingMulti::fit(
            &returns_excess,
            &factors,
            None,
            CovarianceType::NonRobust,
            None,
            12.0,
        )
        .unwrap();

        assert_eq!(ivol_tracking.results.len(), 2);
    }

    #[test]
    fn test_ivol_tracking_with_benchmark() {
        let returns_excess = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
                0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
            ],
        )
        .unwrap();

        let factors = Array2::from_shape_vec(
            (12, 1),
            vec![
                0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025,
                0.012,
            ],
        )
        .unwrap();

        let benchmark = array![
            0.009, 0.018, -0.008, 0.028, 0.014, -0.004, 0.022, 0.010, 0.018, -0.008, 0.028, 0.014
        ];

        let ivol_tracking = IVOLTrackingMulti::fit(
            &returns_excess,
            &factors,
            Some(&benchmark),
            CovarianceType::NonRobust,
            Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
            12.0,
        )
        .unwrap();

        let asset1 = ivol_tracking.get_asset("Asset1").unwrap();
        assert!(asset1.tracking_error_monthly.is_some());
        assert!(asset1.tracking_error_annual.is_some());
    }

    #[test]
    fn test_ivol_tracking_to_table() {
        let returns_excess = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
                0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
            ],
        )
        .unwrap();

        let factors = Array2::from_shape_vec(
            (12, 1),
            vec![
                0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025,
                0.012,
            ],
        )
        .unwrap();

        let ivol_tracking = IVOLTrackingMulti::fit(
            &returns_excess,
            &factors,
            None,
            CovarianceType::NonRobust,
            None,
            12.0,
        )
        .unwrap();

        let table = ivol_tracking.to_table();
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_ivol_tracking_csv_export() {
        let returns_excess = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
                0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
            ],
        )
        .unwrap();

        let factors = Array2::from_shape_vec(
            (12, 1),
            vec![
                0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025,
                0.012,
            ],
        )
        .unwrap();

        let ivol_tracking = IVOLTrackingMulti::fit(
            &returns_excess,
            &factors,
            None,
            CovarianceType::NonRobust,
            None,
            12.0,
        )
        .unwrap();

        let csv = ivol_tracking.to_csv_string();
        assert!(csv.contains("asset,mean_excess_annual"));
        assert!(csv.contains("Asset1"));
        assert!(csv.contains("Asset2"));
    }
}
