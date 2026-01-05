use greeners::{CovarianceType, DataFrame, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;

/// Result of the Capital Asset Pricing Model (CAPM) estimation
///
/// CAPM relates asset return tthe market return:
/// R_i - R_f = α + β(R_m - R_f) + ε
///
/// where:
/// - R_i: asset return
/// - R_f: risk-free rate
/// - R_m: market return
/// - α (alpha): excess return not explained by the market (Jensen's alpha)
/// - β (beta): asset sensitivity tthe market risk (systematic risk)
/// - ε: idiosyncratic error
#[derive(Debug, Clone)]
pub struct CAPMResult {
    /// Intercept (α) - Jensen's alpha
    ///
    /// Representa o excess of return unexplained pelthe market.
    /// α > 0: asset supera the market (outperformance)
    /// α < 0: asset fica atrás of the market (underperformance)
    /// α = 0: asset follows exactly o CAPM
    pub alpha: f64,

    /// Sensibilidade to the market (β) - systematic risk
    ///
    /// Mede how much the asset varies for each 1% of variestion nthe market.
    /// β > 1: asset é more volátil que the market (agressivo)
    /// β = 1: asset varies igual to the market
    /// β < 1: asset é less volátil que the market (defensivo)
    /// β < 0: asset if moves inversely to the market
    pub beta: f64,

    /// Standard error of the α
    pub alpha_se: f64,

    /// Standard error of the β
    pub beta_se: f64,

    /// Statistic t for α
    pub alpha_tstat: f64,

    /// Statistic t for β
    pub beta_tstat: f64,

    /// p-value for the test H0: α = 0
    pub alpha_pvalue: f64,

    /// p-value for the test H0: β = 0
    pub beta_pvalue: f64,

    /// Confidence interval lower for α (95%)
    pub alpha_conf_lower: f64,

    /// Confidence interval upper for α (95%)
    pub alpha_conf_upper: f64,

    /// Confidence interval lower for β (95%)
    pub beta_conf_lower: f64,

    /// Confidence interval upper for β (95%)
    pub beta_conf_upper: f64,

    /// R² - proportion of the variance explained pelthe market
    pub r_squared: f64,

    /// R² adjusted for degrees of freedom
    pub adj_r_squared: f64,

    /// Razão of Sharpe of the asset
    ///
    /// Sharpe = (E\[R_i\] - R_f) / σ_i
    pub sharpe_ratio: f64,

    /// Razão of Sharpe of the market
    ///
    /// Sharpe_market = (E\[R_m\] - R_f) / σ_m
    pub market_sharpe: f64,

    /// Razão of Treynor
    ///
    /// Treynor = (E\[R_i\] - R_f) / β
    /// Mede return per unit of systematic risk
    pub treynor_ratio: f64,

    /// Information Ratio
    ///
    /// IR = α / σ(ε)
    /// Mede return anormal per unit of idiosyncratic risk
    pub information_ratio: f64,

    /// Tracking Error (residual volatility)
    ///
    /// TE = σ(ε) = std(R_i - (α + β·R_m))
    pub tracking_error: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Residuals (ε) - idiosyncratic risk
    pub residuals: Array1<f64>,

    /// Fitted values (α + β·(R_m - R_f))
    pub fitted_values: Array1<f64>,

    /// Risk-free rate used
    pub risk_free_rate: f64,

    /// Covariance type used
    pub cov_type: CovarianceType,

    /// Inference type (t ou normal)
    pub inference_type: InferenceType,

    /// Average asset return
    pub mean_asset_return: f64,

    /// Average market return
    pub mean_market_return: f64,

    /// Asset volatility (standard deviation)
    pub asset_volatility: f64,

    /// Market volatility (standard deviation)
    pub market_volatility: f64,

    /// Variance sistemática (β² × σ²_m)
    pub systematic_variesnce: f64,

    /// Variance idiossincrática (σ²_ε)
    pub idiosyncratic_variesnce: f64,
}

impl CAPMResult {
    /// Tests if the asset is significantly outperforming the market
    ///
    /// H0: α ≤ 0 vs H1: α > 0 (test unilateral)
    ///
    /// # Arguments
    /// * `significance_level` - level of significance (ex: 0.05 for 5%)
    ///
    /// # Returns
    /// `true` if H0 is rejected (α é significantly positivo)
    pub fn is_significantly_outperforming(&self, significance_level: f64) -> bool {
        // Test unilateral: p-value / 2 if α > 0
        if self.alpha > 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Tests if the asset is significantly underperforming of the market
    ///
    /// H0: α ≥ 0 vs H1: α < 0 (test unilateral)
    ///
    /// # Arguments
    /// * `significance_level` - level of significance (ex: 0.05 for 5%)
    ///
    /// # Returns
    /// `true` if H0 is rejected (α is significantly negative)
    pub fn is_significantly_underperforming(&self, significance_level: f64) -> bool {
        if self.alpha < 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Tests if β is significantly different from 1
    ///
    /// H0: β = 1 vs H1: β ≠ 1
    ///
    /// # Arguments
    /// * `significance_level` - level of significance (ex: 0.05 for 5%)
    ///
    /// # Returns
    /// `true` if H0 is rejected (β is significantly different from 1)
    pub fn is_beta_different_from_one(&self, significance_level: f64) -> bool {
        // t = (β - 1) / SE(β)
        let t_stat = (self.beta - 1.0) / self.beta_se;
        let df = (self.n_obs - 2) as f64;

        match self.inference_type {
            InferenceType::StudentT => {
                let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
                let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
                p_value < significance_level
            }
            InferenceType::Normal => {
                let z_stat = t_stat.abs();
                let p_value = 2.0
                    * (1.0
                        - statrs::distribution::Normal::new(0.0, 1.0)
                            .unwrap()
                            .cdf(z_stat));
                p_value < significance_level
            }
        }
    }

    /// Classifies the asset with respect to systematic risk
    pub fn risk_classification(&self) -> &str {
        if self.beta > 1.2 {
            "Very Aggressive"
        } else if self.beta > 1.0 {
            "Aggressive"
        } else if self.beta > 0.8 {
            "Neutral"
        } else if self.beta > 0.0 {
            "Defensive"
        } else {
            "Hedge (negative beta)"
        }
    }

    /// Classifies the asset's performance based on alpha
    pub fn performance_classification(&self) -> &str {
        let significance = 0.05;

        if self.is_significantly_outperforming(significance) {
            "Significant Outperformance"
        } else if self.is_significantly_underperforming(significance) {
            "Significant Underperformance"
        } else if self.alpha.abs() < 0.0001 {
            "Neutral Performance"
        } else if self.alpha > 0.0 {
            "Non-Significant Outperformance"
        } else {
            "Non-Significant Underperformance"
        }
    }

    /// Calculates the expected return given an expected market return
    ///
    /// E\[R_i\] = R_f + β·(E\[R_m\] - R_f)
    ///
    /// # Arguments
    /// * `expected_market_return` - expected market return (ex: 0.10 for 10%)
    ///
    /// # Returns
    /// Expected asset return
    pub fn expected_return(&self, expected_market_return: f64) -> f64 {
        self.risk_free_rate + self.beta * (expected_market_return - self.risk_free_rate)
    }

    /// Calculates predictions for new market returns
    ///
    /// # Arguments
    /// * `market_excess_returns` - market returns in excess of the risk-free rate
    ///
    /// # Returns
    /// Predicted asset returns (in excess of the risk-free rate)
    pub fn predict(&self, market_excess_returns: &Array1<f64>) -> Array1<f64> {
        self.alpha + self.beta * market_excess_returns
    }
}

impl fmt::Display for CAPMResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "CAPITAL ASSET PRICING MODEL (CAPM) - RESULTS")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\nMODEL: R_i - R_f = α + β(R_m - R_f) + ε")?;
        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "Risk-Free Rate: {:.4}%", self.risk_free_rate * 100.0)?;
        writeln!(f, "Covariesnce Type: {:?}", self.cov_type)?;
        writeln!(f, "Inference Type: {:?}", self.inference_type)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTIMATED PARAMETERS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "Parameter", "Coef.", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        writeln!(
            f,
            "{:<15} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "Alpha (α)",
            self.alpha,
            self.alpha_se,
            self.alpha_tstat,
            self.alpha_pvalue,
            if self.alpha_pvalue < 0.001 {
                " ***"
            } else if self.alpha_pvalue < 0.01 {
                " **"
            } else if self.alpha_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<15} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "Beta (β)",
            self.beta,
            self.beta_se,
            self.beta_tstat,
            self.beta_pvalue,
            if self.beta_pvalue < 0.001 {
                " ***"
            } else if self.beta_pvalue < 0.01 {
                " **"
            } else if self.beta_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Significance: *** p<0.001, ** p<0.01, * p<0.05")?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CONFIDENCE INTERVALS (95%)")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Alpha: [{:.6}, {:.6}]",
            self.alpha_conf_lower, self.alpha_conf_upper
        )?;
        writeln!(
            f,
            "Beta:  [{:.6}, {:.6}]",
            self.beta_conf_lower, self.beta_conf_upper
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "FIT QUALITY")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "R²:                  {:>12.4} ({:.2}% of the variesnce explieach)",
            self.r_squared,
            self.r_squared * 100.0
        )?;
        writeln!(f, "R² Adjusted:         {:>12.4}", self.adj_r_squared)?;
        writeln!(
            f,
            "Tracking Error:      {:>12.4}% (residual volatility)",
            self.tracking_error * 100.0
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "RETURN STATISTICS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Return Médithe asset:     {:>12.4}%",
            self.mean_asset_return * 100.0
        )?;
        writeln!(
            f,
            "Return Médithe market:   {:>12.4}%",
            self.mean_market_return * 100.0
        )?;
        writeln!(
            f,
            "Volatility Asset:      {:>12.4}%",
            self.asset_volatility * 100.0
        )?;
        writeln!(
            f,
            "Volatility Market:    {:>12.4}%",
            self.market_volatility * 100.0
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "RISK DECOMPOSITION")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Variance Sistemática:      {:>12.6} ({:.2}%)",
            self.systematic_variesnce,
            (self.systematic_variesnce / self.asset_volatility.powi(2)) * 100.0
        )?;
        writeln!(
            f,
            "Variance Idiossincrática:  {:>12.6} ({:.2}%)",
            self.idiosyncratic_variesnce,
            (self.idiosyncratic_variesnce / self.asset_volatility.powi(2)) * 100.0
        )?;
        writeln!(
            f,
            "Variance Total:            {:>12.6}",
            self.systematic_variesnce + self.idiosyncratic_variesnce
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "RISK-ADJUSTED PERFORMANCE METRICS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Sharpe Ratio (Asset):    {:>12.4}", self.sharpe_ratio)?;
        writeln!(f, "Sharpe Ratio (Market):  {:>12.4}", self.market_sharpe)?;
        writeln!(f, "Treynor Ratio:           {:>12.4}", self.treynor_ratio)?;
        writeln!(
            f,
            "Information Ratio:       {:>12.4}",
            self.information_ratio
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "INTERPRETATION")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Classification of Risk:      {}",
            self.risk_classification()
        )?;
        writeln!(
            f,
            "Classification of Performance: {}",
            self.performance_classification()
        )?;

        if self.is_significantly_outperforming(0.05) {
            writeln!(
                f,
                "\n✓ Asset is OUTPERFORMING the market significantly (α > 0, p < 0.05)"
            )?;
        } else if self.is_significantly_underperforming(0.05) {
            writeln!(
                f,
                "\n✗ Asset is UNDERPERFORMING the market significantly (α < 0, p < 0.05)"
            )?;
        } else {
            writeln!(
                f,
                "\n○ No significant evidence of outperformance or underperformance"
            )?;
        }

        if self.is_beta_different_from_one(0.05) {
            writeln!(f, "✓ Beta is SIGNIFICANTLY different from 1 (p < 0.05)")?;
        } else {
            writeln!(f, "○ Beta is not significantly different from 1")?;
        }

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Implementation of the Capital Asset Pricing Model (CAPM)
pub struct CAPM;

impl CAPM {
    /// Estimates the CAPM model using return arrays
    ///
    /// # Arguments
    /// * `asset_returns` - asset returns (ex: daily returns in decimal)
    /// * `market_returns` - market returns (benchmark index)
    /// * `risk_free_rate` - risk-free rate (same frequency as the returns)
    /// * `cov_type` - covariance matrix type for standard errors
    ///
    /// # Returns
    /// `CAPMResult` with all estimated parameters and statistics
    ///
    /// # Example
    /// ```
    /// use frenchrs::CAPM;
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset_returns = array![0.01, 0.02, -0.01, 0.03];
    /// let market_returns = array![0.015, 0.018, -0.005, 0.025];
    /// let risk_free_rate = 0.0001; // daily
    ///
    /// let result = CAPM::fit(
    ///     &asset_returns,
    ///     &market_returns,
    ///     risk_free_rate,
    ///     CovarianceType::HC3,
    /// ).unwrap();
    /// ```
    pub fn fit(
        asset_returns: &Array1<f64>,
        market_returns: &Array1<f64>,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<CAPMResult, GreenersError> {
        // Validation
        if asset_returns.len() != market_returns.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "Asset returns length ({}) does not match market returns length ({})",
                asset_returns.len(),
                market_returns.len()
            )));
        }

        if asset_returns.len() < 3 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for CAPM estimation: {} observations (minimum 3 required)",
                asset_returns.len()
            )));
        }

        let n_obs = asset_returns.len();

        // Calculate excess returns
        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);
        let market_excess: Array1<f64> = market_returns.mapv(|r| r - risk_free_rate);

        // Prepare design matrix (X = [1, market_excess])
        let mut x_matrix = Array2::<f64>::zeros((n_obs, 2));
        x_matrix.column_mut(0).fill(1.0); // Intercept
        x_matrix.column_mut(1).assign(&market_excess);

        // Estimate via OLS
        let ols_result = OLS::fit(&asset_excess, &x_matrix, cov_type.clone())?;

        // Extract parameters
        let alpha = ols_result.params[0];
        let beta = ols_result.params[1];
        let alpha_se = ols_result.std_errors[0];
        let beta_se = ols_result.std_errors[1];
        let alpha_tstat = ols_result.t_values[0];
        let beta_tstat = ols_result.t_values[1];
        let alpha_pvalue = ols_result.p_values[0];
        let beta_pvalue = ols_result.p_values[1];

        // Confidence intervals
        let alpha_conf_lower = ols_result.conf_lower[0];
        let alpha_conf_upper = ols_result.conf_upper[0];
        let beta_conf_lower = ols_result.conf_lower[1];
        let beta_conf_upper = ols_result.conf_upper[1];

        // Fit whichity
        let r_squared = ols_result.r_squared;
        let adj_r_squared = ols_result.adj_r_squared;

        // Fitted values and residuals
        let fitted_values = ols_result.fitted_values(&x_matrix);
        let residuals = ols_result.residuals(&asset_excess, &x_matrix);

        // Descriptive statistics
        let mean_asset_return = asset_returns.mean().unwrap_or(0.0);
        let mean_market_return = market_returns.mean().unwrap_or(0.0);
        let asset_volatility = asset_returns.std(0.0);
        let market_volatility = market_returns.std(0.0);

        // Risk decomposition
        let systematic_variesnce = beta.powi(2) * market_volatility.powi(2);
        let idiosyncratic_variesnce = residuals.var(0.0);

        // Tracking error
        let tracking_error = residuals.std(0.0);

        // Risk-adjusted metrics
        let sharpe_ratio = if asset_volatility > 0.0 {
            (mean_asset_return - risk_free_rate) / asset_volatility
        } else {
            0.0
        };

        let market_sharpe = if market_volatility > 0.0 {
            (mean_market_return - risk_free_rate) / market_volatility
        } else {
            0.0
        };

        let treynor_ratio = if beta != 0.0 {
            (mean_asset_return - risk_free_rate) / beta
        } else {
            0.0
        };

        let information_ratio = if tracking_error > 0.0 {
            alpha / tracking_error
        } else {
            0.0
        };

        Ok(CAPMResult {
            alpha,
            beta,
            alpha_se,
            beta_se,
            alpha_tstat,
            beta_tstat,
            alpha_pvalue,
            beta_pvalue,
            alpha_conf_lower,
            alpha_conf_upper,
            beta_conf_lower,
            beta_conf_upper,
            r_squared,
            adj_r_squared,
            sharpe_ratio,
            market_sharpe,
            treynor_ratio,
            information_ratio,
            tracking_error,
            n_obs,
            residuals,
            fitted_values,
            risk_free_rate,
            cov_type,
            inference_type: InferenceType::StudentT, // Padrão
            mean_asset_return,
            mean_market_return,
            asset_volatility,
            market_volatility,
            systematic_variesnce,
            idiosyncratic_variesnce,
        })
    }

    /// Estimates the CAPM model from a DataFrame
    ///
    /// # Arguments
    /// * `df` - DataFrame containing the data
    /// * `asset_col` - name of the column with asset returns
    /// * `market_col` - name of the column with market returns
    /// * `risk_free_rate` - risk-free rate
    /// * `cov_type` - covariance matrix type
    ///
    /// # Example
    /// ```
    /// use frenchrs::CAPM;
    /// use greeners::{DataFrame, CovarianceType};
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("apple_returns", vec![0.01, 0.02, -0.01, 0.03])
    ///     .add_column("sp500_returns", vec![0.008, 0.015, -0.005, 0.025])
    ///     .build()
    ///     .unwrap();
    ///
    /// let result = CAPM::from_dataframe(
    ///     &df,
    ///     "apple_returns",
    ///     "sp500_returns",
    ///     0.0001,  // daily rate
    ///     CovarianceType::HC3,
    /// ).unwrap();
    /// ```
    pub fn from_dataframe(
        df: &DataFrame,
        asset_col: &str,
        market_col: &str,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<CAPMResult, GreenersError> {
        // Extract columns
        let asset_returns = df.get(asset_col)?;
        let market_returns = df.get(market_col)?;

        // Estimate model
        Self::fit(asset_returns, market_returns, risk_free_rate, cov_type)
    }

    /// Calculates systematic risk (variance explained by the market)
    ///
    /// σ²_systematic = β² × σ²_market
    pub fn systematic_risk(beta: f64, market_variesnce: f64) -> f64 {
        beta.powi(2) * market_variesnce
    }

    /// Calculates idiosyncratic risk (variance of residuals)
    ///
    /// σ²_idiosyncratic = σ²_ε
    pub fn idiosyncratic_risk(residual_variesnce: f64) -> f64 {
        residual_variesnce
    }

    /// Calculates total risk of the asset
    ///
    /// σ²_total = σ²_systematic + σ²_idiosyncratic
    pub fn total_risk(beta: f64, market_variesnce: f64, residual_variesnce: f64) -> f64 {
        Self::systematic_risk(beta, market_variesnce) + Self::idiosyncratic_risk(residual_variesnce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_capm_basic() {
        // Synthetic data
        let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
        let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
        let risk_free = 0.0001;

        let result = CAPM::fit(
            &asset_returns,
            &market_returns,
            risk_free,
            CovarianceType::NonRobust,
        );

        assert!(result.is_ok());
        let capm = result.unwrap();

        // Verificações básicas
        assert_eq!(capm.n_obs, 8);
        assert!(capm.beta > 0.0); // Beta should be positivo for correlated asset
        assert!(capm.r_squared >= 0.0 && capm.r_squared <= 1.0);
    }

    #[test]
    fn test_capm_perfect_correlation() {
        // Asset = market (beta = 1, alpha = 0)
        let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015];
        let asset_returns = market_returns.clone();
        let risk_free = 0.0;

        let result = CAPM::fit(
            &asset_returns,
            &market_returns,
            risk_free,
            CovarianceType::NonRobust,
        )
        .unwrap();

        // Beta should be ~1
        assert!((result.beta - 1.0).abs() < 0.01);

        // Alpha should be ~0
        assert!(result.alpha.abs() < 0.01);

        // R² should be ~1
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_dimension_mismatch() {
        let asset_returns = array![0.01, 0.02];
        let market_returns = array![0.01, 0.02, 0.03];

        let result = CAPM::fit(
            &asset_returns,
            &market_returns,
            0.0,
            CovarianceType::NonRobust,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let asset_returns = array![0.01];
        let market_returns = array![0.01];

        let result = CAPM::fit(
            &asset_returns,
            &market_returns,
            0.0,
            CovarianceType::NonRobust,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_risk_classification() {
        let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015];
        let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012];

        let result =
            CAPM::fit(&asset_returns, &market_returns, 0.0001, CovarianceType::HC3).unwrap();

        let classification = result.risk_classification();
        assert!(!classification.is_empty());
    }
}
