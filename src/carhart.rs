use greeners::{CovarianceType, DataFrame, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the Carhart 4 Factor model estimation
///
/// The Carhart 4 Factor model extends the Fama-French 3 Factor model by adding the Momentum factor:
/// R_i - R_f = α + β_MKT(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + β_MOM(MOM) + ε
///
/// where:
/// - R_i: asset return
/// - R_f: risk-free rate
/// - R_m: market return
/// - SMB (Small Minus Big): size factor
/// - HML (High Minus Low): value factor
/// - MOM (Momentum): momentum factor (winners - losers)
/// - α (alpha): excess return unexplained by the 4 factors
/// - β_MKT: sensitivity to market risk
/// - β_SMB: sensitivity to the size factor
/// - β_HML: sensitivity to the value factor
/// - β_MOM: sensitivity to the momentum factor
#[derive(Debug, Clone)]
pub struct Carhart4FactorResult {
    /// Intercept (α) - Jensen's alpha
    pub alpha: f64,

    /// Beta of market (β_MKT)
    pub beta_market: f64,

    /// Beta SMB (β_SMB) - sensibilidade ao factor size
    pub beta_smb: f64,

    /// Beta HML (β_HML) - sensibilidade ao factor value
    pub beta_hml: f64,

    /// Beta MOM (β_MOM) - sensibilidade ao factor momentum
    ///
    /// β_MOM > 0: asset exibe momentum positivo (follows tendências)
    /// β_MOM < 0: asset exibe reverare (contrarian)
    /// β_MOM = 0: asset é neutro ao momentum
    pub beta_mom: f64,

    /// Standard error of the α
    pub alpha_se: f64,

    /// Standard error of the β_MKT
    pub beta_market_se: f64,

    /// Standard error of the β_SMB
    pub beta_smb_se: f64,

    /// Standard error of the β_HML
    pub beta_hml_se: f64,

    /// Standard error of the β_MOM
    pub beta_mom_se: f64,

    /// Statistic t for α
    pub alpha_tstat: f64,

    /// Statistic t for β_MKT
    pub beta_market_tstat: f64,

    /// Statistic t for β_SMB
    pub beta_smb_tstat: f64,

    /// Statistic t for β_HML
    pub beta_hml_tstat: f64,

    /// Statistic t for β_MOM
    pub beta_mom_tstat: f64,

    /// p-value for the test H0: α = 0
    pub alpha_pvalue: f64,

    /// p-value for the test H0: β_MKT = 0
    pub beta_market_pvalue: f64,

    /// p-value for the test H0: β_SMB = 0
    pub beta_smb_pvalue: f64,

    /// p-value for the test H0: β_HML = 0
    pub beta_hml_pvalue: f64,

    /// p-value for the test H0: β_MOM = 0
    pub beta_mom_pvalue: f64,

    /// Confidence interval lower for α (95%)
    pub alpha_conf_lower: f64,

    /// Confidence interval upper for α (95%)
    pub alpha_conf_upper: f64,

    /// Confidence interval lower for β_MKT (95%)
    pub beta_market_conf_lower: f64,

    /// Confidence interval upper for β_MKT (95%)
    pub beta_market_conf_upper: f64,

    /// Confidence interval lower for β_SMB (95%)
    pub beta_smb_conf_lower: f64,

    /// Confidence interval upper for β_SMB (95%)
    pub beta_smb_conf_upper: f64,

    /// Confidence interval lower for β_HML (95%)
    pub beta_hml_conf_lower: f64,

    /// Confidence interval upper for β_HML (95%)
    pub beta_hml_conf_upper: f64,

    /// Confidence interval lower for β_MOM (95%)
    pub beta_mom_conf_lower: f64,

    /// Confidence interval upper for β_MOM (95%)
    pub beta_mom_conf_upper: f64,

    /// R² - proportion of the variance explained by the 4 factors
    pub r_squared: f64,

    /// R² adjusted for degrees of freedom
    pub adj_r_squared: f64,

    /// Tracking Error (residual volatility)
    pub tracking_error: f64,

    /// Information Ratio (α / tracking_error)
    pub information_ratio: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Residuals (ε) - idiosyncratic risk
    pub residuals: Array1<f64>,

    /// Fitted values
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

    /// Return médio of the factor SMB
    pub mean_smb: f64,

    /// Return médio of the factor HML
    pub mean_hml: f64,

    /// Return médio of the factor MOM
    pub mean_mom: f64,

    /// Asset volatility (standard deviation)
    pub asset_volatility: f64,
}

impl Carhart4FactorResult {
    /// Tests if the asset is significantly outperforming the model Carhart
    pub fn is_significantly_outperforming(&self, significance_level: f64) -> bool {
        if self.alpha > 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Tests if the asset is significantly underperforming of the model Carhart
    pub fn is_significantly_underperforming(&self, significance_level: f64) -> bool {
        if self.alpha < 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Tests if β_SMB é significantly diferente of zero
    pub fn is_smb_significant(&self, significance_level: f64) -> bool {
        self.beta_smb_pvalue < significance_level
    }

    /// Tests if β_HML é significantly diferente of zero
    pub fn is_hml_significant(&self, significance_level: f64) -> bool {
        self.beta_hml_pvalue < significance_level
    }

    /// Tests if β_MOM é significantly diferente of zero
    pub fn is_mom_significant(&self, significance_level: f64) -> bool {
        self.beta_mom_pvalue < significance_level
    }

    /// Classifies the asset with respect to size factor (SMB)
    pub fn size_classification(&self) -> &str {
        if !self.is_smb_significant(0.05) {
            "Neutral (SMB not significant)"
        } else if self.beta_smb > 0.5 {
            "Strongly Small Cap"
        } else if self.beta_smb > 0.0 {
            "Small Cap"
        } else if self.beta_smb > -0.5 {
            "Large Cap"
        } else {
            "Strongly Large Cap"
        }
    }

    /// Classifies the asset with respect to value factor (HML)
    pub fn value_classification(&self) -> &str {
        if !self.is_hml_significant(0.05) {
            "Neutral (HML not significant)"
        } else if self.beta_hml > 0.5 {
            "Strongly Value"
        } else if self.beta_hml > 0.0 {
            "Value"
        } else if self.beta_hml > -0.5 {
            "Growth"
        } else {
            "Strongly Growth"
        }
    }

    /// Classifies the asset with respect to momentum factor (MOM)
    pub fn momentum_classification(&self) -> &str {
        if !self.is_mom_significant(0.05) {
            "Neutral (Momentum not significant)"
        } else if self.beta_mom > 0.5 {
            "Strong Momentum (Winner)"
        } else if self.beta_mom > 0.0 {
            "Momentum (Trend Following)"
        } else if self.beta_mom > -0.5 {
            "Reversal (Contrarian)"
        } else {
            "Strong Reversal (Strong Contrarian)"
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

    /// Calculates the expected return given expected factor returns
    pub fn expected_return(
        &self,
        expected_market_return: f64,
        expected_smb: f64,
        expected_hml: f64,
        expected_mom: f64,
    ) -> f64 {
        self.risk_free_rate
            + self.beta_market * (expected_market_return - self.risk_free_rate)
            + self.beta_smb * expected_smb
            + self.beta_hml * expected_hml
            + self.beta_mom * expected_mom
    }

    /// Calculates predições for novthe data of the factors
    pub fn predict(
        &self,
        market_excess_returns: &Array1<f64>,
        smb_returns: &Array1<f64>,
        hml_returns: &Array1<f64>,
        mom_returns: &Array1<f64>,
    ) -> Array1<f64> {
        self.alpha
            + self.beta_market * market_excess_returns
            + self.beta_smb * smb_returns
            + self.beta_hml * hml_returns
            + self.beta_mom * mom_returns
    }

    /// Calculates a contribution of each factor for the return esperado
    pub fn factor_contributions(&self) -> (f64, f64, f64, f64) {
        let market_contrib = self.beta_market * (self.mean_market_return - self.risk_free_rate);
        let smb_contrib = self.beta_smb * self.mean_smb;
        let hml_contrib = self.beta_hml * self.mean_hml;
        let mom_contrib = self.beta_mom * self.mean_mom;

        (market_contrib, smb_contrib, hml_contrib, mom_contrib)
    }
}

impl fmt::Display for Carhart4FactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "CARHART 4 FACTOR MODEL - RESULTS")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(
            f,
            "\nMODEL: R_i - R_f = α + β_MKT(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + β_MOM(MOM) + ε"
        )?;
        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "Risk-Free Rate: {:.4}%", self.risk_free_rate * 100.0)?;
        writeln!(f, "Covariesnce Type: {:?}", self.cov_type)?;
        writeln!(f, "Inference Type: {:?}", self.inference_type)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTIMATED PARAMETERS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Parameter", "Coef.", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        let params = [
            (
                "Alpha (α)",
                self.alpha,
                self.alpha_se,
                self.alpha_tstat,
                self.alpha_pvalue,
            ),
            (
                "Beta Market (β_MKT)",
                self.beta_market,
                self.beta_market_se,
                self.beta_market_tstat,
                self.beta_market_pvalue,
            ),
            (
                "Beta SMB (β_SMB)",
                self.beta_smb,
                self.beta_smb_se,
                self.beta_smb_tstat,
                self.beta_smb_pvalue,
            ),
            (
                "Beta HML (β_HML)",
                self.beta_hml,
                self.beta_hml_se,
                self.beta_hml_tstat,
                self.beta_hml_pvalue,
            ),
            (
                "Beta MOM (β_MOM)",
                self.beta_mom,
                self.beta_mom_se,
                self.beta_mom_tstat,
                self.beta_mom_pvalue,
            ),
        ];

        for (name, coef, se, tstat, pval) in params.iter() {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
                name,
                coef,
                se,
                tstat,
                pval,
                if *pval < 0.001 {
                    " ***"
                } else if *pval < 0.01 {
                    " **"
                } else if *pval < 0.05 {
                    " *"
                } else {
                    ""
                }
            )?;
        }

        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Significance: *** p<0.001, ** p<0.01, * p<0.05")?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CONFIDENCE INTERVALS (95%)")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Alpha:       [{:.6}, {:.6}]",
            self.alpha_conf_lower, self.alpha_conf_upper
        )?;
        writeln!(
            f,
            "Beta Market: [{:.6}, {:.6}]",
            self.beta_market_conf_lower, self.beta_market_conf_upper
        )?;
        writeln!(
            f,
            "Beta SMB:    [{:.6}, {:.6}]",
            self.beta_smb_conf_lower, self.beta_smb_conf_upper
        )?;
        writeln!(
            f,
            "Beta HML:    [{:.6}, {:.6}]",
            self.beta_hml_conf_lower, self.beta_hml_conf_upper
        )?;
        writeln!(
            f,
            "Beta MOM:    [{:.6}, {:.6}]",
            self.beta_mom_conf_lower, self.beta_mom_conf_upper
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
        writeln!(f, "FACTOR STATISTICS")?;
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
            "Return Médio SMB:       {:>12.4}%",
            self.mean_smb * 100.0
        )?;
        writeln!(
            f,
            "Return Médio HML:       {:>12.4}%",
            self.mean_hml * 100.0
        )?;
        writeln!(
            f,
            "Return Médio MOM:       {:>12.4}%",
            self.mean_mom * 100.0
        )?;
        writeln!(
            f,
            "Volatility Asset:      {:>12.4}%",
            self.asset_volatility * 100.0
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "FACTOR CONTRIBUTIONS PARA O RETORNO")?;
        writeln!(f, "{}", "-".repeat(80))?;

        let (market_contrib, smb_contrib, hml_contrib, mom_contrib) = self.factor_contributions();
        let total_contrib = market_contrib + smb_contrib + hml_contrib + mom_contrib;

        writeln!(
            f,
            "Market (β_MKT × MRP):   {:>12.4}% ({:>5.1}%)",
            market_contrib * 100.0,
            if total_contrib != 0.0 {
                (market_contrib / total_contrib) * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "SMB (β_SMB × SMB):       {:>12.4}% ({:>5.1}%)",
            smb_contrib * 100.0,
            if total_contrib != 0.0 {
                (smb_contrib / total_contrib) * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "HML (β_HML × HML):       {:>12.4}% ({:>5.1}%)",
            hml_contrib * 100.0,
            if total_contrib != 0.0 {
                (hml_contrib / total_contrib) * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "MOM (β_MOM × MOM):       {:>12.4}% ({:>5.1}%)",
            mom_contrib * 100.0,
            if total_contrib != 0.0 {
                (mom_contrib / total_contrib) * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Total Explained:         {:>12.4}%",
            total_contrib * 100.0
        )?;
        writeln!(f, "Alpha (unexplained):   {:>12.4}%", self.alpha * 100.0)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "PERFORMANCE METRICS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Information Ratio:       {:>12.4}",
            self.information_ratio
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CLASSIFICATIONS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Size (SMB):           {}", self.size_classification())?;
        writeln!(
            f,
            "Value (HML):             {}",
            self.value_classification()
        )?;
        writeln!(
            f,
            "Momentum (MOM):          {}",
            self.momentum_classification()
        )?;
        writeln!(
            f,
            "Performance (Alpha):      {}",
            self.performance_classification()
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "INTERPRETATION")?;
        writeln!(f, "{}", "-".repeat(80))?;

        if self.is_significantly_outperforming(0.05) {
            writeln!(
                f,
                "✓ the asset is OUTPERFORMING the model Carhart significantly (α > 0, p < 0.05)"
            )?;
        } else if self.is_significantly_underperforming(0.05) {
            writeln!(
                f,
                "✗ the asset is UNDERPERFORMING of the model Carhart significantly (α < 0, p < 0.05)"
            )?;
        } else {
            writeln!(
                f,
                "○ Não significant evidence of outperformance ou underperformance"
            )?;
        }

        if self.is_smb_significant(0.05) {
            if self.beta_smb > 0.0 {
                writeln!(
                    f,
                    "✓ Exposição significativa a SMALL CAPS (β_SMB = {:.4}, p < 0.05)",
                    self.beta_smb
                )?;
            } else {
                writeln!(
                    f,
                    "✓ Exposição significativa a LARGE CAPS (β_SMB = {:.4}, p < 0.05)",
                    self.beta_smb
                )?;
            }
        } else {
            writeln!(f, "○ Sem exposição significativa ao factor size (SMB)")?;
        }

        if self.is_hml_significant(0.05) {
            if self.beta_hml > 0.0 {
                writeln!(
                    f,
                    "✓ Exposição significativa a VALUE STOCKS (β_HML = {:.4}, p < 0.05)",
                    self.beta_hml
                )?;
            } else {
                writeln!(
                    f,
                    "✓ Exposição significativa a GROWTH STOCKS (β_HML = {:.4}, p < 0.05)",
                    self.beta_hml
                )?;
            }
        } else {
            writeln!(f, "○ Sem exposição significativa ao factor value (HML)")?;
        }

        if self.is_mom_significant(0.05) {
            if self.beta_mom > 0.0 {
                writeln!(
                    f,
                    "✓ Exposição significativa a MOMENTUM (β_MOM = {:.4}, p < 0.05)",
                    self.beta_mom
                )?;
                writeln!(f, "  → Estrup togia follows tendências (trend following)")?;
            } else {
                writeln!(
                    f,
                    "✓ Exposição significativa a REVERSÃO (β_MOM = {:.4}, p < 0.05)",
                    self.beta_mom
                )?;
                writeln!(f, "  → Estrup togia contrarian (compra perdedores)")?;
            }
        } else {
            writeln!(f, "○ Sem exposição significativa ao factor momentum (MOM)")?;
        }

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Implementation of the model Carhart 4 Factor
pub struct Carhart4Factor;

impl Carhart4Factor {
    /// Estimates the model Carhart 4 Factor using arrays
    ///
    /// # Arguments
    /// * `asset_returns` - asset returns
    /// * `market_returns` - market returns
    /// * `smb_returns` - returns of the factor SMB (Small Minus Big)
    /// * `hml_returns` - returns of the factor HML (High Minus Low)
    /// * `mom_returns` - returns of the factor MOM (Momentum)
    /// * `risk_free_rate` - risk-free rate
    /// * `cov_type` - covariance matrix type
    ///
    /// # Example
    /// ```
    /// use frenchrs::Carhart4Factor;
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    /// let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    /// let smb_returns = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    /// let hml_returns = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
    /// let mom_returns = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];
    /// let risk_free_rate = 0.0001;
    ///
    /// let result = Carhart4Factor::fit(
    ///     &asset_returns,
    ///     &market_returns,
    ///     &smb_returns,
    ///     &hml_returns,
    ///     &mom_returns,
    ///     risk_free_rate,
    ///     CovarianceType::HC3,
    /// ).unwrap();
    /// ```
    pub fn fit(
        asset_returns: &Array1<f64>,
        market_returns: &Array1<f64>,
        smb_returns: &Array1<f64>,
        hml_returns: &Array1<f64>,
        mom_returns: &Array1<f64>,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<Carhart4FactorResult, GreenersError> {
        // Validation
        let n = asset_returns.len();
        if market_returns.len() != n
            || smb_returns.len() != n
            || hml_returns.len() != n
            || mom_returns.len() != n
        {
            return Err(GreenersError::ShapeMismatch(
                "All return arrays must have the same length".to_string(),
            ));
        }

        if n < 6 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for Carhart 4F estimation: {} observations (minimum 6 required)",
                n
            )));
        }

        // Calculate returns in excess
        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);
        let market_excess: Array1<f64> = market_returns.mapv(|r| r - risk_free_rate);

        // Prepare design matrix: X = [1, market_excess, SMB, HML, MOM]
        let mut x_matrix = Array2::<f64>::zeros((n, 5));
        x_matrix.column_mut(0).fill(1.0); // Intercept
        x_matrix.column_mut(1).assign(&market_excess);
        x_matrix.column_mut(2).assign(smb_returns);
        x_matrix.column_mut(3).assign(hml_returns);
        x_matrix.column_mut(4).assign(mom_returns);

        // Estimate via OLS
        let ols_result = OLS::fit(&asset_excess, &x_matrix, cov_type.clone())?;

        // Extract parameters
        let alpha = ols_result.params[0];
        let beta_market = ols_result.params[1];
        let beta_smb = ols_result.params[2];
        let beta_hml = ols_result.params[3];
        let beta_mom = ols_result.params[4];

        let alpha_se = ols_result.std_errors[0];
        let beta_market_se = ols_result.std_errors[1];
        let beta_smb_se = ols_result.std_errors[2];
        let beta_hml_se = ols_result.std_errors[3];
        let beta_mom_se = ols_result.std_errors[4];

        let alpha_tstat = ols_result.t_values[0];
        let beta_market_tstat = ols_result.t_values[1];
        let beta_smb_tstat = ols_result.t_values[2];
        let beta_hml_tstat = ols_result.t_values[3];
        let beta_mom_tstat = ols_result.t_values[4];

        let alpha_pvalue = ols_result.p_values[0];
        let beta_market_pvalue = ols_result.p_values[1];
        let beta_smb_pvalue = ols_result.p_values[2];
        let beta_hml_pvalue = ols_result.p_values[3];
        let beta_mom_pvalue = ols_result.p_values[4];

        // Confidence intervals
        let alpha_conf_lower = ols_result.conf_lower[0];
        let alpha_conf_upper = ols_result.conf_upper[0];
        let beta_market_conf_lower = ols_result.conf_lower[1];
        let beta_market_conf_upper = ols_result.conf_upper[1];
        let beta_smb_conf_lower = ols_result.conf_lower[2];
        let beta_smb_conf_upper = ols_result.conf_upper[2];
        let beta_hml_conf_lower = ols_result.conf_lower[3];
        let beta_hml_conf_upper = ols_result.conf_upper[3];
        let beta_mom_conf_lower = ols_result.conf_lower[4];
        let beta_mom_conf_upper = ols_result.conf_upper[4];

        // Fit whichity
        let r_squared = ols_result.r_squared;
        let adj_r_squared = ols_result.adj_r_squared;

        // Fitted values and residuals
        let fitted_values = ols_result.fitted_values(&x_matrix);
        let residuals = ols_result.residuals(&asset_excess, &x_matrix);

        // Descriptive statistics
        let mean_asset_return = asset_returns.mean().unwrap_or(0.0);
        let mean_market_return = market_returns.mean().unwrap_or(0.0);
        let mean_smb = smb_returns.mean().unwrap_or(0.0);
        let mean_hml = hml_returns.mean().unwrap_or(0.0);
        let mean_mom = mom_returns.mean().unwrap_or(0.0);
        let asset_volatility = asset_returns.std(0.0);

        // Tracking error
        let tracking_error = residuals.std(0.0);

        // Information ratio
        let information_ratio = if tracking_error > 0.0 {
            alpha / tracking_error
        } else {
            0.0
        };

        Ok(Carhart4FactorResult {
            alpha,
            beta_market,
            beta_smb,
            beta_hml,
            beta_mom,
            alpha_se,
            beta_market_se,
            beta_smb_se,
            beta_hml_se,
            beta_mom_se,
            alpha_tstat,
            beta_market_tstat,
            beta_smb_tstat,
            beta_hml_tstat,
            beta_mom_tstat,
            alpha_pvalue,
            beta_market_pvalue,
            beta_smb_pvalue,
            beta_hml_pvalue,
            beta_mom_pvalue,
            alpha_conf_lower,
            alpha_conf_upper,
            beta_market_conf_lower,
            beta_market_conf_upper,
            beta_smb_conf_lower,
            beta_smb_conf_upper,
            beta_hml_conf_lower,
            beta_hml_conf_upper,
            beta_mom_conf_lower,
            beta_mom_conf_upper,
            r_squared,
            adj_r_squared,
            tracking_error,
            information_ratio,
            n_obs: n,
            residuals,
            fitted_values,
            risk_free_rate,
            cov_type,
            inference_type: InferenceType::StudentT,
            mean_asset_return,
            mean_market_return,
            mean_smb,
            mean_hml,
            mean_mom,
            asset_volatility,
        })
    }

    /// Estimates the model Carhart 4 Factor from a DataFrame
    ///
    /// # Example
    /// ```
    /// use frenchrs::Carhart4Factor;
    /// use greeners::{DataFrame, CovarianceType};
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("asset", vec![0.01, 0.02, -0.01, 0.03, 0.015, -0.005])
    ///     .add_column("market", vec![0.008, 0.015, -0.005, 0.025, 0.012, -0.003])
    ///     .add_column("smb", vec![0.002, -0.001, 0.003, 0.001, -0.002, 0.001])
    ///     .add_column("hml", vec![0.001, 0.002, -0.002, 0.003, 0.001, -0.001])
    ///     .add_column("mom", vec![0.003, 0.002, -0.003, 0.004, 0.001, -0.002])
    ///     .build()
    ///     .unwrap();
    ///
    /// let result = Carhart4Factor::from_dataframe(
    ///     &df,
    ///     "asset",
    ///     "market",
    ///     "smb",
    ///     "hml",
    ///     "mom",
    ///     0.0001,
    ///     CovarianceType::HC3,
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn from_dataframe(
        df: &DataFrame,
        asset_col: &str,
        market_col: &str,
        smb_col: &str,
        hml_col: &str,
        mom_col: &str,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<Carhart4FactorResult, GreenersError> {
        // Extract columns
        let asset_returns = df.get(asset_col)?;
        let market_returns = df.get(market_col)?;
        let smb_returns = df.get(smb_col)?;
        let hml_returns = df.get(hml_col)?;
        let mom_returns = df.get(mom_col)?;

        // Estimate model
        Self::fit(
            asset_returns,
            market_returns,
            smb_returns,
            hml_returns,
            mom_returns,
            risk_free_rate,
            cov_type,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_carhart_basic_fit() {
        let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
        let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
        let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001];
        let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001];
        let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001];

        let result = Carhart4Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &mom,
            0.0001,
            CovarianceType::NonRobust,
        );

        assert!(result.is_ok());
        let c4f = result.unwrap();

        assert_eq!(c4f.n_obs, 8);
        assert!(c4f.r_squared >= 0.0 && c4f.r_squared <= 1.0);
        assert!(c4f.beta_market.is_finite());
        assert!(c4f.beta_smb.is_finite());
        assert!(c4f.beta_hml.is_finite());
        assert!(c4f.beta_mom.is_finite());
    }

    #[test]
    fn test_carhart_dimension_mismatch() {
        let asset = array![0.01, 0.02];
        let market = array![0.01, 0.02, 0.03];
        let smb = array![0.001, 0.002];
        let hml = array![0.001, 0.002];
        let mom = array![0.001, 0.002];

        let result = Carhart4Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &mom,
            0.0,
            CovarianceType::NonRobust,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_carhart_predictions() {
        let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
        let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
        let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
        let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
        let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];

        let result = Carhart4Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &mom,
            0.0001,
            CovarianceType::HC3,
        )
        .unwrap();

        let new_market = array![0.01, -0.01];
        let new_smb = array![0.002, -0.001];
        let new_hml = array![0.001, 0.001];
        let new_mom = array![0.003, -0.002];

        let predictions = result.predict(&new_market, &new_smb, &new_hml, &new_mom);

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&x| x.is_finite()));
    }
}
