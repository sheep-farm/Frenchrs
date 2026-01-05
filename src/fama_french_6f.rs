use greeners::{CovarianceType, DataFrame, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the model Fama-French 6 Factor
///
/// FF6 = FF5 + UMD (Up Minus Down - momentum)
/// R_i - R_f = α + β_MKT(R_m - R_f) + β_SMB·SMB + β_HML·HML + β_RMW·RMW + β_CMA·CMA + β_UMD·UMD + ε
///
/// Where:
/// - MKT: Market premium (excess market return)
/// - SMB: Small Minus Big (size factor)
/// - HML: High Minus Low (value factor)
/// - RMW: Robust Minus Weak (profitability factor)
/// - CMA: Conservative Minus Aggressive (investment factor)
/// - UMD: Up Minus Down (momentum factor)
#[derive(Debug, Clone)]
pub struct FamaFrench6FactorResult {
    /// Alpha (intercepto)
    pub alpha: f64,

    /// Betas of the factors
    pub beta_market: f64,
    pub beta_smb: f64,
    pub beta_hml: f64,
    pub beta_rmw: f64,
    pub beta_cma: f64,
    pub beta_umd: f64,

    /// Erros padrão
    pub alpha_se: f64,
    pub beta_market_se: f64,
    pub beta_smb_se: f64,
    pub beta_hml_se: f64,
    pub beta_rmw_se: f64,
    pub beta_cma_se: f64,
    pub beta_umd_se: f64,

    /// Statistics t
    pub alpha_tstat: f64,
    pub beta_market_tstat: f64,
    pub beta_smb_tstat: f64,
    pub beta_hml_tstat: f64,
    pub beta_rmw_tstat: f64,
    pub beta_cma_tstat: f64,
    pub beta_umd_tstat: f64,

    /// P-values
    pub alpha_pvalue: f64,
    pub beta_market_pvalue: f64,
    pub beta_smb_pvalue: f64,
    pub beta_hml_pvalue: f64,
    pub beta_rmw_pvalue: f64,
    pub beta_cma_pvalue: f64,
    pub beta_umd_pvalue: f64,

    /// Confidence intervals
    pub alpha_conf_lower: f64,
    pub alpha_conf_upper: f64,
    pub beta_market_conf_lower: f64,
    pub beta_market_conf_upper: f64,
    pub beta_smb_conf_lower: f64,
    pub beta_smb_conf_upper: f64,
    pub beta_hml_conf_lower: f64,
    pub beta_hml_conf_upper: f64,
    pub beta_rmw_conf_lower: f64,
    pub beta_rmw_conf_upper: f64,
    pub beta_cma_conf_lower: f64,
    pub beta_cma_conf_upper: f64,
    pub beta_umd_conf_lower: f64,
    pub beta_umd_conf_upper: f64,

    /// Fit whichity
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,

    /// Data
    pub n_obs: usize,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub risk_free_rate: f64,
    pub cov_type: CovarianceType,
    pub inference_type: InferenceType,
}

impl FamaFrench6FactorResult {
    /// Tests if the asset is with performance upper ao esperado pelthe model
    pub fn is_significantly_outperforming(&self, sig: f64) -> bool {
        self.alpha > 0.0 && self.alpha_pvalue / 2.0 < sig
    }

    /// Tests if the asset is with performance lower ao esperado pelthe model
    pub fn is_significantly_underperforming(&self, sig: f64) -> bool {
        self.alpha < 0.0 && self.alpha_pvalue / 2.0 < sig
    }

    /// Test of significance for o factor of market
    pub fn is_market_significant(&self, sig: f64) -> bool {
        self.beta_market_pvalue < sig
    }

    /// Test of significance for SMB
    pub fn is_smb_significant(&self, sig: f64) -> bool {
        self.beta_smb_pvalue < sig
    }

    /// Test of significance for HML
    pub fn is_hml_significant(&self, sig: f64) -> bool {
        self.beta_hml_pvalue < sig
    }

    /// Test of significance for RMW
    pub fn is_rmw_significant(&self, sig: f64) -> bool {
        self.beta_rmw_pvalue < sig
    }

    /// Test of significance for CMA
    pub fn is_cma_significant(&self, sig: f64) -> bool {
        self.beta_cma_pvalue < sig
    }

    /// Test of significance for UMD
    pub fn is_umd_significant(&self, sig: f64) -> bool {
        self.beta_umd_pvalue < sig
    }

    /// Classification by size based on SMB
    pub fn size_classification(&self) -> &str {
        if !self.is_smb_significant(0.05) {
            "Neutral"
        } else if self.beta_smb > 0.3 {
            "Small Cap"
        } else if self.beta_smb < -0.3 {
            "Large Cap"
        } else {
            "Mid Cap"
        }
    }

    /// Classification by value based on HML
    pub fn value_classification(&self) -> &str {
        if !self.is_hml_significant(0.05) {
            "Neutral"
        } else if self.beta_hml > 0.3 {
            "Value"
        } else if self.beta_hml < -0.3 {
            "Growth"
        } else {
            "Blend"
        }
    }

    /// Classification by profitability based on RMW
    pub fn profitability_classification(&self) -> &str {
        if !self.is_rmw_significant(0.05) {
            "Neutral"
        } else if self.beta_rmw > 0.2 {
            "High Profitability"
        } else if self.beta_rmw < -0.2 {
            "Low Profitability"
        } else {
            "Average Profitability"
        }
    }

    /// Classification by investment based on CMA
    pub fn investment_classification(&self) -> &str {
        if !self.is_cma_significant(0.05) {
            "Neutral"
        } else if self.beta_cma > 0.2 {
            "Conservative"
        } else if self.beta_cma < -0.2 {
            "Aggressive"
        } else {
            "Moderate"
        }
    }

    /// Classification by momentum based on UMD
    pub fn momentum_classification(&self) -> &str {
        if !self.is_umd_significant(0.05) {
            "Neutral"
        } else if self.beta_umd > 0.3 {
            "High Momentum"
        } else if self.beta_umd < -0.3 {
            "Low Momentum"
        } else {
            "Moderate Momentum"
        }
    }

    /// Overall performance classification
    pub fn performance_classification(&self) -> &str {
        if self.is_significantly_outperforming(0.05) {
            "Superior"
        } else if self.is_significantly_underperforming(0.05) {
            "Inferior"
        } else {
            "In line with model"
        }
    }

    /// Expected return given the risk factors
    pub fn expected_return(
        &self,
        market_premium: f64,
        smb: f64,
        hml: f64,
        rmw: f64,
        cma: f64,
        umd: f64,
    ) -> f64 {
        self.risk_free_rate
            + self.alpha
            + self.beta_market * market_premium
            + self.beta_smb * smb
            + self.beta_hml * hml
            + self.beta_rmw * rmw
            + self.beta_cma * cma
            + self.beta_umd * umd
    }

    /// Predições for multiple periods
    pub fn predict(
        &self,
        market_premium: &Array1<f64>,
        smb: &Array1<f64>,
        hml: &Array1<f64>,
        rmw: &Array1<f64>,
        cma: &Array1<f64>,
        umd: &Array1<f64>,
    ) -> Array1<f64> {
        let n = market_premium.len();
        let mut predictions = Array1::<f64>::zeros(n);

        for i in 0..n {
            predictions[i] =
                self.expected_return(market_premium[i], smb[i], hml[i], rmw[i], cma[i], umd[i]);
        }

        predictions
    }

    /// Contribuição of each factor for the return esperado
    pub fn factor_contributions(&self) -> (f64, f64, f64, f64, f64, f64) {
        // Usando mean histórica of the factors (assumindo normalized)
        let market_contrib = self.beta_market;
        let smb_contrib = self.beta_smb;
        let hml_contrib = self.beta_hml;
        let rmw_contrib = self.beta_rmw;
        let cma_contrib = self.beta_cma;
        let umd_contrib = self.beta_umd;

        (
            market_contrib,
            smb_contrib,
            hml_contrib,
            rmw_contrib,
            cma_contrib,
            umd_contrib,
        )
    }
}

impl fmt::Display for FamaFrench6FactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "FAMA-FRENCH 6 FACTOR MODEL - RESULTS")?;
        writeln!(f, "{}", "=".repeat(80))?;
        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "Risk-Free Rate: {:.4}%", self.risk_free_rate * 100.0)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTIMATED PARAMETERS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Parameter", "Coef.", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
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
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_MKT (Market)",
            self.beta_market,
            self.beta_market_se,
            self.beta_market_tstat,
            self.beta_market_pvalue,
            if self.beta_market_pvalue < 0.001 {
                " ***"
            } else if self.beta_market_pvalue < 0.01 {
                " **"
            } else if self.beta_market_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_SMB (Size)",
            self.beta_smb,
            self.beta_smb_se,
            self.beta_smb_tstat,
            self.beta_smb_pvalue,
            if self.beta_smb_pvalue < 0.001 {
                " ***"
            } else if self.beta_smb_pvalue < 0.01 {
                " **"
            } else if self.beta_smb_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_HML (Value)",
            self.beta_hml,
            self.beta_hml_se,
            self.beta_hml_tstat,
            self.beta_hml_pvalue,
            if self.beta_hml_pvalue < 0.001 {
                " ***"
            } else if self.beta_hml_pvalue < 0.01 {
                " **"
            } else if self.beta_hml_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_RMW (Rentab.)",
            self.beta_rmw,
            self.beta_rmw_se,
            self.beta_rmw_tstat,
            self.beta_rmw_pvalue,
            if self.beta_rmw_pvalue < 0.001 {
                " ***"
            } else if self.beta_rmw_pvalue < 0.01 {
                " **"
            } else if self.beta_rmw_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_CMA (Invest.)",
            self.beta_cma,
            self.beta_cma_se,
            self.beta_cma_tstat,
            self.beta_cma_pvalue,
            if self.beta_cma_pvalue < 0.001 {
                " ***"
            } else if self.beta_cma_pvalue < 0.01 {
                " **"
            } else if self.beta_cma_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
            "β_UMD (Momentum)",
            self.beta_umd,
            self.beta_umd_se,
            self.beta_umd_tstat,
            self.beta_umd_pvalue,
            if self.beta_umd_pvalue < 0.001 {
                " ***"
            } else if self.beta_umd_pvalue < 0.01 {
                " **"
            } else if self.beta_umd_pvalue < 0.05 {
                " *"
            } else {
                ""
            }
        )?;

        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "R²: {:.4} ({:.2}%)",
            self.r_squared,
            self.r_squared * 100.0
        )?;
        writeln!(f, "R² Adjusted: {:.4}", self.adj_r_squared)?;
        writeln!(f, "Tracking Error: {:.4}%", self.tracking_error * 100.0)?;
        writeln!(f, "Information Ratio: {:.4}", self.information_ratio)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CLASSIFICATIONS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Performance: {}", self.performance_classification())?;
        writeln!(f, "Size: {}", self.size_classification())?;
        writeln!(f, "Value: {}", self.value_classification())?;
        writeln!(f, "Profitability: {}", self.profitability_classification())?;
        writeln!(f, "Investment: {}", self.investment_classification())?;
        writeln!(f, "Momentum: {}", self.momentum_classification())?;

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Implementation of the model Fama-French 6 Factor
pub struct FamaFrench6Factor;

impl FamaFrench6Factor {
    /// Estimates the model FF6
    ///
    /// # Arguments
    /// * `asset_returns` - asset returns
    /// * `market_returns` - returns of market
    /// * `smb_returns` - returns of the factor SMB (Small Minus Big)
    /// * `hml_returns` - returns of the factor HML (High Minus Low)
    /// * `rmw_returns` - returns of the factor RMW (Robust Minus Weak)
    /// * `cma_returns` - returns of the factor CMA (Conbevative Minus Aggressive)
    /// * `umd_returns` - returns of the factor UMD (Up Minus Down - momentum)
    /// * `risk_free_rate` - risk-free rate
    /// * `cov_type` - covariance matrix type
    ///
    /// # Example
    /// ```
    /// use frenchrs::FamaFrench6Factor;
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01];
    /// let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005];
    /// let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002];
    /// let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001];
    /// let rmw = array![0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001];
    /// let cma = array![0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001];
    /// let umd = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002];
    ///
    /// let result = FamaFrench6Factor::fit(
    ///     &asset, &market, &smb, &hml, &rmw, &cma, &umd,
    ///     0.0001, CovarianceType::HC3
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        asset_returns: &Array1<f64>,
        market_returns: &Array1<f64>,
        smb_returns: &Array1<f64>,
        hml_returns: &Array1<f64>,
        rmw_returns: &Array1<f64>,
        cma_returns: &Array1<f64>,
        umd_returns: &Array1<f64>,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<FamaFrench6FactorResult, GreenersError> {
        let n = asset_returns.len();

        if market_returns.len() != n
            || smb_returns.len() != n
            || hml_returns.len() != n
            || rmw_returns.len() != n
            || cma_returns.len() != n
            || umd_returns.len() != n
        {
            return Err(GreenersError::ShapeMismatch(format!(
                "All input arrays must have the same length: asset={}, market={}, smb={}, hml={}, rmw={}, cma={}, umd={}",
                n,
                market_returns.len(),
                smb_returns.len(),
                hml_returns.len(),
                rmw_returns.len(),
                cma_returns.len(),
                umd_returns.len()
            )));
        }

        if n < 10 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for FF6: {} observations (need at least 10)",
                n
            )));
        }

        // Returns in excess
        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);
        let market_excess: Array1<f64> = market_returns.mapv(|r| r - risk_free_rate);

        // Matriz of design: [1, MKT, SMB, HML, RMW, CMA, UMD]
        let mut x = Array2::<f64>::zeros((n, 7));
        x.column_mut(0).fill(1.0);
        x.column_mut(1).assign(&market_excess);
        x.column_mut(2).assign(smb_returns);
        x.column_mut(3).assign(hml_returns);
        x.column_mut(4).assign(rmw_returns);
        x.column_mut(5).assign(cma_returns);
        x.column_mut(6).assign(umd_returns);

        let ols = OLS::fit(&asset_excess, &x, cov_type.clone())?;

        let residuals = ols.residuals(&asset_excess, &x);
        let tracking_error = residuals.std(0.0);

        Ok(FamaFrench6FactorResult {
            alpha: ols.params[0],
            beta_market: ols.params[1],
            beta_smb: ols.params[2],
            beta_hml: ols.params[3],
            beta_rmw: ols.params[4],
            beta_cma: ols.params[5],
            beta_umd: ols.params[6],
            alpha_se: ols.std_errors[0],
            beta_market_se: ols.std_errors[1],
            beta_smb_se: ols.std_errors[2],
            beta_hml_se: ols.std_errors[3],
            beta_rmw_se: ols.std_errors[4],
            beta_cma_se: ols.std_errors[5],
            beta_umd_se: ols.std_errors[6],
            alpha_tstat: ols.t_values[0],
            beta_market_tstat: ols.t_values[1],
            beta_smb_tstat: ols.t_values[2],
            beta_hml_tstat: ols.t_values[3],
            beta_rmw_tstat: ols.t_values[4],
            beta_cma_tstat: ols.t_values[5],
            beta_umd_tstat: ols.t_values[6],
            alpha_pvalue: ols.p_values[0],
            beta_market_pvalue: ols.p_values[1],
            beta_smb_pvalue: ols.p_values[2],
            beta_hml_pvalue: ols.p_values[3],
            beta_rmw_pvalue: ols.p_values[4],
            beta_cma_pvalue: ols.p_values[5],
            beta_umd_pvalue: ols.p_values[6],
            alpha_conf_lower: ols.conf_lower[0],
            alpha_conf_upper: ols.conf_upper[0],
            beta_market_conf_lower: ols.conf_lower[1],
            beta_market_conf_upper: ols.conf_upper[1],
            beta_smb_conf_lower: ols.conf_lower[2],
            beta_smb_conf_upper: ols.conf_upper[2],
            beta_hml_conf_lower: ols.conf_lower[3],
            beta_hml_conf_upper: ols.conf_upper[3],
            beta_rmw_conf_lower: ols.conf_lower[4],
            beta_rmw_conf_upper: ols.conf_upper[4],
            beta_cma_conf_lower: ols.conf_lower[5],
            beta_cma_conf_upper: ols.conf_upper[5],
            beta_umd_conf_lower: ols.conf_lower[6],
            beta_umd_conf_upper: ols.conf_upper[6],
            r_squared: ols.r_squared,
            adj_r_squared: ols.adj_r_squared,
            tracking_error,
            information_ratio: if tracking_error > 0.0 {
                ols.params[0] / tracking_error
            } else {
                0.0
            },
            n_obs: n,
            residuals,
            fitted_values: ols.fitted_values(&x),
            risk_free_rate,
            cov_type,
            inference_type: InferenceType::StudentT,
        })
    }

    /// Estimates the model from a DataFrame
    #[allow(clippy::too_many_arguments)]
    pub fn from_dataframe(
        df: &DataFrame,
        asset_col: &str,
        market_col: &str,
        smb_col: &str,
        hml_col: &str,
        rmw_col: &str,
        cma_col: &str,
        umd_col: &str,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<FamaFrench6FactorResult, GreenersError> {
        let asset = df.get(asset_col)?;
        let market = df.get(market_col)?;
        let smb = df.get(smb_col)?;
        let hml = df.get(hml_col)?;
        let rmw = df.get(rmw_col)?;
        let cma = df.get(cma_col)?;
        let umd = df.get(umd_col)?;

        Self::fit(
            asset,
            market,
            smb,
            hml,
            rmw,
            cma,
            umd,
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
    fn test_ff6_basic() {
        let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01];
        let market =
            array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005];
        let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002];
        let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001];
        let rmw = array![0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001];
        let cma = array![0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001];
        let umd = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002];

        let result = FamaFrench6Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &rmw,
            &cma,
            &umd,
            0.0001,
            CovarianceType::NonRobust,
        );

        assert!(result.is_ok());
        let ff6 = result.unwrap();
        assert_eq!(ff6.n_obs, 10);
        assert!(ff6.r_squared >= 0.0 && ff6.r_squared <= 1.0);
    }
}
