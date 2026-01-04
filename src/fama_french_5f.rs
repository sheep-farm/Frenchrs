use greeners::{CovarianceType, DataFrame, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the model Fama-French 5 Factor (2015)
///
/// Model: R_i - R_f = α + β_MKT(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + β_RMW(RMW) + β_CMA(CMA) + ε
///
/// Factors:
/// - MKT: market
/// - SMB: Small Minus Big (size)
/// - HML: High Minus Low (value)
/// - RMW: Robust Minus Weak (profitability) - NOVO
/// - CMA: Conbevative Minus Aggressive (investment) - NOVO
#[derive(Debug, Clone)]
pub struct FamaFrench5FactorResult {
    pub alpha: f64,
    pub beta_market: f64,
    pub beta_smb: f64,
    pub beta_hml: f64,
    pub beta_rmw: f64, // Profitability: profitable - unprofitable
    pub beta_cma: f64, // Investment: low investment - high investment

    pub alpha_se: f64,
    pub beta_market_se: f64,
    pub beta_smb_se: f64,
    pub beta_hml_se: f64,
    pub beta_rmw_se: f64,
    pub beta_cma_se: f64,

    pub alpha_tstat: f64,
    pub beta_market_tstat: f64,
    pub beta_smb_tstat: f64,
    pub beta_hml_tstat: f64,
    pub beta_rmw_tstat: f64,
    pub beta_cma_tstat: f64,

    pub alpha_pvalue: f64,
    pub beta_market_pvalue: f64,
    pub beta_smb_pvalue: f64,
    pub beta_hml_pvalue: f64,
    pub beta_rmw_pvalue: f64,
    pub beta_cma_pvalue: f64,

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

    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub n_obs: usize,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub risk_free_rate: f64,
    pub cov_type: CovarianceType,
    pub inference_type: InferenceType,

    pub mean_asset_return: f64,
    pub mean_market_return: f64,
    pub mean_smb: f64,
    pub mean_hml: f64,
    pub mean_rmw: f64,
    pub mean_cma: f64,
    pub asset_volatility: f64,
}

impl FamaFrench5FactorResult {
    pub fn is_significantly_outperforming(&self, sig: f64) -> bool {
        self.alpha > 0.0 && self.alpha_pvalue / 2.0 < sig
    }

    pub fn is_rmw_significant(&self, sig: f64) -> bool {
        self.beta_rmw_pvalue < sig
    }

    pub fn is_cma_significant(&self, sig: f64) -> bool {
        self.beta_cma_pvalue < sig
    }

    pub fn profitability_classification(&self) -> &str {
        if !self.is_rmw_significant(0.05) {
            "Neutral (RMW not significant)"
        } else if self.beta_rmw > 0.3 {
            "Strongly Profitable"
        } else if self.beta_rmw > 0.0 {
            "Profitable"
        } else if self.beta_rmw > -0.3 {
            "Weak"
        } else {
            "Strongly Weak"
        }
    }

    pub fn investment_classification(&self) -> &str {
        if !self.is_cma_significant(0.05) {
            "Neutral (CMA not significant)"
        } else if self.beta_cma > 0.3 {
            "Strongly Conservative"
        } else if self.beta_cma > 0.0 {
            "Conservative"
        } else if self.beta_cma > -0.3 {
            "Aggressive"
        } else {
            "Strongly Aggressive"
        }
    }

    pub fn expected_return(
        &self,
        exp_mkt: f64,
        exp_smb: f64,
        exp_hml: f64,
        exp_rmw: f64,
        exp_cma: f64,
    ) -> f64 {
        self.risk_free_rate
            + self.beta_market * (exp_mkt - self.risk_free_rate)
            + self.beta_smb * exp_smb
            + self.beta_hml * exp_hml
            + self.beta_rmw * exp_rmw
            + self.beta_cma * exp_cma
    }

    pub fn factor_contributions(&self) -> (f64, f64, f64, f64, f64) {
        (
            self.beta_market * (self.mean_market_return - self.risk_free_rate),
            self.beta_smb * self.mean_smb,
            self.beta_hml * self.mean_hml,
            self.beta_rmw * self.mean_rmw,
            self.beta_cma * self.mean_cma,
        )
    }
}

impl fmt::Display for FamaFrench5FactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "FAMA-FRENCH 5 FACTOR MODEL (2015) - RESULTS")?;
        writeln!(f, "{}", "=".repeat(80))?;
        writeln!(
            f,
            "\nMODEL: R_i - R_f = α + β_MKT(R_m - R_f) + β_SMB(SMB) + β_HML(HML)"
        )?;
        writeln!(f, "                  + β_RMW(RMW) + β_CMA(CMA) + ε")?;
        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "Risk-Free Rate: {:.4}%", self.risk_free_rate * 100.0)?;
        writeln!(f, "Covariesnce Type: {:?}", self.cov_type)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTIMATED PARAMETERS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Parameter", "Coef.", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        for (name, coef, se, tstat, pval) in [
            (
                "Alpha (α)",
                self.alpha,
                self.alpha_se,
                self.alpha_tstat,
                self.alpha_pvalue,
            ),
            (
                "Beta MKT",
                self.beta_market,
                self.beta_market_se,
                self.beta_market_tstat,
                self.beta_market_pvalue,
            ),
            (
                "Beta SMB",
                self.beta_smb,
                self.beta_smb_se,
                self.beta_smb_tstat,
                self.beta_smb_pvalue,
            ),
            (
                "Beta HML",
                self.beta_hml,
                self.beta_hml_se,
                self.beta_hml_tstat,
                self.beta_hml_pvalue,
            ),
            (
                "Beta RMW",
                self.beta_rmw,
                self.beta_rmw_se,
                self.beta_rmw_tstat,
                self.beta_rmw_pvalue,
            ),
            (
                "Beta CMA",
                self.beta_cma,
                self.beta_cma_se,
                self.beta_cma_tstat,
                self.beta_cma_pvalue,
            ),
        ] {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
                name,
                coef,
                se,
                tstat,
                pval,
                if pval < 0.001 {
                    " ***"
                } else if pval < 0.01 {
                    " **"
                } else if pval < 0.05 {
                    " *"
                } else {
                    ""
                }
            )?;
        }

        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "R²: {:.4} ({:.2}%)",
            self.r_squared,
            self.r_squared * 100.0
        )?;
        writeln!(f, "Tracking Error: {:.4}%", self.tracking_error * 100.0)?;
        writeln!(f, "Information Ratio: {:.4}", self.information_ratio)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CONTRIBUIÇÕES DOS FATORES")?;
        writeln!(f, "{}", "-".repeat(80))?;
        let (mkt, smb, hml, rmw, cma) = self.factor_contributions();
        let total = mkt + smb + hml + rmw + cma;
        writeln!(
            f,
            "Market: {:>8.4}% ({:>5.1}%)",
            mkt * 100.0,
            if total != 0.0 {
                mkt / total * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "SMB:     {:>8.4}% ({:>5.1}%)",
            smb * 100.0,
            if total != 0.0 {
                smb / total * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "HML:     {:>8.4}% ({:>5.1}%)",
            hml * 100.0,
            if total != 0.0 {
                hml / total * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "RMW:     {:>8.4}% ({:>5.1}%)",
            rmw * 100.0,
            if total != 0.0 {
                rmw / total * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "CMA:     {:>8.4}% ({:>5.1}%)",
            cma * 100.0,
            if total != 0.0 {
                cma / total * 100.0
            } else {
                0.0
            }
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CLASSIFICATIONS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Profitability (RMW): {}",
            self.profitability_classification()
        )?;
        writeln!(f, "Investment (CMA):  {}", self.investment_classification())?;

        writeln!(f, "\n{}", "=".repeat(80))?;
        Ok(())
    }
}

pub struct FamaFrench5Factor;

impl FamaFrench5Factor {
    /// Estimates the model Fama-French 5 Factor
    ///
    /// # Example
    /// ```
    /// use frenchrs::FamaFrench5Factor;
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    /// let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    /// let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    /// let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];
    /// let rmw = array![0.002, 0.001, -0.001, 0.002, 0.001, -0.001, 0.002];
    /// let cma = array![0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001];
    ///
    /// let result = FamaFrench5Factor::fit(
    ///     &asset, &market, &smb, &hml, &rmw, &cma,
    ///     0.0001, CovarianceType::HC3,
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
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<FamaFrench5FactorResult, GreenersError> {
        let n = asset_returns.len();
        if market_returns.len() != n
            || smb_returns.len() != n
            || hml_returns.len() != n
            || rmw_returns.len() != n
            || cma_returns.len() != n
        {
            return Err(GreenersError::ShapeMismatch(
                "All arrays must have the same length".to_string(),
            ));
        }

        if n < 7 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for FF5: {} observations (minimum 7 required)",
                n
            )));
        }

        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);
        let market_excess: Array1<f64> = market_returns.mapv(|r| r - risk_free_rate);

        let mut x = Array2::<f64>::zeros((n, 6));
        x.column_mut(0).fill(1.0);
        x.column_mut(1).assign(&market_excess);
        x.column_mut(2).assign(smb_returns);
        x.column_mut(3).assign(hml_returns);
        x.column_mut(4).assign(rmw_returns);
        x.column_mut(5).assign(cma_returns);

        let ols = OLS::fit(&asset_excess, &x, cov_type.clone())?;

        Ok(FamaFrench5FactorResult {
            alpha: ols.params[0],
            beta_market: ols.params[1],
            beta_smb: ols.params[2],
            beta_hml: ols.params[3],
            beta_rmw: ols.params[4],
            beta_cma: ols.params[5],
            alpha_se: ols.std_errors[0],
            beta_market_se: ols.std_errors[1],
            beta_smb_se: ols.std_errors[2],
            beta_hml_se: ols.std_errors[3],
            beta_rmw_se: ols.std_errors[4],
            beta_cma_se: ols.std_errors[5],
            alpha_tstat: ols.t_values[0],
            beta_market_tstat: ols.t_values[1],
            beta_smb_tstat: ols.t_values[2],
            beta_hml_tstat: ols.t_values[3],
            beta_rmw_tstat: ols.t_values[4],
            beta_cma_tstat: ols.t_values[5],
            alpha_pvalue: ols.p_values[0],
            beta_market_pvalue: ols.p_values[1],
            beta_smb_pvalue: ols.p_values[2],
            beta_hml_pvalue: ols.p_values[3],
            beta_rmw_pvalue: ols.p_values[4],
            beta_cma_pvalue: ols.p_values[5],
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
            r_squared: ols.r_squared,
            adj_r_squared: ols.adj_r_squared,
            tracking_error: ols.residuals(&asset_excess, &x).std(0.0),
            information_ratio: {
                let te = ols.residuals(&asset_excess, &x).std(0.0);
                if te > 0.0 { ols.params[0] / te } else { 0.0 }
            },
            n_obs: n,
            residuals: ols.residuals(&asset_excess, &x),
            fitted_values: ols.fitted_values(&x),
            risk_free_rate,
            cov_type,
            inference_type: InferenceType::StudentT,
            mean_asset_return: asset_returns.mean().unwrap_or(0.0),
            mean_market_return: market_returns.mean().unwrap_or(0.0),
            mean_smb: smb_returns.mean().unwrap_or(0.0),
            mean_hml: hml_returns.mean().unwrap_or(0.0),
            mean_rmw: rmw_returns.mean().unwrap_or(0.0),
            mean_cma: cma_returns.mean().unwrap_or(0.0),
            asset_volatility: asset_returns.std(0.0),
        })
    }

    /// Estimates FF5 from DataFrame
    #[allow(clippy::too_many_arguments)]
    pub fn from_dataframe(
        df: &DataFrame,
        asset_col: &str,
        market_col: &str,
        smb_col: &str,
        hml_col: &str,
        rmw_col: &str,
        cma_col: &str,
        risk_free_rate: f64,
        cov_type: CovarianceType,
    ) -> Result<FamaFrench5FactorResult, GreenersError> {
        Self::fit(
            df.get(asset_col)?,
            df.get(market_col)?,
            df.get(smb_col)?,
            df.get(hml_col)?,
            df.get(rmw_col)?,
            df.get(cma_col)?,
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
    fn test_ff5_basic() {
        let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
        let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
        let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001];
        let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001];
        let rmw = array![0.002, 0.001, -0.001, 0.002, 0.001, -0.001, 0.002, 0.001];
        let cma = array![0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, -0.001];

        let result = FamaFrench5Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &rmw,
            &cma,
            0.0001,
            CovarianceType::NonRobust,
        );

        assert!(result.is_ok());
        let ff5 = result.unwrap();
        assert_eq!(ff5.n_obs, 8);
        assert!(ff5.r_squared >= 0.0 && ff5.r_squared <= 1.0);
    }
}
