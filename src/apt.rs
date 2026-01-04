use greeners::{CovarianceType, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the model APT (Arbitrage Pricing Theory)
///
/// APT é um model multi-fatorial genérico:
/// R_i - R_f = α + β₁F₁ + β₂F₂ + ... + βₙFₙ + ε
///
/// Diferente of the models Fama-French, o APT not especifica which factors usar.
/// Esta implementação permite N factors arbitrários.
#[derive(Debug, Clone)]
pub struct APTResult {
    /// Alpha (intercepto)
    pub alpha: f64,

    /// Betas of the factors (β₁, β₂, ..., βₙ)
    pub betas: Array1<f64>,

    /// Erros padrão
    pub alpha_se: f64,
    pub betas_se: Array1<f64>,

    /// Statistics t
    pub alpha_tstat: f64,
    pub betas_tstat: Array1<f64>,

    /// P-values
    pub alpha_pvalue: f64,
    pub betas_pvalue: Array1<f64>,

    /// Confidence intervals
    pub alpha_conf_lower: f64,
    pub alpha_conf_upper: f64,
    pub betas_conf_lower: Array1<f64>,
    pub betas_conf_upper: Array1<f64>,

    /// Fit whichity
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,

    /// Data
    pub n_obs: usize,
    pub n_factors: usize,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub risk_free_rate: f64,
    pub cov_type: CovarianceType,
    pub inference_type: InferenceType,

    /// Nomes of the factors (opcional)
    pub factor_names: Option<Vec<String>>,
}

impl APTResult {
    pub fn is_significantly_outperforming(&self, sig: f64) -> bool {
        self.alpha > 0.0 && self.alpha_pvalue / 2.0 < sig
    }

    pub fn factor_is_significant(&self, factor_idx: usize, sig: f64) -> bool {
        if factor_idx < self.n_factors {
            self.betas_pvalue[factor_idx] < sig
        } else {
            false
        }
    }

    pub fn expected_return(&self, factor_returns: &Array1<f64>) -> f64 {
        if factor_returns.len() != self.n_factors {
            return 0.0;
        }
        self.risk_free_rate + self.alpha + self.betas.dot(factor_returns)
    }
}

impl fmt::Display for APTResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "ARBITRAGE PRICING THEORY (APT) - RESULTS")?;
        writeln!(f, "{}", "=".repeat(80))?;
        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "Número of Factors: {}", self.n_factors)?;
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

        for i in 0..self.n_factors {
            let name = if let Some(ref names) = self.factor_names {
                names.get(i).map(|s| s.as_str()).unwrap_or("Factor")
            } else {
                "Factor"
            };

            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>12.4} {:>12.4}{}",
                format!("β{} ({})", i + 1, name),
                self.betas[i],
                self.betas_se[i],
                self.betas_tstat[i],
                self.betas_pvalue[i],
                if self.betas_pvalue[i] < 0.001 {
                    " ***"
                } else if self.betas_pvalue[i] < 0.01 {
                    " **"
                } else if self.betas_pvalue[i] < 0.05 {
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
        writeln!(f, "R² Adjusted: {:.4}", self.adj_r_squared)?;
        writeln!(f, "Tracking Error: {:.4}%", self.tracking_error * 100.0)?;
        writeln!(f, "Information Ratio: {:.4}", self.information_ratio)?;
        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Implementation of the APT (Arbitrage Pricing Theory)
pub struct APT;

impl APT {
    /// Estimates the model APT with N factors
    ///
    /// # Arguments
    /// * `asset_returns` - asset returns
    /// * `factor_returns` - matriz of factors (n_obs × n_factors)
    /// * `risk_free_rate` - risk-free rate
    /// * `cov_type` - tipo of covariance
    /// * `factor_names` - names of the factors (opcional)
    ///
    /// # Example
    /// ```
    /// use frenchrs::APT;
    /// use greeners::CovarianceType;
    /// use ndarray::{array, Array2};
    ///
    /// let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    /// let factors = Array2::from_shape_vec((7, 3), vec![
    ///     0.008, 0.002, 0.001,
    ///     0.015, -0.001, 0.002,
    ///     -0.005, 0.003, -0.002,
    ///     0.025, 0.001, 0.003,
    ///     0.012, -0.002, 0.001,
    ///     -0.003, 0.001, -0.001,
    ///     0.020, 0.002, 0.002,
    /// ]).unwrap();
    ///
    /// let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::HC3, None).unwrap();
    /// assert_eq!(result.n_factors, 3);
    /// ```
    pub fn fit(
        asset_returns: &Array1<f64>,
        factor_returns: &Array2<f64>,
        risk_free_rate: f64,
        cov_type: CovarianceType,
        factor_names: Option<Vec<String>>,
    ) -> Result<APTResult, GreenersError> {
        let n_obs = asset_returns.len();
        let n_factors = factor_returns.ncols();

        if factor_returns.nrows() != n_obs {
            return Err(GreenersError::ShapeMismatch(format!(
                "Asset returns ({}) and factor returns ({}) must have same length",
                n_obs,
                factor_returns.nrows()
            )));
        }

        if n_obs < n_factors + 2 {
            return Err(GreenersError::InvalidOperation(format!(
                "Insufficient data for APT: {} obs, {} factors (need at least {})",
                n_obs,
                n_factors,
                n_factors + 2
            )));
        }

        // Returns in excess
        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);

        // Matriz of design: [1, F1, F2, ..., Fn]
        let mut x = Array2::<f64>::zeros((n_obs, n_factors + 1));
        x.column_mut(0).fill(1.0);
        for i in 0..n_factors {
            x.column_mut(i + 1).assign(&factor_returns.column(i));
        }

        let ols = OLS::fit(&asset_excess, &x, cov_type.clone())?;

        let alpha = ols.params[0];
        let betas = ols.params.slice(ndarray::s![1..]).to_owned();
        let betas_se = ols.std_errors.slice(ndarray::s![1..]).to_owned();
        let betas_tstat = ols.t_values.slice(ndarray::s![1..]).to_owned();
        let betas_pvalue = ols.p_values.slice(ndarray::s![1..]).to_owned();
        let betas_conf_lower = ols.conf_lower.slice(ndarray::s![1..]).to_owned();
        let betas_conf_upper = ols.conf_upper.slice(ndarray::s![1..]).to_owned();

        let residuals = ols.residuals(&asset_excess, &x);
        let tracking_error = residuals.std(0.0);

        Ok(APTResult {
            alpha,
            betas,
            alpha_se: ols.std_errors[0],
            betas_se,
            alpha_tstat: ols.t_values[0],
            betas_tstat,
            alpha_pvalue: ols.p_values[0],
            betas_pvalue,
            alpha_conf_lower: ols.conf_lower[0],
            alpha_conf_upper: ols.conf_upper[0],
            betas_conf_lower,
            betas_conf_upper,
            r_squared: ols.r_squared,
            adj_r_squared: ols.adj_r_squared,
            tracking_error,
            information_ratio: if tracking_error > 0.0 {
                alpha / tracking_error
            } else {
                0.0
            },
            n_obs,
            n_factors,
            residuals,
            fitted_values: ols.fitted_values(&x),
            risk_free_rate,
            cov_type,
            inference_type: InferenceType::StudentT,
            factor_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_apt_basic() {
        let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
        let factors = Array2::from_shape_vec(
            (7, 2),
            vec![
                0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003,
                0.001, 0.020, 0.002,
            ],
        )
        .unwrap();

        let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::NonRobust, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().n_factors, 2);
    }
}
