use greeners::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Analysis of IVOL (Idiosyncratic Volatility)
///
/// IVOL é a volatility específica of the asset que not é explieach
/// by the factors of systematic risks. Representa the risk diversificável.
#[derive(Debug, Clone)]
pub struct IVOLAnalysis {
    /// IVOL (volatility idiossincrática) - standard deviation of the residuals
    pub ivol: f64,

    /// IVOL annualized (assumindo periods diários)
    pub ivol_annualized_daily: f64,

    /// IVOL annualized (assumindo periods mensais)
    pub ivol_annualized_monthly: f64,

    /// Mean of the residuals (should be ~0)
    pub residual_mean: f64,

    /// Median of the residuals
    pub residual_median: f64,

    /// Minimum of the residuals
    pub residual_min: f64,

    /// Maximum of the residuals
    pub residual_max: f64,

    /// Percentil 5 of the residuals
    pub residual_p5: f64,

    /// Percentil 95 of the residuals
    pub residual_p95: f64,

    /// Skewness of the residuals
    pub residual_skewness: f64,

    /// Kurtosis of the residuals
    pub residual_kurtosis: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Residuals (for análises adicionais)
    pub residuals: Array1<f64>,
}

impl IVOLAnalysis {
    /// Creates analysis of IVOL from thes residuals of um model
    ///
    /// # Arguments
    /// * `residuals` - Residuals of the model of factors
    ///
    /// # Example
    /// ```
    /// use frenchrs::{CAPM, IVOLAnalysis};
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    /// let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    /// let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();
    ///
    /// let ivol = IVOLAnalysis::from_residuals(&capm.residuals).unwrap();
    /// assert!(ivol.ivol > 0.0);
    /// ```
    pub fn from_residuals(residuals: &Array1<f64>) -> Result<Self, GreenersError> {
        let n = residuals.len();

        if n < 3 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 3 residuals for IVOL analysis".to_string(),
            ));
        }

        // IVOL = standard deviation of the residuals
        let ivol = residuals.std(1.0); // ddof=1 for amostra

        // IVOL annualized (252 dias úteis, 12 meses)
        let ivol_annualized_daily = ivol * (252.0_f64).sqrt();
        let ivol_annualized_monthly = ivol * (12.0_f64).sqrt();

        // Descriptive statistics
        let residual_mean = residuals.mean().unwrap_or(0.0);

        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let residual_median = sorted_residuals[n / 2];
        let residual_min = sorted_residuals[0];
        let residual_max = sorted_residuals[n - 1];
        let residual_p5 = sorted_residuals[(n as f64 * 0.05) as usize];
        let residual_p95 = sorted_residuals[(n as f64 * 0.95) as usize];

        // Skewness
        let residual_skewness = if ivol > 0.0 {
            let m3: f64 = residuals
                .iter()
                .map(|&x| (x - residual_mean).powi(3))
                .sum::<f64>()
                / n as f64;
            m3 / ivol.powi(3)
        } else {
            0.0
        };

        // Kurtosis (excess kurtosis)
        let residual_kurtosis = if ivol > 0.0 {
            let m4: f64 = residuals
                .iter()
                .map(|&x| (x - residual_mean).powi(4))
                .sum::<f64>()
                / n as f64;
            (m4 / ivol.powi(4)) - 3.0
        } else {
            0.0
        };

        Ok(IVOLAnalysis {
            ivol,
            ivol_annualized_daily,
            ivol_annualized_monthly,
            residual_mean,
            residual_median,
            residual_min,
            residual_max,
            residual_p5,
            residual_p95,
            residual_skewness,
            residual_kurtosis,
            n_obs: n,
            residuals: residuals.clone(),
        })
    }

    /// Classifies the level of IVOL
    pub fn ivol_classification(&self) -> &str {
        // Classification baseada em IVOL annualized (monthly)
        if self.ivol_annualized_monthly < 0.10 {
            "Baixo"
        } else if self.ivol_annualized_monthly < 0.20 {
            "Moderado"
        } else if self.ivol_annualized_monthly < 0.30 {
            "Alto"
        } else {
            "Muito Alto"
        }
    }

    /// Tests if the residuals are normalmente distribuídos (Jarque-Bera test)
    pub fn is_residuals_normal(&self, sig: f64) -> bool {
        // Jarque-Bera statistic
        let n = self.n_obs as f64;
        let jb =
            (n / 6.0) * (self.residual_skewness.powi(2) + (self.residual_kurtosis.powi(2) / 4.0));

        // Chi-squared critical value for df=2 at 5% is ~5.99
        // At 1% is ~9.21
        let critical_value = if sig < 0.01 { 9.21 } else { 5.99 };

        jb < critical_value
    }
}

impl fmt::Display for IVOLAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "ANALYSIS OF IVOL (IDIOSYNCRATIC VOLATILITY)")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\nObbevations: {}", self.n_obs)?;
        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "VOLATILIDADE IDIOSSINCRÁTICA")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "IVOL:                    {:.4} ({:.2}%)",
            self.ivol,
            self.ivol * 100.0
        )?;
        writeln!(
            f,
            "IVOL Annualized (daily): {:.4} ({:.2}%)",
            self.ivol_annualized_daily,
            self.ivol_annualized_daily * 100.0
        )?;
        writeln!(
            f,
            "IVOL Annualized (monthly): {:.4} ({:.2}%)",
            self.ivol_annualized_monthly,
            self.ivol_annualized_monthly * 100.0
        )?;
        writeln!(
            f,
            "Classification:           {}",
            self.ivol_classification()
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "RESIDUAL STATISTICS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Mean:      {:>10.6}", self.residual_mean)?;
        writeln!(f, "Median:    {:>10.6}", self.residual_median)?;
        writeln!(f, "Minimum:     {:>10.6}", self.residual_min)?;
        writeln!(f, "Maximum:     {:>10.6}", self.residual_max)?;
        writeln!(f, "P5:         {:>10.6}", self.residual_p5)?;
        writeln!(f, "P95:        {:>10.6}", self.residual_p95)?;
        writeln!(f, "Skewness:   {:>10.4}", self.residual_skewness)?;
        writeln!(f, "Kurtosis:   {:>10.4}", self.residual_kurtosis)?;

        let is_normal = self.is_residuals_normal(0.05);
        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(
            f,
            "Distribuição Normal (Jarque-Bera): {}",
            if is_normal { "SIM" } else { "NÃO" }
        )?;

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Analysis of Tracking Error
///
/// Tracking error measures how much the return of um portfólio desvia
/// of the return of um benchmark ou model.
#[derive(Debug, Clone)]
pub struct TrackingErrorAnalysis {
    /// Tracking error (standard deviation of the residuals)
    pub tracking_error: f64,

    /// Tracking error annualized (daily)
    pub tracking_error_annualized_daily: f64,

    /// Tracking error annualized (monthly)
    pub tracking_error_annualized_monthly: f64,

    /// Information ratio (alpha / tracking error)
    pub information_ratio: f64,

    /// Alpha of the model
    pub alpha: f64,

    /// R² of the model
    pub r_squared: f64,

    /// Correlação between returns obbevados and prevthiss
    pub correlation: f64,

    /// RMSE (Root Mean Squared Error)
    pub rmse: f64,

    /// MAE (Mean Absolute Error)
    pub mae: f64,

    /// Proporção of periods with tracking error > 1%
    pub periods_above_1pct: f64,

    /// Proporção of periods with tracking error > 2%
    pub periods_above_2pct: f64,

    /// Rolling tracking error (window of 12 periods)
    pub rolling_te: Option<Array1<f64>>,

    /// Number of observations
    pub n_obs: usize,

    /// Residuals
    pub residuals: Array1<f64>,
}

impl TrackingErrorAnalysis {
    /// Creates analysis of tracking error
    ///
    /// # Arguments
    /// * `actual_returns` - Returns obbevados
    /// * `fitted_values` - Returns prevthiss pelthe model
    /// * `alpha` - Alpha of the model
    /// * `r_squared` - R² of the model
    ///
    /// # Example
    /// ```
    /// use frenchrs::{CAPM, TrackingErrorAnalysis};
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    /// let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    /// let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();
    ///
    /// let te = TrackingErrorAnalysis::new(
    ///     &asset,
    ///     &capm.fitted_values,
    ///     capm.alpha,
    ///     capm.r_squared
    /// ).unwrap();
    /// assert!(te.tracking_error > 0.0);
    /// ```
    pub fn new(
        actual_returns: &Array1<f64>,
        fitted_values: &Array1<f64>,
        alpha: f64,
        r_squared: f64,
    ) -> Result<Self, GreenersError> {
        let n = actual_returns.len();

        if fitted_values.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "actual_returns and fitted_values must have same length".to_string(),
            ));
        }

        if n < 3 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 3 observations for tracking error analysis".to_string(),
            ));
        }

        // Calculates residuals
        let residuals = actual_returns - fitted_values;

        // Tracking error = standard deviation of the residuals
        let tracking_error = residuals.std(1.0);

        // Annualizeds
        let tracking_error_annualized_daily = tracking_error * (252.0_f64).sqrt();
        let tracking_error_annualized_monthly = tracking_error * (12.0_f64).sqrt();

        // Information ratio
        let information_ratio = if tracking_error > 0.0 {
            alpha / tracking_error
        } else {
            0.0
        };

        // Correlação
        let mean_actual = actual_returns.mean().unwrap_or(0.0);
        let mean_fitted = fitted_values.mean().unwrap_or(0.0);

        let cov: f64 = actual_returns
            .iter()
            .zip(fitted_values.iter())
            .map(|(&a, &f)| (a - mean_actual) * (f - mean_fitted))
            .sum::<f64>()
            / (n - 1) as f64;

        let std_actual = actual_returns.std(1.0);
        let std_fitted = fitted_values.std(1.0);

        let correlation = if std_actual > 0.0 && std_fitted > 0.0 {
            cov / (std_actual * std_fitted)
        } else {
            0.0
        };

        // RMSE
        let rmse = (residuals.iter().map(|&x| x.powi(2)).sum::<f64>() / n as f64).sqrt();

        // MAE
        let mae = residuals.iter().map(|x| x.abs()).sum::<f64>() / n as f64;

        // Periods with TE alto
        let periods_above_1pct =
            residuals.iter().filter(|&&x| x.abs() > 0.01).count() as f64 / n as f64;
        let periods_above_2pct =
            residuals.iter().filter(|&&x| x.abs() > 0.02).count() as f64 / n as f64;

        // Rolling tracking error (if tiver data suficientes)
        let rolling_te = if n >= 12 {
            let mut rolling = Array1::<f64>::zeros(n - 11);
            for i in 0..(n - 11) {
                let window = residuals.slice(ndarray::s![i..i + 12]);
                rolling[i] = window.std(1.0);
            }
            Some(rolling)
        } else {
            None
        };

        Ok(TrackingErrorAnalysis {
            tracking_error,
            tracking_error_annualized_daily,
            tracking_error_annualized_monthly,
            information_ratio,
            alpha,
            r_squared,
            correlation,
            rmse,
            mae,
            periods_above_1pct,
            periods_above_2pct,
            rolling_te,
            n_obs: n,
            residuals,
        })
    }

    /// Classifies the level of tracking error
    pub fn te_classification(&self) -> &str {
        // Classification baseada em TE annualized (monthly)
        if self.tracking_error_annualized_monthly < 0.02 {
            "Muito Baixo (< 2%)"
        } else if self.tracking_error_annualized_monthly < 0.05 {
            "Baixo (2-5%)"
        } else if self.tracking_error_annualized_monthly < 0.10 {
            "Moderado (5-10%)"
        } else if self.tracking_error_annualized_monthly < 0.15 {
            "Alto (10-15%)"
        } else {
            "Muito Alto (> 15%)"
        }
    }

    /// Classifies the information ratio
    pub fn ir_classification(&self) -> &str {
        if self.information_ratio > 1.0 {
            "Excelente (> 1.0)"
        } else if self.information_ratio > 0.5 {
            "Bom (0.5-1.0)"
        } else if self.information_ratio > 0.0 {
            "Moderado (0-0.5)"
        } else if self.information_ratio > -0.5 {
            "Fraco (-0.5-0)"
        } else {
            "Ruim (< -0.5)"
        }
    }
}

impl fmt::Display for TrackingErrorAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "ANALYSIS OF TRACKING ERROR")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\nObbevations: {}", self.n_obs)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "TRACKING ERROR")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "TE:                       {:.4} ({:.2}%)",
            self.tracking_error,
            self.tracking_error * 100.0
        )?;
        writeln!(
            f,
            "TE Annualized (daily):    {:.4} ({:.2}%)",
            self.tracking_error_annualized_daily,
            self.tracking_error_annualized_daily * 100.0
        )?;
        writeln!(
            f,
            "TE Annualized (monthly):    {:.4} ({:.2}%)",
            self.tracking_error_annualized_monthly,
            self.tracking_error_annualized_monthly * 100.0
        )?;
        writeln!(f, "Classification:            {}", self.te_classification())?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "METRICS DE AJUSTE")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Alpha:              {:>10.6}", self.alpha)?;
        writeln!(f, "R²:                 {:>10.4}", self.r_squared)?;
        writeln!(f, "Correlação:         {:>10.4}", self.correlation)?;
        writeln!(f, "RMSE:               {:>10.6}", self.rmse)?;
        writeln!(f, "MAE:                {:>10.6}", self.mae)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "INFORMATION RATIO")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "IR:                 {:>10.4}", self.information_ratio)?;
        writeln!(f, "Classification:      {}", self.ir_classification())?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ANALYSIS OF DESVIOS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Periods with |erro| > 1%:  {:.1}%",
            self.periods_above_1pct * 100.0
        )?;
        writeln!(
            f,
            "Periods with |erro| > 2%:  {:.1}%",
            self.periods_above_2pct * 100.0
        )?;

        if let Some(ref rolling) = self.rolling_te {
            writeln!(f, "\n{}", "-".repeat(80))?;
            writeln!(f, "TRACKING ERROR ROLLING (12 periods)")?;
            writeln!(f, "{}", "-".repeat(80))?;
            writeln!(f, "Mean:      {:>10.4}", rolling.mean().unwrap_or(0.0))?;
            writeln!(
                f,
                "Minimum:     {:>10.4}",
                rolling.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            )?;
            writeln!(
                f,
                "Maximum:     {:>10.4}",
                rolling.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            )?;
            writeln!(f, "Std Dev:    {:>10.4}", rolling.std(1.0))?;
        }

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ivol_basic() {
        let residuals = array![0.001, -0.002, 0.003, -0.001, 0.002, -0.003];
        let ivol = IVOLAnalysis::from_residuals(&residuals).unwrap();

        assert!(ivol.ivol > 0.0);
        assert!(ivol.ivol_annualized_daily > ivol.ivol);
        assert_eq!(ivol.n_obs, 6);
    }

    #[test]
    fn test_tracking_error_basic() {
        let actual = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
        let fitted = array![0.009, 0.019, -0.011, 0.029, 0.014, -0.006];

        let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

        assert!(te.tracking_error > 0.0);
        assert!(te.correlation > 0.0);
        assert_eq!(te.n_obs, 6);
    }
}
