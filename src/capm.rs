use greeners::{CovarianceType, DataFrame, GreenersError, InferenceType, OLS};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fmt;

/// Resultado da estimação do Capital Asset Pricing Model (CAPM)
///
/// O CAPM relaciona o retorno de um ativo ao retorno do mercado:
/// R_i - R_f = α + β(R_m - R_f) + ε
///
/// onde:
/// - R_i: retorno do ativo
/// - R_f: taxa livre de risco
/// - R_m: retorno do mercado
/// - α (alpha): excesso de retorno não explicado pelo mercado (Jensen's alpha)
/// - β (beta): sensibilidade do ativo ao risco de mercado (risco sistemático)
/// - ε: erro idiossincrático
#[derive(Debug, Clone)]
pub struct CAPMResult {
    /// Intercepto (α) - Jensen's alpha
    ///
    /// Representa o excesso de retorno não explicado pelo mercado.
    /// α > 0: ativo supera o mercado (outperformance)
    /// α < 0: ativo fica atrás do mercado (underperformance)
    /// α = 0: ativo segue exatamente o CAPM
    pub alpha: f64,

    /// Sensibilidade ao mercado (β) - risco sistemático
    ///
    /// Mede quanto o ativo varia para cada 1% de variação no mercado.
    /// β > 1: ativo é mais volátil que o mercado (agressivo)
    /// β = 1: ativo varia igual ao mercado
    /// β < 1: ativo é menos volátil que o mercado (defensivo)
    /// β < 0: ativo se move inversamente ao mercado
    pub beta: f64,

    /// Erro padrão do α
    pub alpha_se: f64,

    /// Erro padrão do β
    pub beta_se: f64,

    /// Estatística t para α
    pub alpha_tstat: f64,

    /// Estatística t para β
    pub beta_tstat: f64,

    /// p-value para teste H0: α = 0
    pub alpha_pvalue: f64,

    /// p-value para teste H0: β = 0
    pub beta_pvalue: f64,

    /// Intervalo de confiança inferior para α (95%)
    pub alpha_conf_lower: f64,

    /// Intervalo de confiança superior para α (95%)
    pub alpha_conf_upper: f64,

    /// Intervalo de confiança inferior para β (95%)
    pub beta_conf_lower: f64,

    /// Intervalo de confiança superior para β (95%)
    pub beta_conf_upper: f64,

    /// R² - proporção da variância explicada pelo mercado
    pub r_squared: f64,

    /// R² ajustado por graus de liberdade
    pub adj_r_squared: f64,

    /// Razão de Sharpe do ativo
    ///
    /// Sharpe = (E[R_i] - R_f) / σ_i
    pub sharpe_ratio: f64,

    /// Razão de Sharpe do mercado
    ///
    /// Sharpe_market = (E[R_m] - R_f) / σ_m
    pub market_sharpe: f64,

    /// Razão de Treynor
    ///
    /// Treynor = (E[R_i] - R_f) / β
    /// Mede retorno por unidade de risco sistemático
    pub treynor_ratio: f64,

    /// Information Ratio
    ///
    /// IR = α / σ(ε)
    /// Mede retorno anormal por unidade de risco idiossincrático
    pub information_ratio: f64,

    /// Tracking Error (volatilidade dos resíduos)
    ///
    /// TE = σ(ε) = std(R_i - (α + β·R_m))
    pub tracking_error: f64,

    /// Número de observações
    pub n_obs: usize,

    /// Resíduos (ε) - risco idiossincrático
    pub residuals: Array1<f64>,

    /// Valores ajustados (α + β·(R_m - R_f))
    pub fitted_values: Array1<f64>,

    /// Taxa livre de risco utilizada
    pub risk_free_rate: f64,

    /// Tipo de covariância utilizado
    pub cov_type: CovarianceType,

    /// Tipo de inferência (t ou normal)
    pub inference_type: InferenceType,

    /// Retorno médio do ativo
    pub mean_asset_return: f64,

    /// Retorno médio do mercado
    pub mean_market_return: f64,

    /// Volatilidade do ativo (desvio padrão)
    pub asset_volatility: f64,

    /// Volatilidade do mercado (desvio padrão)
    pub market_volatility: f64,

    /// Variância sistemática (β² × σ²_m)
    pub systematic_variance: f64,

    /// Variância idiossincrática (σ²_ε)
    pub idiosyncratic_variance: f64,
}

impl CAPMResult {
    /// Testa se o ativo está significativamente superando o mercado
    ///
    /// H0: α ≤ 0 vs H1: α > 0 (teste unilateral)
    ///
    /// # Arguments
    /// * `significance_level` - nível de significância (ex: 0.05 para 5%)
    ///
    /// # Returns
    /// `true` se H0 é rejeitada (α é significativamente positivo)
    pub fn is_significantly_outperforming(&self, significance_level: f64) -> bool {
        // Teste unilateral: p-value / 2 se α > 0
        if self.alpha > 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Testa se o ativo está significativamente ficando atrás do mercado
    ///
    /// H0: α ≥ 0 vs H1: α < 0 (teste unilateral)
    ///
    /// # Arguments
    /// * `significance_level` - nível de significância (ex: 0.05 para 5%)
    ///
    /// # Returns
    /// `true` se H0 é rejeitada (α é significativamente negativo)
    pub fn is_significantly_underperforming(&self, significance_level: f64) -> bool {
        if self.alpha < 0.0 {
            self.alpha_pvalue / 2.0 < significance_level
        } else {
            false
        }
    }

    /// Testa se β é significativamente diferente de 1
    ///
    /// H0: β = 1 vs H1: β ≠ 1
    ///
    /// # Arguments
    /// * `significance_level` - nível de significância (ex: 0.05 para 5%)
    ///
    /// # Returns
    /// `true` se H0 é rejeitada (β é significativamente diferente de 1)
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

    /// Classifica o ativo quanto ao risco sistemático
    pub fn risk_classification(&self) -> &str {
        if self.beta > 1.2 {
            "Muito Agressivo"
        } else if self.beta > 1.0 {
            "Agressivo"
        } else if self.beta > 0.8 {
            "Neutro"
        } else if self.beta > 0.0 {
            "Defensivo"
        } else {
            "Hedge (beta negativo)"
        }
    }

    /// Classifica o desempenho do ativo quanto ao alpha
    pub fn performance_classification(&self) -> &str {
        let significance = 0.05;

        if self.is_significantly_outperforming(significance) {
            "Outperformance Significativa"
        } else if self.is_significantly_underperforming(significance) {
            "Underperformance Significativa"
        } else if self.alpha.abs() < 0.0001 {
            "Desempenho Neutro"
        } else if self.alpha > 0.0 {
            "Outperformance Não Significativa"
        } else {
            "Underperformance Não Significativa"
        }
    }

    /// Calcula o retorno esperado dado um retorno de mercado esperado
    ///
    /// E[R_i] = R_f + β·(E[R_m] - R_f)
    ///
    /// # Arguments
    /// * `expected_market_return` - retorno esperado do mercado (ex: 0.10 para 10%)
    ///
    /// # Returns
    /// Retorno esperado do ativo
    pub fn expected_return(&self, expected_market_return: f64) -> f64 {
        self.risk_free_rate + self.beta * (expected_market_return - self.risk_free_rate)
    }

    /// Calcula predições para novos retornos de mercado
    ///
    /// # Arguments
    /// * `market_excess_returns` - retornos de mercado em excesso da taxa livre de risco
    ///
    /// # Returns
    /// Retornos preditos do ativo (em excesso da taxa livre de risco)
    pub fn predict(&self, market_excess_returns: &Array1<f64>) -> Array1<f64> {
        self.alpha + self.beta * market_excess_returns
    }
}

impl fmt::Display for CAPMResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "CAPITAL ASSET PRICING MODEL (CAPM) - RESULTADOS")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\nMODELO: R_i - R_f = α + β(R_m - R_f) + ε")?;
        writeln!(f, "\nObservações: {}", self.n_obs)?;
        writeln!(
            f,
            "Taxa Livre de Risco: {:.4}%",
            self.risk_free_rate * 100.0
        )?;
        writeln!(f, "Tipo de Covariância: {:?}", self.cov_type)?;
        writeln!(f, "Tipo de Inferência: {:?}", self.inference_type)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "PARÂMETROS ESTIMADOS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "Parâmetro", "Coef.", "Std Err", "t-stat", "P>|t|"
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
        writeln!(f, "Significância: *** p<0.001, ** p<0.01, * p<0.05")?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "INTERVALOS DE CONFIANÇA (95%)")?;
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
        writeln!(f, "QUALIDADE DO AJUSTE")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "R²:                  {:>12.4} ({:.2}% da variância explicada)",
            self.r_squared,
            self.r_squared * 100.0
        )?;
        writeln!(f, "R² Ajustado:         {:>12.4}", self.adj_r_squared)?;
        writeln!(
            f,
            "Tracking Error:      {:>12.4}% (volatilidade dos resíduos)",
            self.tracking_error * 100.0
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTATÍSTICAS DE RETORNO")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Retorno Médio Ativo:     {:>12.4}%",
            self.mean_asset_return * 100.0
        )?;
        writeln!(
            f,
            "Retorno Médio Mercado:   {:>12.4}%",
            self.mean_market_return * 100.0
        )?;
        writeln!(
            f,
            "Volatilidade Ativo:      {:>12.4}%",
            self.asset_volatility * 100.0
        )?;
        writeln!(
            f,
            "Volatilidade Mercado:    {:>12.4}%",
            self.market_volatility * 100.0
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "DECOMPOSIÇÃO DE RISCO")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Variância Sistemática:      {:>12.6} ({:.2}%)",
            self.systematic_variance,
            (self.systematic_variance / self.asset_volatility.powi(2)) * 100.0
        )?;
        writeln!(
            f,
            "Variância Idiossincrática:  {:>12.6} ({:.2}%)",
            self.idiosyncratic_variance,
            (self.idiosyncratic_variance / self.asset_volatility.powi(2)) * 100.0
        )?;
        writeln!(
            f,
            "Variância Total:            {:>12.6}",
            self.systematic_variance + self.idiosyncratic_variance
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "MÉTRICAS DE DESEMPENHO AJUSTADAS POR RISCO")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Sharpe Ratio (Ativo):    {:>12.4}", self.sharpe_ratio)?;
        writeln!(f, "Sharpe Ratio (Mercado):  {:>12.4}", self.market_sharpe)?;
        writeln!(f, "Treynor Ratio:           {:>12.4}", self.treynor_ratio)?;
        writeln!(
            f,
            "Information Ratio:       {:>12.4}",
            self.information_ratio
        )?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "INTERPRETAÇÃO")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Classificação de Risco:      {}",
            self.risk_classification()
        )?;
        writeln!(
            f,
            "Classificação de Desempenho: {}",
            self.performance_classification()
        )?;

        if self.is_significantly_outperforming(0.05) {
            writeln!(
                f,
                "\n✓ O ativo está SUPERANDO o mercado significativamente (α > 0, p < 0.05)"
            )?;
        } else if self.is_significantly_underperforming(0.05) {
            writeln!(
                f,
                "\n✗ O ativo está FICANDO ATRÁS do mercado significativamente (α < 0, p < 0.05)"
            )?;
        } else {
            writeln!(
                f,
                "\n○ Não há evidência significativa de outperformance ou underperformance"
            )?;
        }

        if self.is_beta_different_from_one(0.05) {
            writeln!(f, "✓ O beta é SIGNIFICATIVAMENTE diferente de 1 (p < 0.05)")?;
        } else {
            writeln!(f, "○ O beta não é significativamente diferente de 1")?;
        }

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

/// Implementação do Capital Asset Pricing Model (CAPM)
pub struct CAPM;

impl CAPM {
    /// Estima o modelo CAPM usando arrays de retornos
    ///
    /// # Arguments
    /// * `asset_returns` - retornos do ativo (ex: retornos diários em decimal)
    /// * `market_returns` - retornos do mercado (índice de referência)
    /// * `risk_free_rate` - taxa livre de risco (mesma frequência dos retornos)
    /// * `cov_type` - tipo de matriz de covariância para erros padrão
    ///
    /// # Returns
    /// `CAPMResult` com todos os parâmetros estimados e estatísticas
    ///
    /// # Example
    /// ```
    /// use frenchrs::CAPM;
    /// use greeners::CovarianceType;
    /// use ndarray::array;
    ///
    /// let asset_returns = array![0.01, 0.02, -0.01, 0.03];
    /// let market_returns = array![0.015, 0.018, -0.005, 0.025];
    /// let risk_free_rate = 0.0001; // diária
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
        // Validação
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

        // Calcular retornos em excesso
        let asset_excess: Array1<f64> = asset_returns.mapv(|r| r - risk_free_rate);
        let market_excess: Array1<f64> = market_returns.mapv(|r| r - risk_free_rate);

        // Preparar matriz de design (X = [1, market_excess])
        let mut x_matrix = Array2::<f64>::zeros((n_obs, 2));
        x_matrix.column_mut(0).fill(1.0); // Intercepto
        x_matrix.column_mut(1).assign(&market_excess);

        // Estimar via OLS
        let ols_result = OLS::fit(&asset_excess, &x_matrix, cov_type.clone())?;

        // Extrair parâmetros
        let alpha = ols_result.params[0];
        let beta = ols_result.params[1];
        let alpha_se = ols_result.std_errors[0];
        let beta_se = ols_result.std_errors[1];
        let alpha_tstat = ols_result.t_values[0];
        let beta_tstat = ols_result.t_values[1];
        let alpha_pvalue = ols_result.p_values[0];
        let beta_pvalue = ols_result.p_values[1];

        // Intervalos de confiança
        let alpha_conf_lower = ols_result.conf_lower[0];
        let alpha_conf_upper = ols_result.conf_upper[0];
        let beta_conf_lower = ols_result.conf_lower[1];
        let beta_conf_upper = ols_result.conf_upper[1];

        // Qualidade do ajuste
        let r_squared = ols_result.r_squared;
        let adj_r_squared = ols_result.adj_r_squared;

        // Valores ajustados e resíduos
        let fitted_values = ols_result.fitted_values(&x_matrix);
        let residuals = ols_result.residuals(&asset_excess, &x_matrix);

        // Estatísticas descritivas
        let mean_asset_return = asset_returns.mean().unwrap_or(0.0);
        let mean_market_return = market_returns.mean().unwrap_or(0.0);
        let asset_volatility = asset_returns.std(0.0);
        let market_volatility = market_returns.std(0.0);

        // Decomposição de risco
        let systematic_variance = beta.powi(2) * market_volatility.powi(2);
        let idiosyncratic_variance = residuals.var(0.0);

        // Tracking error
        let tracking_error = residuals.std(0.0);

        // Métricas ajustadas por risco
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
            systematic_variance,
            idiosyncratic_variance,
        })
    }

    /// Estima o modelo CAPM a partir de um DataFrame
    ///
    /// # Arguments
    /// * `df` - DataFrame contendo os dados
    /// * `asset_col` - nome da coluna com retornos do ativo
    /// * `market_col` - nome da coluna com retornos do mercado
    /// * `risk_free_rate` - taxa livre de risco
    /// * `cov_type` - tipo de matriz de covariância
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
    ///     0.0001,  // taxa diária
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
        // Extrair colunas
        let asset_returns = df.get(asset_col)?;
        let market_returns = df.get(market_col)?;

        // Estimar modelo
        Self::fit(asset_returns, market_returns, risk_free_rate, cov_type)
    }

    /// Calcula o risco sistemático (variância explicada pelo mercado)
    ///
    /// σ²_sistemático = β² × σ²_mercado
    pub fn systematic_risk(beta: f64, market_variance: f64) -> f64 {
        beta.powi(2) * market_variance
    }

    /// Calcula o risco idiossincrático (variância dos resíduos)
    ///
    /// σ²_idiossincrático = σ²_ε
    pub fn idiosyncratic_risk(residual_variance: f64) -> f64 {
        residual_variance
    }

    /// Calcula o risco total do ativo
    ///
    /// σ²_total = σ²_sistemático + σ²_idiossincrático
    pub fn total_risk(beta: f64, market_variance: f64, residual_variance: f64) -> f64 {
        Self::systematic_risk(beta, market_variance) + Self::idiosyncratic_risk(residual_variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_capm_basic() {
        // Dados sintéticos
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
        assert!(capm.beta > 0.0); // Beta deve ser positivo para ativo correlacionado
        assert!(capm.r_squared >= 0.0 && capm.r_squared <= 1.0);
    }

    #[test]
    fn test_capm_perfect_correlation() {
        // Ativo = mercado (beta = 1, alpha = 0)
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

        // Beta deve ser ~1
        assert!((result.beta - 1.0).abs() < 0.01);

        // Alpha deve ser ~0
        assert!(result.alpha.abs() < 0.01);

        // R² deve ser ~1
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
