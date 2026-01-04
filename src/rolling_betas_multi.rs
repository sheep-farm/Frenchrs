use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Resultado de Rolling Betas para múltiplos ativos
///
/// Estrutura similar ao output do Python: DataFrame com índice (asset, date)
#[derive(Debug, Clone)]
pub struct RollingBetasMulti {
    /// Resultados por ativo
    pub results: HashMap<String, RollingBetasAsset>,

    /// Tamanho da janela
    pub window_size: usize,

    /// Número de fatores
    pub n_factors: usize,

    /// Nomes dos fatores (opcional)
    pub factor_names: Option<Vec<String>>,
}

/// Resultado de Rolling Betas para um único ativo
#[derive(Debug, Clone)]
pub struct RollingBetasAsset {
    /// Nome do ativo
    pub asset_name: String,

    /// Datas de cada janela (índice final de cada janela)
    pub dates: Vec<usize>,

    /// Alpha ao longo do tempo
    pub alphas: Array1<f64>,

    /// Betas ao longo do tempo (n_windows × n_factors)
    pub betas: Array2<f64>,

    /// R² ao longo do tempo
    pub r_squared: Array1<f64>,

    /// Número de janelas
    pub n_windows: usize,
}

impl RollingBetasMulti {
    /// Calcula rolling betas para múltiplos ativos
    ///
    /// # Arguments
    /// * `returns` - Matriz de retornos (n_obs × n_assets)
    /// * `factors` - Matriz de fatores (n_obs × n_factors)
    /// * `window_size` - Tamanho da janela móvel
    /// * `cov_type` - Tipo de covariância
    /// * `asset_names` - Nomes dos ativos (opcional)
    /// * `factor_names` - Nomes dos fatores (opcional)
    ///
    /// # Example
    /// ```
    /// use frenchrs::RollingBetasMulti;
    /// use greeners::CovarianceType;
    /// use ndarray::{array, Array2};
    ///
    /// // 12 observações, 2 ativos
    /// let returns = Array2::from_shape_vec((12, 2), vec![
    ///     0.01, 0.015,  // t=1
    ///     0.02, 0.025,  // t=2
    ///     -0.01, -0.005, // t=3
    ///     0.03, 0.035,  // ... etc
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
    /// // 12 observações, 1 fator (mercado)
    /// let factors = Array2::from_shape_vec((12, 1), vec![
    ///     0.008, 0.015, -0.005, 0.025, 0.012, -0.003,
    ///     0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
    /// ]).unwrap();
    ///
    /// let rolling = RollingBetasMulti::fit(
    ///     &returns,
    ///     &factors,
    ///     6,
    ///     CovarianceType::NonRobust,
    ///     Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
    ///     Some(vec!["Market".to_string()]),
    /// ).unwrap();
    ///
    /// assert_eq!(rolling.results.len(), 2);
    /// ```
    pub fn fit(
        returns: &Array2<f64>,
        factors: &Array2<f64>,
        window_size: usize,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
        factor_names: Option<Vec<String>>,
    ) -> Result<Self, GreenersError> {
        let n_obs = returns.nrows();
        let n_assets = returns.ncols();
        let n_factors = factors.ncols();

        if factors.nrows() != n_obs {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns ({} obs) and factors ({} obs) must have same number of observations",
                n_obs,
                factors.nrows()
            )));
        }

        if window_size < n_factors + 2 {
            return Err(GreenersError::InvalidOperation(format!(
                "Window size ({}) must be at least {} (n_factors + 2)",
                window_size,
                n_factors + 2
            )));
        }

        if n_obs < window_size {
            return Err(GreenersError::InvalidOperation(format!(
                "Not enough data: {} observations, window size {}",
                n_obs, window_size
            )));
        }

        // Nomes padrão se não fornecidos
        let asset_names = asset_names
            .unwrap_or_else(|| (0..n_assets).map(|i| format!("Asset{}", i + 1)).collect());

        let mut results = HashMap::new();

        // Processar cada ativo
        for (asset_idx, asset_name) in asset_names.iter().enumerate() {
            let asset_returns = returns.column(asset_idx);

            let rolling_asset = Self::fit_single_asset(
                &asset_returns.to_owned(),
                factors,
                window_size,
                cov_type.clone(),
                asset_name.clone(),
            )?;

            results.insert(asset_name.clone(), rolling_asset);
        }

        Ok(RollingBetasMulti {
            results,
            window_size,
            n_factors,
            factor_names,
        })
    }

    /// Calcula rolling betas para um único ativo
    fn fit_single_asset(
        asset_returns: &Array1<f64>,
        factors: &Array2<f64>,
        window_size: usize,
        cov_type: CovarianceType,
        asset_name: String,
    ) -> Result<RollingBetasAsset, GreenersError> {
        let n_obs = asset_returns.len();
        let n_factors = factors.ncols();
        let n_windows = n_obs - window_size + 1;

        let mut alphas = Array1::<f64>::zeros(n_windows);
        let mut betas = Array2::<f64>::zeros((n_windows, n_factors));
        let mut r_squared = Array1::<f64>::zeros(n_windows);
        let mut dates = Vec::with_capacity(n_windows);

        for i in 0..n_windows {
            let y_window = asset_returns.slice(ndarray::s![i..i + window_size]);
            let x_factors = factors.slice(ndarray::s![i..i + window_size, ..]);

            // Matriz de design: [1, factors]
            let mut x = Array2::<f64>::zeros((window_size, n_factors + 1));
            x.column_mut(0).fill(1.0);
            for j in 0..n_factors {
                x.column_mut(j + 1).assign(&x_factors.column(j));
            }

            // Estimar OLS
            let ols = OLS::fit(&y_window.to_owned(), &x, cov_type.clone())?;

            alphas[i] = ols.params[0];
            for j in 0..n_factors {
                betas[[i, j]] = ols.params[j + 1];
            }
            r_squared[i] = ols.r_squared;

            // Data é o índice final da janela
            dates.push(i + window_size - 1);
        }

        Ok(RollingBetasAsset {
            asset_name,
            dates,
            alphas,
            betas,
            r_squared,
            n_windows,
        })
    }

    /// Obtém resultados para um ativo específico
    pub fn get_asset(&self, asset_name: &str) -> Option<&RollingBetasAsset> {
        self.results.get(asset_name)
    }

    /// Lista todos os ativos
    pub fn asset_names(&self) -> Vec<String> {
        self.results.keys().cloned().collect()
    }

    /// Converte para formato tabular (similar ao DataFrame do Python)
    ///
    /// Retorna: Vec de (asset_name, date_idx, alpha, betas..., r_squared)
    pub fn to_table(&self) -> Vec<RollingBetasRow> {
        let mut rows = Vec::new();

        for (asset_name, asset_result) in &self.results {
            for i in 0..asset_result.n_windows {
                let betas_vec: Vec<f64> = (0..self.n_factors)
                    .map(|j| asset_result.betas[[i, j]])
                    .collect();

                rows.push(RollingBetasRow {
                    asset: asset_name.clone(),
                    date_idx: asset_result.dates[i],
                    alpha: asset_result.alphas[i],
                    betas: betas_vec,
                    r_squared: asset_result.r_squared[i],
                });
            }
        }

        rows
    }

    /// Exporta para CSV-like string
    pub fn to_csv_string(&self) -> String {
        let mut result = String::new();

        // Header
        let mut header = vec![
            "asset".to_string(),
            "date_idx".to_string(),
            "alpha".to_string(),
        ];

        if let Some(ref names) = self.factor_names {
            header.extend(names.iter().cloned());
        } else {
            for i in 0..self.n_factors {
                header.push(format!("factor_{}", i + 1));
            }
        }
        header.push("r_squared".to_string());

        result.push_str(&header.join(","));
        result.push('\n');

        // Data
        let table = self.to_table();
        for row in table {
            let mut line = vec![
                row.asset,
                row.date_idx.to_string(),
                format!("{:.6}", row.alpha),
            ];

            for beta in &row.betas {
                line.push(format!("{:.6}", beta));
            }
            line.push(format!("{:.6}", row.r_squared));

            result.push_str(&line.join(","));
            result.push('\n');
        }

        result
    }
}

/// Linha de resultado rolling betas (formato tabular)
#[derive(Debug, Clone)]
pub struct RollingBetasRow {
    /// Nome do ativo
    pub asset: String,

    /// Índice da data (índice final da janela)
    pub date_idx: usize,

    /// Alpha
    pub alpha: f64,

    /// Betas dos fatores
    pub betas: Vec<f64>,

    /// R²
    pub r_squared: f64,
}

impl RollingBetasAsset {
    /// Retorna o beta médio para um fator específico
    pub fn mean_beta(&self, factor_idx: usize) -> f64 {
        if factor_idx >= self.betas.ncols() {
            return 0.0;
        }
        self.betas.column(factor_idx).mean().unwrap_or(0.0)
    }

    /// Retorna o desvio padrão do beta para um fator específico
    pub fn std_beta(&self, factor_idx: usize) -> f64 {
        if factor_idx >= self.betas.ncols() {
            return 0.0;
        }
        self.betas.column(factor_idx).std(1.0)
    }

    /// Coeficiente de variação do beta
    pub fn cv_beta(&self, factor_idx: usize) -> f64 {
        let mean = self.mean_beta(factor_idx);
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        self.std_beta(factor_idx) / mean.abs()
    }

    /// Estatísticas de estabilidade para um fator específico
    pub fn beta_stability(&self, factor_idx: usize) -> BetaStability {
        if factor_idx >= self.betas.ncols() {
            // Retornar estabilidade vazia se índice inválido
            return BetaStability::default();
        }
        let beta_series = self.betas.column(factor_idx).to_owned();
        BetaStability::from_series(&beta_series)
    }

    /// Estatísticas de estabilidade do alpha
    pub fn alpha_stability(&self) -> BetaStability {
        BetaStability::from_series(&self.alphas)
    }

    /// Verifica se beta é estável ao longo do tempo
    pub fn is_beta_stable(&self, factor_idx: usize, threshold: f64) -> bool {
        let stability = self.beta_stability(factor_idx);
        stability.coefficient_of_variation < threshold
    }
}

/// Estatísticas de estabilidade de uma série temporal
#[derive(Debug, Clone)]
pub struct BetaStability {
    /// Média da série
    pub mean: f64,

    /// Mediana da série
    pub median: f64,

    /// Desvio padrão
    pub std_dev: f64,

    /// Mínimo
    pub min: f64,

    /// Máximo
    pub max: f64,

    /// Range (max - min)
    pub range: f64,

    /// Coeficiente de variação (std_dev / mean)
    pub coefficient_of_variation: f64,

    /// Tendência linear (slope)
    pub trend: f64,

    /// Autocorrelação de lag 1
    pub autocorrelation: f64,
}

impl Default for BetaStability {
    fn default() -> Self {
        BetaStability {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            range: 0.0,
            coefficient_of_variation: 0.0,
            trend: 0.0,
            autocorrelation: 0.0,
        }
    }
}

impl BetaStability {
    /// Calcula estatísticas de estabilidade de uma série
    pub fn from_series(series: &Array1<f64>) -> Self {
        let n = series.len();

        if n == 0 {
            return BetaStability::default();
        }

        let mean = series.mean().unwrap_or(0.0);

        let mut sorted = series.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n > 0 { sorted[n / 2] } else { 0.0 };

        let std_dev = series.std(1.0);
        let min = sorted[0];
        let max = sorted[n - 1];
        let range = max - min;

        let coefficient_of_variation = if mean.abs() > 1e-10 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Tendência linear (regressão simples: y = a + bx)
        let x_mean = (n - 1) as f64 / 2.0;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in series.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - mean);
            denominator += (x - x_mean).powi(2);
        }

        let trend = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Autocorrelação lag 1
        let mut autocorr_num = 0.0;
        let mut autocorr_den = 0.0;

        if n > 1 {
            for i in 0..n - 1 {
                autocorr_num += (series[i] - mean) * (series[i + 1] - mean);
            }

            for &val in series.iter() {
                autocorr_den += (val - mean).powi(2);
            }
        }

        let autocorrelation = if autocorr_den > 1e-10 {
            autocorr_num / autocorr_den
        } else {
            0.0
        };

        BetaStability {
            mean,
            median,
            std_dev,
            min,
            max,
            range,
            coefficient_of_variation,
            trend,
            autocorrelation,
        }
    }

    /// Classifica o nível de estabilidade
    pub fn stability_classification(&self) -> &str {
        if self.coefficient_of_variation < 0.05 {
            "Muito Estável"
        } else if self.coefficient_of_variation < 0.10 {
            "Estável"
        } else if self.coefficient_of_variation < 0.20 {
            "Moderadamente Estável"
        } else if self.coefficient_of_variation < 0.50 {
            "Instável"
        } else {
            "Muito Instável"
        }
    }

    /// Classifica a tendência
    pub fn trend_classification(&self) -> &str {
        if self.trend > 0.01 {
            "Tendência Crescente Forte"
        } else if self.trend > 0.001 {
            "Tendência Crescente"
        } else if self.trend > -0.001 {
            "Estável (sem tendência)"
        } else if self.trend > -0.01 {
            "Tendência Decrescente"
        } else {
            "Tendência Decrescente Forte"
        }
    }
}

impl std::fmt::Display for BetaStability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "ANÁLISE DE ESTABILIDADE")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "ESTATÍSTICAS DESCRITIVAS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "Média:               {:>10.4}", self.mean)?;
        writeln!(f, "Mediana:             {:>10.4}", self.median)?;
        writeln!(f, "Desvio Padrão:       {:>10.4}", self.std_dev)?;
        writeln!(f, "Mínimo:              {:>10.4}", self.min)?;
        writeln!(f, "Máximo:              {:>10.4}", self.max)?;
        writeln!(f, "Range:               {:>10.4}", self.range)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "MÉTRICAS DE ESTABILIDADE")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Coef. Variação:      {:>10.4}",
            self.coefficient_of_variation
        )?;
        writeln!(f, "Tendência:           {:>10.6}", self.trend)?;
        writeln!(f, "Autocorrelação:      {:>10.4}", self.autocorrelation)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "CLASSIFICAÇÕES")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Estabilidade:        {}",
            self.stability_classification()
        )?;
        writeln!(f, "Tendência:           {}", self.trend_classification())?;

        writeln!(f, "\n{}", "=".repeat(80))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_rolling_betas_multi_basic() {
        let returns = Array2::from_shape_vec(
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

        let rolling =
            RollingBetasMulti::fit(&returns, &factors, 6, CovarianceType::NonRobust, None, None)
                .unwrap();

        assert_eq!(rolling.results.len(), 2);
        assert_eq!(rolling.window_size, 6);
        assert_eq!(rolling.n_factors, 1);
    }

    #[test]
    fn test_rolling_betas_to_table() {
        let returns = Array2::from_shape_vec(
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

        let rolling = RollingBetasMulti::fit(
            &returns,
            &factors,
            6,
            CovarianceType::NonRobust,
            Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
            Some(vec!["Market".to_string()]),
        )
        .unwrap();

        let table = rolling.to_table();
        assert_eq!(table.len(), 14); // 7 windows × 2 assets
    }

    #[test]
    fn test_rolling_betas_csv_export() {
        let returns = Array2::from_shape_vec(
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

        let rolling =
            RollingBetasMulti::fit(&returns, &factors, 6, CovarianceType::NonRobust, None, None)
                .unwrap();

        let csv = rolling.to_csv_string();
        assert!(csv.contains("asset,date_idx,alpha"));
        assert!(csv.contains("Asset1"));
        assert!(csv.contains("Asset2"));
    }
}
