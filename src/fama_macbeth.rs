use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the model Fama-MacBeth Two-Pass
///
/// Implementa a metodologia of Fama-MacBeth (1973) for estimar prêmios of risk
/// of the factors usando regresare em duas passagens.
#[derive(Debug, Clone)]
pub struct FamaMacBethResult {
    /// Lambdas médios (prêmios of risk of the factors) - (K+1) incluindo constante
    pub lambda_mean: Array1<f64>,

    /// Statistics t of the lambdas (Fama-MacBeth padrão)
    pub tstat_fm: Array1<f64>,

    /// P-values of the lambdas (Fama-MacBeth padrão)
    pub pval_fm: Array1<f64>,

    /// Statistics t with correção of Shanken
    pub tstat_shanken: Array1<f64>,

    /// P-values with correção of Shanken
    pub pval_shanken: Array1<f64>,

    /// Betas estimated por asset (N × K)
    pub betas: Array2<f64>,

    /// Lambdas over time (T_eff × (K+1))
    pub lambda_t: Array2<f64>,

    /// Returns médios realizados por asset
    pub mean_returns: Array1<f64>,

    /// Returns prevthiss pelthe model
    pub model_returns: Array1<f64>,

    /// Pricing errors (alphas)
    pub pricing_errors: Array1<f64>,

    /// R² cross-sectional médio
    pub r2_cross_sectional_mean: f64,

    /// Número of assets
    pub n_assets: usize,

    /// Número of factors
    pub n_factors: usize,

    /// Número efetivo of periods (T_eff)
    pub t_eff: usize,

    /// Nomes of the factors (opcional)
    pub factor_names: Option<Vec<String>>,

    /// Nomes of the assets (opcional)
    pub asset_names: Option<Vec<String>>,
}

impl FamaMacBethResult {
    /// Returns the lambda of um specific factor
    pub fn lambda(&self, factor_idx: usize) -> f64 {
        // índice 0 é a constante, factors começam em 1
        if factor_idx + 1 >= self.lambda_mean.len() {
            return 0.0;
        }
        self.lambda_mean[factor_idx + 1]
    }

    /// Returns a constante (lambda_0)
    pub fn lambda_const(&self) -> f64 {
        self.lambda_mean[0]
    }

    /// Checks if um factor é significant (Shanken, α=5%)
    pub fn is_significant_shanken(&self, factor_idx: usize, alpha: f64) -> bool {
        if factor_idx + 1 >= self.pval_shanken.len() {
            return false;
        }
        self.pval_shanken[factor_idx + 1] < alpha
    }

    /// R² médio of the pricing errors
    pub fn r2_pricing(&self) -> f64 {
        let tss: f64 = self
            .mean_returns
            .iter()
            .map(|&r| {
                let mean = self.mean_returns.mean().unwrap_or(0.0);
                (r - mean).powi(2)
            })
            .sum();

        let rss: f64 = self.pricing_errors.iter().map(|&e| e.powi(2)).sum();

        if tss > 1e-10 {
            1.0 - (rss / tss)
        } else {
            0.0
        }
    }

    /// RMSE of the pricing errors
    pub fn rmse_pricing(&self) -> f64 {
        let mse: f64 =
            self.pricing_errors.iter().map(|&e| e.powi(2)).sum::<f64>() / self.n_assets as f64;
        mse.sqrt()
    }

    /// MAE of the pricing errors
    pub fn mae_pricing(&self) -> f64 {
        self.pricing_errors.iter().map(|&e| e.abs()).sum::<f64>() / self.n_assets as f64
    }
}

impl fmt::Display for FamaMacBethResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "FAMA-MACBETH TWO-PASS REGRESSION")?;
        writeln!(f, "{}", "=".repeat(80))?;

        writeln!(f, "\nDimensões:")?;
        writeln!(f, "  • Assets: {}", self.n_assets)?;
        writeln!(f, "  • Factors: {}", self.n_factors)?;
        writeln!(f, "  • Periods efetivos: {}", self.t_eff)?;

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "PRÊMIOS DE RISCO (λ) - Risk Premia")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "{:<15} {:>12} {:>10} {:>10} {:>10} {:>10}",
            "Factor", "Lambda", "t(FM)", "p(FM)", "t(Shank)", "p(Shank)"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;

        // Constante
        writeln!(
            f,
            "{:<15} {:>12.6} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            "Constant",
            self.lambda_mean[0],
            self.tstat_fm[0],
            self.pval_fm[0],
            self.tstat_shanken[0],
            self.pval_shanken[0]
        )?;

        // Factors
        for i in 0..self.n_factors {
            let name = if let Some(ref names) = self.factor_names {
                names.get(i).map(|s| s.as_str()).unwrap_or("Factor")
            } else {
                "Factor"
            };

            let sig = if self.is_significant_shanken(i, 0.05) {
                "*"
            } else {
                ""
            };

            writeln!(
                f,
                "{:<15} {:>12.6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {}",
                name,
                self.lambda_mean[i + 1],
                self.tstat_fm[i + 1],
                self.pval_fm[i + 1],
                self.tstat_shanken[i + 1],
                self.pval_shanken[i + 1],
                sig
            )?;
        }

        writeln!(f, "\n{}", "-".repeat(80))?;
        writeln!(f, "PRICING ERRORS")?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "R² Cross-Sectional:  {:>10.4}",
            self.r2_cross_sectional_mean
        )?;
        writeln!(f, "R² Pricing:          {:>10.4}", self.r2_pricing())?;
        writeln!(f, "RMSE:                {:>10.6}", self.rmse_pricing())?;
        writeln!(f, "MAE:                 {:>10.6}", self.mae_pricing())?;

        writeln!(f, "\n{}", "=".repeat(80))?;
        writeln!(f, "* = Significativo a 5% (Shanken correction)")?;

        Ok(())
    }
}

/// Fama-MacBeth Two-Pass Regression
pub struct FamaMacBeth;

impl FamaMacBeth {
    /// Estimates the model Fama-MacBeth
    ///
    /// # Arguments
    /// * `returns_excess` - Matriz of returns excedentes (T × N)
    /// * `factors` - Matriz of factors (T × K)
    /// * `cov_type` - Tipo for covariesnce primeira passagem
    /// * `asset_names` - Nomes of the assets (opcional)
    /// * `factor_names` - Nomes of the factors (opcional)
    ///
    /// # Returns
    /// `FamaMacBethResult` with lambdas, betas, pricing errors, etc.
    ///
    /// # Example
    /// ```
    /// use frenchrs::FamaMacBeth;
    /// use greeners::CovarianceType;
    /// use ndarray::Array2;
    ///
    /// // Gerar Synthetic data
    /// let mut rng = 42u64;
    /// let mut rand = || {
    ///     rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
    ///     ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
    /// };
    ///
    /// // 60 meses, 25 assets, 3 factors
    /// let factors = Array2::from_shape_fn((60, 3), |_| rand() * 0.03);
    /// let mut returns = Array2::from_shape_fn((60, 25), |_| rand() * 0.02);
    ///
    /// // Adicionar exposição aos factors
    /// for i in 0..60 {
    ///     for j in 0..25 {
    ///         for f in 0..3 {
    ///             returns[[i, j]] += factors[[i, f]] * (0.5 + j as f64 / 25.0);
    ///         }
    ///     }
    /// }
    ///
    /// let result = FamaMacBeth::fit(
    ///     &returns,
    ///     &factors,
    ///     CovarianceType::HC3,
    ///     None,
    ///     Some(vec!["Market".to_string(), "SMB".to_string(), "HML".to_string()])
    /// ).unwrap();
    ///
    /// assert_eq!(result.n_assets, 25);
    /// assert_eq!(result.n_factors, 3);
    /// ```
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
        factor_names: Option<Vec<String>>,
    ) -> Result<FamaMacBethResult, GreenersError> {
        let t = returns_excess.nrows();
        let n_assets = returns_excess.ncols();
        let n_factors = factors.ncols();

        if factors.nrows() != t {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns ({} obs) and factors ({} obs) must have same number of observations",
                t,
                factors.nrows()
            )));
        }

        if t <= n_factors + 5 {
            return Err(GreenersError::InvalidOperation(format!(
                "Too few observations ({}) for {} factors",
                t, n_factors
            )));
        }

        // ========================================================================
        // PRIMEIRA PASSAGEM: Time-Series Regressions (estimar betas)
        // ========================================================================

        let mut betas = Array2::<f64>::zeros((n_assets, n_factors));
        let mut valid_assets = vec![true; n_assets];

        // Matriz of design for time-beies: [1, factors]
        let mut x_ts = Array2::<f64>::zeros((t, n_factors + 1));
        x_ts.column_mut(0).fill(1.0);
        for j in 0..n_factors {
            x_ts.column_mut(j + 1).assign(&factors.column(j));
        }

        for asset_idx in 0..n_assets {
            let y = returns_excess.column(asset_idx).to_owned();

            // Checar if tem data suficientes
            let valid_count = y.iter().filter(|&&v| v.is_finite()).count();
            if valid_count <= n_factors + 2 {
                valid_assets[asset_idx] = false;
                continue;
            }

            match OLS::fit(&y, &x_ts, cov_type.clone()) {
                Ok(ols) => {
                    // Extract betas (pular a constante)
                    for j in 0..n_factors {
                        betas[[asset_idx, j]] = ols.params[j + 1];
                    }
                }
                Err(_) => {
                    valid_assets[asset_idx] = false;
                }
            }
        }

        // Verificar if temos assets válidos
        let n_valid = valid_assets.iter().filter(|&&v| v).count();
        if n_valid <= n_factors + 1 {
            return Err(GreenersError::InvalidOperation(
                "Too few valid assets after first-pass regression".to_string(),
            ));
        }

        // ========================================================================
        // SEGUNDA PASSAGEM: Cross-Sectional Regressions (estimar lambdas_t)
        // ========================================================================

        let mut lambda_t_list = Vec::new();
        let mut r2_cs_list = Vec::new();

        for time_idx in 0..t {
            let y_t = returns_excess.row(time_idx);

            // Filtrar assets válidos and without NaN nthis period
            let mut assets_t = Vec::new();
            let mut betas_t = Vec::new();
            let mut returns_t = Vec::new();

            for asset_idx in 0..n_assets {
                if !valid_assets[asset_idx] {
                    continue;
                }
                let ret = y_t[asset_idx];
                if !ret.is_finite() {
                    continue;
                }

                assets_t.push(asset_idx);
                returns_t.push(ret);

                let beta_row: Vec<f64> = (0..n_factors).map(|j| betas[[asset_idx, j]]).collect();
                betas_t.push(beta_row);
            }

            if assets_t.len() <= n_factors + 1 {
                continue;
            }

            // Construir X_cs: [1, betas]
            let n_cs = assets_t.len();
            let mut x_cs = Array2::<f64>::zeros((n_cs, n_factors + 1));
            x_cs.column_mut(0).fill(1.0);

            for (i, beta_vec) in betas_t.iter().enumerate() {
                for j in 0..n_factors {
                    x_cs[[i, j + 1]] = beta_vec[j];
                }
            }

            let y_cs = Array1::from_vec(returns_t);

            match OLS::fit(&y_cs, &x_cs, CovarianceType::NonRobust) {
                Ok(ols_cs) => {
                    lambda_t_list.push(ols_cs.params.clone());
                    r2_cs_list.push(ols_cs.r_squared);
                }
                Err(_) => continue,
            }
        }

        if lambda_t_list.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "No valid cross-sectional regressions".to_string(),
            ));
        }

        let t_eff = lambda_t_list.len();

        // Converter lambda_t_list for Array2
        let mut lambda_t = Array2::<f64>::zeros((t_eff, n_factors + 1));
        for (i, lam) in lambda_t_list.iter().enumerate() {
            for j in 0..(n_factors + 1) {
                lambda_t[[i, j]] = lam[j];
            }
        }

        // ========================================================================
        // ESTIMADORES FAMA-MACBETH: λ̄, SE padrão, Shanken correction
        // ========================================================================

        let mut lambda_mean = Array1::<f64>::zeros(n_factors + 1);
        let mut tstat_fm = Array1::<f64>::zeros(n_factors + 1);
        let mut pval_fm = Array1::<f64>::zeros(n_factors + 1);
        let mut tstat_shanken = Array1::<f64>::zeros(n_factors + 1);
        let mut pval_shanken = Array1::<f64>::zeros(n_factors + 1);

        // Calculates covariance of the factors (for Shanken)
        let sigma_f = Self::cov_matrix(factors);
        let sigma_f_inv = match Self::invert_matrix(&sigma_f) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(GreenersError::InvalidOperation(
                    "Cannot invert factor covariesnce matrix".to_string(),
                ));
            }
        };

        for param_idx in 0..(n_factors + 1) {
            let lam_beies = lambda_t.column(param_idx).to_owned();
            let lam_mean_val = lam_beies.mean().unwrap_or(0.0);
            lambda_mean[param_idx] = lam_mean_val;

            if t_eff < 3 {
                continue;
            }

            let std_dev = lam_beies.std(1.0);
            let se_fm = std_dev / (t_eff as f64).sqrt();

            // t-stat padrão Fama-MacBeth
            if se_fm > 1e-10 {
                let t_val = lam_mean_val / se_fm;
                tstat_fm[param_idx] = t_val;
                pval_fm[param_idx] = Self::t_test_pvalue(t_val, t_eff - 1);
            }

            // Shanken correction (apenas for factors, not for constante)
            if param_idx == 0 {
                // Constante: not aplica Shanken
                tstat_shanken[param_idx] = tstat_fm[param_idx];
                pval_shanken[param_idx] = pval_fm[param_idx];
            } else {
                // Factor: aplicar Shanken
                let lambda_factors = lambda_mean.slice(ndarray::s![1..]).to_owned();

                // correction_factor = 1 + λ' Σ_f^{-1} λ
                let mut correction_factor = 1.0;
                for i in 0..n_factors {
                    for j in 0..n_factors {
                        correction_factor +=
                            lambda_factors[i] * sigma_f_inv[[i, j]] * lambda_factors[j];
                    }
                }

                let se_shanken = se_fm * correction_factor.max(1.0).sqrt();

                if se_shanken > 1e-10 {
                    let t_shanken = lam_mean_val / se_shanken;
                    tstat_shanken[param_idx] = t_shanken;
                    pval_shanken[param_idx] = Self::t_test_pvalue(t_shanken, t_eff - 1);
                }
            }
        }

        // ========================================================================
        // PRICING ERRORS
        // ========================================================================

        // Returns médios por asset
        let mut mean_returns = Array1::<f64>::zeros(n_assets);
        for asset_idx in 0..n_assets {
            let col = returns_excess.column(asset_idx);
            let valid: Vec<f64> = col.iter().filter(|&&v| v.is_finite()).copied().collect();
            if !valid.is_empty() {
                mean_returns[asset_idx] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        // Returns prevthiss: model_ret = betas × lambda_factors
        let lambda_factors = lambda_mean.slice(ndarray::s![1..]).to_owned();
        let mut model_returns = Array1::<f64>::zeros(n_assets);

        for asset_idx in 0..n_assets {
            let mut pred = 0.0;
            for j in 0..n_factors {
                pred += betas[[asset_idx, j]] * lambda_factors[j];
            }
            model_returns[asset_idx] = pred;
        }

        let pricing_errors = &mean_returns - &model_returns;

        // R² cross-sectional médio
        let r2_cross_sectional_mean = if r2_cs_list.is_empty() {
            0.0
        } else {
            r2_cs_list.iter().sum::<f64>() / r2_cs_list.len() as f64
        };

        Ok(FamaMacBethResult {
            lambda_mean,
            tstat_fm,
            pval_fm,
            tstat_shanken,
            pval_shanken,
            betas,
            lambda_t,
            mean_returns,
            model_returns,
            pricing_errors,
            r2_cross_sectional_mean,
            n_assets,
            n_factors,
            t_eff,
            factor_names,
            asset_names,
        })
    }

    // ========================================================================
    // FUNÇÕES AUXILIARES
    // ========================================================================

    /// Calculates covariance matrix
    fn cov_matrix(data: &Array2<f64>) -> Array2<f64> {
        let n = data.nrows();
        let k = data.ncols();

        if n <= 1 {
            return Array2::zeros((k, k));
        }

        let means: Vec<f64> = (0..k)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect();

        let mut cov = Array2::<f64>::zeros((k, k));

        for i in 0..k {
            for j in 0..k {
                let mut sum = 0.0;
                for t in 0..n {
                    sum += (data[[t, i]] - means[i]) * (data[[t, j]] - means[j]);
                }
                cov[[i, j]] = sum / ((n - 1) as f64);
            }
        }

        cov
    }

    /// Inverte matriz usando decomposição LU
    fn invert_matrix(mat: &Array2<f64>) -> Result<Array2<f64>, GreenersError> {
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(GreenersError::ShapeMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        // Usar pseudo-inversa via SVD beia better, mas por simplicidade usamos LU
        // (assumindo que Greeners ou ndarray-linalg tem that disponível)

        // Por enhow much, implementação simples for 3x3 ou less
        if n == 1 {
            let val = mat[[0, 0]];
            if val.abs() < 1e-10 {
                return Err(GreenersError::InvalidOperation(
                    "Singular matrix".to_string(),
                ));
            }
            let mut inv = Array2::zeros((1, 1));
            inv[[0, 0]] = 1.0 / val;
            return Ok(inv);
        }

        // Para matrizes greateres, usar pseudo-inversa (simplificado)
        // Em produção, usar ndarray-linalg ou similar
        Self::pseudo_inverse(mat)
    }

    /// Pseudo-inversa simplifieach (Moore-Penrose via Gauss-Jordan)
    fn pseudo_inverse(mat: &Array2<f64>) -> Result<Array2<f64>, GreenersError> {
        let n = mat.nrows();

        // Createsr matriz aumentada [A | I]
        let mut aug = Array2::<f64>::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = mat[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }

        // Eliminação of Gauss-Jordan
        for i in 0..n {
            // Encontrar pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            if aug[[max_row, i]].abs() < 1e-10 {
                return Err(GreenersError::InvalidOperation(
                    "Matrix is singular".to_string(),
                ));
            }

            // Trocar linhas
            if max_row != i {
                for j in 0..(2 * n) {
                    let tmp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Normalizar linha pivot
            let pivot = aug[[i, i]];
            for j in 0..(2 * n) {
                aug[[i, j]] /= pivot;
            }

            // Eliminar column i nas outras linhas
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..(2 * n) {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Extract inversa (metade direita of the matriz aumentada)
        let mut inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }

        Ok(inv)
    }

    /// P-value of the test t bilateral
    fn t_test_pvalue(t: f64, df: usize) -> f64 {
        if df == 0 {
            return 1.0;
        }

        // Aproximação simples for p-value of the test t
        // Em produção, usar biblioteca statistic adequada
        let x = df as f64 / (df as f64 + t * t);

        Self::incomplete_beta(df as f64 / 2.0, 0.5, x)
    }

    /// Funçãthe beta incompleta regularizada (aproximação)
    fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }

        // Aproximação simples (not precisa for uso real)
        // Em produção, usar statrs ou similar
        let n = 100;
        let mut sum = 0.0;
        let dx = x / n as f64;

        for i in 0..n {
            let t = (i as f64 + 0.5) * dx;
            sum += t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0);
        }

        sum * dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fama_macbeth_basic() {
        // 60 meses, 25 assets, 3 factors
        let t = 60;
        let n = 25;
        let k = 3;

        let mut rng = 0u64;
        let mut next_rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        // Gerar Synthetic data
        let factors = Array2::from_shape_fn((t, k), |_| next_rand() * 0.05);
        let mut returns = Array2::from_shape_fn((t, n), |_| next_rand() * 0.03);

        // Adicionar exposição aos factors
        for i in 0..t {
            for j in 0..n {
                let mut exposure = 0.0;
                for f in 0..k {
                    exposure += factors[[i, f]] * (0.5 + (j as f64 / n as f64));
                }
                returns[[i, j]] += exposure;
            }
        }

        let result = FamaMacBeth::fit(
            &returns,
            &factors,
            CovarianceType::NonRobust,
            None,
            Some(vec![
                "Market".to_string(),
                "SMB".to_string(),
                "HML".to_string(),
            ]),
        )
        .unwrap();

        assert_eq!(result.n_assets, 25);
        assert_eq!(result.n_factors, 3);
        assert!(result.t_eff > 0);
        assert_eq!(result.lambda_mean.len(), 4); // const + 3 factors
        assert_eq!(result.betas.nrows(), 25);
        assert_eq!(result.betas.ncols(), 3);
    }

    #[test]
    fn test_fama_macbeth_dimensions() {
        let t = 100;
        let n = 50;
        let k = 5;

        let mut rng = 42u64;
        let mut next_rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| next_rand() * 0.03);
        let mut returns = Array2::from_shape_fn((t, n), |_| next_rand() * 0.02);

        // Adicionar exposição aos factors
        for i in 0..t {
            for j in 0..n {
                let mut exposure = 0.0;
                for f in 0..k {
                    exposure += factors[[i, f]] * (0.3 + (j as f64 / n as f64) * 0.5);
                }
                returns[[i, j]] += exposure;
            }
        }

        let result = FamaMacBeth::fit(&returns, &factors, CovarianceType::HC3, None, None).unwrap();

        assert_eq!(result.n_assets, 50);
        assert_eq!(result.n_factors, 5);
        assert_eq!(result.lambda_mean.len(), 6);
        assert_eq!(result.pricing_errors.len(), 50);
    }

    #[test]
    fn test_fama_macbeth_shape_mismatch() {
        let returns = Array2::<f64>::zeros((100, 50));
        let factors = Array2::<f64>::zeros((90, 5)); // diferente!

        let result = FamaMacBeth::fit(&returns, &factors, CovarianceType::NonRobust, None, None);

        assert!(result.is_err());
    }
}
