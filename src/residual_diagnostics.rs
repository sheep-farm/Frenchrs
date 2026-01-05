//! # Residual Diagnostics
//!
//! This module provides comprehensive diagnostic tests for regression residuals.
//! These tests help validate model assumptions and detect specification problems.
//!
//! ## Available Tests
//!
//! 1. **Durbin-Watson**: Tests for first-order autocorrelation
//! 2. **Ljung-Box**: Tests for autocorrelation at multiple lags
//! 3. **Breusch-Pagan**: Tests for heteroscedasticity
//! 4. **White**: Tests for heteroscedasticity (more general than BP)
//! 5. **RESET**: Tests for functional form misspecification
//! 6. **Chow**: Tests for structural breaks
//! 7. **ARCH**: Tests for conditional heteroscedasticity (volatility clustering)
//! 8. **Jarque-Bera**: Tests for normality of residuals
//!
//! ## Example
//!
//! ```rust
//! use frenchrs::ResidualDiagnostics;
//! use greeners::CovarianceType;
//! use ndarray::Array2;
//!
//! let t = 120;
//! let n = 5;
//! let k = 2;
//!
//! let mut rng = 42u64;
//! let mut rand = || {
//!     rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
//!     ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
//! };
//!
//! let returns_excess = Array2::from_shape_fn((t, n), |_| rand() * 0.03);
//! let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
//!
//! let result = ResidualDiagnostics::fit(
//!     &returns_excess,
//!     &factors,
//!     CovarianceType::HC3,
//!     None,
//! ).unwrap();
//!
//! // Check for issues
//! for (asset, diag) in &result.diagnostics {
//!     println!("{}: DW={:.3}, JB p-value={:.4}", asset, diag.durbin_watson, diag.jb_p_value);
//! }
//! ```

use greeners::{CovarianceType, GreenersError, OLS};
use ndarray::{s, Array1, Array2};
use std::collections::HashMap;

/// Diagnostic test results for a single asset
#[derive(Debug, Clone)]
pub struct AssetDiagnostics {
    /// Asset name
    pub asset: String,
    /// Number of observations
    pub n_obs: usize,
    /// Durbin-Watson statistic (tests for autocorrelation)
    pub durbin_watson: f64,
    /// Ljung-Box test statistic (lag 12)
    pub lb_stat: f64,
    /// Ljung-Box p-value
    pub lb_p_value: f64,
    /// Breusch-Pagan test statistic (heteroscedasticity)
    pub bp_stat: f64,
    /// Breusch-Pagan p-value
    pub bp_p_value: f64,
    /// White test statistic (heteroscedasticity)
    pub white_stat: f64,
    /// White p-value
    pub white_p_value: f64,
    /// RESET F-statistic (functional form)
    pub reset_f: f64,
    /// RESET p-value
    pub reset_p_value: f64,
    /// Chow F-statistic (structural break)
    pub chow_f: f64,
    /// Chow p-value
    pub chow_p_value: f64,
    /// ARCH test statistic (conditional heteroscedasticity)
    pub arch_stat: f64,
    /// ARCH p-value
    pub arch_p_value: f64,
    /// Jarque-Bera test statistic (normality)
    pub jb_stat: f64,
    /// Jarque-Bera p-value
    pub jb_p_value: f64,
}

impl AssetDiagnostics {
    /// Returns true if Durbin-Watson indicates positive autocorrelation (DW < 1.5)
    pub fn has_positive_autocorr(&self) -> bool {
        self.durbin_watson < 1.5
    }

    /// Returns true if Durbin-Watson indicates negative autocorrelation (DW > 2.5)
    pub fn has_negative_autocorr(&self) -> bool {
        self.durbin_watson > 2.5
    }

    /// Returns true if residuals are heteroscedastic (Breusch-Pagan test, 5% level)
    pub fn has_heteroscedasticity(&self) -> bool {
        self.bp_p_value < 0.05
    }

    /// Returns true if White test rejects homoscedasticity (5% level)
    pub fn white_rejects(&self) -> bool {
        self.white_p_value < 0.05
    }

    /// Returns true if RESET test rejects correct functional form (5% level)
    pub fn has_misspecification(&self) -> bool {
        self.reset_p_value < 0.05
    }

    /// Returns true if Chow test detects structural break (5% level)
    pub fn has_structural_break(&self) -> bool {
        self.chow_p_value < 0.05
    }

    /// Returns true if ARCH effects detected (5% level)
    pub fn has_arch_effects(&self) -> bool {
        self.arch_p_value < 0.05
    }

    /// Returns true if residuals are not normal (Jarque-Bera test, 5% level)
    pub fn non_normal_residuals(&self) -> bool {
        self.jb_p_value < 0.05
    }

    /// Count number of issues detected
    pub fn count_issues(&self) -> usize {
        let mut count = 0;
        if self.has_positive_autocorr() || self.has_negative_autocorr() {
            count += 1;
        }
        if self.has_heteroscedasticity() {
            count += 1;
        }
        if self.has_misspecification() {
            count += 1;
        }
        if self.has_structural_break() {
            count += 1;
        }
        if self.has_arch_effects() {
            count += 1;
        }
        if self.non_normal_residuals() {
            count += 1;
        }
        count
    }
}

/// Collection of diagnostic tests for multiple assets
#[derive(Debug, Clone)]
pub struct ResidualDiagnostics {
    /// Diagnostics for each asset
    pub diagnostics: HashMap<String, AssetDiagnostics>,
}

impl ResidualDiagnostics {
    /// Perform comprehensive residual diagnostics for all assets
    ///
    /// # Arguments
    ///
    /// * `returns_excess` - T x N matrix of excess returns (time x assets)
    /// * `factors` - T x K matrix of factor returns (time x factors)
    /// * `cov_type` - Type of covariance matrix estimator
    /// * `asset_names` - Optional vector of asset names
    ///
    /// # Returns
    ///
    /// * `ResidualDiagnostics` - Collection of diagnostic tests for each asset
    pub fn fit(
        returns_excess: &Array2<f64>,
        factors: &Array2<f64>,
        cov_type: CovarianceType,
        asset_names: Option<Vec<String>>,
    ) -> Result<Self, GreenersError> {
        let (t, n) = returns_excess.dim();
        let (t_factors, k) = factors.dim();

        if t != t_factors {
            return Err(GreenersError::ShapeMismatch(format!(
                "Returns has {} observations but factors has {}",
                t, t_factors
            )));
        }

        let asset_names = asset_names.unwrap_or_else(|| {
            (0..n)
                .map(|i| format!("Asset{}", i + 1))
                .collect::<Vec<String>>()
        });

        if asset_names.len() != n {
            return Err(GreenersError::ShapeMismatch(format!(
                "Expected {} asset names but got {}",
                n,
                asset_names.len()
            )));
        }

        // Prepare X matrix with intercept
        let mut x_with_intercept = Array2::zeros((t, k + 1));
        x_with_intercept.column_mut(0).fill(1.0);
        for i in 0..k {
            x_with_intercept
                .column_mut(i + 1)
                .assign(&factors.column(i));
        }

        let mut diagnostics = HashMap::new();

        for (j, asset_name) in asset_names.iter().enumerate() {
            let y = returns_excess.column(j).to_owned();

            if t < k + 10 {
                continue;
            }

            match OLS::fit(&y, &x_with_intercept, cov_type.clone()) {
                Ok(model) => {
                    let resid = model.residuals(&y, &x_with_intercept);

                    // Compute all diagnostic tests
                    let dw = durbin_watson(&resid);
                    let (lb_stat, lb_p) = ljung_box(&resid, 12);
                    let (bp_stat, bp_p) = breusch_pagan(&resid, &x_with_intercept);
                    let (white_stat, white_p) = white_test(&resid, &x_with_intercept);
                    let (reset_f, reset_p) = reset_test(&y, &x_with_intercept, &resid);
                    let split_idx = t / 2;
                    let (chow_f, chow_p) = chow_test(&y, &x_with_intercept, split_idx);
                    let (arch_stat, arch_p) = arch_test(&resid, 12);
                    let (jb_stat, jb_p) = jarque_bera(&resid);

                    diagnostics.insert(
                        asset_name.clone(),
                        AssetDiagnostics {
                            asset: asset_name.clone(),
                            n_obs: t,
                            durbin_watson: dw,
                            lb_stat,
                            lb_p_value: lb_p,
                            bp_stat,
                            bp_p_value: bp_p,
                            white_stat,
                            white_p_value: white_p,
                            reset_f,
                            reset_p_value: reset_p,
                            chow_f,
                            chow_p_value: chow_p,
                            arch_stat,
                            arch_p_value: arch_p,
                            jb_stat,
                            jb_p_value: jb_p,
                        },
                    );
                }
                Err(_) => continue,
            }
        }

        Ok(ResidualDiagnostics { diagnostics })
    }

    /// Get diagnostics for a specific asset
    pub fn get(&self, asset: &str) -> Option<&AssetDiagnostics> {
        self.diagnostics.get(asset)
    }

    /// Get list of assets with issues (at least one test rejected at 5%)
    pub fn assets_with_issues(&self) -> Vec<String> {
        self.diagnostics
            .values()
            .filter(|d| d.count_issues() > 0)
            .map(|d| d.asset.clone())
            .collect()
    }

    /// Export to CSV string
    pub fn to_csv_string(&self) -> String {
        let mut csv = String::new();
        csv.push_str("Asset,N_Obs,DW,LB_Stat,LB_P,BP_Stat,BP_P,White_Stat,White_P,RESET_F,RESET_P,Chow_F,Chow_P,ARCH_Stat,ARCH_P,JB_Stat,JB_P,Issues\n");

        let mut assets: Vec<_> = self.diagnostics.keys().collect();
        assets.sort();

        for asset in assets {
            if let Some(diag) = self.diagnostics.get(asset) {
                csv.push_str(&format!(
                    "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}\n",
                    diag.asset,
                    diag.n_obs,
                    diag.durbin_watson,
                    diag.lb_stat,
                    diag.lb_p_value,
                    diag.bp_stat,
                    diag.bp_p_value,
                    diag.white_stat,
                    diag.white_p_value,
                    diag.reset_f,
                    diag.reset_p_value,
                    diag.chow_f,
                    diag.chow_p_value,
                    diag.arch_stat,
                    diag.arch_p_value,
                    diag.jb_stat,
                    diag.jb_p_value,
                    diag.count_issues()
                ));
            }
        }

        csv
    }
}

/// Durbin-Watson test for autocorrelation
fn durbin_watson(residuals: &Array1<f64>) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 1..n {
        let diff = residuals[i] - residuals[i - 1];
        numerator += diff * diff;
    }

    for i in 0..n {
        denominator += residuals[i] * residuals[i];
    }

    if denominator == 0.0 {
        return f64::NAN;
    }

    numerator / denominator
}

/// Ljung-Box test for autocorrelation
fn ljung_box(residuals: &Array1<f64>, max_lag: usize) -> (f64, f64) {
    let n = residuals.len();
    if n < max_lag + 2 {
        return (f64::NAN, f64::NAN);
    }

    // Compute autocorrelations
    let mean = residuals.mean().unwrap_or(0.0);
    let var = residuals.mapv(|x| (x - mean).powi(2)).sum() / n as f64;

    if var == 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let mut q_stat = 0.0;

    for lag in 1..=max_lag {
        let mut acf = 0.0;
        for i in lag..n {
            acf += (residuals[i] - mean) * (residuals[i - lag] - mean);
        }
        acf /= n as f64 * var;

        q_stat += acf * acf / (n - lag) as f64;
    }

    q_stat *= n as f64 * (n + 2) as f64;

    // P-value from chi-squared distribution with max_lag degrees of freedom
    let p_value = chi_squared_cdf_complement(q_stat, max_lag);

    (q_stat, p_value)
}

/// Breusch-Pagan test for heteroscedasticity
fn breusch_pagan(residuals: &Array1<f64>, x: &Array2<f64>) -> (f64, f64) {
    let n = residuals.len();
    let k = x.ncols();

    if n < k + 5 {
        return (f64::NAN, f64::NAN);
    }

    // Squared residuals
    let resid_sq = residuals.mapv(|r| r * r);
    let sigma_sq = resid_sq.sum() / n as f64;

    if sigma_sq == 0.0 {
        return (f64::NAN, f64::NAN);
    }

    // Normalize squared residuals
    let y_bp = resid_sq.mapv(|r| r / sigma_sq);

    // Regress normalized squared residuals on X
    match OLS::fit(&y_bp, x, CovarianceType::NonRobust) {
        Ok(model) => {
            // BP statistic = n * R²
            let bp_stat = n as f64 * model.r_squared;
            let df = k - 1; // Exclude intercept
            let p_value = chi_squared_cdf_complement(bp_stat, df);
            (bp_stat, p_value)
        }
        Err(_) => (f64::NAN, f64::NAN),
    }
}

/// White test for heteroscedasticity
fn white_test(residuals: &Array1<f64>, x: &Array2<f64>) -> (f64, f64) {
    let n = residuals.len();
    let k = x.ncols();

    if n < k * 2 + 5 {
        return (f64::NAN, f64::NAN);
    }

    // Squared residuals
    let resid_sq = residuals.mapv(|r| r * r);

    // Build augmented X matrix with squares and cross-products
    let mut x_white_cols = Vec::new();

    // Original variables
    for j in 0..k {
        x_white_cols.push(x.column(j).to_owned());
    }

    // Squares (excluding constant)
    for j in 1..k {
        let col = x.column(j);
        x_white_cols.push(col.mapv(|v| v * v));
    }

    // Cross products (excluding constant)
    for j in 1..k {
        for l in (j + 1)..k {
            let col_j = x.column(j);
            let col_l = x.column(l);
            let cross: Array1<f64> = col_j.iter().zip(col_l.iter()).map(|(a, b)| a * b).collect();
            x_white_cols.push(cross);
        }
    }

    let k_white = x_white_cols.len();
    let mut x_white = Array2::zeros((n, k_white));
    for (j, col) in x_white_cols.iter().enumerate() {
        x_white.column_mut(j).assign(col);
    }

    // Regress squared residuals on augmented X
    match OLS::fit(&resid_sq, &x_white, CovarianceType::NonRobust) {
        Ok(model) => {
            // White statistic = n * R²
            let white_stat = n as f64 * model.r_squared;
            let df = k_white - 1; // Exclude intercept
            let p_value = chi_squared_cdf_complement(white_stat, df);
            (white_stat, p_value)
        }
        Err(_) => (f64::NAN, f64::NAN),
    }
}

/// RESET test for functional form misspecification
fn reset_test(y: &Array1<f64>, x: &Array2<f64>, residuals: &Array1<f64>) -> (f64, f64) {
    let n = y.len();
    let k = x.ncols();

    if n < k + 5 {
        return (f64::NAN, f64::NAN);
    }

    // Fit original model to get fitted values
    match OLS::fit(y, x, CovarianceType::NonRobust) {
        Ok(model) => {
            let y_hat = x.dot(&model.params);
            let y_hat_sq = y_hat.mapv(|v| v * v);

            // Augment X with squared fitted values
            let mut x_reset = Array2::zeros((n, k + 1));
            for j in 0..k {
                x_reset.column_mut(j).assign(&x.column(j));
            }
            x_reset.column_mut(k).assign(&y_hat_sq);

            // Fit augmented model
            match OLS::fit(y, &x_reset, CovarianceType::NonRobust) {
                Ok(model_aug) => {
                    // F-test for added variable
                    let rss_restricted = residuals.mapv(|r| r * r).sum();
                    let resid_aug = model_aug.residuals(y, &x_reset);
                    let rss_unrestricted = resid_aug.mapv(|r| r * r).sum();

                    let df1 = 1; // One added variable
                    let df2 = n - k - 1;

                    let numerator = (rss_restricted - rss_unrestricted) / df1 as f64;
                    let denominator = rss_unrestricted / df2 as f64;

                    if denominator <= 0.0 {
                        return (f64::NAN, f64::NAN);
                    }

                    let f_stat = numerator / denominator;
                    let p_value = f_cdf_complement(f_stat, df1, df2);

                    (f_stat, p_value)
                }
                Err(_) => (f64::NAN, f64::NAN),
            }
        }
        Err(_) => (f64::NAN, f64::NAN),
    }
}

/// Chow test for structural break
fn chow_test(y: &Array1<f64>, x: &Array2<f64>, split_idx: usize) -> (f64, f64) {
    let n = y.len();
    let k = x.ncols();

    if split_idx < k + 2 || n - split_idx < k + 2 {
        return (f64::NAN, f64::NAN);
    }

    // Full sample
    let rss_full = match OLS::fit(y, x, CovarianceType::NonRobust) {
        Ok(model) => {
            let resid = model.residuals(y, x);
            resid.mapv(|r| r * r).sum()
        }
        Err(_) => return (f64::NAN, f64::NAN),
    };

    // First subsample
    let y1 = y.slice(s![..split_idx]).to_owned();
    let x1 = x.slice(s![..split_idx, ..]).to_owned();

    let rss1 = match OLS::fit(&y1, &x1, CovarianceType::NonRobust) {
        Ok(model) => {
            let resid = model.residuals(&y1, &x1);
            resid.mapv(|r| r * r).sum()
        }
        Err(_) => return (f64::NAN, f64::NAN),
    };

    // Second subsample
    let y2 = y.slice(s![split_idx..]).to_owned();
    let x2 = x.slice(s![split_idx.., ..]).to_owned();

    let rss2 = match OLS::fit(&y2, &x2, CovarianceType::NonRobust) {
        Ok(model) => {
            let resid = model.residuals(&y2, &x2);
            resid.mapv(|r| r * r).sum()
        }
        Err(_) => return (f64::NAN, f64::NAN),
    };

    let rss_split = rss1 + rss2;
    let numerator = (rss_full - rss_split) / k as f64;
    let denominator = rss_split / (n - 2 * k) as f64;

    if denominator <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let f_stat = numerator / denominator;
    let df1 = k;
    let df2 = n - 2 * k;
    let p_value = f_cdf_complement(f_stat, df1, df2);

    (f_stat, p_value)
}

/// ARCH test (Engle's test for conditional heteroscedasticity)
fn arch_test(residuals: &Array1<f64>, lags: usize) -> (f64, f64) {
    let n = residuals.len();
    if n < lags + 10 {
        return (f64::NAN, f64::NAN);
    }

    // Squared residuals
    let resid_sq = residuals.mapv(|r| r * r);

    // Build lagged squared residuals matrix
    let n_eff = n - lags;
    let mut x_arch = Array2::zeros((n_eff, lags + 1));
    x_arch.column_mut(0).fill(1.0); // Intercept

    for lag in 1..=lags {
        for i in 0..n_eff {
            x_arch[[i, lag]] = resid_sq[i + lags - lag];
        }
    }

    let y_arch = resid_sq.slice(s![lags..]).to_owned();

    // Regress squared residuals on lagged squared residuals
    match OLS::fit(&y_arch, &x_arch, CovarianceType::NonRobust) {
        Ok(model) => {
            // ARCH statistic = n * R²
            let arch_stat = n_eff as f64 * model.r_squared;
            let p_value = chi_squared_cdf_complement(arch_stat, lags);
            (arch_stat, p_value)
        }
        Err(_) => (f64::NAN, f64::NAN),
    }
}

/// Jarque-Bera test for normality
fn jarque_bera(residuals: &Array1<f64>) -> (f64, f64) {
    let n = residuals.len();
    if n < 4 {
        return (f64::NAN, f64::NAN);
    }

    let mean = residuals.mean().unwrap_or(0.0);
    let std = residuals.std(0.0);

    if std == 0.0 {
        return (f64::NAN, f64::NAN);
    }

    // Standardized residuals
    let std_resid = residuals.mapv(|r| (r - mean) / std);

    // Skewness
    let skew = std_resid.mapv(|r| r.powi(3)).sum() / n as f64;

    // Kurtosis
    let kurt = std_resid.mapv(|r| r.powi(4)).sum() / n as f64;

    // JB statistic
    let jb = (n as f64 / 6.0) * (skew.powi(2) + (kurt - 3.0).powi(2) / 4.0);

    // P-value from chi-squared with 2 df
    let p_value = chi_squared_cdf_complement(jb, 2);

    (jb, p_value)
}

/// Chi-squared CDF complement (reuse from previous modules)
fn chi_squared_cdf_complement(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    let k = df as f64 / 2.0;
    let x_half = x / 2.0;

    // For large df, use normal approximation
    if df > 100 {
        let mean = df as f64;
        let variance = 2.0 * df as f64;
        let z = (x - mean) / variance.sqrt();
        return normal_cdf_complement(z);
    }

    gamma_p_complement(k, x_half)
}

/// F-distribution CDF complement (reuse from GRS test)
fn f_cdf_complement(x: f64, df1: usize, df2: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    let d1 = df1 as f64;
    let d2 = df2 as f64;
    let y = d1 * x / (d1 * x + d2);

    incomplete_beta_complement(d2 / 2.0, d1 / 2.0, y)
}

// Statistical helper functions (reuse from HJ distance and GRS test modules)
fn normal_cdf_complement(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        let t = 1.0 / (1.0 + 0.5 * x);
        t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp()
    } else {
        2.0 - erfc(-x)
    }
}

fn gamma_p_complement(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..100 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-10 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..100 {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

fn ln_gamma(x: f64) -> f64 {
    if x > 10.0 {
        return (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x)
            - 1.0 / (360.0 * x.powi(3));
    }

    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;

    for &coef in coefficients.iter() {
        y += 1.0;
        ser += coef / y;
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn incomplete_beta_complement(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }
    1.0 - incomplete_beta(a, b, x)
}

fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta_ab = ln_beta(a, b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta_ab).exp();
    let cf = beta_continued_fraction(a, b, x);
    front * cf / a
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let epsilon = 1e-10;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use greeners::CovarianceType;
    use ndarray::Array2;

    #[test]
    fn test_residual_diagnostics_basic() {
        let t = 120;
        let n = 5;
        let k = 2;

        let mut rng = 12345u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result =
            ResidualDiagnostics::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        assert_eq!(result.diagnostics.len(), n);

        for diag in result.diagnostics.values() {
            assert!((0.0..=4.0).contains(&diag.durbin_watson));
            assert!((0.0..=1.0).contains(&diag.lb_p_value));
            assert!((0.0..=1.0).contains(&diag.jb_p_value));
        }
    }

    #[test]
    fn test_csv_export() {
        let t = 100;
        let n = 3;
        let k = 2;

        let mut rng = 42u64;
        let mut rand = || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        let factors = Array2::from_shape_fn((t, k), |_| rand() * 0.02);
        let returns = Array2::from_shape_fn((t, n), |_| rand() * 0.03);

        let result =
            ResidualDiagnostics::fit(&returns, &factors, CovarianceType::HC3, None).unwrap();

        let csv = result.to_csv_string();
        assert!(csv.contains("Asset,N_Obs,DW"));
        assert!(csv.contains("Asset1"));
    }

    #[test]
    fn test_durbin_watson() {
        // Test with residuals (DW should be between 0 and 4)
        let resid = Array1::from_vec(vec![0.1, -0.1, 0.05, -0.05, 0.08, -0.08]);
        let dw = durbin_watson(&resid);
        // DW statistic is bounded between 0 and 4
        assert!((0.0..=4.0).contains(&dw));

        // Test with positive autocorrelation (low DW)
        let resid_pos = Array1::from_vec(vec![0.1, 0.11, 0.12, 0.11, 0.10, 0.09]);
        let dw_pos = durbin_watson(&resid_pos);
        assert!(dw_pos < 2.0); // Should indicate positive autocorrelation

        // Test with negative autocorrelation (high DW)
        let resid_neg = Array1::from_vec(vec![0.1, -0.1, 0.1, -0.1, 0.1, -0.1]);
        let dw_neg = durbin_watson(&resid_neg);
        assert!(dw_neg > 2.0); // Should indicate negative autocorrelation
    }

    #[test]
    fn test_jarque_bera() {
        // Approximately normal
        let resid = Array1::from_vec(vec![
            0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.02, -0.02, 0.12, -0.12,
        ]);
        let (jb, p) = jarque_bera(&resid);
        assert!(jb >= 0.0);
        assert!((0.0..=1.0).contains(&p));
    }
}
