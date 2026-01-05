
# Frenchrs v0.1.0

A high-performance Rust library for asset pricing and financial analysis, built on the robust econometric infrastructure of [Greeners](https://crates.io/crates/greeners).

## 📊 Implemented Models

### Classic Models

* **CAPM** (Capital Asset Pricing Model, 1964): A fundamental pricing model based on systematic risk.
* **Formula**: R_i - R_f = α + β(R_m - R_f) + ε

### Fama-French Models

* **Fama-French 3 Factor** (1993): Extends CAPM with size (SMB) and value (HML) factors. It provides a significant improvement in explanatory power.
* **Fama-French 5 Factor** (2015): Adds profitability (RMW) and investment (CMA) factors to the FF3 model. This represents the state-of-the-art in asset pricing.
* **Fama-French 6 Factor** (2018): Incorporates the momentum factor (UMD/Up Minus Down) into the FF5 framework. This is the most comprehensive model available in the library.

### Multi-factor Models

* **Carhart 4 Factor** (1997): Combines FF3 with the momentum factor (MOM). It is popular for analyzing investment funds.
* **APT** (Arbitrage Pricing Theory, 1976): A generic framework utilizing N arbitrary factors. Offers maximum flexibility for research and custom models.

### Cross-Sectional Analysis

* **Fama-MacBeth Two-Pass Regression** (1973): Cross-sectional analysis for estimating risk premia.
  * First pass: Time-series regressions to estimate betas
  * Second pass: Cross-sectional regressions to estimate factor risk premia
  * Includes Shanken (1992) correction for standard errors

* **Hansen-Jagannathan Distance** (1997): Tests whether a factor model correctly prices assets.
  * Measures distance between model SDF and set of valid SDFs
  * Chi-squared test for model specification (H₀: d = 0)
  * Identifies pricing errors (alphas) across assets
  * Assesses overall model fit quality
  * Classification system for model adequacy

* **GRS Test** (Gibbons, Ross & Shanken, 1989): Joint test of whether all alphas are zero.
  * F-test for H₀: α₁ = α₂ = ... = αₙ = 0
  * Accounts for cross-correlation of residuals
  * More powerful than testing alphas individually
  * Controls for factor sampling variability
  * Standard test for model evaluation

### Model Diagnostics

* **Residual Diagnostics**: Comprehensive suite of 8 diagnostic tests to validate regression assumptions
  * **Durbin-Watson**: Tests for first-order autocorrelation in residuals
  * **Ljung-Box**: Tests for autocorrelation at multiple lags (default: 12)
  * **Breusch-Pagan**: Tests for heteroscedasticity (non-constant variance)
  * **White**: General heteroscedasticity test with squares and cross-products
  * **RESET**: Ramsey's test for functional form misspecification
  * **Chow**: Tests for structural breaks at midpoint
  * **ARCH**: Engle's test for conditional heteroscedasticity (volatility clustering)
  * **Jarque-Bera**: Tests for normality of residuals (skewness and kurtosis)
  * **Multi-asset batch processing** with CSV export and issue detection
  * Automatic classification of violations at 5% significance level

---

## 🛡️ Risk Metrics

### Idiosyncratic Volatility (IVOL)
* Measures asset-specific volatility not explained by model factors
* Provides annualized IVOL (monthly and daily data)
* Includes complete residual statistics (skewness, kurtosis)
* Features Jarque-Bera normality test
* **Multi-asset batch processing** for portfolio-wide analysis

### Tracking Error Analysis
* Ex-post tracking error and information ratio
* Optional benchmark comparison
* Rolling tracking error with configurable windows
* Fit quality metrics (RMSE, MAE, correlation)
* **Multi-asset batch processing** with CSV export

### Residual Correlation Analysis
* Computes correlation matrix of model residuals
* Detects common unmodeled factors
* Identifies asset clusters with correlated idiosyncratic risk
* Summary statistics: average/min/max off-diagonal correlation
* Maximum eigenvalue analysis
* Classification of correlation levels (low/moderate/high)
* **Multi-asset batch processing** with CSV export

---

## 🕒 Temporal Analysis

### Rolling Betas
* Moving window analysis for multi-factor models (generic N-factor framework)
* Tracks temporal evolution of alphas and betas
* **Beta Stability Analysis**:
  * Coefficient of Variation (CV)
  * Linear trend detection
  * Autocorrelation (lag-1)
  * Automatic stability classification
* Identifies structural changes in factor loadings
* **Multi-asset batch processing** with table/CSV export

### Out-of-Sample (OOS) Performance
* Evaluates true predictive power of factor models
* In-sample vs. out-of-sample metrics (R², RMSE, MAE)
* **Campbell-Thompson R²_OOS**: Measures predictive power vs. historical mean benchmark
  * Positive values indicate model beats naive forecast
  * Critical for assessing real-world model utility
* Automatic overfitting detection
* Configurable train/test split ratios
* **Multi-asset batch processing** with summary statistics

---

## 🚀 Key Features

* ✅ **High Performance**: Built in Rust using BLAS/LAPACK for speed
* ✅ **Statistically Robust**: Supports multiple standard error types (HC0-HC4, Newey-West, Clustering)
* ✅ **Comprehensive**: Provides t-statistics, p-values, confidence intervals, and performance metrics
* ✅ **Flexible**: Compatible with DataFrames and `ndarray` arrays
* ✅ **Batch Processing**: Multi-asset analysis with single function calls
* ✅ **Thoroughly Tested**: Over **176 unit and integration tests**
* ✅ **Well-Documented**: Full examples and inline documentation in English

---

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
frenchrs = "0.1.0"
greeners = "1.3.2"
ndarray = "0.17.1"
```

---

## 📚 Basic Usage

### CAPM Example

```rust
use frenchrs::CAPM;
use greeners::CovarianceType;
use ndarray::array;

let asset_returns = array![0.01, 0.02, -0.01, 0.03];
let market_returns = array![0.008, 0.015, -0.005, 0.025];
let risk_free_rate = 0.0001;

let result = CAPM::fit(
    &asset_returns,
    &market_returns,
    risk_free_rate,
    CovarianceType::HC3,
).unwrap();

println!("Beta: {:.4}", result.beta);
println!("Alpha: {:.4}", result.alpha);
println!("R²: {:.4}", result.r_squared);
```

### APT (Arbitrage Pricing Theory) Example

```rust
use frenchrs::APT;
use greeners::CovarianceType;
use ndarray::{array, Array2};

let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];

// Factor Matrix (n_obs × n_factors)
let factors = Array2::from_shape_vec((7, 3), vec![
    0.008, 0.002, 0.001,
    0.015, -0.001, 0.002,
    -0.005, 0.003, -0.002,
    0.025, 0.001, 0.003,
    0.012, -0.002, 0.001,
    -0.003, 0.001, -0.001,
    0.020, 0.002, 0.002,
]).unwrap();

let result = APT::fit(
    &returns,
    &factors,
    0.0001,
    CovarianceType::HC3,
    Some(vec!["Market".into(), "Size".into(), "Value".into()]),
).unwrap();
```

### Multi-Asset Out-of-Sample Performance

```rust
use frenchrs::OOSPerformance;
use greeners::CovarianceType;

// returns_excess: T x N matrix (time x assets)
// factors: T x K matrix (time x factors)
let result = OOSPerformance::fit(
    &returns_excess,
    &factors,
    0.7,  // 70% in-sample, 30% out-of-sample
    CovarianceType::HC3,
    Some(asset_names),
).unwrap();

// Get summary statistics
let stats = result.summary_stats();
println!("Assets beating benchmark: {} ({:.1}%)",
    stats.assets_beating_benchmark,
    stats.pct_beating_benchmark
);

// Export to CSV
let csv = result.to_csv_string();
```

### Multi-Asset IVOL & Tracking Error

```rust
use frenchrs::IVOLTrackingMulti;
use greeners::CovarianceType;

// Analyze multiple assets at once
let result = IVOLTrackingMulti::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    Some(&benchmark), // Optional benchmark for tracking error
    CovarianceType::HC3,
    Some(asset_names),
    12.0,  // 12 periods per year (monthly data)
).unwrap();

// Export results
let csv = result.to_csv_string();
```

### Multi-Asset Rolling Betas

```rust
use frenchrs::RollingBetasMulti;
use greeners::CovarianceType;

// Rolling betas for multiple assets
let result = RollingBetasMulti::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    36,               // 36-month window
    CovarianceType::HC3,
    Some(asset_names),
    Some(factor_names),
).unwrap();

// Analyze beta stability
for (asset_name, asset_result) in &result.results {
    for (i, factor_name) in factor_names.iter().enumerate() {
        let stability = asset_result.beta_stability(i);
        println!("{} - {}: CV = {:.4}, Trend = {:.4}",
            asset_name, factor_name,
            stability.coefficient_of_variation,
            stability.trend
        );
    }
}
```

### Residual Correlation Analysis

```rust
use frenchrs::ResidualCorrelation;
use greeners::CovarianceType;

// Analyze residual correlation to detect missing factors
let result = ResidualCorrelation::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    CovarianceType::HC3,
    Some(asset_names),
).unwrap();

// Check model specification quality
let summary = result.summary_stats();
println!("Avg off-diagonal correlation: {:.4}", summary.avg_off_diag_corr);
println!("Classification: {}", summary.correlation_classification());

// Find most correlated assets (may indicate missing common factor)
let top_correlated = result.most_correlated_assets("Asset1", 3);
for (name, corr) in top_correlated {
    println!("{}: {:.4}", name, corr);
}

// Export correlation matrix
let csv = result.correlation_to_csv_string();
```

### Hansen-Jagannathan Distance Test

```rust
use frenchrs::HJDistance;
use greeners::CovarianceType;

// Test whether factor model correctly prices assets
let result = HJDistance::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    CovarianceType::HC3,
    Some(asset_names),
).unwrap();

// Check model quality
println!("HJ Distance: {:.6}", result.hj_distance);
println!("P-value: {:.4}", result.p_value);
println!("Model Quality: {}", result.model_quality_classification());

// Test model specification
if result.reject_model(0.05) {
    println!("Model has significant pricing errors");

    // Identify assets with largest pricing errors
    for name in &result.asset_names {
        let alpha = result.get_alpha(name).unwrap();
        println!("{}: alpha = {:.6}", name, alpha);
    }
}

// Export results
let csv = result.to_csv_string();
```

### GRS Test (Gibbons, Ross & Shanken)

```rust
use frenchrs::GRSTest;
use greeners::CovarianceType;

// Joint test: Are all alphas simultaneously zero?
let result = GRSTest::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    CovarianceType::HC3,
    Some(asset_names),
).unwrap();

// Check test results
println!("GRS F-statistic: {:.4}", result.grs_f_stat);
println!("P-value: {:.4}", result.p_value);
println!("DF: F({}, {})", result.df1, result.df2);

// Test decision
if result.reject_model(0.05) {
    println!("Reject H₀: At least one alpha ≠ 0");
    println!("Model fails to price all assets jointly");
} else {
    println!("Do not reject H₀: All alphas jointly zero");
    println!("Model adequately prices the cross-section");
}

// Individual alphas (for diagnostics)
for name in &result.asset_names {
    let alpha = result.get_alpha(name).unwrap();
    println!("{}: {:.6}", name, alpha);
}
```

### Residual Diagnostics

```rust
use frenchrs::ResidualDiagnostics;
use greeners::CovarianceType;

// Run comprehensive diagnostic tests on residuals
let result = ResidualDiagnostics::fit(
    &returns_excess,  // T x N
    &factors,         // T x K
    CovarianceType::HC3,
    Some(asset_names),
).unwrap();

// Check diagnostics for each asset
for (asset_name, diag) in &result.diagnostics {
    println!("\nAsset: {}", asset_name);

    // Autocorrelation tests
    println!("Durbin-Watson: {:.4}", diag.durbin_watson);
    if diag.has_positive_autocorr() {
        println!("  ⚠ Positive autocorrelation detected");
    }

    println!("Ljung-Box: {:.4} (p = {:.4})", diag.lb_stat, diag.lb_p_value);

    // Heteroscedasticity tests
    println!("Breusch-Pagan: {:.4} (p = {:.4})", diag.bp_stat, diag.bp_p_value);
    if diag.has_heteroscedasticity() {
        println!("  ⚠ Heteroscedasticity detected");
    }

    println!("White: {:.4} (p = {:.4})", diag.white_stat, diag.white_p_value);
    println!("ARCH: {:.4} (p = {:.4})", diag.arch_stat, diag.arch_p_value);

    // Specification tests
    println!("RESET: {:.4} (p = {:.4})", diag.reset_f, diag.reset_p_value);
    println!("Chow: {:.4} (p = {:.4})", diag.chow_f, diag.chow_p_value);

    // Normality test
    println!("Jarque-Bera: {:.4} (p = {:.4})", diag.jb_stat, diag.jb_p_value);
    if diag.non_normal_residuals() {
        println!("  ⚠ Non-normal residuals");
    }

    // Overall assessment
    let issues = diag.count_issues();
    if issues == 0 {
        println!("  ✓ All diagnostic tests passed");
    } else {
        println!("  ✗ {} diagnostic test(s) failed", issues);
    }
}

// Export to CSV
let csv = result.to_csv_string();
```

---

## 🔬 Provided Statistics

All models return:

* **Parameters**: α (alpha) and β (factor betas)
* **Inference**: Standard errors, t-statistics, p-values, and confidence intervals
* **Fit Quality**: R², Adjusted R², tracking error, and information ratio
* **Diagnostics**: Residuals and fitted values
* **Classifications**: Categorizations for performance, size, value, profitability, momentum, etc.

---

## 📈 Performance and Optimization

Frenchrs is optimized for maximum efficiency:

* Uses **BLAS/LAPACK** via `ndarray-linalg`
* Supports **multi-core processing** when available
* Utilizes **zero-copy** operations wherever possible
* Batch processing for analyzing multiple assets efficiently
* Supports **LTO (Link Time Optimization)** for release builds

---

## 🗺️ Roadmap

Implemented:
* ✅ Classic asset pricing models (CAPM, FF3/5/6, Carhart, APT)
* ✅ Fama-MacBeth two-pass regression
* ✅ GRS test (Gibbons, Ross & Shanken)
* ✅ Hansen-Jagannathan distance test
* ✅ Residual diagnostics (8 comprehensive tests)
* ✅ IVOL and Tracking Error analysis (single and multi-asset)
* ✅ Residual correlation analysis (multi-asset)
* ✅ Rolling betas with stability analysis (multi-asset)
* ✅ Out-of-sample performance evaluation

Planned:
* [ ] Value-at-Risk (VaR) and Conditional VaR (CVaR)
* [ ] Portfolio Optimization (Markowitz, Black-Litterman)
* [ ] Python Bindings (PyO3)
* [ ] Support for irregular time series
* [ ] Additional cross-sectional tests

---

## 📚 References

1. **Sharpe, W. F.** (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk". *Journal of Finance*.
2. **Ross, S. A.** (1976). "The Arbitrage Theory of Capital Asset Pricing". *Journal of Economic Theory*.
3. **Fama, E. F., & MacBeth, J. D.** (1973). "Risk, Return, and Equilibrium: Empirical Tests". *Journal of Political Economy*.
4. **Shanken, J.** (1992). "On the Estimation of Beta-Pricing Models". *Review of Financial Studies*.
5. **Gibbons, M. R., Ross, S. A., & Shanken, J.** (1989). "A Test of the Efficiency of a Given Portfolio". *Econometrica*.
6. **Fama, E. F., & French, K. R.** (1993). "Common Risk Factors in the Returns on Stocks and Bonds". *Journal of Financial Economics*.
7. **Carhart, M. M.** (1997). "On Persistence in Mutual Fund Performance". *Journal of Finance*.
8. **Hansen, L. P., & Jagannathan, R.** (1997). "Assessing Specification Errors in Stochastic Discount Factor Models". *Journal of Finance*.
9. **Fama, E. F., & French, K. R.** (2015). "A Five-Factor Asset Pricing Model". *Journal of Financial Economics*.
10. **Campbell, J. Y., & Thompson, S. B.** (2008). "Predicting Excess Stock Returns Out of Sample". *Journal of Financial Economics*.

---

**Developed with ❤️ in Rust for the quantitative finance community.**

