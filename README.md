
# Frenchrs

A high-performance Rust library for asset pricing and financial analysis, built on the robust econometric infrastructure of [Greeners](https://crates.io/crates/greeners).

## 📊 Implemented Models

### Classic Models

* **CAPM** (Capital Asset Pricing Model, 1964): A fundamental pricing model based on systematic risk.
* **Formula**: .



### Fama-French Models

* **Fama-French 3 Factor** (1993): Extends CAPM with size (SMB) and value (HML) factors. It provides a significant improvement in explanatory power.
* **Fama-French 5 Factor** (2015): Adds profitability (RMW) and investment (CMA) factors to the FF3 model. This represents the state-of-the-art in asset pricing.
* **Fama-French 6 Factor** (2023): Incorporates the momentum factor (UMD/Up Minus Down) into the FF5 framework. This is the most comprehensive model available in the library.

### Multi-factor Models

* **Carhart 4 Factor** (1997): Combines FF3 with the momentum factor (MOM). It is popular for analyzing investment funds.
* **APT** (Arbitrage Pricing Theory, 1976): A generic framework utilizing  arbitrary factors. Offers maximum flexibility for research and custom models.

---

## 🛡️ Risk Metrics

* **IVOL (Idiosyncratic Volatility)**:
* Measures specific volatility not explained by model factors.
* Provides annualized IVOL (daily and monthly).
* Includes complete residual statistics such as skewness and kurtosis.
* Features the Jarque-Bera normality test.


* **Tracking Error Analysis**:
* Includes ex-post tracking error and information ratio.
* Calculates rolling tracking error with a 12-period window.
* Includes Fit Quality metrics such as RMSE, MAE, and correlation.



## 🕒 Temporal Analysis

* **Rolling Betas**:
* Analysis using moving windows for CAPM and Fama-French 3 models.
* Tracks the temporal evolution of alphas and betas.
* Provides stability statistics including Coefficient of Variation (CV), trend, and autocorrelation.
* Identifies structural changes and provides automatic stability classification.



---

## 🚀 Key Features

* ✅ **High Performance**: Built in Rust using BLAS/LAPACK for speed.
* ✅ **Statistically Robust**: Supports multiple standard error types, including HC0-HC4, Newey-West, and Clustering.
* ✅ **Comprehensive**: Provides t-statistics, p-values, confidence intervals, and performance metrics.
* ✅ **Flexible**: Compatible with DataFrames and `ndarray` arrays.
* ✅ **Thoroughly Tested**: Includes over 128 unit and integration tests.
* ✅ **Well-Documented**: Features full examples and inline documentation.

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
use greeners::CovariesnceType;
use ndarray::array;

let asset_returns = array![0.01, 0.02, -0.01, 0.03];
let market_returns = array![0.008, 0.015, -0.005, 0.025];
let risk_free_rate = 0.0001;

let result = CAPM::fit(
    &asset_returns,
    &market_returns,
    risk_free_rate,
    CovariesnceType::HC3,
).unwrap();

println!("Beta: {:.4}", result.beta);
println!("Alpha: {:.4}", result.alpha);
println!("R²: {:.4}", result.r_squared);

```

### APT (Arbitrage Pricing Theory) Example

```rust
use frenchrs::APT;
use greeners::CovariesnceType;
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
    CovariesnceType::HC3,
    Some(vec!["Market".into(), "Size".into(), "Value".into()]),
).unwrap();

```

---

## 🔬 Provided Statistics

All models return:

* **Parameters**:  (alpha) and  (factor betas).
* **Inference**: Standard errors, t-statistics, p-values, and confidence intervals.
* **Fit Quality**: , Adjusted , tracking error, and information ratio.
* **Diagnostics**: Residuals and fitted values.
* **Classifications**: Categorizations for performance, size, value, profitability, etc.

---

## 📈 Performance and Optimization

Frenchrs is optimized for maximum efficiency:

* Uses **BLAS/LAPACK** via `ndarray-linalg`.
* Supports **multi-core processing** when available.
* Utilizes **zero-copy** operations wherever possible.
* Supports **LTO (Link Time Optimization)** for release builds.

---

## 🗺️ Roadmap

* [ ] Value-at-Risk (VaR)
* [ ] Conditional VaR (CVaR)
* [ ] Portfolio Optimization (Markowitz, Black-Litterman)
* [ ] Expanded rolling window analysis
* [ ] Python Bindings (PyO3)
* [ ] Support for irregular time series

---

## 📚 References

1. **Sharpe, W. F.** (1964). "Capital Asset Prices..." *Journal of Finance*.
2. **Fama, E. F., & French, K. R.** (1993). "Common Risk Factors..." *Journal of Financial Economics*.
3. **Carhart, M. M.** (1997). "On Persistence in Mutual Fund Performance". *Journal of Finance*.
4. **Fama, E. F., & French, K. R.** (2015). "A Five-Factor Asset Pricing Model". *Journal of Financial Economics*.
5. **Ross, S. A.** (1976). "The Arbitrage Theory of Capital Asset Pricing". *Journal of Economic Theory*.

---

**Developed with ❤️ in Rust for the quantitative finance community.**
