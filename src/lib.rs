//! # Frenchrs - Rust Library for Asset Pricing
//!
//! Frenchrs is a high-performance library for asset pricing and financial analysis,
//! built in Rust and leveraging the robust econametric infrastructure of Greeners.
//!
//! ## Implemented Models
//!
//! - **CAPM** (Capital Asset Pricing Model): Asset pricing model based on systematic risk
//! - **Fama-French 3 Factor**: CAPM + size (SMB) and value (HML) factors
//! - **Carhart 4 Factor**: Fama-French 3 Factor + momentum factor (MOM)
//! - **Fama-French 5 Factor**: FF3 + profitability (RMW) and investment (CMA) factors
//! - **Fama-French 6 Factor**: FF5 + momentum factor (UMD)
//! - **APT** (Arbitrage Pricing Theory): Generic multi-factor model with N factors
//!
//! ## Roadmap
//!
//! - Value-at-Risk (VaR)
//! - Conditional VaR (CVaR)
//! - Portfolio Optimization
//! - Black-Litterman Model
//!
//! ## Basic Example
//!
//! ```rust
//! use frenchrs::CAPM;
//! use greeners::CovarianceType;
//! use ndarray::array;
//!
//! // Daily returns
//! let asset_returns = array![0.01, 0.02, -0.01, 0.03];
//! let market_returns = array![0.008, 0.015, -0.005, 0.025];
//! let risk_free_rate = 0.0001; // daily rate (~2.5% per year)
//!
//! // Estimate CAPM with robust standard errors (HC3)
//! let result = CAPM::fit(
//!     &asset_returns,
//!     &market_returns,
//!     risk_free_rate,
//!     CovarianceType::HC3,
//! ).unwrap();
//!
//! println!("{}", result);
//! println!("Beta: {:.4}", result.beta);
//! println!("Alpha: {:.4}", result.alpha);
//! ```

// Re-exports from Greeners for convenience
pub use greeners::{CovarianceType, DataFrame, Formula, GreenersError, InferenceType, OLS};

// Library modules
pub mod apt;
pub mod capm;
pub mod carhart;
pub mod fama_french_3f;
pub mod fama_french_5f;
pub mod fama_french_6f;
pub mod fama_macbeth;
pub mod grs_test;
pub mod hj_distance;
pub mod ivol_tracking_multi;
pub mod oos_performance;
pub mod residual_correlation;
pub mod residual_diagnostics;
pub mod risk_metrics;
pub mod rolling_betas_multi;

// Main re-exports
pub use apt::{APTResult, APT};
pub use capm::{CAPMResult, CAPM};
pub use carhart::{Carhart4Factor, Carhart4FactorResult};
pub use fama_french_3f::{FamaFrench3Factor, FamaFrench3FactorResult};
pub use fama_french_5f::{FamaFrench5Factor, FamaFrench5FactorResult};
pub use fama_french_6f::{FamaFrench6Factor, FamaFrench6FactorResult};
pub use fama_macbeth::{FamaMacBeth, FamaMacBethResult};
pub use grs_test::GRSTest;
pub use hj_distance::HJDistance;
pub use ivol_tracking_multi::{IVOLTrackingAsset, IVOLTrackingMulti, IVOLTrackingRow};
pub use oos_performance::{
    OOSPerformance, OOSPerformanceAsset, OOSPerformanceRow, OOSSummaryStats,
};
pub use residual_correlation::{ResidualCorrSummary, ResidualCorrelation};
pub use residual_diagnostics::{AssetDiagnostics, ResidualDiagnostics};
pub use risk_metrics::{IVOLAnalysis, TrackingErrorAnalysis};
pub use rolling_betas_multi::{
    BetaStability, RollingBetasAsset, RollingBetasMulti, RollingBetasRow,
};
