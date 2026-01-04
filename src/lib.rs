//! # Frenchrs - Biblioteca Rust para Precificação de Ativos
//!
//! Frenchrs é uma biblioteca de alto desempenho para precificação de ativos e análise financeira,
//! construída em Rust e aproveitando a infraestrutura econométrica robusta do Greeners.
//!
//! ## Modelos Implementados
//!
//! - **CAPM** (Capital Asset Pricing Model): Modelo de precificação de ativos baseado no risco sistemático
//! - **Fama-French 3 Factor**: CAPM + fatores tamanho (SMB) e valor (HML)
//! - **Carhart 4 Factor**: Fama-French 3 Factor + fator momentum (MOM)
//! - **Fama-French 5 Factor**: FF3 + fatores rentabilidade (RMW) e investimento (CMA)
//! - **Fama-French 6 Factor**: FF5 + fator momentum (UMD)
//! - **APT** (Arbitrage Pricing Theory): Modelo multi-fatorial genérico com N fatores
//!
//! ## Roadmap
//!
//! - Value-at-Risk (VaR)
//! - Conditional VaR (CVaR)
//! - Portfolio Optimization
//! - Black-Litterman Model
//!
//! ## Exemplo Básico
//!
//! ```rust
//! use frenchrs::CAPM;
//! use greeners::CovarianceType;
//! use ndarray::array;
//!
//! // Retornos diários
//! let asset_returns = array![0.01, 0.02, -0.01, 0.03];
//! let market_returns = array![0.008, 0.015, -0.005, 0.025];
//! let risk_free_rate = 0.0001; // taxa diária (~2.5% ao ano)
//!
//! // Estimar CAPM com erros padrão robustos (HC3)
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

// Re-exports do Greeners para conveniência
pub use greeners::{CovarianceType, DataFrame, Formula, GreenersError, InferenceType, OLS};

// Módulos da biblioteca
pub mod apt;
pub mod capm;
pub mod carhart;
pub mod fama_french_3f;
pub mod fama_french_5f;
pub mod fama_french_6f;
pub mod fama_macbeth;
pub mod ivol_tracking_multi;
pub mod risk_metrics;
pub mod rolling_betas_multi;

// Re-exports principais
pub use apt::{APT, APTResult};
pub use capm::{CAPM, CAPMResult};
pub use carhart::{Carhart4Factor, Carhart4FactorResult};
pub use fama_french_3f::{FamaFrench3Factor, FamaFrench3FactorResult};
pub use fama_french_5f::{FamaFrench5Factor, FamaFrench5FactorResult};
pub use fama_french_6f::{FamaFrench6Factor, FamaFrench6FactorResult};
pub use fama_macbeth::{FamaMacBeth, FamaMacBethResult};
pub use ivol_tracking_multi::{IVOLTrackingAsset, IVOLTrackingMulti, IVOLTrackingRow};
pub use risk_metrics::{IVOLAnalysis, TrackingErrorAnalysis};
pub use rolling_betas_multi::{
    BetaStability, RollingBetasAsset, RollingBetasMulti, RollingBetasRow,
};
