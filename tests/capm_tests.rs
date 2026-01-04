use frenchrs::CAPM;
use greeners::{CovarianceType, DataFrame};
use ndarray::array;

#[test]
fn test_capm_basic_fit() {
    // Synthetic data simples
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
    assert!(capm.beta > 0.0); // Beta should be positivo for correlated asset
    assert!(capm.r_squared >= 0.0 && capm.r_squared <= 1.0);
    assert!(capm.adj_r_squared <= capm.r_squared);
}

#[test]
fn test_capm_perfect_correlation() {
    // Asset idêntico to the market: beta = 1, alpha = 0, R² = 1
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015, -0.008, 0.022];
    let asset_returns = market_returns.clone();
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Beta should be ~1
    assert!(
        (result.beta - 1.0).abs() < 0.01,
        "Beta shouldria be ~1, got {}",
        result.beta
    );

    // Alpha should be ~0
    assert!(
        result.alpha.abs() < 0.01,
        "Alpha shouldria be ~0, got {}",
        result.alpha
    );

    // R² should be ~1
    assert!(
        result.r_squared > 0.99,
        "R² shouldria be ~1, got {}",
        result.r_squared
    );

    // Residuals should be ~0
    let max_residual = result.residuals.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(max_residual < 0.01, "Residuals shouldriam be ~0");
}

#[test]
fn test_capm_defensive_asset() {
    // Asset defensivo: beta < 1
    // Market varies 2x more que the asset
    let market_returns = array![0.02, 0.04, -0.02, 0.06, 0.00, 0.03];
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015]; // Metade of the volatility
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Beta should be < 1 (defensive)
    assert!(
        result.beta < 1.0,
        "Defensive asset should have beta < 1, got {}",
        result.beta
    );
    assert!(result.beta > 0.0, "Beta should be positive");

    // Classification
    assert!(
        result.risk_classification().contains("Defensive")
            || result.risk_classification().contains("Neutral")
    );
}

#[test]
fn test_capm_aggressive_asset() {
    // Asset agressivo: beta > 1
    // Asset varies 1.5x more que the market
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015, -0.005];
    let asset_returns = array![0.015, 0.03, -0.015, 0.045, 0.00, 0.0225, -0.0075];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Beta should be > 1 (aggressive)
    assert!(
        result.beta > 1.0,
        "Aggressive asset should have beta > 1, got {}",
        result.beta
    );

    // Classification
    assert!(result.risk_classification().contains("Aggressive"));
}

#[test]
fn test_capm_with_alpha() {
    // Asset with alpha positivo (outperformance)
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015];

    // Adicionar alpha of 0.5% ao mês to the asset
    let asset_returns = array![0.015, 0.025, -0.005, 0.035, 0.005, 0.020];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Alpha should be positivo
    assert!(
        result.alpha > 0.0,
        "Alpha shouldria be positivo, got {}",
        result.alpha
    );
}

#[test]
fn test_capm_dimension_mismatch() {
    let asset_returns = array![0.01, 0.02];
    let market_returns = array![0.01, 0.02, 0.03]; // Size diferente

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_capm_insufficient_data() {
    let asset_returns = array![0.01, 0.02]; // Apenas 2 obbevations
    let market_returns = array![0.01, 0.02];

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_capm_covariesnce_types() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let risk_free = 0.0001;

    // Tthisr diferentes tipos of covariância
    let cov_types = vec![
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC2,
        CovarianceType::HC3,
        CovarianceType::HC4,
    ];

    for cov_type in cov_types {
        let result = CAPM::fit(&asset_returns, &market_returns, risk_free, cov_type.clone());
        assert!(result.is_ok(), "Falhou with covariância {:?}", cov_type);

        let capm = result.unwrap();
        // Beta e alpha should be os mesmos, apenas SE muda
        assert!(capm.beta.is_finite());
        assert!(capm.alpha.is_finite());
        assert!(capm.beta_se > 0.0);
        assert!(capm.alpha_se > 0.0);
    }
}

#[test]
fn test_capm_predictions() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012];
    let risk_free = 0.0001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Tthisr predições
    let new_market_excess = array![0.01, -0.01, 0.02];
    let predictions = result.predict(&new_market_excess);

    assert_eq!(predictions.len(), 3);
    assert!(predictions.iter().all(|&x| x.is_finite()));

    // Predição for market = 0 should be aproximadamente alpha
    let zero_market = array![0.0];
    let pred_zero = result.predict(&zero_market);
    assert!((pred_zero[0] - result.alpha).abs() < 0.001);
}

#[test]
fn test_capm_expected_return() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03];
    let market_returns = array![0.008, 0.015, -0.005, 0.025];
    let risk_free = 0.001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Test of return esperado with diferentes cenários
    let expected_10pct = result.expected_return(0.10);
    let expected_0pct = result.expected_return(0.0);
    let expected_neg5pct = result.expected_return(-0.05);

    assert!(expected_10pct.is_finite());
    assert!(expected_0pct.is_finite());
    assert!(expected_neg5pct.is_finite());

    // Com beta positivo: return esperado should aumentar with return of the market
    if result.beta > 0.0 {
        assert!(expected_10pct > expected_0pct);
        assert!(expected_0pct > expected_neg5pct);
    }
}

#[test]
fn test_capm_risk_decomposition() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Verificar decomposição of variesnce
    let total_variesnce = result.asset_volatility.powi(2);
    let explained_variesnce = result.systematic_variesnce + result.idiosyncratic_variesnce;

    // A soma should be próxima of the variesnce total
    assert!(
        (total_variesnce - explained_variesnce).abs() / total_variesnce < 0.1,
        "Decomposition of variesnce inconsistente: total={}, explieach={}",
        total_variesnce,
        explained_variesnce
    );

    // Variances should be not-negativas
    assert!(result.systematic_variesnce >= 0.0);
    assert!(result.idiosyncratic_variesnce >= 0.0);
}

#[test]
fn test_capm_sharpe_ratio() {
    let asset_returns = array![0.02, 0.03, 0.01, 0.04, 0.02, 0.03];
    let market_returns = array![0.015, 0.025, 0.008, 0.03, 0.015, 0.025];
    let risk_free = 0.001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Sharpe ratio should be finito
    assert!(result.sharpe_ratio.is_finite());
    assert!(result.market_sharpe.is_finite());

    // Com returns positivos, Sharpe should be positivo
    assert!(result.sharpe_ratio > 0.0);
    assert!(result.market_sharpe > 0.0);
}

#[test]
fn test_capm_from_dataframe() {
    let df = DataFrame::builder()
        .add_column("asset", vec![0.01, 0.02, -0.01, 0.03, 0.015, -0.005])
        .add_column("market", vec![0.008, 0.015, -0.005, 0.025, 0.012, -0.003])
        .build()
        .unwrap();

    let result = CAPM::from_dataframe(&df, "asset", "market", 0.0001, CovarianceType::HC3);

    assert!(result.is_ok());
    let capm = result.unwrap();
    assert_eq!(capm.n_obs, 6);
}

#[test]
fn test_capm_significance_tests() {
    // Createsr data where alpha é significant
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015, -0.005, 0.025];

    // Adicionar alpha substancial
    let asset_returns = array![0.02, 0.03, 0.00, 0.04, 0.01, 0.025, 0.005, 0.035];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Tthisr métodos of significance
    let is_outperforming = result.is_significantly_outperforming(0.05);
    let is_underperforming = result.is_significantly_underperforming(0.05);

    // Não can be ambos
    assert!(!(is_outperforming && is_underperforming));
}

#[test]
fn test_capm_beta_different_from_one() {
    // Asset with beta claramente diferente of 1
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015];
    let asset_returns = array![0.02, 0.04, -0.02, 0.06, 0.00, 0.03]; // Beta ~ 2
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Com beta ~2, should be significantly diferente of 1
    // (mas depende of the size of the amostra e SE)
    assert!(
        result.beta > 1.5,
        "Beta shouldria be ~2, got {}",
        result.beta
    );
}

#[test]
fn test_capm_performance_classification() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012];
    let risk_free = 0.0001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    let classification = result.performance_classification();

    // Deve retornar a string not vazia
    assert!(!classification.is_empty());

    // Deve conter a of the palavras-chave esperadas
    assert!(
        classification.contains("Outperformance")
            || classification.contains("Underperformance")
            || classification.contains("Neutro")
    );
}

#[test]
fn test_capm_tracking_error() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Tracking error should be positivo e finito
    assert!(result.tracking_error > 0.0);
    assert!(result.tracking_error.is_finite());

    // Tracking error é o standard deviation of the residuals
    let residuals_std = result.residuals.std(0.0);
    assert!((result.tracking_error - residuals_std).abs() < 0.0001);
}

#[test]
fn test_capm_information_ratio() {
    let asset_returns = array![0.015, 0.025, -0.005, 0.035, 0.020];
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.015];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Information ratio should be finito
    assert!(result.information_ratio.is_finite());

    // IR = alpha / tracking_error
    if result.tracking_error > 0.0 {
        let expected_ir = result.alpha / result.tracking_error;
        assert!((result.information_ratio - expected_ir).abs() < 0.0001);
    }
}

#[test]
fn test_capm_treynor_ratio() {
    let asset_returns = array![0.02, 0.03, 0.01, 0.04, 0.025];
    let market_returns = array![0.015, 0.025, 0.008, 0.03, 0.020];
    let risk_free = 0.001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Treynor ratio should be finito
    assert!(result.treynor_ratio.is_finite());

    // Treynor = (E[R] - Rf) / beta
    if result.beta != 0.0 {
        let expected_treynor = (result.mean_asset_return - risk_free) / result.beta;
        assert!((result.treynor_ratio - expected_treynor).abs() < 0.0001);
    }
}

#[test]
fn test_capm_confidence_intervals() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let risk_free = 0.0001;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Confidence intervals should fazer sentido
    assert!(result.alpha_conf_lower < result.alpha_conf_upper);
    assert!(result.beta_conf_lower < result.beta_conf_upper);

    // Estimatestiva pontual should thisr dentro of the intervalo
    assert!(result.alpha >= result.alpha_conf_lower);
    assert!(result.alpha <= result.alpha_conf_upper);
    assert!(result.beta >= result.beta_conf_lower);
    assert!(result.beta <= result.beta_conf_upper);
}
