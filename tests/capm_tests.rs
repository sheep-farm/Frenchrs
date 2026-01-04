use frenchrs::CAPM;
use greeners::{CovarianceType, DataFrame};
use ndarray::array;

#[test]
fn test_capm_basic_fit() {
    // Dados sintéticos simples
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
    assert!(capm.adj_r_squared <= capm.r_squared);
}

#[test]
fn test_capm_perfect_correlation() {
    // Ativo idêntico ao mercado: beta = 1, alpha = 0, R² = 1
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

    // Beta deve ser ~1
    assert!(
        (result.beta - 1.0).abs() < 0.01,
        "Beta deveria ser ~1, got {}",
        result.beta
    );

    // Alpha deve ser ~0
    assert!(
        result.alpha.abs() < 0.01,
        "Alpha deveria ser ~0, got {}",
        result.alpha
    );

    // R² deve ser ~1
    assert!(
        result.r_squared > 0.99,
        "R² deveria ser ~1, got {}",
        result.r_squared
    );

    // Resíduos devem ser ~0
    let max_residual = result.residuals.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(max_residual < 0.01, "Resíduos deveriam ser ~0");
}

#[test]
fn test_capm_defensive_asset() {
    // Ativo defensivo: beta < 1
    // Mercado varia 2x mais que o ativo
    let market_returns = array![0.02, 0.04, -0.02, 0.06, 0.00, 0.03];
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015]; // Metade da volatilidade
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::HC3,
    )
    .unwrap();

    // Beta deve ser < 1 (defensivo)
    assert!(
        result.beta < 1.0,
        "Ativo defensivo deveria ter beta < 1, got {}",
        result.beta
    );
    assert!(result.beta > 0.0, "Beta deveria ser positivo");

    // Classificação
    assert!(
        result.risk_classification().contains("Defensivo")
            || result.risk_classification().contains("Neutro")
    );
}

#[test]
fn test_capm_aggressive_asset() {
    // Ativo agressivo: beta > 1
    // Ativo varia 1.5x mais que o mercado
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

    // Beta deve ser > 1 (agressivo)
    assert!(
        result.beta > 1.0,
        "Ativo agressivo deveria ter beta > 1, got {}",
        result.beta
    );

    // Classificação
    assert!(result.risk_classification().contains("Agressivo"));
}

#[test]
fn test_capm_with_alpha() {
    // Ativo com alpha positivo (outperformance)
    let market_returns = array![0.01, 0.02, -0.01, 0.03, 0.00, 0.015];

    // Adicionar alpha de 0.5% ao mês ao ativo
    let asset_returns = array![0.015, 0.025, -0.005, 0.035, 0.005, 0.020];
    let risk_free = 0.0;

    let result = CAPM::fit(
        &asset_returns,
        &market_returns,
        risk_free,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Alpha deve ser positivo
    assert!(
        result.alpha > 0.0,
        "Alpha deveria ser positivo, got {}",
        result.alpha
    );
}

#[test]
fn test_capm_dimension_mismatch() {
    let asset_returns = array![0.01, 0.02];
    let market_returns = array![0.01, 0.02, 0.03]; // Tamanho diferente

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
    let asset_returns = array![0.01, 0.02]; // Apenas 2 observações
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
fn test_capm_covariance_types() {
    let asset_returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market_returns = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let risk_free = 0.0001;

    // Testar diferentes tipos de covariância
    let cov_types = vec![
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC2,
        CovarianceType::HC3,
        CovarianceType::HC4,
    ];

    for cov_type in cov_types {
        let result = CAPM::fit(&asset_returns, &market_returns, risk_free, cov_type.clone());
        assert!(result.is_ok(), "Falhou com covariância {:?}", cov_type);

        let capm = result.unwrap();
        // Beta e alpha devem ser os mesmos, apenas SE muda
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

    // Testar predições
    let new_market_excess = array![0.01, -0.01, 0.02];
    let predictions = result.predict(&new_market_excess);

    assert_eq!(predictions.len(), 3);
    assert!(predictions.iter().all(|&x| x.is_finite()));

    // Predição para mercado = 0 deve ser aproximadamente alpha
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

    // Teste de retorno esperado com diferentes cenários
    let expected_10pct = result.expected_return(0.10);
    let expected_0pct = result.expected_return(0.0);
    let expected_neg5pct = result.expected_return(-0.05);

    assert!(expected_10pct.is_finite());
    assert!(expected_0pct.is_finite());
    assert!(expected_neg5pct.is_finite());

    // Com beta positivo: retorno esperado deve aumentar com retorno do mercado
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

    // Verificar decomposição de variância
    let total_variance = result.asset_volatility.powi(2);
    let explained_variance = result.systematic_variance + result.idiosyncratic_variance;

    // A soma deve ser próxima da variância total
    assert!(
        (total_variance - explained_variance).abs() / total_variance < 0.1,
        "Decomposição de variância inconsistente: total={}, explicada={}",
        total_variance,
        explained_variance
    );

    // Variâncias devem ser não-negativas
    assert!(result.systematic_variance >= 0.0);
    assert!(result.idiosyncratic_variance >= 0.0);
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

    // Sharpe ratio deve ser finito
    assert!(result.sharpe_ratio.is_finite());
    assert!(result.market_sharpe.is_finite());

    // Com retornos positivos, Sharpe deve ser positivo
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
    // Criar dados onde alpha é significativo
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

    // Testar métodos de significância
    let is_outperforming = result.is_significantly_outperforming(0.05);
    let is_underperforming = result.is_significantly_underperforming(0.05);

    // Não pode ser ambos
    assert!(!(is_outperforming && is_underperforming));
}

#[test]
fn test_capm_beta_different_from_one() {
    // Ativo com beta claramente diferente de 1
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

    // Com beta ~2, deve ser significativamente diferente de 1
    // (mas depende do tamanho da amostra e SE)
    assert!(
        result.beta > 1.5,
        "Beta deveria ser ~2, got {}",
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

    // Deve retornar uma string não vazia
    assert!(!classification.is_empty());

    // Deve conter uma das palavras-chave esperadas
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

    // Tracking error deve ser positivo e finito
    assert!(result.tracking_error > 0.0);
    assert!(result.tracking_error.is_finite());

    // Tracking error é o desvio padrão dos resíduos
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

    // Information ratio deve ser finito
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

    // Treynor ratio deve ser finito
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

    // Intervalos de confiança devem fazer sentido
    assert!(result.alpha_conf_lower < result.alpha_conf_upper);
    assert!(result.beta_conf_lower < result.beta_conf_upper);

    // Estimativa pontual deve estar dentro do intervalo
    assert!(result.alpha >= result.alpha_conf_lower);
    assert!(result.alpha <= result.alpha_conf_upper);
    assert!(result.beta >= result.beta_conf_lower);
    assert!(result.beta <= result.beta_conf_upper);
}
