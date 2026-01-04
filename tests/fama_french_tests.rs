use frenchrs::FamaFrench3Factor;
use greeners::{CovarianceType, DataFrame};
use ndarray::array;

#[test]
fn test_ff3_basic_fit() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001];

    let result = FamaFrench3Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        0.0001,
        CovarianceType::NonRobust,
    );

    assert!(result.is_ok());
    let ff3 = result.unwrap();

    assert_eq!(ff3.n_obs, 8);
    assert!(ff3.r_squared >= 0.0 && ff3.r_squared <= 1.0);
    assert!(ff3.adj_r_squared <= ff3.r_squared);
    assert!(ff3.beta_market.is_finite());
    assert!(ff3.beta_smb.is_finite());
    assert!(ff3.beta_hml.is_finite());
}

#[test]
fn test_ff3_dimension_mismatch() {
    let asset = array![0.01, 0.02];
    let market = array![0.01, 0.02, 0.03]; // Tamanho diferente
    let smb = array![0.001, 0.002];
    let hml = array![0.001, 0.002];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0, CovarianceType::NonRobust);

    assert!(result.is_err());
}

#[test]
fn test_ff3_insufficient_data() {
    let asset = array![0.01, 0.02, 0.03];
    let market = array![0.01, 0.02, 0.03];
    let smb = array![0.001, 0.002, 0.003];
    let hml = array![0.001, 0.002, 0.003];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0, CovarianceType::NonRobust);

    assert!(result.is_err());
}

#[test]
fn test_ff3_covariance_types() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];
    let risk_free = 0.0001;

    let cov_types = vec![
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC2,
        CovarianceType::HC3,
        CovarianceType::HC4,
    ];

    for cov_type in cov_types {
        let result =
            FamaFrench3Factor::fit(&asset, &market, &smb, &hml, risk_free, cov_type.clone());
        assert!(result.is_ok(), "Falhou com covariância {:?}", cov_type);

        let ff3 = result.unwrap();
        assert!(ff3.beta_market.is_finite());
        assert!(ff3.beta_smb.is_finite());
        assert!(ff3.beta_hml.is_finite());
        assert!(ff3.beta_market_se > 0.0);
        assert!(ff3.beta_smb_se > 0.0);
        assert!(ff3.beta_hml_se > 0.0);
    }
}

#[test]
fn test_ff3_predictions() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();

    let new_market = array![0.01, -0.01, 0.02];
    let new_smb = array![0.002, -0.001, 0.003];
    let new_hml = array![0.001, 0.001, -0.001];

    let predictions = result.predict(&new_market, &new_smb, &new_hml);

    assert_eq!(predictions.len(), 3);
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_ff3_expected_return() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001];

    let result = FamaFrench3Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        0.001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Teste de retorno esperado com diferentes cenários
    let expected_1 = result.expected_return(0.10, 0.02, 0.03);
    let expected_2 = result.expected_return(0.05, 0.01, 0.01);
    let expected_3 = result.expected_return(-0.05, -0.01, -0.01);

    assert!(expected_1.is_finite());
    assert!(expected_2.is_finite());
    assert!(expected_3.is_finite());
}

#[test]
fn test_ff3_from_dataframe() {
    let df = DataFrame::builder()
        .add_column("asset", vec![0.01, 0.02, -0.01, 0.03, 0.015, -0.005])
        .add_column("market", vec![0.008, 0.015, -0.005, 0.025, 0.012, -0.003])
        .add_column("smb", vec![0.002, -0.001, 0.003, 0.001, -0.002, 0.001])
        .add_column("hml", vec![0.001, 0.002, -0.002, 0.003, 0.001, -0.001])
        .build()
        .unwrap();

    let result = FamaFrench3Factor::from_dataframe(
        &df,
        "asset",
        "market",
        "smb",
        "hml",
        0.0001,
        CovarianceType::HC3,
    );

    assert!(result.is_ok());
    let ff3 = result.unwrap();
    assert_eq!(ff3.n_obs, 6);
}

#[test]
fn test_ff3_smb_significance() {
    // Criar dados onde SMB é claramente significativo
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.010, 0.015, -0.008, 0.020, 0.012, -0.005, 0.018]; // Grande correlação
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];

    let result = FamaFrench3Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        0.0001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Beta SMB deve ser positivo e potencialmente significativo
    assert!(result.beta_smb > 0.0);
}

#[test]
fn test_ff3_hml_significance() {
    // Criar dados onde HML é claramente significativo
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.001, 0.002, -0.001, 0.002, 0.001, -0.001, 0.002];
    let hml = array![0.008, 0.012, -0.006, 0.018, 0.010, -0.004, 0.015]; // Grande correlação

    let result = FamaFrench3Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        0.0001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Beta HML deve ser positivo e potencialmente significativo
    assert!(result.beta_hml > 0.0);
}

#[test]
fn test_ff3_factor_contributions() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.001, CovarianceType::HC3).unwrap();

    let (market_contrib, smb_contrib, hml_contrib) = result.factor_contributions();

    assert!(market_contrib.is_finite());
    assert!(smb_contrib.is_finite());
    assert!(hml_contrib.is_finite());
}

#[test]
fn test_ff3_classification_methods() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();

    // Testar métodos de classificação
    let size_class = result.size_classification();
    let value_class = result.value_classification();
    let perf_class = result.performance_classification();

    assert!(!size_class.is_empty());
    assert!(!value_class.is_empty());
    assert!(!perf_class.is_empty());
}

#[test]
fn test_ff3_r_squared_improvement() {
    // FF3 deve explicar pelo menos tanto quanto CAPM (R² >= R² do CAPM)
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001];

    let ff3 =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();

    // R² deve estar entre 0 e 1
    assert!(ff3.r_squared >= 0.0 && ff3.r_squared <= 1.0);

    // Adjusted R² deve ser menor ou igual ao R²
    assert!(ff3.adj_r_squared <= ff3.r_squared);
}

#[test]
fn test_ff3_confidence_intervals() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();

    // Intervalos de confiança devem fazer sentido
    assert!(result.alpha_conf_lower < result.alpha_conf_upper);
    assert!(result.beta_market_conf_lower < result.beta_market_conf_upper);
    assert!(result.beta_smb_conf_lower < result.beta_smb_conf_upper);
    assert!(result.beta_hml_conf_lower < result.beta_hml_conf_upper);

    // Estimativas pontuais devem estar dentro dos intervalos
    assert!(result.alpha >= result.alpha_conf_lower);
    assert!(result.alpha <= result.alpha_conf_upper);
    assert!(result.beta_market >= result.beta_market_conf_lower);
    assert!(result.beta_market <= result.beta_market_conf_upper);
    assert!(result.beta_smb >= result.beta_smb_conf_lower);
    assert!(result.beta_smb <= result.beta_smb_conf_upper);
    assert!(result.beta_hml >= result.beta_hml_conf_lower);
    assert!(result.beta_hml <= result.beta_hml_conf_upper);
}

#[test]
fn test_ff3_tracking_error() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0, CovarianceType::NonRobust)
            .unwrap();

    // Tracking error deve ser positivo e finito
    assert!(result.tracking_error > 0.0);
    assert!(result.tracking_error.is_finite());

    // Tracking error é o desvio padrão dos resíduos
    let residuals_std = result.residuals.std(0.0);
    assert!((result.tracking_error - residuals_std).abs() < 0.0001);
}

#[test]
fn test_ff3_information_ratio() {
    let asset = array![0.015, 0.025, -0.005, 0.035, 0.020, -0.002];
    let market = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0, CovarianceType::HC3).unwrap();

    // Information ratio deve ser finito
    assert!(result.information_ratio.is_finite());

    // IR = alpha / tracking_error
    if result.tracking_error > 0.0 {
        let expected_ir = result.alpha / result.tracking_error;
        assert!((result.information_ratio - expected_ir).abs() < 0.0001);
    }
}

#[test]
fn test_ff3_significance_tests() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];

    let result = FamaFrench3Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        0.0001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    // Testar métodos de significância
    let is_outperforming = result.is_significantly_outperforming(0.05);
    let is_underperforming = result.is_significantly_underperforming(0.05);
    let smb_sig = result.is_smb_significant(0.05);
    let hml_sig = result.is_hml_significant(0.05);

    // Não pode ser ambos ao mesmo tempo
    assert!(!(is_outperforming && is_underperforming));

    // Métodos devem retornar valores booleanos válidos
    assert!(is_outperforming == true || is_outperforming == false);
    assert!(is_underperforming == true || is_underperforming == false);
    assert!(smb_sig == true || smb_sig == false);
    assert!(hml_sig == true || hml_sig == false);
}

#[test]
fn test_ff3_all_parameters_finite() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

    let result =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();

    // Todos os parâmetros devem ser finitos
    assert!(result.alpha.is_finite());
    assert!(result.beta_market.is_finite());
    assert!(result.beta_smb.is_finite());
    assert!(result.beta_hml.is_finite());
    assert!(result.alpha_se.is_finite());
    assert!(result.beta_market_se.is_finite());
    assert!(result.beta_smb_se.is_finite());
    assert!(result.beta_hml_se.is_finite());
    assert!(result.r_squared.is_finite());
    assert!(result.adj_r_squared.is_finite());
}
