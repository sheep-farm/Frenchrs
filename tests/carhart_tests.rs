use frenchrs::Carhart4Factor;
use greeners::{CovarianceType, DataFrame};
use ndarray::array;

#[test]
fn test_carhart_basic_fit() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0001,
        CovarianceType::NonRobust,
    );

    assert!(result.is_ok());
    let c4f = result.unwrap();
    assert_eq!(c4f.n_obs, 8);
    assert!(c4f.r_squared >= 0.0 && c4f.r_squared <= 1.0);
}

#[test]
fn test_carhart_dimension_mismatch() {
    let asset = array![0.01, 0.02];
    let market = array![0.01, 0.02, 0.03];
    let smb = array![0.001, 0.002];
    let hml = array![0.001, 0.002];
    let mom = array![0.001, 0.002];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_carhart_insufficient_data() {
    let asset = array![0.01, 0.02, 0.03, 0.04];
    let market = array![0.01, 0.02, 0.03, 0.04];
    let smb = array![0.001, 0.002, 0.003, 0.004];
    let hml = array![0.001, 0.002, 0.003, 0.004];
    let mom = array![0.001, 0.002, 0.003, 0.004];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_carhart_covariance_types() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003];

    for cov_type in [
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC3,
    ] {
        let result =
            Carhart4Factor::fit(&asset, &market, &smb, &hml, &mom, 0.0001, cov_type.clone());
        assert!(result.is_ok());
    }
}

#[test]
fn test_carhart_predictions() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    let predictions = result.predict(
        &array![0.01, -0.01],
        &array![0.002, -0.001],
        &array![0.001, 0.001],
        &array![0.003, -0.002],
    );

    assert_eq!(predictions.len(), 2);
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_carhart_expected_return() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    let expected = result.expected_return(0.10, 0.02, 0.03, 0.01);
    assert!(expected.is_finite());
}

#[test]
fn test_carhart_from_dataframe() {
    let df = DataFrame::builder()
        .add_column("asset", vec![0.01, 0.02, -0.01, 0.03, 0.015, -0.005])
        .add_column("market", vec![0.008, 0.015, -0.005, 0.025, 0.012, -0.003])
        .add_column("smb", vec![0.002, -0.001, 0.003, 0.001, -0.002, 0.001])
        .add_column("hml", vec![0.001, 0.002, -0.002, 0.003, 0.001, -0.001])
        .add_column("mom", vec![0.003, 0.002, -0.003, 0.004, 0.001, -0.002])
        .build()
        .unwrap();

    let result = Carhart4Factor::from_dataframe(
        &df,
        "asset",
        "market",
        "smb",
        "hml",
        "mom",
        0.0001,
        CovarianceType::HC3,
    );

    assert!(result.is_ok());
    assert_eq!(result.unwrap().n_obs, 6);
}

#[test]
fn test_carhart_factor_contributions() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.001,
        CovarianceType::HC3,
    )
    .unwrap();

    let (market_contrib, smb_contrib, hml_contrib, mom_contrib) = result.factor_contributions();

    assert!(market_contrib.is_finite());
    assert!(smb_contrib.is_finite());
    assert!(hml_contrib.is_finite());
    assert!(mom_contrib.is_finite());
}

#[test]
fn test_carhart_classification_methods() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    assert!(!result.size_classification().is_empty());
    assert!(!result.value_classification().is_empty());
    assert!(!result.momentum_classification().is_empty());
    assert!(!result.performance_classification().is_empty());
}

#[test]
fn test_carhart_confidence_intervals() {
    let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020];
    let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002];
    let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002];
    let mom = array![0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003];

    let result = Carhart4Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &mom,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    assert!(result.alpha_conf_lower < result.alpha_conf_upper);
    assert!(result.beta_market_conf_lower < result.beta_market_conf_upper);
    assert!(result.beta_smb_conf_lower < result.beta_smb_conf_upper);
    assert!(result.beta_hml_conf_lower < result.beta_hml_conf_upper);
    assert!(result.beta_mom_conf_lower < result.beta_mom_conf_upper);
}
