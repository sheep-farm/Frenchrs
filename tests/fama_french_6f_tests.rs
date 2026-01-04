use frenchrs::FamaFrench6Factor;
use greeners::{CovarianceType, DataFrame};
use ndarray::array;

#[test]
fn test_ff6_basic_fit() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0001,
        CovarianceType::NonRobust,
    );

    assert!(result.is_ok());
    let ff6 = result.unwrap();
    assert_eq!(ff6.n_obs, 10);
    assert!(ff6.r_squared >= 0.0 && ff6.r_squared <= 1.0);
}

#[test]
fn test_ff6_dimension_mismatch() {
    let asset = array![0.01, 0.02];
    let market = array![0.01, 0.02, 0.03];
    let smb = array![0.001, 0.002];
    let hml = array![0.001, 0.002];
    let rmw = array![0.001, 0.002];
    let cma = array![0.001, 0.002];
    let umd = array![0.001, 0.002];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_ff6_insufficient_data() {
    let asset = array![0.01, 0.02, 0.03, 0.04, 0.05];
    let market = array![0.01, 0.02, 0.03, 0.04, 0.05];
    let smb = array![0.001, 0.002, 0.003, 0.004, 0.005];
    let hml = array![0.001, 0.002, 0.003, 0.004, 0.005];
    let rmw = array![0.001, 0.002, 0.003, 0.004, 0.005];
    let cma = array![0.001, 0.002, 0.003, 0.004, 0.005];
    let umd = array![0.001, 0.002, 0.003, 0.004, 0.005];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0,
        CovarianceType::NonRobust,
    );

    assert!(result.is_err());
}

#[test]
fn test_ff6_covariesnce_types() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    for cov_type in [
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC3,
    ] {
        let result = FamaFrench6Factor::fit(
            &asset,
            &market,
            &smb,
            &hml,
            &rmw,
            &cma,
            &umd,
            0.0001,
            cov_type.clone(),
        );
        assert!(result.is_ok());
    }
}

#[test]
fn test_ff6_predictions() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    let predictions = result.predict(
        &array![0.01, -0.01],
        &array![0.002, -0.001],
        &array![0.001, 0.001],
        &array![0.001, -0.001],
        &array![0.001, 0.001],
        &array![0.003, -0.002],
    );

    assert_eq!(predictions.len(), 2);
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_ff6_expected_return() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.001,
        CovarianceType::NonRobust,
    )
    .unwrap();

    let expected = result.expected_return(0.10, 0.02, 0.03, 0.01, 0.02, 0.04);
    assert!(expected.is_finite());
}

#[test]
fn test_ff6_from_dataframe() {
    let df = DataFrame::builder()
        .add_column(
            "asset",
            vec![
                0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01,
            ],
        )
        .add_column(
            "market",
            vec![
                0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005,
            ],
        )
        .add_column(
            "smb",
            vec![
                0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002,
            ],
        )
        .add_column(
            "hml",
            vec![
                0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001,
            ],
        )
        .add_column(
            "rmw",
            vec![
                0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001,
            ],
        )
        .add_column(
            "cma",
            vec![
                0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001,
            ],
        )
        .add_column(
            "umd",
            vec![
                0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002,
            ],
        )
        .build()
        .unwrap();

    let result = FamaFrench6Factor::from_dataframe(
        &df,
        "asset",
        "market",
        "smb",
        "hml",
        "rmw",
        "cma",
        "umd",
        0.0001,
        CovarianceType::HC3,
    );

    assert!(result.is_ok());
    assert_eq!(result.unwrap().n_obs, 10);
}

#[test]
fn test_ff6_factor_contributions() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.001,
        CovarianceType::HC3,
    )
    .unwrap();

    let (market_contrib, smb_contrib, hml_contrib, rmw_contrib, cma_contrib, umd_contrib) =
        result.factor_contributions();

    assert!(market_contrib.is_finite());
    assert!(smb_contrib.is_finite());
    assert!(hml_contrib.is_finite());
    assert!(rmw_contrib.is_finite());
    assert!(cma_contrib.is_finite());
    assert!(umd_contrib.is_finite());
}

#[test]
fn test_ff6_classification_methods() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    assert!(!result.size_classification().is_empty());
    assert!(!result.value_classification().is_empty());
    assert!(!result.profitability_classification().is_empty());
    assert!(!result.investment_classification().is_empty());
    assert!(!result.momentum_classification().is_empty());
    assert!(!result.performance_classification().is_empty());
}

#[test]
fn test_ff6_confidence_intervals() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    assert!(result.alpha_conf_lower < result.alpha_conf_upper);
    assert!(result.beta_market_conf_lower < result.beta_market_conf_upper);
    assert!(result.beta_smb_conf_lower < result.beta_smb_conf_upper);
    assert!(result.beta_hml_conf_lower < result.beta_hml_conf_upper);
    assert!(result.beta_rmw_conf_lower < result.beta_rmw_conf_upper);
    assert!(result.beta_cma_conf_lower < result.beta_cma_conf_upper);
    assert!(result.beta_umd_conf_lower < result.beta_umd_conf_upper);
}

#[test]
fn test_ff6_significance_tests() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];
    let smb = array![
        0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 0.002, -0.001, 0.001, 0.002
    ];
    let hml = array![
        0.001, 0.002, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, 0.002, -0.001
    ];
    let rmw = array![
        0.001, 0.001, -0.001, 0.002, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001
    ];
    let cma = array![
        0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001, -0.001, 0.001, 0.001
    ];
    let umd = array![
        0.003, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003, 0.001, 0.002, -0.002
    ];

    let result = FamaFrench6Factor::fit(
        &asset,
        &market,
        &smb,
        &hml,
        &rmw,
        &cma,
        &umd,
        0.0001,
        CovarianceType::HC3,
    )
    .unwrap();

    // Just ensure these methods don't panic
    let _ = result.is_market_significant(0.05);
    let _ = result.is_smb_significant(0.05);
    let _ = result.is_hml_significant(0.05);
    let _ = result.is_rmw_significant(0.05);
    let _ = result.is_cma_significant(0.05);
    let _ = result.is_umd_significant(0.05);
}
