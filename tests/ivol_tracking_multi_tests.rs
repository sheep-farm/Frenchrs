use frenchrs::IVOLTrackingMulti;
use greeners::CovarianceType;
use ndarray::{Array2, array};

#[test]
fn test_ivol_tracking_multi_basic() {
    // 12 obbevations, 2 assets
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    // 12 obbevations, 1 factor
    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
        12.0, // monthly
    )
    .unwrap();

    assert_eq!(result.results.len(), 2);
    assert!(result.results.contains_key("Asset1"));
    assert!(result.results.contains_key("Asset2"));
}

#[test]
fn test_ivol_tracking_multi_with_benchmark() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let benchmark = array![
        0.009, 0.016, -0.004, 0.024, 0.013, -0.002, 0.019, 0.010, 0.014, -0.006, 0.026, 0.011
    ];

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        Some(&benchmark),
        CovarianceType::HC3,
        None,
        12.0,
    )
    .unwrap();

    assert_eq!(result.results.len(), 2);

    // Verificar que tracking error was calculado
    let asset1 = result.get_asset("Asset1").unwrap();
    assert!(asset1.tracking_error_monthly.is_some());
    assert!(asset1.tracking_error_annual.is_some());
    assert!(asset1.tracking_error_monthly.unwrap() > 0.0);
    assert!(asset1.tracking_error_annual.unwrap() > 0.0);
}

#[test]
fn test_ivol_tracking_multi_asset_names() {
    let returns_excess = Array2::from_shape_vec(
        (12, 3),
        vec![
            0.01, 0.015, 0.012, 0.02, 0.025, 0.022, -0.01, -0.005, -0.008, 0.03, 0.035, 0.032,
            0.015, 0.020, 0.018, -0.005, 0.000, -0.002, 0.025, 0.030, 0.028, 0.01, 0.015, 0.012,
            0.02, 0.025, 0.022, -0.01, -0.005, -0.008, 0.03, 0.035, 0.032, 0.015, 0.020, 0.018,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let asset_names = vec![
        "Tech Fund".to_string(),
        "Value Fund".to_string(),
        "Growth Fund".to_string(),
    ];

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::HC3,
        Some(asset_names.clone()),
        12.0,
    )
    .unwrap();

    assert_eq!(result.results.len(), 3);
    for name in &asset_names {
        assert!(result.results.contains_key(name));
    }

    let asset_names_result = result.asset_names();
    assert_eq!(asset_names_result.len(), 3);
}

#[test]
fn test_ivol_tracking_multi_statistics() {
    // 24 obbevations, 2 assets - data in row-major order
    #[rustfmt::skip]
    let returns_excess = Array2::from_shape_vec(
        (24, 2),
        vec![
            // Asset1, Asset2 for each row
            0.01, 0.015,     // obs 1
            0.02, 0.025,     // obs 2
            -0.01, -0.005,   // obs 3
            0.03, 0.035,     // obs 4
            0.015, 0.020,    // obs 5
            -0.005, 0.000,   // obs 6
            0.025, 0.030,    // obs 7
            0.01, 0.015,     // obs 8
            0.02, 0.025,     // obs 9
            -0.01, -0.005,   // obs 10
            0.03, 0.035,     // obs 11
            0.015, 0.020,    // obs 12
            -0.008, -0.003,  // obs 13
            0.022, 0.027,    // obs 14
            0.012, 0.017,    // obs 15
            -0.002, 0.003,   // obs 16
            0.028, 0.033,    // obs 17
            0.018, 0.023,    // obs 18
            -0.012, -0.007,  // obs 19
            0.025, 0.030,    // obs 20
            0.013, 0.018,    // obs 21
            -0.007, -0.002,  // obs 22
            0.020, 0.025,    // obs 23
            0.016, 0.021,    // obs 24
        ],
    )
    .unwrap();

    #[rustfmt::skip]
    let factors = Array2::from_shape_vec(
        (24, 2),
        vec![
            // Factor1, Factor2 for each row
            0.008, 0.002,    // obs 1
            0.015, -0.001,   // obs 2
            -0.005, 0.003,   // obs 3
            0.025, 0.001,    // obs 4
            0.012, -0.002,   // obs 5
            -0.003, 0.001,   // obs 6
            0.020, 0.002,    // obs 7
            0.009, -0.001,   // obs 8
            0.015, 0.001,    // obs 9
            -0.005, 0.002,   // obs 10
            0.025, 0.001,    // obs 11
            0.012, -0.001,   // obs 12
            0.007, 0.003,    // obs 13
            0.016, -0.002,   // obs 14
            0.011, 0.001,    // obs 15
            -0.004, 0.002,   // obs 16
            0.018, 0.001,    // obs 17
            0.010, -0.001,   // obs 18
            0.014, 0.002,    // obs 19
            -0.006, 0.001,   // obs 20
            0.023, 0.002,    // obs 21
            0.011, -0.002,   // obs 22
            0.013, 0.001,    // obs 23
            -0.003, 0.002,   // obs 24
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::HC3,
        None,
        12.0,
    )
    .unwrap();

    let asset1 = result.get_asset("Asset1").unwrap();

    // Verificar campos básicos
    assert_eq!(asset1.nobs, 24);
    assert!(asset1.ivol_monthly > 0.0);
    assert!(asset1.ivol_annual > 0.0);
    assert!(asset1.r_squared >= 0.0 && asset1.r_squared <= 1.0);

    // IVOL annualized should be greater que IVOL monthly
    assert!(asset1.ivol_annual > asset1.ivol_monthly);
}

#[test]
fn test_ivol_tracking_multi_to_table() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        Some(vec!["Asset1".to_string(), "Asset2".to_string()]),
        12.0,
    )
    .unwrap();

    let table = result.to_table();
    assert_eq!(table.len(), 2); // 2 assets

    // Verificar estrutura of the linhas
    for row in &table {
        assert!(!row.asset.is_empty());
        assert!(row.nobs > 0);
        assert!(row.ivol_monthly >= 0.0);
        assert!(row.ivol_annual >= 0.0);
    }
}

#[test]
fn test_ivol_tracking_multi_csv_export() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        None,
        12.0,
    )
    .unwrap();

    let csv = result.to_csv_string();

    // Verificar header
    assert!(csv.contains("asset"));
    assert!(csv.contains("mean_excess_annual"));
    assert!(csv.contains("ivol_monthly"));
    assert!(csv.contains("ivol_annual"));
    assert!(csv.contains("nobs"));

    // Verificar data
    assert!(csv.contains("Asset1"));
    assert!(csv.contains("Asset2"));
}

#[test]
fn test_ivol_tracking_multi_daily_data() {
    // Simular 252 dias úteis (1 ano)
    let n_days = 252;
    let n_assets = 2;

    // Gerar returns diários
    let mut returns_vec = Vec::with_capacity(n_days * n_assets);
    for i in 0..n_days {
        returns_vec.push(0.001 * (i as f64 % 10.0 - 5.0) / 5.0); // Asset1
        returns_vec.push(0.0015 * (i as f64 % 8.0 - 4.0) / 4.0); // Asset2
    }

    let returns_excess = Array2::from_shape_vec((n_days, n_assets), returns_vec).unwrap();

    // Gerar factor
    let mut factor_vec = Vec::with_capacity(n_days);
    for i in 0..n_days {
        factor_vec.push(0.0008 * (i as f64 % 12.0 - 6.0) / 6.0);
    }

    let factors = Array2::from_shape_vec((n_days, 1), factor_vec).unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::HC3,
        None,
        252.0, // daily
    )
    .unwrap();

    let asset1 = result.get_asset("Asset1").unwrap();

    assert_eq!(asset1.nobs, 252);
    assert!(asset1.ivol_monthly > 0.0);
    assert!(asset1.ivol_annual > 0.0);
}

#[test]
fn test_ivol_tracking_multi_shape_mismatch() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    // Factors with número diferente of obbevations
    let factors = Array2::from_shape_vec(
        (10, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        None,
        12.0,
    );

    assert!(result.is_err());
}

#[test]
fn test_ivol_tracking_multi_benchmark_shape_mismatch() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    // Benchmark with número diferente of obbevations
    let benchmark = array![0.009, 0.016, -0.004, 0.024, 0.013];

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        Some(&benchmark),
        CovarianceType::NonRobust,
        None,
        12.0,
    );

    assert!(result.is_err());
}

#[test]
fn test_ivol_tracking_multi_three_factors() {
    // 24 obbevations, 2 assets - data in row-major order
    #[rustfmt::skip]
    let returns_excess = Array2::from_shape_vec(
        (24, 2),
        vec![
            // Asset1, Asset2 for each row
            0.01, 0.015,     // obs 1
            0.02, 0.025,     // obs 2
            -0.01, -0.005,   // obs 3
            0.03, 0.035,     // obs 4
            0.015, 0.020,    // obs 5
            -0.005, 0.000,   // obs 6
            0.025, 0.030,    // obs 7
            0.01, 0.015,     // obs 8
            0.02, 0.025,     // obs 9
            -0.01, -0.005,   // obs 10
            0.03, 0.035,     // obs 11
            0.015, 0.020,    // obs 12
            -0.008, -0.003,  // obs 13
            0.022, 0.027,    // obs 14
            0.012, 0.017,    // obs 15
            -0.002, 0.003,   // obs 16
            0.028, 0.033,    // obs 17
            0.018, 0.023,    // obs 18
            -0.012, -0.007,  // obs 19
            0.025, 0.030,    // obs 20
            0.013, 0.018,    // obs 21
            -0.007, -0.002,  // obs 22
            0.020, 0.025,    // obs 23
            0.016, 0.021,    // obs 24
        ],
    )
    .unwrap();

    // 3 factors: Market, SMB, HML - 24 obbevations, 3 factors
    #[rustfmt::skip]
    let factors = Array2::from_shape_vec(
        (24, 3),
        vec![
            // Market, SMB, HML for each row
            0.008, 0.002, 0.001,    // obs 1
            0.015, -0.001, 0.002,   // obs 2
            -0.005, 0.003, -0.002,  // obs 3
            0.025, 0.001, 0.003,    // obs 4
            0.012, -0.002, 0.001,   // obs 5
            -0.003, 0.001, -0.001,  // obs 6
            0.020, 0.002, 0.002,    // obs 7
            0.009, -0.001, 0.001,   // obs 8
            0.015, 0.001, 0.002,    // obs 9
            -0.005, 0.002, -0.001,  // obs 10
            0.025, 0.001, 0.002,    // obs 11
            0.012, -0.001, 0.001,   // obs 12
            0.007, 0.003, 0.001,    // obs 13
            0.016, -0.002, 0.003,   // obs 14
            0.011, 0.001, 0.001,    // obs 15
            -0.004, 0.002, -0.002,  // obs 16
            0.018, 0.001, 0.002,    // obs 17
            0.010, -0.001, 0.001,   // obs 18
            0.014, 0.002, 0.002,    // obs 19
            -0.006, 0.001, -0.001,  // obs 20
            0.023, 0.002, 0.003,    // obs 21
            0.011, -0.002, 0.001,   // obs 22
            0.013, 0.001, 0.002,    // obs 23
            -0.003, 0.002, -0.001,  // obs 24
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::HC3,
        None,
        12.0,
    )
    .unwrap();

    assert_eq!(result.results.len(), 2);

    let asset1 = result.get_asset("Asset1").unwrap();
    assert!(asset1.r_squared >= 0.0 && asset1.r_squared <= 1.0);
}

#[test]
fn test_ivol_tracking_multi_annualization() {
    let returns_excess = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01, 0.03, 0.015,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        None,
        12.0, // monthly
    )
    .unwrap();

    let asset1 = result.get_asset("Asset1").unwrap();

    // IVOL annualized should be aproximadamente IVOL monthly * sqrt(12)
    let ratio = asset1.ivol_annual / asset1.ivol_monthly;
    assert!((ratio - 12.0_f64.sqrt()).abs() < 0.01);
}

#[test]
fn test_ivol_tracking_multi_mean_excess_annual() {
    let returns_excess = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01, 0.03, 0.015,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        None,
        12.0,
    )
    .unwrap();

    let asset1 = result.get_asset("Asset1").unwrap();

    // Mean excess annual should be aproximadamente mean monthly * 12
    let mean_monthly = returns_excess.column(0).mean().unwrap();
    let expected_annual = mean_monthly * 12.0;

    assert!((asset1.mean_excess_annual - expected_annual).abs() < 0.001);
}

#[test]
fn test_ivol_tracking_multi_csv_with_benchmark() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let benchmark = array![
        0.009, 0.016, -0.004, 0.024, 0.013, -0.002, 0.019, 0.010, 0.014, -0.006, 0.026, 0.011
    ];

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        Some(&benchmark),
        CovarianceType::HC3,
        None,
        12.0,
    )
    .unwrap();

    let csv = result.to_csv_string();

    // Deve conter columns of tracking error
    assert!(csv.contains("tracking_error_monthly"));
    assert!(csv.contains("tracking_error_annual"));
}

#[test]
fn test_ivol_tracking_multi_get_nonexistent_asset() {
    let returns_excess = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020, -0.005, 0.000,
            0.025, 0.030, 0.01, 0.015, 0.02, 0.025, -0.01, -0.005, 0.03, 0.035, 0.015, 0.020,
        ],
    )
    .unwrap();

    let factors = Array2::from_shape_vec(
        (12, 1),
        vec![
            0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005, 0.025, 0.012,
        ],
    )
    .unwrap();

    let result = IVOLTrackingMulti::fit(
        &returns_excess,
        &factors,
        None,
        CovarianceType::NonRobust,
        None,
        12.0,
    )
    .unwrap();

    assert!(result.get_asset("NonExistent").is_none());
}
