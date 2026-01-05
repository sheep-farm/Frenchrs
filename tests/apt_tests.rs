use frenchrs::APT;
use greeners::CovarianceType;
use ndarray::{array, Array2};

#[test]
fn test_apt_basic_fit() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::NonRobust, None);
    assert!(result.is_ok());
    let apt = result.unwrap();
    assert_eq!(apt.n_factors, 2);
    assert_eq!(apt.n_obs, 7);
}

#[test]
fn test_apt_three_factors() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 3),
        vec![
            0.008, 0.002, 0.001, 0.015, -0.001, 0.002, -0.005, 0.003, -0.002, 0.025, 0.001, 0.003,
            0.012, -0.002, 0.001, -0.003, 0.001, -0.001, 0.020, 0.002, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::NonRobust, None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().n_factors, 3);
}

#[test]
fn test_apt_with_factor_names() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let factor_names = Some(vec!["Market".to_string(), "Size".to_string()]);

    let result = APT::fit(
        &returns,
        &factors,
        0.0001,
        CovarianceType::HC3,
        factor_names.clone(),
    );
    assert!(result.is_ok());

    let apt = result.unwrap();
    assert_eq!(apt.factor_names, factor_names);
}

#[test]
fn test_apt_dimension_mismatch() {
    let returns = array![0.01, 0.02];
    let factors = Array2::from_shape_vec((3, 2), vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06]).unwrap();

    let result = APT::fit(&returns, &factors, 0.0, CovarianceType::NonRobust, None);
    assert!(result.is_err());
}

#[test]
fn test_apt_insufficient_data() {
    let returns = array![0.01, 0.02, 0.03];
    let factors = Array2::from_shape_vec((3, 2), vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06]).unwrap();

    let result = APT::fit(&returns, &factors, 0.0, CovarianceType::NonRobust, None);
    assert!(result.is_err());
}

#[test]
fn test_apt_covariesnce_types() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    for cov_type in [
        CovarianceType::NonRobust,
        CovarianceType::HC1,
        CovarianceType::HC3,
    ] {
        let result = APT::fit(&returns, &factors, 0.0001, cov_type.clone(), None);
        assert!(result.is_ok());
    }
}

#[test]
fn test_apt_expected_return() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 3),
        vec![
            0.008, 0.002, 0.001, 0.015, -0.001, 0.002, -0.005, 0.003, -0.002, 0.025, 0.001, 0.003,
            0.012, -0.002, 0.001, -0.003, 0.001, -0.001, 0.020, 0.002, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.001, CovarianceType::NonRobust, None).unwrap();

    let factor_returns = array![0.01, 0.02, 0.015];
    let expected = result.expected_return(&factor_returns);
    assert!(expected.is_finite());
}

#[test]
fn test_apt_expected_return_wrong_dimensions() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 3),
        vec![
            0.008, 0.002, 0.001, 0.015, -0.001, 0.002, -0.005, 0.003, -0.002, 0.025, 0.001, 0.003,
            0.012, -0.002, 0.001, -0.003, 0.001, -0.001, 0.020, 0.002, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.001, CovarianceType::NonRobust, None).unwrap();

    // Wrong number of factors
    let factor_returns = array![0.01, 0.02];
    let expected = result.expected_return(&factor_returns);
    assert_eq!(expected, 0.0);
}

#[test]
fn test_apt_factor_significance() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::HC3, None).unwrap();

    // Test factor significance
    let _is_sig_0 = result.factor_is_significant(0, 0.05); // Just checking it doesn't panic
    let _is_sig_1 = result.factor_is_significant(1, 0.05); // Just checking it doesn't panic
    let is_sig_invalid = result.factor_is_significant(5, 0.05);

    assert!(!is_sig_invalid); // Invalid index should return false
}

#[test]
fn test_apt_performance_test() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::HC3, None).unwrap();

    // Just ensure these methods work
    let _ = result.is_significantly_outperforming(0.05);
}

#[test]
fn test_apt_r_squared() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::NonRobust, None).unwrap();

    assert!(result.r_squared >= 0.0);
    assert!(result.r_squared <= 1.0);
    assert!(result.adj_r_squared <= result.r_squared);
}

#[test]
fn test_apt_tracking_error() {
    let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];
    let factors = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.008, 0.002, 0.015, -0.001, -0.005, 0.003, 0.025, 0.001, 0.012, -0.002, -0.003, 0.001,
            0.020, 0.002,
        ],
    )
    .unwrap();

    let result = APT::fit(&returns, &factors, 0.0001, CovarianceType::HC3, None).unwrap();

    assert!(result.tracking_error >= 0.0);
    assert!(result.information_ratio.is_finite());
}
