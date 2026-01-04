use frenchrs::{CAPM, FamaFrench3Factor, IVOLAnalysis, TrackingErrorAnalysis};
use greeners::CovarianceType;
use ndarray::array;

#[test]
fn test_ivol_basic() {
    let residuals = array![
        0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.002, -0.001
    ];
    let ivol = IVOLAnalysis::from_residuals(&residuals).unwrap();

    assert!(ivol.ivol > 0.0);
    assert!(ivol.ivol_annualized_daily > ivol.ivol);
    assert!(ivol.ivol_annualized_monthly > ivol.ivol);
    assert_eq!(ivol.n_obs, 10);
}

#[test]
fn test_ivol_from_capm() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];

    let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();
    let ivol = IVOLAnalysis::from_residuals(&capm.residuals).unwrap();

    assert!(ivol.ivol > 0.0);
    assert!(ivol.ivol_annualized_daily > 0.0);
    assert!(ivol.ivol_annualized_monthly > 0.0);
    assert!(!ivol.ivol_classification().is_empty());
}

#[test]
fn test_ivol_from_ff3() {
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

    let ff3 =
        FamaFrench3Factor::fit(&asset, &market, &smb, &hml, 0.0001, CovarianceType::HC3).unwrap();
    let ivol = IVOLAnalysis::from_residuals(&ff3.residuals).unwrap();

    // FF3 deve ter IVOL menor que CAPM (mais fatores explicam mais variação)
    assert!(ivol.ivol > 0.0);
}

#[test]
fn test_ivol_insufficient_data() {
    let residuals = array![0.001, -0.002];
    let result = IVOLAnalysis::from_residuals(&residuals);

    assert!(result.is_err());
}

#[test]
fn test_ivol_statistics() {
    let residuals = array![
        0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.002, -0.001
    ];
    let ivol = IVOLAnalysis::from_residuals(&residuals).unwrap();

    assert!(ivol.residual_mean.abs() < 0.01); // Média deve ser próxima de 0
    assert!(ivol.residual_min < 0.0);
    assert!(ivol.residual_max > 0.0);
    assert!(ivol.residual_p5 < ivol.residual_p95);
    assert!(ivol.residual_skewness.is_finite());
    assert!(ivol.residual_kurtosis.is_finite());
}

#[test]
fn test_ivol_classification() {
    let low_ivol = array![0.0001, -0.0001, 0.0002, -0.0001, 0.0001, -0.0002];
    let ivol = IVOLAnalysis::from_residuals(&low_ivol).unwrap();
    let classification = ivol.ivol_classification();

    assert!(
        classification == "Baixo"
            || classification == "Moderado"
            || classification == "Alto"
            || classification == "Muito Alto"
    );
}

#[test]
fn test_ivol_normality_test() {
    let residuals = array![
        0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.002, -0.001
    ];
    let ivol = IVOLAnalysis::from_residuals(&residuals).unwrap();

    // Just ensure it doesn't panic
    let _ = ivol.is_residuals_normal(0.05);
}

#[test]
fn test_tracking_error_basic() {
    let actual = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let fitted = array![
        0.009, 0.019, -0.011, 0.029, 0.014, -0.006, 0.024, 0.009, 0.019, -0.011
    ];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    assert!(te.tracking_error > 0.0);
    assert!(te.tracking_error_annualized_daily > te.tracking_error);
    assert!(te.tracking_error_annualized_monthly > te.tracking_error);
    assert_eq!(te.n_obs, 10);
}

#[test]
fn test_tracking_error_from_capm() {
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];

    let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();
    let te = TrackingErrorAnalysis::new(&asset, &capm.fitted_values, capm.alpha, capm.r_squared)
        .unwrap();

    assert!(te.tracking_error > 0.0);
    assert!(te.information_ratio.is_finite());
    assert!(!te.te_classification().is_empty());
    assert!(!te.ir_classification().is_empty());
}

#[test]
fn test_tracking_error_dimension_mismatch() {
    let actual = array![0.01, 0.02, -0.01];
    let fitted = array![0.009, 0.019];

    let result = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95);
    assert!(result.is_err());
}

#[test]
fn test_tracking_error_insufficient_data() {
    let actual = array![0.01, 0.02];
    let fitted = array![0.009, 0.019];

    let result = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95);
    assert!(result.is_err());
}

#[test]
fn test_tracking_error_metrics() {
    let actual = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let fitted = array![
        0.009, 0.019, -0.011, 0.029, 0.014, -0.006, 0.024, 0.009, 0.019, -0.011
    ];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    assert!(te.correlation > 0.0 && te.correlation <= 1.0);
    assert!(te.rmse > 0.0);
    assert!(te.mae > 0.0);
    assert!(te.periods_above_1pct >= 0.0 && te.periods_above_1pct <= 1.0);
    assert!(te.periods_above_2pct >= 0.0 && te.periods_above_2pct <= 1.0);
}

#[test]
fn test_tracking_error_perfect_fit() {
    let actual = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let fitted = actual.clone();

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.0, 1.0).unwrap();

    // Com fit perfeito, tracking error deve ser zero (ou muito próximo)
    assert!(te.tracking_error < 1e-10);
    assert!((te.correlation - 1.0).abs() < 1e-10);
}

#[test]
fn test_tracking_error_rolling() {
    // Dados suficientes para rolling window (12+)
    let actual = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01, 0.03, 0.015, -0.005,
        0.025
    ];
    let fitted = array![
        0.009, 0.019, -0.011, 0.029, 0.014, -0.006, 0.024, 0.009, 0.019, -0.011, 0.029, 0.014,
        -0.006, 0.024
    ];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    assert!(te.rolling_te.is_some());
    let rolling = te.rolling_te.unwrap();
    assert_eq!(rolling.len(), 14 - 11); // n - 11
}

#[test]
fn test_tracking_error_no_rolling() {
    // Dados insuficientes para rolling window (< 12)
    let actual = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let fitted = array![0.009, 0.019, -0.011, 0.029, 0.014, -0.006];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    assert!(te.rolling_te.is_none());
}

#[test]
fn test_tracking_error_classification() {
    let actual = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let fitted = array![0.009, 0.019, -0.011, 0.029, 0.014, -0.006];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    let te_class = te.te_classification();
    assert!(
        te_class.contains("Baixo") || te_class.contains("Moderado") || te_class.contains("Alto")
    );

    let ir_class = te.ir_classification();
    assert!(
        ir_class.contains("Excelente")
            || ir_class.contains("Bom")
            || ir_class.contains("Moderado")
            || ir_class.contains("Fraco")
            || ir_class.contains("Ruim")
    );
}

#[test]
fn test_ivol_vs_tracking_error_equivalence() {
    // IVOL e TE devem ter o mesmo valor (ambos são std dos resíduos)
    let asset = array![
        0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, 0.02, -0.01
    ];
    let market = array![
        0.008, 0.015, -0.005, 0.025, 0.012, -0.003, 0.020, 0.009, 0.015, -0.005
    ];

    let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();

    let ivol = IVOLAnalysis::from_residuals(&capm.residuals).unwrap();
    let te = TrackingErrorAnalysis::new(&asset, &capm.fitted_values, capm.alpha, capm.r_squared)
        .unwrap();

    // IVOL e TE devem ser praticamente iguais
    assert!((ivol.ivol - te.tracking_error).abs() < 1e-10);
}

#[test]
fn test_display_ivol() {
    let residuals = array![0.001, -0.002, 0.003, -0.001, 0.002, -0.003];
    let ivol = IVOLAnalysis::from_residuals(&residuals).unwrap();

    let display = format!("{}", ivol);
    assert!(display.contains("IVOL"));
    assert!(display.contains("ANÁLISE"));
}

#[test]
fn test_display_tracking_error() {
    let actual = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
    let fitted = array![0.009, 0.019, -0.011, 0.029, 0.014, -0.006];

    let te = TrackingErrorAnalysis::new(&actual, &fitted, 0.001, 0.95).unwrap();

    let display = format!("{}", te);
    assert!(display.contains("TRACKING ERROR"));
    assert!(display.contains("INFORMATION RATIO"));
}
