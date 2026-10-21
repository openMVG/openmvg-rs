// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::Command;

use openmvg_sfm::{config::*, *};

const ENV_ISOLATION_CHILD: &str = "OPENMVG_CONFIG_ENV_ISOLATION_CHILD";
const LEGACY_POLICY_ENV: &[(&str, &str)] = &[
    ("TRANS_MIN_PARALLAX_DEG", "off"),
    ("REG_MIN_OBS", "999"),
    ("REG_MIN_INLIERS", "998"),
    ("REG_MIN_INLIER_RATIO", "0.99"),
    ("REG_MAX_ROT_EDGE", "none"),
    ("REG_MAX_TRANS_EDGE", "88"),
    ("GROWTH_MIN_VISIBLE", "997"),
    ("GROWTH_MIN_VERIFIED_NEIGHBORS", "996"),
    ("GP_RELATIVE_DIRS", "0"),
    ("GP_DIR_REWEIGHT", "9.9"),
    ("GP_MIN_VIEWS", "99"),
    ("STELLAR_REPROJ_GATE", "9.8"),
    ("STELLAR_DENSITY_GATE", "0.01"),
];

#[test]
fn partial_incremental_config_inherits_the_safe_neighbor_default() {
    let mut serialized = serde_json::to_value(IncrementalSfmConfig::default()).unwrap();
    let fields = serialized.as_object_mut().unwrap();
    fields.remove("registration_min_verified_neighbors");

    let restored: IncrementalSfmConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(restored.registration_min_verified_neighbors(), 1);
}

#[test]
fn removed_incremental_config_fields_are_rejected() {
    for field in [
        "second_pass",
        "recurrent_average_every",
        "extension_scheme",
        "registration_min_observations",
        "min_visible_landmarks",
    ] {
        let mut serialized = serde_json::to_value(IncrementalSfmConfig::default()).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .insert(field.to_owned(), serde_json::json!(true));

        let error = serde_json::from_value::<IncrementalSfmConfig>(serialized).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }
}

#[test]
fn removed_global_config_fields_are_rejected() {
    for field in ["preset", "require_connected_graph"] {
        let mut serialized = serde_json::to_value(GlobalAveragingConfig::default()).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .insert(field.to_owned(), serde_json::json!(true));

        let error = serde_json::from_value::<GlobalAveragingConfig>(serialized).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }
}

#[test]
fn incremental_config_rejects_invalid_numeric_policy() {
    assert!(openmvg_core::PixelThreshold::try_new(0.0).is_err());
    assert!(openmvg_core::AngularThreshold::from_degrees(f64::INFINITY).is_err());
    assert!(openmvg_core::AngularThreshold::from_degrees(-1.0).is_err());
    for invalid in [
        IncrementalSfmConfig::default().with_registration_min_inlier_ratio(f64::NAN),
        IncrementalSfmConfig::default().with_registration_min_inlier_ratio(1.01),
        IncrementalSfmConfig::default().with_bundle_adjust_growth_ratio(f64::NAN),
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn global_config_rejects_invalid_numeric_policy() {
    assert!(openmvg_core::AngularResidual::from_degrees(181.0).is_err());
    let default = GlobalAveragingConfig::default();
    for invalid in [
        default.with_gauge_scale_policy(GaugeScalePolicy::FixedBaseline {
            first: ViewId::new(0),
            second: ViewId::new(1),
            distance: 0.0,
            unit: LengthUnit::Meters,
        }),
        default.with_min_edge_isotropy(f64::NAN),
        default.with_min_edge_isotropy(0.34),
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn environment_variables_do_not_override_typed_sfm_policy() {
    if std::env::var_os(ENV_ISOLATION_CHILD).is_none() {
        let output = Command::new(std::env::current_exe().expect("current test executable"))
            .arg("--exact")
            .arg("environment_variables_do_not_override_typed_sfm_policy")
            .arg("--nocapture")
            .env(ENV_ISOLATION_CHILD, "1")
            .envs(LEGACY_POLICY_ENV.iter().copied())
            .output()
            .expect("isolated config-policy child starts");
        assert!(
            output.status.success(),
            "isolated config-policy child failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let config = IncrementalSfmConfig::default()
        .with_min_registration_support(13)
        .with_registration_min_inliers(10)
        .with_registration_min_inlier_ratio(0.42)
        .with_registration_max_rotation_edge_error(Some(
            openmvg_core::AngularThreshold::from_degrees(12.5).unwrap(),
        ))
        .with_registration_max_translation_edge_error(None)
        .with_registration_min_verified_neighbors(2);

    let registration = config.registration_config();
    assert_eq!(registration.min_observations(), 13);
    assert_eq!(registration.min_inliers(), 10);
    assert_eq!(registration.min_inlier_ratio(), 0.42);
    assert_eq!(
        registration.max_rotation_edge_error(),
        Some(openmvg_core::AngularThreshold::from_degrees(12.5).unwrap())
    );
    assert_eq!(registration.max_translation_edge_error(), None);

    let growth = config.growth_config();
    assert_eq!(growth.min_visible_landmarks(), 13);
    assert_eq!(growth.min_verified_registered_neighbors(), 2);

    let global = GlobalAveragingConfig::default().with_min_translation_parallax(Some(
        openmvg_core::AngularResidual::from_degrees(7.5).unwrap(),
    ));
    assert_eq!(
        global.min_translation_parallax(),
        Some(openmvg_core::AngularResidual::from_degrees(7.5).unwrap())
    );
}

#[test]
fn partial_global_config_inherits_the_current_parallax_default() {
    let mut serialized = serde_json::to_value(GlobalAveragingConfig::default()).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("min_translation_parallax_deg");

    let restored: GlobalAveragingConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        restored.min_translation_parallax(),
        GlobalAveragingConfig::default().min_translation_parallax()
    );
}
