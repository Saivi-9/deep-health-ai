export type TriageUrgency = 'LOW' | 'EXPEDITED' | 'CRITICAL' | 'STAT';

export interface UncertaintyDetails {
  predictive_variance: number;
  normalized_entropy: number;
  uncertainty_risk_level: 'LOW' | 'MODERATE' | 'HIGH';
  human_referral_mandatory: boolean;
  monte_carlo_samples: number;
}

export interface SeverityDetails {
  radiological_severity: string;
  radiological_grade: string;
  infiltrate_burden_percentage: number;
  clinical_triage_urgency: TriageUrgency;
}

export interface VisionDiagnosisResponse {
  filename: string;
  diagnosis: string;
  confidence: number;
  normal_probability: number;
  pneumonia_probability: number;
  uncertainty: UncertaintyDetails;
  severity: SeverityDetails;
  gradcam_available: boolean;
  timestamp: string;
}

export interface ECGTelemetry {
  heart_rate_bpm: number;
  qrs_duration_ms: number;
  pr_interval_ms: number;
  qt_interval_ms: number;
  qtc_interval_ms: number;
  st_segment_mv: number;
  r_peak_amplitude: number;
  clinical_flags: {
    wide_qrs_ventricular_conduction_delay: boolean;
    prolonged_repolarization_qtc: boolean;
    st_segment_abnormality_ischemia: boolean;
  };
}

export interface ECGScreenResponse {
  diagnosis: string;
  is_anomaly: boolean;
  reconstruction_mse: number;
  decision_threshold: number;
  telemetry: ECGTelemetry;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  device: string;
  models_loaded: {
    ResidualChestCNN: boolean;
    Conv1dAutoencoder: boolean;
  };
}
