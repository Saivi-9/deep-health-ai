export type TriageUrgency = 'LOW' | 'EXPEDITED' | 'CRITICAL' | 'STAT' | string;

export interface UncertaintyDetails {
  predictive_variance: number;
  normalized_entropy: number;
  uncertainty_risk_level: 'LOW' | 'MODERATE' | 'HIGH' | string;
  human_referral_mandatory: boolean;
  monte_carlo_samples: number;
}

export interface SeverityDetails {
  radiological_severity: string;
  radiological_grade: string;
  infiltrate_burden_percentage: number;
  clinical_triage_urgency: TriageUrgency;
}

export interface HotspotPoint {
  x_norm: number;
  y_norm: number;
  peak_activation: number;
}

export interface VisionDiagnosisResponse {
  filename: string;
  model_used?: string;
  diagnosis: string;
  confidence: number;
  normal_probability: number;
  pneumonia_probability: number;
  uncertainty: UncertaintyDetails;
  severity: SeverityDetails;
  gradcam_available: boolean;
  gradcam_heatmap_base64?: string | null;
  gradcam_overlay_base64?: string | null;
  hotspot?: HotspotPoint | null;
  radiomics?: Record<string, number>;
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
  p_wave_amplitude?: number;
  t_wave_amplitude?: number;
  r_s_ratio?: number;
  signal_rms?: number;
  dominant_freq_hz?: number;
  spectral_entropy?: number;
  clinical_flags: {
    wide_qrs_ventricular_conduction_delay: boolean;
    prolonged_repolarization_qtc: boolean;
    st_segment_abnormality_ischemia: boolean;
    atrial_fibrillation_absent_p_wave?: boolean;
  };
  [key: string]: unknown;
}

export interface ECGScreenResponse {
  model_used?: string;
  diagnosis: string;
  is_anomaly: boolean;
  reconstruction_mse: number;
  decision_threshold: number;
  rhythm_subtype?: string;
  subtype_probabilities?: Record<string, number>;
  reconstructed_signal?: number[];
  residual_profile?: number[];
  telemetry: ECGTelemetry;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  device: string;
  models_loaded: Record<string, boolean>;
}

export interface VisionSample {
  id: string;
  title: string;
  source: string;
  ground_truth: string;
  image_base64: string;
}

export interface ECGSample {
  id: string;
  title: string;
  source: string;
  sampling_rate_hz: number;
  signal: number[];
}

export interface ClinicalFeaturesInput {
  age: number;
  sex: number;
  chest_pain_type: number;
  resting_bp: number;
  cholesterol: number;
  fasting_bs: number;
  resting_ecg: number;
  max_hr: number;
  exercise_angina: number;
  oldpeak: number;
  st_slope: number;
  cxr_pneumonia_probability?: number | null;
  ecg_is_anomaly?: boolean | null;
  ecg_reconstruction_mse?: number | null;
}

export interface ClinicalPreset {
  id: string;
  title: string;
  source: string;
  true_outcome: number;
  features: ClinicalFeaturesInput;
}

export interface DatasetSamplesResponse {
  vision_samples: VisionSample[];
  ecg_samples: ECGSample[];
  clinical_presets: ClinicalPreset[];
}

export interface ClinicalTriageResponse {
  model_used: string;
  cad_risk_probability: number;
  predicted_label: string;
  fused_risk_score: number;
  triage_tier: string;
  recommended_urgency: string;
  multimodal_modifiers: string[];
  clinical_risk_drivers: { factor: string; value: string; impact: string }[];
  engineered_biomarkers: Record<string, number>;
  hf_consensus?: {
    repo_id: string;
    predicted_label: string;
    risk_probability: number;
    healthy_probability: number;
  } | null;
  timestamp: string;
}

export interface PretrainedCatalogEntry {
  repo_id: string;
  domain: 'vision' | 'ecg' | 'clinical';
  display_name: string;
  architecture: string;
  input_spec: string;
  classes: string[];
  license: string;
  summary: string;
}

export interface HFSearchResult {
  repo_id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string;
  library_name: string;
  verified_in_studio: boolean;
  url: string;
}

export type ActiveTab = 'radiology' | 'ecg' | 'clinical' | 'pretrained' | 'batch' | 'governance';
