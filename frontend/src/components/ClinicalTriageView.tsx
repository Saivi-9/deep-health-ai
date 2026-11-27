import React, { useState, useEffect } from 'react';
import {
  HeartPulse,
  Activity,
  AlertTriangle,
  ShieldCheck,
  Stethoscope,
  Sparkles,
  UserCheck,
  CheckCircle2,
  Gauge,
  Heart,
} from 'lucide-react';
import type {
  ClinicalPreset,
  ClinicalFeaturesInput,
  ClinicalTriageResponse,
  VisionDiagnosisResponse,
  ECGScreenResponse,
  ExplainMode,
  CardiovascularProfileResponse,
  VascularStiffnessResponse,
} from '../types';
import {
  runClinicalTriage,
  fetchCardiovascularProfile,
  fetchVascularStiffness,
} from '../api/client';

interface ClinicalTriageViewProps {
  presets: ClinicalPreset[];
  latestVision: VisionDiagnosisResponse | null;
  latestEcg: ECGScreenResponse | null;
  explainMode?: ExplainMode;
}

const DEFAULT_FEATURES: ClinicalFeaturesInput = {
  age: 58,
  sex: 1,
  chest_pain_type: 4,
  resting_bp: 142,
  cholesterol: 256,
  fasting_bs: 0,
  resting_ecg: 1,
  max_hr: 128,
  exercise_angina: 1,
  oldpeak: 1.9,
  st_slope: 2,
};

export const ClinicalTriageView: React.FC<ClinicalTriageViewProps> = ({
  presets,
  latestVision,
  latestEcg,
  explainMode = 'simple',
}) => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const [customForm, setCustomForm] = useState<ClinicalFeaturesInput | null>(null);
  const [includeMultimodal, setIncludeMultimodal] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ClinicalTriageResponse | null>(null);
  const [cvProfile, setCvProfile] = useState<CardiovascularProfileResponse | null>(null);
  const [stiffness, setStiffness] = useState<VascularStiffnessResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const activePreset = presets.find((p) => p.id === selectedPresetId) || presets[0];
  const effectivePresetId = selectedPresetId || activePreset?.id || '';
  const form = customForm || activePreset?.features || DEFAULT_FEATURES;

  useEffect(() => {
    let active = true;
    const payload: ClinicalFeaturesInput = {
      ...form,
      cxr_pneumonia_probability: includeMultimodal && latestVision ? latestVision.pneumonia_probability : null,
      ecg_is_anomaly: includeMultimodal && latestEcg ? latestEcg.is_anomaly : null,
      ecg_reconstruction_mse: includeMultimodal && latestEcg ? latestEcg.reconstruction_mse : null,
    };

    runClinicalTriage(payload)
      .then((res) => {
        if (active) {
          setErrorMsg(null);
          setResult(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (active) {
          setErrorMsg(err instanceof Error ? err.message : 'Clinical triage failed');
          setLoading(false);
        }
      });

    fetchCardiovascularProfile(payload)
      .then((p) => {
        if (active) setCvProfile(p);
      })
      .catch(() => {});

    fetchVascularStiffness(payload)
      .then((s) => {
        if (active) setStiffness(s);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [form, includeMultimodal, latestVision, latestEcg]);

  const handlePresetChange = (id: string) => {
    setSelectedPresetId(id);
    setCustomForm(null);
  };

  const updateNum = (field: keyof ClinicalFeaturesInput, value: number) => {
    setCustomForm({ ...form, [field]: value });
  };

  const bpStatus =
    form.resting_bp >= 140
      ? { label: 'High BP', cls: 'bg-clinical-rose/20 text-clinical-rose' }
      : form.resting_bp > 120
      ? { label: 'Elevated', cls: 'bg-amber-500/20 text-amber-300' }
      : { label: 'Optimal', cls: 'bg-clinical-green/20 text-clinical-green' };

  const cholStatus =
    form.cholesterol >= 240
      ? { label: 'High Lipid', cls: 'bg-clinical-rose/20 text-clinical-rose' }
      : form.cholesterol > 200
      ? { label: 'Borderline', cls: 'bg-amber-500/20 text-amber-300' }
      : { label: 'Healthy', cls: 'bg-clinical-green/20 text-clinical-green' };

  const oldpeakStatus =
    form.oldpeak >= 1.5
      ? { label: 'High Strain', cls: 'bg-clinical-rose/20 text-clinical-rose' }
      : form.oldpeak >= 0.6
      ? { label: 'Mild Shift', cls: 'bg-amber-500/20 text-amber-300' }
      : { label: 'Normal', cls: 'bg-clinical-green/20 text-clinical-green' };

  return (
    <div id="panel-clinical" role="tabpanel" aria-labelledby="tab-clinical" className="space-y-6">
      {/* Interactive Patient Persona Cards */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-4 shadow-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <UserCheck className="w-4 h-4 text-clinical-rose" />
            <span className="text-xs font-bold uppercase tracking-wider text-white">
              Choose a Sample Patient Profile (or Adjust Sliders Below to Test Any Scenario)
            </span>
          </div>

          <label className="flex items-center gap-2 bg-pacs-950 px-3 py-1.5 rounded-xl border border-pacs-700 cursor-pointer text-xs">
            <input
              type="checkbox"
              checked={includeMultimodal}
              onChange={(e) => setIncludeMultimodal(e.target.checked)}
            />
            <span className="font-semibold text-clinical-cyan">
              Combine Step 1 (X-Ray) &amp; Step 2 (ECG) Results
            </span>
          </label>
        </div>

        {presets.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {presets.map((p) => {
              const active = effectivePresetId === p.id && !customForm;
              const isHighRisk = p.id.includes('critical');
              const isLowRisk = p.id.includes('low');
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handlePresetChange(p.id)}
                  className={`p-3.5 rounded-xl border text-left transition cursor-pointer flex items-center justify-between gap-3 ${
                    active
                      ? 'bg-pacs-800 border-clinical-cyan shadow-md ring-1 ring-clinical-cyan/40'
                      : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                  }`}
                >
                  <div>
                    <span
                      className={`inline-block text-[10px] font-bold uppercase px-2 py-0.5 rounded-full mb-1 ${
                        isLowRisk
                          ? 'bg-clinical-green/20 text-clinical-green'
                          : isHighRisk
                          ? 'bg-clinical-rose/20 text-clinical-rose'
                          : 'bg-amber-500/20 text-amber-300'
                      }`}
                    >
                      {isLowRisk ? 'Healthy Checkup' : isHighRisk ? 'High-Risk Case' : 'Moderate Symptoms'}
                    </span>
                    <div className="text-xs font-bold text-white">{p.title}</div>
                    <div className="text-[11px] text-pacs-300 mt-0.5">
                      BP {p.features.resting_bp} mmHg &bull; Chol {p.features.cholesterol} mg/dL &bull; Max HR{' '}
                      {p.features.max_hr}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="bg-clinical-rose/15 border border-clinical-rose/40 text-clinical-rose px-4 py-2.5 rounded-xl text-xs font-mono">
          {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Interactive Vitals & Lab Sliders (6 Cols) */}
        <div className="lg:col-span-6 bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-5">
          <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
            <span className="text-xs uppercase font-bold tracking-wider text-white flex items-center gap-2">
              <Gauge className="w-4 h-4 text-clinical-cyan" />
              Interactive Vitals, Blood Labs &amp; Exercise Stress Inputs
            </span>
            {customForm && (
              <button
                type="button"
                onClick={() => setCustomForm(null)}
                className="text-[11px] text-clinical-cyan hover:underline cursor-pointer"
              >
                Reset to Preset
              </button>
            )}
          </div>

          {/* Group 1: Basic Vitals & Blood Pressure */}
          <div className="space-y-3">
            <div className="text-[11px] font-bold uppercase tracking-wider text-clinical-teal">
              1. Demographics &amp; Blood Pressure
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-pacs-200 font-semibold">Patient Age</span>
                  <span className="text-white font-bold">{form.age} years</span>
                </div>
                <input
                  type="range"
                  min={25}
                  max={85}
                  value={form.age}
                  onChange={(e) => updateNum('age', Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <span className="text-pacs-200 font-semibold block">Biological Sex</span>
                <select
                  value={form.sex}
                  onChange={(e) => updateNum('sex', Number(e.target.value))}
                  className="w-full bg-pacs-900 border border-pacs-700 rounded-lg px-2.5 py-1.5 text-white font-medium"
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5 sm:col-span-2">
                <div className="flex items-center justify-between">
                  <span className="text-pacs-200 font-semibold">
                    Resting Systolic Blood Pressure (Normal: &lt;120 mmHg)
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-white font-bold">{form.resting_bp} mmHg</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${bpStatus.cls}`}>
                      {bpStatus.label}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min={85}
                  max={210}
                  value={form.resting_bp}
                  onChange={(e) => updateNum('resting_bp', Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </label>
            </div>
          </div>

          {/* Group 2: Cholesterol & Blood Sugar */}
          <div className="space-y-3">
            <div className="text-[11px] font-bold uppercase tracking-wider text-clinical-cyan">
              2. Cholesterol &amp; Blood Sugar Labs
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-pacs-200 font-semibold">Total Cholesterol</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${cholStatus.cls}`}>
                    {form.cholesterol} mg/dL
                  </span>
                </div>
                <input
                  type="range"
                  min={120}
                  max={420}
                  value={form.cholesterol}
                  onChange={(e) => updateNum('cholesterol', Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
                <span className="text-[10px] text-pacs-400 block">Healthy target: &lt;200 mg/dL</span>
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <span className="text-pacs-200 font-semibold block">Fasting Blood Sugar</span>
                <select
                  value={form.fasting_bs}
                  onChange={(e) => updateNum('fasting_bs', Number(e.target.value))}
                  className="w-full bg-pacs-900 border border-pacs-700 rounded-lg px-2.5 py-1.5 text-white font-medium"
                >
                  <option value={0}>Normal (≤ 120 mg/dL)</option>
                  <option value={1}>Elevated (&gt; 120 mg/dL / Pre-Diabetes)</option>
                </select>
                <span className="text-[10px] text-pacs-400 block">High glucose accelerates vascular aging</span>
              </label>
            </div>
          </div>

          {/* Group 3: Exercise Heart Response & Symptoms */}
          <div className="space-y-3">
            <div className="text-[11px] font-bold uppercase tracking-wider text-clinical-rose">
              3. Symptoms &amp; Exercise Stress Response
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <span className="text-pacs-200 font-semibold block">Chest Discomfort Type</span>
                <select
                  value={form.chest_pain_type}
                  onChange={(e) => updateNum('chest_pain_type', Number(e.target.value))}
                  className="w-full bg-pacs-900 border border-pacs-700 rounded-lg px-2.5 py-1.5 text-white font-medium"
                >
                  <option value={1}>Typical Exertional Angina</option>
                  <option value={2}>Mild / Atypical Chest Discomfort</option>
                  <option value={3}>Non-Cardiac Chest Pain</option>
                  <option value={4}>Silent / Asymptomatic Ischemia</option>
                </select>
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <span className="text-pacs-200 font-semibold block">Chest Pain During Exercise?</span>
                <select
                  value={form.exercise_angina}
                  onChange={(e) => updateNum('exercise_angina', Number(e.target.value))}
                  className="w-full bg-pacs-900 border border-pacs-700 rounded-lg px-2.5 py-1.5 text-white font-medium"
                >
                  <option value={0}>No Pain During Exercise</option>
                  <option value={1}>Yes — Exercise Triggers Discomfort</option>
                </select>
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-pacs-200 font-semibold">Peak Exercise Heart Rate</span>
                  <span className="text-white font-bold">{form.max_hr} BPM</span>
                </div>
                <input
                  type="range"
                  min={70}
                  max={205}
                  value={form.max_hr}
                  onChange={(e) => updateNum('max_hr', Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </label>

              <label className="bg-pacs-950/70 border border-pacs-700 rounded-xl p-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-pacs-200 font-semibold">ECG Stress Shift (Oldpeak)</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${oldpeakStatus.cls}`}>
                    {form.oldpeak.toFixed(1)} mm ({oldpeakStatus.label})
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={5.5}
                  step={0.1}
                  value={form.oldpeak}
                  onChange={(e) => updateNum('oldpeak', Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </label>
            </div>
          </div>

          {/* Connected Step 1 & Step 2 Summary Box */}
          <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5 text-xs space-y-2">
            <div className="text-[11px] font-bold text-white">
              Connected Scans from Step 1 (X-Ray) &amp; Step 2 (ECG):
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
              <div className="flex items-center gap-2 text-pacs-300 bg-pacs-900 p-2 rounded-lg border border-pacs-800">
                <Stethoscope className="w-4 h-4 text-clinical-cyan shrink-0" />
                <span>
                  Lung X-Ray:{' '}
                  <strong className="text-white">
                    {latestVision
                      ? `${(latestVision.pneumonia_probability * 100).toFixed(0)}% Pneumonia Risk`
                      : 'Not scanned yet'}
                  </strong>
                </span>
              </div>
              <div className="flex items-center gap-2 text-pacs-300 bg-pacs-900 p-2 rounded-lg border border-pacs-800">
                <Activity className="w-4 h-4 text-clinical-teal shrink-0" />
                <span>
                  ECG Rhythm:{' '}
                  <strong className={latestEcg?.is_anomaly ? 'text-clinical-rose' : 'text-clinical-green'}>
                    {latestEcg
                      ? latestEcg.is_anomaly
                        ? `Irregular (${latestEcg.rhythm_subtype})`
                        : 'Normal Sinus'
                      : 'Not tested yet'}
                  </strong>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Heart Risk, Vascular Age & Prevention Readout (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-clinical-rose" />
                {explainMode === 'simple'
                  ? 'COMPLETE HEART HEALTH & RISK ASSESSMENT'
                  : 'MULTIMODAL CARDIOPULMONARY RISK READOUT'}
              </span>
              <span className="text-[11px] font-mono text-clinical-cyan">
                {result?.model_used || (loading ? 'Computing...' : '--')}
              </span>
            </div>

            {result && (
              <>
                {/* Top Triage Recommendation Banner */}
                <div
                  className={`p-4 rounded-2xl border flex flex-wrap items-center justify-between gap-3 ${
                    result.fused_risk_score >= 0.55
                      ? 'bg-clinical-rose/10 border-clinical-rose/40'
                      : result.fused_risk_score >= 0.35
                      ? 'bg-amber-500/10 border-amber-500/40'
                      : 'bg-clinical-green/10 border-clinical-green/40'
                  }`}
                >
                  <div>
                    <div className="text-[11px] font-bold uppercase tracking-wider text-pacs-300">
                      Overall Triage Recommendation
                    </div>
                    <h3 className="text-lg font-extrabold text-white mt-0.5">{result.triage_tier}</h3>
                  </div>
                  <span
                    className={`px-3.5 py-1.5 rounded-full text-xs font-bold flex items-center gap-1.5 border ${
                      result.fused_risk_score >= 0.55
                        ? 'bg-clinical-rose/20 text-clinical-rose border-clinical-rose/40'
                        : 'bg-clinical-green/20 text-clinical-green border-clinical-green/40'
                    }`}
                  >
                    {result.fused_risk_score >= 0.55 ? (
                      <AlertTriangle className="w-4 h-4" />
                    ) : (
                      <ShieldCheck className="w-4 h-4" />
                    )}
                    {result.recommended_urgency}
                  </span>
                </div>

                {/* 3 Key Cards: Combined Risk, Heart Age, and Blood Pressure Load */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5">
                    <span className="text-[11px] text-pacs-300 font-semibold block">Combined Risk Score</span>
                    <span className="text-2xl font-black text-clinical-cyan mt-0.5 block">
                      {(result.fused_risk_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[11px] text-pacs-400">
                      EHR Alone: {(result.cad_risk_probability * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5">
                    <span className="text-[11px] text-pacs-300 font-semibold flex items-center gap-1">
                      <Heart className="w-3.5 h-3.5 text-clinical-rose" /> Estimated Heart Age
                    </span>
                    <span className="text-2xl font-black text-white mt-0.5 block">
                      {cvProfile ? `${cvProfile.estimated_vascular_age} yrs` : `${form.age} yrs`}
                    </span>
                    <span
                      className={`text-[11px] font-semibold ${
                        (cvProfile?.vascular_age_acceleration_years || 0) > 5
                          ? 'text-clinical-rose'
                          : 'text-clinical-green'
                      }`}
                    >
                      {cvProfile
                        ? cvProfile.vascular_age_acceleration_years > 0
                          ? `+${cvProfile.vascular_age_acceleration_years} yrs vs actual age (${form.age})`
                          : 'Matches chronological age'
                        : `Actual Age: ${form.age} yrs`}
                    </span>
                  </div>

                  <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5">
                    <span className="text-[11px] text-pacs-300 font-semibold block">Estimated BP &amp; Pulse</span>
                    <span className="text-xl font-black text-white mt-1 block">
                      {stiffness
                        ? `${stiffness.resting_systolic_bp_mmhg.toFixed(0)}/${stiffness.estimated_diastolic_bp_mmhg.toFixed(0)}`
                        : `${form.resting_bp}/80`}{' '}
                      <span className="text-xs font-normal text-pacs-300">mmHg</span>
                    </span>
                    <span className="text-[11px] text-pacs-400 block">
                      Pulse Pressure: {stiffness ? `${stiffness.estimated_pulse_pressure_mmhg} mmHg` : '--'}
                    </span>
                  </div>
                </div>

                {/* Why the AI gave this score (Primary Drivers) */}
                <div className="bg-pacs-950/80 border border-pacs-700 rounded-xl p-4 space-y-2.5 text-xs">
                  <span className="text-white font-bold block">
                    Why the AI Calculated This Risk (Top Health Factors):
                  </span>
                  {result.clinical_risk_drivers.map((d, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between text-xs border-b border-pacs-800/80 pb-2 last:border-none last:pb-0"
                    >
                      <span className="text-pacs-200">
                        {d.factor}: <strong className="text-white">{d.value}</strong>
                      </span>
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                          d.impact === 'HIGH'
                            ? 'bg-clinical-rose/20 text-clinical-rose'
                            : d.impact === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-300'
                            : 'bg-clinical-green/20 text-clinical-green'
                        }`}
                      >
                        {d.impact} IMPACT
                      </span>
                    </div>
                  ))}
                </div>

                {/* Actionable Prevention & Next Steps Checklist */}
                {cvProfile && cvProfile.preventive_recommendations.length > 0 && (
                  <div className="bg-clinical-teal/10 border border-clinical-teal/30 rounded-xl p-4 space-y-2 text-xs">
                    <span className="text-white font-bold flex items-center gap-1.5">
                      <HeartPulse className="w-4 h-4 text-clinical-teal" />
                      Recommended Action Plan &amp; Prevention Steps:
                    </span>
                    <ul className="space-y-1.5">
                      {cvProfile.preventive_recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-pacs-100">
                          <CheckCircle2 className="w-4 h-4 text-clinical-teal shrink-0 mt-0.5" />
                          <span>{rec}</span>
                        </li>
                      ))}
                      {stiffness && (
                        <li className="flex items-start gap-2 text-pacs-100">
                          <CheckCircle2 className="w-4 h-4 text-clinical-cyan shrink-0 mt-0.5" />
                          <span>{stiffness.clinical_recommendation}</span>
                        </li>
                      )}
                    </ul>
                  </div>
                )}

                {/* Pro View: 9 Engineered Biomarkers & Multimodal Modifiers */}
                {explainMode === 'pro' && (
                  <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-2">
                    <div className="text-[10px] uppercase font-bold text-pacs-400">
                      Engineered Cardiovascular Biomarkers (9 Derived Indices)
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] font-mono">
                      {Object.entries(result.engineered_biomarkers).map(([k, v]) => (
                        <div key={k} className="flex justify-between gap-2 text-pacs-300">
                          <span className="truncate">{k.replace(/_/g, ' ')}:</span>
                          <span className="text-white font-bold">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

