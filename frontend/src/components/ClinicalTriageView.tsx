import React, { useState, useEffect } from 'react';
import { HeartPulse, Activity, AlertTriangle, ShieldCheck, Stethoscope, Sparkles } from 'lucide-react';
import type {
  ClinicalPreset,
  ClinicalFeaturesInput,
  ClinicalTriageResponse,
  VisionDiagnosisResponse,
  ECGScreenResponse,
} from '../types';
import { runClinicalTriage } from '../api/client';

interface ClinicalTriageViewProps {
  presets: ClinicalPreset[];
  latestVision: VisionDiagnosisResponse | null;
  latestEcg: ECGScreenResponse | null;
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
}) => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const [customForm, setCustomForm] = useState<ClinicalFeaturesInput | null>(null);
  const [includeMultimodal, setIncludeMultimodal] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ClinicalTriageResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const activePreset = presets.find((p) => p.id === selectedPresetId) || presets[presets.length - 1];
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

  return (
    <div id="panel-clinical" role="tabpanel" aria-labelledby="tab-clinical" className="space-y-6">
      {/* Header Banner */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-pacs-800 border border-pacs-600 flex items-center justify-center">
            <HeartPulse className="w-5 h-5 text-clinical-rose" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">REAL CARDIOLOGY EHR &amp; MULTI-MODAL RISK FUSION</div>
            <div className="text-pacs-300 font-mono">
              1,305 Real Deduplicated Multi-Center Patients • 20 Clinical Biomarkers • HF Consensus
            </div>
          </div>
        </div>

        <label className="flex items-center gap-2 bg-pacs-950 px-3 py-1.5 rounded-lg border border-pacs-700 cursor-pointer">
          <input
            type="checkbox"
            checked={includeMultimodal}
            onChange={(e) => setIncludeMultimodal(e.target.checked)}
          />
          <span className="font-mono text-clinical-cyan">Fuse Live CXR &amp; ECG Findings</span>
        </label>
      </div>

      {errorMsg && (
        <div className="bg-clinical-rose/15 border border-clinical-rose/40 text-clinical-rose px-4 py-2.5 rounded-xl text-xs font-mono">
          {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Clinical EHR Form (6 Cols) */}
        <div className="lg:col-span-6 bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-pacs-700/60 pb-3">
            <span className="text-xs uppercase font-bold tracking-wider text-pacs-300">
              Patient Hemodynamic &amp; Stress Profile
            </span>
            {presets.length > 0 && (
              <select
                aria-label="Real Cardiology Cohort Preset"
                value={effectivePresetId}
                onChange={(e) => handlePresetChange(e.target.value)}
                className="bg-pacs-800 text-clinical-cyan border border-pacs-600 rounded-lg px-2.5 py-1 text-xs outline-none"
              >
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Age (yrs): {form.age}</span>
              <input type="range" min={25} max={85} value={form.age} onChange={(e) => updateNum('age', Number(e.target.value))} className="w-full" />
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Biological Sex</span>
              <select value={form.sex} onChange={(e) => updateNum('sex', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={1}>Male (1)</option>
                <option value={0}>Female (0)</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Chest Pain Type</span>
              <select value={form.chest_pain_type} onChange={(e) => updateNum('chest_pain_type', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={1}>1: Typical Angina</option>
                <option value={2}>2: Atypical Angina</option>
                <option value={3}>3: Non-Anginal</option>
                <option value={4}>4: Asymptomatic Ischemia</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Resting BP: {form.resting_bp} mmHg</span>
              <input type="range" min={85} max={210} value={form.resting_bp} onChange={(e) => updateNum('resting_bp', Number(e.target.value))} className="w-full" />
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Cholesterol: {form.cholesterol} mg/dL</span>
              <input type="range" min={120} max={450} value={form.cholesterol} onChange={(e) => updateNum('cholesterol', Number(e.target.value))} className="w-full" />
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Max HR: {form.max_hr} bpm</span>
              <input type="range" min={70} max={205} value={form.max_hr} onChange={(e) => updateNum('max_hr', Number(e.target.value))} className="w-full" />
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">ST Oldpeak: {form.oldpeak.toFixed(1)} mm</span>
              <input type="range" min={0} max={5.5} step={0.1} value={form.oldpeak} onChange={(e) => updateNum('oldpeak', Number(e.target.value))} className="w-full" />
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Exercise Angina</span>
              <select value={form.exercise_angina} onChange={(e) => updateNum('exercise_angina', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={0}>No (0)</option>
                <option value={1}>Yes (1)</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">ST Slope</span>
              <select value={form.st_slope} onChange={(e) => updateNum('st_slope', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={1}>1: Upsloping</option>
                <option value={2}>2: Flat (Ischemic)</option>
                <option value={3}>3: Downsloping</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Fasting Glucose &gt;120</span>
              <select value={form.fasting_bs} onChange={(e) => updateNum('fasting_bs', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={0}>Normal (&le;120 mg/dL)</option>
                <option value={1}>Elevated (&gt;120 mg/dL)</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-pacs-400 font-mono block">Resting ECG</span>
              <select value={form.resting_ecg} onChange={(e) => updateNum('resting_ecg', Number(e.target.value))} className="w-full bg-pacs-950 border border-pacs-700 rounded-lg px-2 py-1.5 text-white">
                <option value={0}>0: Normal</option>
                <option value={1}>1: ST-T Wave Abnormality</option>
                <option value={2}>2: LV Hypertrophy</option>
              </select>
            </label>
          </div>

          {/* Active Multi-Modal Session Feeds */}
          <div className="bg-pacs-950/90 border border-pacs-700/80 rounded-xl p-3.5 text-xs space-y-2">
            <div className="text-[10px] uppercase font-bold tracking-wider text-pacs-400">
              Live Session Modality Feeds Attached to Triage
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 font-mono text-[11px]">
              <div className="flex items-center gap-2 text-pacs-300">
                <Stethoscope className="w-3.5 h-3.5 text-clinical-cyan" />
                <span>
                  CXR Pneumonia Prob:{' '}
                  <strong className="text-white">
                    {latestVision ? `${(latestVision.pneumonia_probability * 100).toFixed(1)}%` : 'Not Run'}
                  </strong>
                </span>
              </div>
              <div className="flex items-center gap-2 text-pacs-300">
                <Activity className="w-3.5 h-3.5 text-clinical-teal" />
                <span>
                  ECG Rhythm:{' '}
                  <strong className={latestEcg?.is_anomaly ? 'text-clinical-rose' : 'text-clinical-green'}>
                    {latestEcg ? (latestEcg.is_anomaly ? `Anomaly (${latestEcg.rhythm_subtype})` : 'Normal Sinus') : 'Not Run'}
                  </strong>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Triage & Risk Output (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-clinical-rose" />
                MULTIMODAL CARDIOPULMONARY RISK READOUT
              </span>
              <span className="text-[11px] font-mono text-clinical-cyan">{result?.model_used || (loading ? 'Computing...' : '--')}</span>
            </div>

            {result && (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-xs text-pacs-400 font-mono">FUSED CLINICAL TRIAGE TIER</div>
                    <h3 className="text-lg font-extrabold text-white mt-0.5">{result.triage_tier}</h3>
                  </div>
                  <span
                    className={`px-3 py-1.5 rounded-full text-xs font-bold flex items-center gap-1.5 border ${
                      result.fused_risk_score >= 0.55
                        ? 'bg-clinical-rose/20 text-clinical-rose border-clinical-rose/40'
                        : 'bg-clinical-green/20 text-clinical-green border-clinical-green/40'
                    }`}
                  >
                    {result.fused_risk_score >= 0.55 ? <AlertTriangle className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
                    {result.recommended_urgency}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3">
                    <span className="text-[10px] text-pacs-400 uppercase font-bold block">EHR Risk Prob</span>
                    <span className="text-xl font-black text-white mt-0.5 block">
                      {(result.cad_risk_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-pacs-400 font-mono">{result.predicted_label}</span>
                  </div>
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3">
                    <span className="text-[10px] text-pacs-400 uppercase font-bold block">Multimodal Score</span>
                    <span className="text-xl font-black text-clinical-cyan mt-0.5 block">
                      {(result.fused_risk_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-pacs-400 font-mono">EHR + CXR + ECG</span>
                  </div>
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3">
                    <span className="text-[10px] text-pacs-400 uppercase font-bold block">HF RF Consensus</span>
                    <span className="text-xl font-black text-amber-300 mt-0.5 block">
                      {result.hf_consensus ? `${(result.hf_consensus.risk_probability * 100).toFixed(1)}%` : '--'}
                    </span>
                    <span className="text-[10px] text-pacs-400 font-mono truncate block">nexusbert/RF-300</span>
                  </div>
                </div>

                {result.multimodal_modifiers.length > 0 && (
                  <div className="bg-pacs-950/90 border border-pacs-700/80 rounded-xl p-3 space-y-1 text-[11px] font-mono">
                    <span className="text-pacs-400 uppercase font-bold block">Multimodal Modifiers:</span>
                    {result.multimodal_modifiers.map((m, i) => (
                      <div key={i} className="text-clinical-cyan">&bull; {m}</div>
                    ))}
                  </div>
                )}

                <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-2 text-xs">
                  <span className="text-slate-300 font-bold block">Primary Clinical Drivers:</span>
                  {result.clinical_risk_drivers.map((d, i) => (
                    <div key={i} className="flex items-center justify-between text-[11px] font-mono border-b border-pacs-800/80 pb-1 last:border-none last:pb-0">
                      <span className="text-slate-200">{d.factor}: <strong className="text-white">{d.value}</strong></span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        d.impact === 'HIGH' ? 'bg-clinical-rose/20 text-clinical-rose' :
                        d.impact === 'MODERATE' ? 'bg-amber-500/20 text-amber-300' : 'bg-clinical-green/20 text-clinical-green'
                      }`}>
                        {d.impact}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5">
                  <div className="text-[10px] uppercase font-bold text-pacs-400 mb-2">Engineered Cardiovascular Biomarkers (9 Derived Features)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] font-mono">
                    {Object.entries(result.engineered_biomarkers).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2 text-pacs-300">
                        <span className="truncate">{k.replace(/_/g, ' ')}:</span>
                        <span className="text-white font-bold">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
