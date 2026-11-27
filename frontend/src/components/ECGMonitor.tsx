import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  AlertOctagon,
  Heart,
  Zap,
  Clock,
  ShieldCheck,
  BarChart2,
  Upload,
  Play,
  Pause,
  ArrowRight,
  CheckCircle2,
  Info,
} from 'lucide-react';
import type { ECGScreenResponse, ECGSample, ExplainMode, ECGRiskSummaryResponse } from '../types';
import { screenECGSignal, fetchECGClinicalSummary } from '../api/client';

interface ECGMonitorProps {
  samples: ECGSample[];
  availableModels: string[];
  onEcgUpdate?: (res: ECGScreenResponse) => void;
  explainMode?: ExplainMode;
  onNextStep?: () => void;
}

const EMPTY_SIGNAL: number[] = [];

const RHYTHM_PLAIN_GUIDE: Record<string, string> = {
  'Normal Sinus (NSR)': 'Steady, healthy electrical heartbeat originating normally from the heart’s natural pacemaker.',
  'Normal Sinus (N)': 'Normal, regular heartbeat rhythm with healthy electrical conduction.',
  'PVC Ectopy': 'An early, extra heartbeat originating in the lower pumping chambers (ventricles).',
  'PVC (V)': 'Premature Ventricular Contraction — an early extra beat felt like a brief skipped or forceful beat.',
  'Atrial Fibrillation': 'Irregular, rapid electrical firing in the upper chambers (atria) without a distinct P-wave.',
  'Atrial Premature (A)': 'An early beat triggered in the upper chambers (atria) of the heart.',
  'ST-Elevation Ischemia': 'Elevated ST wave segment suggesting acute heart muscle strain or reduced blood flow.',
  LBBB: 'Left Bundle Branch Block — electrical impulse travels more slowly through the left side of the heart.',
  'LBBB (L)': 'Left Bundle Branch Block — widened QRS wave due to delayed left ventricular electrical activation.',
  'RBBB (R)': 'Right Bundle Branch Block — delayed electrical conduction along the right bundle pathway.',
};

export const ECGMonitor: React.FC<ECGMonitorProps> = ({
  samples,
  availableModels,
  onEcgUpdate,
  explainMode = 'simple',
  onNextStep,
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [modelId, setModelId] = useState<string>('resnet1d');
  const [threshold, setThreshold] = useState<number>(0.01037);
  const [customSignal, setCustomSignal] = useState<number[] | null>(null);
  const [customRate, setCustomRate] = useState<number | null>(null);
  const [screeningResult, setScreeningResult] = useState<ECGScreenResponse | null>(null);
  const [clinicalSummary, setClinicalSummary] = useState<ECGRiskSummaryResponse | null>(null);
  const [showReconstruction, setShowReconstruction] = useState<boolean>(true);
  const [showResiduals, setShowResiduals] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [liveSweep, setLiveSweep] = useState<boolean>(true);
  const [sweepIndex, setSweepIndex] = useState<number>(0);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const activeSample =
    samples.find((s) => s.id === selectedSampleId) ||
    samples.find((s) => s.id === 'leadii_0') ||
    samples[0];
  const effectiveSampleId = selectedSampleId || activeSample?.id || '';
  const currentSignal = customSignal || activeSample?.signal || EMPTY_SIGNAL;
  const samplingRate = customRate || activeSample?.sampling_rate_hz || 250;

  useEffect(() => {
    if (currentSignal.length < 32) return;
    let active = true;
    screenECGSignal(currentSignal, samplingRate, threshold, modelId)
      .then((res) => {
        if (!active) return;
        setScreeningResult(res);
        setLoading(false);
        onEcgUpdate?.(res);
      })
      .catch(() => {
        if (active) setLoading(false);
      });

    fetchECGClinicalSummary(currentSignal, samplingRate, threshold)
      .then((sum) => {
        if (active) setClinicalSummary(sum);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [currentSignal, samplingRate, threshold, modelId, onEcgUpdate]);

  // Live animated hospital monitor sweep cursor
  useEffect(() => {
    if (!liveSweep || currentSignal.length === 0) return;
    const timer = window.setInterval(() => {
      setSweepIndex((prev) => (prev + 4) % currentSignal.length);
    }, 45);
    return () => window.clearInterval(timer);
  }, [liveSweep, currentSignal.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || currentSignal.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = '#050C16';
    ctx.fillRect(0, 0, width, height);

    // Subtle ECG millimeter paper grid
    ctx.strokeStyle = 'rgba(20, 184, 166, 0.11)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const recon = screeningResult?.reconstructed_signal || [];
    const maxAbs = Math.max(2.0, ...currentSignal.map((v) => Math.abs(v)), ...recon.map((v) => Math.abs(v)));
    const midY = height / 2 + 10;
    const scaleY = (height * 0.40) / maxAbs;
    const stepX = width / Math.max(1, currentSignal.length - 1);

    // Isoelectric 0 mV baseline
    ctx.strokeStyle = '#1E3A5F';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Real Autoencoder Residual Area
    if (showResiduals && recon.length > 0 && screeningResult?.is_anomaly) {
      const nPlot = Math.min(currentSignal.length, recon.length);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.beginPath();
      for (let i = 0; i < nPlot; i++) {
        const x = i * stepX;
        const y = midY - currentSignal[i] * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      for (let i = nPlot - 1; i >= 0; i--) {
        ctx.lineTo(i * stepX, midY - recon[i] * scaleY);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Real Reconstructed Expected Normal Waveform
    if (showReconstruction && recon.length > 0) {
      const nPlot = Math.min(currentSignal.length, recon.length);
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let i = 0; i < nPlot; i++) {
        const x = i * stepX;
        const y = midY - recon[i] * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Input ECG waveform
    ctx.strokeStyle = screeningResult?.is_anomaly ? '#FB7185' : '#10B981';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    for (let i = 0; i < currentSignal.length; i++) {
      const x = i * stepX;
      const y = midY - currentSignal[i] * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Live Sweep Bar or Hover Scrub Cursor
    const activeIdx = hoverIndex !== null ? hoverIndex : liveSweep ? sweepIndex : null;
    if (activeIdx !== null && activeIdx >= 0 && activeIdx < currentSignal.length) {
      const cx = activeIdx * stepX;
      const cy = midY - currentSignal[activeIdx] * scaleY;
      ctx.save();
      ctx.strokeStyle = hoverIndex !== null ? '#FBBF24' : 'rgba(45, 212, 191, 0.55)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, height);
      ctx.stroke();

      ctx.fillStyle = hoverIndex !== null ? '#FBBF24' : '#2DD4BF';
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }, [currentSignal, screeningResult, showReconstruction, showResiduals, liveSweep, sweepIndex, hoverIndex]);

  const handleSampleChange = (id: string) => {
    setSelectedSampleId(id);
    setCustomSignal(null);
    setCustomRate(null);
    setLoading(true);
    if (id.startsWith('mitbih_')) setModelId('mitbih_resnet1d');
    else if (modelId === 'mitbih_resnet1d') setModelId('resnet1d');
  };

  const handleCsvUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = String(ev.target?.result || '');
      const nums = text
        .split(/[\s,;]+/)
        .map((s) => parseFloat(s))
        .filter((n) => Number.isFinite(n));
      if (nums.length >= 32) {
        setSelectedSampleId('custom_ecg');
        setCustomSignal(nums.slice(0, 256));
        setCustomRate(250);
        setLoading(true);
      }
    };
    reader.readAsText(file);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (currentSignal.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setHoverIndex(Math.round(ratio * (currentSignal.length - 1)));
  };

  const inspectedIdx = hoverIndex !== null ? hoverIndex : sweepIndex;
  const inspectedTimeMs = Math.round((inspectedIdx / Math.max(1, samplingRate)) * 1000);
  const inspectedMv = currentSignal[inspectedIdx] ?? 0;
  const inspectedReconMv = screeningResult?.reconstructed_signal?.[inspectedIdx] ?? 0;

  const hrOk =
    (screeningResult?.telemetry.heart_rate_bpm ?? 75) >= 60 &&
    (screeningResult?.telemetry.heart_rate_bpm ?? 75) <= 100;
  const qrsOk = !screeningResult?.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay;
  const qtcOk = !screeningResult?.telemetry.clinical_flags.prolonged_repolarization_qtc;
  const stOk = !screeningResult?.telemetry.clinical_flags.st_segment_abnormality_ischemia;

  return (
    <div id="panel-ecg" role="tabpanel" aria-labelledby="tab-ecg" className="space-y-6">
      {/* Clickable Visual Rhythm Strip Preset Selector */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-4 shadow-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Heart className="w-4 h-4 text-clinical-rose animate-heartbeat" />
            <span className="text-xs font-bold uppercase tracking-wider text-white">
              Select a Heartbeat Rhythm Pattern to Inspect Live
            </span>
          </div>
          {screeningResult && (
            <span
              className={`px-3.5 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 ${
                screeningResult.is_anomaly
                  ? 'bg-clinical-rose/20 text-clinical-rose border-clinical-rose/40'
                  : 'bg-clinical-green/20 text-clinical-green border-clinical-green/40'
              }`}
            >
              {screeningResult.is_anomaly ? <AlertOctagon className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
              {screeningResult.is_anomaly
                ? `Irregular Rhythm Detected: ${screeningResult.rhythm_subtype}`
                : `Healthy Rhythm: ${screeningResult.rhythm_subtype}`}
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-9 gap-2">
          {samples.map((s) => {
            const active = effectiveSampleId === s.id;
            const isNormalBeat = s.title.toLowerCase().includes('normal');
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => handleSampleChange(s.id)}
                className={`p-2.5 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                  active
                    ? 'bg-pacs-800 border-clinical-teal shadow-md ring-1 ring-clinical-teal/40'
                    : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                }`}
              >
                <span
                  className={`inline-block w-max text-[9px] font-bold uppercase px-1.5 py-0.5 rounded mb-1 ${
                    isNormalBeat
                      ? 'bg-clinical-green/20 text-clinical-green'
                      : 'bg-amber-500/20 text-amber-300'
                  }`}
                >
                  {s.source.includes('MIT-BIH') ? 'MIT-BIH Real' : isNormalBeat ? 'Normal' : 'Arrhythmia'}
                </span>
                <div className="text-[11px] font-bold text-white leading-tight line-clamp-2">{s.title}</div>
              </button>
            );
          })}

          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="p-2.5 rounded-xl border border-dashed border-pacs-600 bg-pacs-950/50 hover:border-clinical-teal text-pacs-300 hover:text-white text-xs font-bold flex flex-col items-center justify-center gap-1 transition cursor-pointer"
          >
            <Upload className="w-4 h-4 text-clinical-teal" />
            <span className="text-[11px]">Upload CSV</span>
          </button>
          <input type="file" ref={fileRef} onChange={handleCsvUpload} accept=".csv,.txt" className="hidden" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Interactive Waveform Monitor + Vital Cards (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl overflow-hidden shadow-2xl">
            {/* Waveform Top Bar */}
            <div className="bg-pacs-950 px-4 py-3 border-b border-pacs-700 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setLiveSweep(!liveSweep)}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-lg font-bold transition cursor-pointer ${
                    liveSweep
                      ? 'bg-clinical-teal text-slate-950'
                      : 'bg-pacs-800 text-pacs-300 border border-pacs-700 hover:text-white'
                  }`}
                >
                  {liveSweep ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  <span>{liveSweep ? 'Live Sweep: ON' : 'Paused'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => setShowReconstruction(!showReconstruction)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${
                    showReconstruction
                      ? 'bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/40'
                      : 'bg-pacs-800 text-pacs-300 border border-pacs-700'
                  }`}
                >
                  Show Expected Normal Wave
                </button>
                <button
                  type="button"
                  onClick={() => setShowResiduals(!showResiduals)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${
                    showResiduals
                      ? 'bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/40'
                      : 'bg-pacs-800 text-pacs-300 border border-pacs-700'
                  }`}
                >
                  Highlight Rhythm Difference
                </button>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[11px] text-pacs-400">Classifier:</span>
                <select
                  aria-label="ECG Classifier Architecture"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="bg-pacs-800 text-clinical-teal border border-pacs-700 rounded-lg px-2.5 py-1 text-[11px] font-semibold"
                >
                  <option value="resnet1d">Lead-II 5-Class ResNet1D + Autoencoder</option>
                  {availableModels.includes('mitbih_resnet1d') && (
                    <option value="mitbih_resnet1d">MIT-BIH Hospital ResNet1D + Autoencoder</option>
                  )}
                  <option value="autoencoder">Autoencoder Anomaly Detector Only</option>
                </select>
              </div>
            </div>

            {/* Interactive Canvas with Hover Scrubbing */}
            <div className="p-4 bg-slate-950 relative">
              <canvas
                ref={canvasRef}
                width={720}
                height={290}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoverIndex(null)}
                className="w-full h-auto rounded-xl border border-slate-800 cursor-crosshair"
              />
            </div>

            {/* Waveform Legend & Live Cursor Readout */}
            <div className="bg-pacs-950 px-4 py-2.5 border-t border-pacs-700 flex flex-wrap items-center justify-between gap-2 text-xs text-pacs-300">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span
                    className={`w-3.5 h-1 rounded inline-block ${
                      screeningResult?.is_anomaly ? 'bg-clinical-rose' : 'bg-clinical-green'
                    }`}
                  />
                  <span>Patient Heartbeat Wave</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3.5 h-0.5 bg-clinical-cyan border-t border-dashed inline-block" />
                  <span>AI Expected Normal Wave</span>
                </span>
              </div>
              <div className="flex items-center gap-3 font-mono text-[11px]">
                <span className="text-clinical-teal">
                  Time: <strong>{inspectedTimeMs} ms</strong> | Voltage: <strong>{inspectedMv.toFixed(2)} mV</strong>
                  {showReconstruction && ` (Expected: ${inspectedReconMv.toFixed(2)} mV)`}
                </span>
                <span className="flex items-center gap-1 text-pacs-400">
                  <BarChart2 className="w-3.5 h-3.5" />
                  {loading ? 'Analyzing...' : `Diff Score: ${screeningResult?.reconstruction_mse.toFixed(4) || '--'}`}
                </span>
              </div>
            </div>
          </div>

          {/* 4 Clear, Reference-Guided Vital Sign Cards */}
          {screeningResult && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3.5 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-pacs-300 font-bold flex items-center gap-1">
                    <Heart className="w-3.5 h-3.5 text-clinical-rose" /> Heart Rate
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      hrOk ? 'bg-clinical-green/20 text-clinical-green' : 'bg-amber-500/20 text-amber-300'
                    }`}
                  >
                    {hrOk ? 'Normal' : 'Watch'}
                  </span>
                </div>
                <div className="text-2xl font-extrabold text-white">
                  {screeningResult.telemetry.heart_rate_bpm}{' '}
                  <span className="text-xs font-normal text-pacs-300">BPM</span>
                </div>
                <div className="text-[11px] text-pacs-400">Healthy range: 60–100 BPM</div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3.5 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-pacs-300 font-bold flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5 text-amber-400" /> Beat Width (QRS)
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      qrsOk ? 'bg-clinical-green/20 text-clinical-green' : 'bg-clinical-rose/20 text-clinical-rose'
                    }`}
                  >
                    {qrsOk ? 'Normal' : 'Wide'}
                  </span>
                </div>
                <div className="text-2xl font-extrabold text-white">
                  {screeningResult.telemetry.qrs_duration_ms}{' '}
                  <span className="text-xs font-normal text-pacs-300">ms</span>
                </div>
                <div className="text-[11px] text-pacs-400">Healthy range: &lt;120 ms</div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3.5 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-pacs-300 font-bold flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-clinical-teal" /> Recharge (QTc)
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      qtcOk ? 'bg-clinical-green/20 text-clinical-green' : 'bg-clinical-rose/20 text-clinical-rose'
                    }`}
                  >
                    {qtcOk ? 'Normal' : 'Prolonged'}
                  </span>
                </div>
                <div className="text-2xl font-extrabold text-white">
                  {screeningResult.telemetry.qtc_interval_ms}{' '}
                  <span className="text-xs font-normal text-pacs-300">ms</span>
                </div>
                <div className="text-[11px] text-pacs-400">Healthy range: &lt;450 ms</div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3.5 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-pacs-300 font-bold flex items-center gap-1">
                    <Activity className="w-3.5 h-3.5 text-clinical-cyan" /> ST Wave Shift
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      stOk ? 'bg-clinical-green/20 text-clinical-green' : 'bg-clinical-rose/20 text-clinical-rose'
                    }`}
                  >
                    {stOk ? 'Normal' : 'Elevated'}
                  </span>
                </div>
                <div className="text-2xl font-extrabold text-white">
                  {screeningResult.telemetry.st_segment_mv.toFixed(2)}{' '}
                  <span className="text-xs font-normal text-pacs-300">mV</span>
                </div>
                <div className="text-[11px] text-pacs-400">Healthy: -0.25 to +0.25 mV</div>
              </div>
            </div>
          )}
        </div>

        {/* Right Plain-English Rhythm Explanation & Differential (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300">
                {explainMode === 'simple' ? 'HEARTBEAT ANALYSIS EXPLAINED' : '5-CLASS RHYTHM DIFFERENTIAL'}
              </span>
              <span className="text-[11px] font-mono text-clinical-teal">18 BIOMARKERS</span>
            </div>

            {screeningResult && (
              <>
                {/* Plain-English Interpretation Box */}
                <div
                  className={`p-3.5 rounded-xl border space-y-1.5 ${
                    screeningResult.is_anomaly
                      ? 'bg-clinical-rose/10 border-clinical-rose/40'
                      : 'bg-clinical-green/10 border-clinical-green/40'
                  }`}
                >
                  <div className="text-xs font-bold text-white flex items-center gap-1.5">
                    <Info className="w-4 h-4 text-clinical-cyan shrink-0" />
                    <span>{screeningResult.rhythm_subtype || screeningResult.diagnosis}</span>
                  </div>
                  <p className="text-xs text-pacs-200 leading-relaxed">
                    {RHYTHM_PLAIN_GUIDE[screeningResult.rhythm_subtype || ''] ||
                      (screeningResult.is_anomaly
                        ? 'The heartbeat wave shape differs from a standard sinus beat, suggesting an electrical conduction or rhythm variation.'
                        : 'The heartbeat wave closely matches the expected normal sinus template.')}
                  </p>
                  {clinicalSummary && (
                    <div className="pt-1.5 mt-1.5 border-t border-pacs-700/60 text-[11px] text-clinical-cyan font-semibold">
                      Recommended Action: {clinicalSummary.recommended_action}
                    </div>
                  )}
                </div>

                {/* 5-Class Rhythm Likelihood Bars */}
                <div className="space-y-2">
                  <div className="text-xs font-bold text-pacs-200">AI Rhythm Match Breakdown:</div>
                  {Object.entries(screeningResult.subtype_probabilities || {}).map(([clsName, p]) => (
                    <div key={clsName} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-pacs-100 font-medium">{clsName}</span>
                        <span className="text-clinical-cyan font-bold font-mono">{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-pacs-950 rounded-full overflow-hidden border border-pacs-800">
                        <div
                          style={{ width: `${Math.min(100, p * 100)}%` }}
                          className="h-full bg-gradient-to-r from-clinical-teal to-clinical-cyan"
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Friendly Conduction Checklist */}
                <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-2 text-xs">
                  <span className="text-white font-bold block">Electrical Conduction Safety Checks:</span>
                  {[
                    {
                      key: 'wide_qrs',
                      flagged: screeningResult.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay,
                      okText: 'Normal ventricular conduction speed (QRS ≤ 120 ms)',
                      flagText: 'Delayed ventricular impulse (Wide QRS > 120 ms)',
                    },
                    {
                      key: 'qtc',
                      flagged: screeningResult.telemetry.clinical_flags.prolonged_repolarization_qtc,
                      okText: 'Healthy electrical recovery time (QTc ≤ 450 ms)',
                      flagText: 'Prolonged recharge interval (QTc > 450 ms)',
                    },
                    {
                      key: 'st',
                      flagged: screeningResult.telemetry.clinical_flags.st_segment_abnormality_ischemia,
                      okText: 'Baseline ST wave (no acute ischemic shift)',
                      flagText: 'Significant ST-segment elevation or depression',
                    },
                    {
                      key: 'afib',
                      flagged: Boolean(screeningResult.telemetry.clinical_flags.atrial_fibrillation_absent_p_wave),
                      okText: 'Coordinated atrial P-wave activity present',
                      flagText: 'Low P-wave with high-frequency atrial fibrillation pattern',
                    },
                  ].map((item) => (
                    <div key={item.key} className="flex items-start gap-2 text-[11px]">
                      {item.flagged ? (
                        <AlertOctagon className="w-4 h-4 text-clinical-rose shrink-0 mt-0.5" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4 text-clinical-green shrink-0 mt-0.5" />
                      )}
                      <span className={item.flagged ? 'text-clinical-rose font-semibold' : 'text-pacs-200'}>
                        {item.flagged ? item.flagText : item.okText}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Pro Threshold Slider */}
                {explainMode === 'pro' && (
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3.5 space-y-2 text-xs">
                    <div className="flex justify-between font-mono">
                      <span className="text-pacs-300">Autoencoder MSE:</span>
                      <span
                        className={
                          screeningResult.is_anomaly ? 'text-clinical-rose font-bold' : 'text-clinical-green font-bold'
                        }
                      >
                        {screeningResult.reconstruction_mse.toFixed(5)}
                      </span>
                    </div>
                    <input
                      type="range"
                      aria-label="Anomaly detection threshold"
                      min="0.005"
                      max="0.060"
                      step="0.001"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-pacs-400 font-mono">
                      <span>Strict (0.005)</span>
                      <span className="text-amber-300 font-bold">Threshold = {threshold.toFixed(4)}</span>
                      <span>Relaxed (0.060)</span>
                    </div>
                  </div>
                )}

                {onNextStep && (
                  <button
                    type="button"
                    onClick={onNextStep}
                    className="w-full py-2.5 px-3 rounded-xl bg-gradient-to-r from-clinical-teal to-clinical-cyan text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 shadow-md transition cursor-pointer"
                  >
                    <span>Continue to Step 3: Heart Risk &amp; Heart Age</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

