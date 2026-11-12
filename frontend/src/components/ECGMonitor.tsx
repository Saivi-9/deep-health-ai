import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertOctagon, Heart, Zap, Clock, ShieldCheck, BarChart2, Upload } from 'lucide-react';
import type { ECGScreenResponse, ECGSample } from '../types';
import { screenECGSignal } from '../api/client';

interface ECGMonitorProps {
  samples: ECGSample[];
  availableModels: string[];
  onEcgUpdate?: (res: ECGScreenResponse) => void;
}

const EMPTY_SIGNAL: number[] = [];

export const ECGMonitor: React.FC<ECGMonitorProps> = ({ samples, availableModels, onEcgUpdate }) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [modelId, setModelId] = useState<string>('resnet1d');
  const [threshold, setThreshold] = useState<number>(0.01037);
  const [customSignal, setCustomSignal] = useState<number[] | null>(null);
  const [customRate, setCustomRate] = useState<number | null>(null);
  const [screeningResult, setScreeningResult] = useState<ECGScreenResponse | null>(null);
  const [showReconstruction, setShowReconstruction] = useState<boolean>(true);
  const [showResiduals, setShowResiduals] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const activeSample = samples.find((s) => s.id === selectedSampleId) || samples.find((s) => s.id === 'leadii_1') || samples[0];
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
    return () => {
      active = false;
    };
  }, [currentSignal, samplingRate, threshold, modelId, onEcgUpdate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || currentSignal.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = '#060A12';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#121C2F';
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

    ctx.strokeStyle = '#1E293B';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Real Autoencoder Residual Area
    if (showResiduals && recon.length > 0 && screeningResult?.is_anomaly) {
      const nPlot = Math.min(currentSignal.length, recon.length);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.20)';
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

    // Real Reconstructed Waveform from Conv1dAutoencoder
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
    ctx.strokeStyle = screeningResult?.is_anomaly ? '#F43F5E' : '#22C55E';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    for (let i = 0; i < currentSignal.length; i++) {
      const x = i * stepX;
      const y = midY - currentSignal[i] * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [currentSignal, screeningResult, showReconstruction, showResiduals]);

  const handleSampleChange = (id: string) => {
    setSelectedSampleId(id);
    setCustomSignal(null);
    setCustomRate(null);
    setLoading(true);
    if (id.startsWith('mitbih_')) setModelId('mitbih_resnet1d');
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

  return (
    <div id="panel-ecg" role="tabpanel" aria-labelledby="tab-ecg" className="space-y-6">
      <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-pacs-800 border border-pacs-600 flex items-center justify-center">
            <Heart className="w-5 h-5 text-clinical-rose" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">ELECTROPHYSIOLOGY TELEMETRY &amp; 5-CLASS ECG LAB</div>
            <div className="text-pacs-300 font-mono">
              MODEL: {screeningResult?.model_used || modelId} | RATE: {samplingRate} Hz
            </div>
          </div>
        </div>
        {screeningResult && (
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-pacs-800 border border-pacs-600 text-clinical-cyan">
              SUBTYPE: {screeningResult.rhythm_subtype}
            </span>
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 ${
                screeningResult.is_anomaly
                  ? 'bg-clinical-rose/20 text-clinical-rose border-clinical-rose/40'
                  : 'bg-clinical-green/20 text-clinical-green border-clinical-green/40'
              }`}
            >
              {screeningResult.is_anomaly ? <AlertOctagon className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
              {screeningResult.is_anomaly ? 'ARRHYTHMIA / ECTOPY' : 'NORMAL SINUS'}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl overflow-hidden shadow-2xl">
            <div className="bg-pacs-950 px-4 py-2.5 border-b border-pacs-700 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <label className="text-pacs-400 font-mono" htmlFor="ecg-sample-select">STREAM:</label>
                <select
                  id="ecg-sample-select"
                  value={effectiveSampleId}
                  onChange={(e) => handleSampleChange(e.target.value)}
                  className="bg-pacs-800 text-slate-200 border border-pacs-600 rounded-lg px-2.5 py-1 text-xs outline-none"
                >
                  {samples.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.title}
                    </option>
                  ))}
                  {selectedSampleId === 'custom_ecg' && <option value="custom_ecg">Uploaded ECG Array</option>}
                </select>

                <input type="file" ref={fileRef} onChange={handleCsvUpload} accept=".csv,.txt" className="hidden" />
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 text-xs transition"
                >
                  <Upload className="w-3.5 h-3.5 text-clinical-teal" /> Upload CSV
                </button>
              </div>

              <div className="flex items-center gap-2">
                <select
                  aria-label="ECG Classifier Architecture"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="bg-pacs-800 text-clinical-teal border border-pacs-700 rounded-lg px-2 py-1 text-[11px] font-mono"
                >
                  <option value="resnet1d">Conv1D-AE + Lead-II ResNet1D</option>
                  {availableModels.includes('mitbih_resnet1d') && <option value="mitbih_resnet1d">Conv1D-AE + MIT-BIH ResNet1D</option>}
                  <option value="autoencoder">Conv1D-AE Only</option>
                </select>
                <button
                  type="button"
                  onClick={() => setShowReconstruction(!showReconstruction)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition ${
                    showReconstruction ? 'bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/40' : 'bg-pacs-800 text-slate-400 border border-pacs-700'
                  }`}
                >
                  Real AE Manifold
                </button>
                <button
                  type="button"
                  onClick={() => setShowResiduals(!showResiduals)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition ${
                    showResiduals ? 'bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/40' : 'bg-pacs-800 text-slate-400 border border-pacs-700'
                  }`}
                >
                  Residual Band
                </button>
              </div>
            </div>

            <div className="p-4 bg-pacs-950">
              <canvas ref={canvasRef} width={720} height={300} className="w-full h-auto rounded-xl border border-pacs-700/80" />
            </div>

            <div className="bg-pacs-950 px-4 py-2.5 border-t border-pacs-700 flex flex-wrap items-center justify-between text-xs text-pacs-300 font-mono">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-clinical-rose inline-block" /> Live Signal
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-clinical-cyan border-t border-dashed inline-block" /> Real Autoencoder x&#770;
                </span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart2 className="w-3.5 h-3.5 text-pacs-400" />
                <span>{loading ? 'Screening...' : `MSE: ${screeningResult?.reconstruction_mse.toFixed(5) || '--'}`}</span>
              </div>
            </div>
          </div>

          {screeningResult && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold flex items-center gap-1"><Heart className="w-3.5 h-3.5 text-clinical-rose" /> Heart Rate</div>
                <div className="text-xl font-extrabold text-white mt-1">{screeningResult.telemetry.heart_rate_bpm} <span className="text-xs font-normal text-slate-400">BPM</span></div>
                <div className="text-[11px] text-pacs-300 font-mono">Dom Freq: {screeningResult.telemetry.dominant_freq_hz ?? 0} Hz</div>
              </div>
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold flex items-center gap-1"><Zap className="w-3.5 h-3.5 text-amber-400" /> QRS / R-S</div>
                <div className="text-xl font-extrabold text-white mt-1">{screeningResult.telemetry.qrs_duration_ms} <span className="text-xs font-normal text-slate-400">ms</span></div>
                <div className="text-[11px] text-pacs-300 font-mono">R/S Ratio: {screeningResult.telemetry.r_s_ratio ?? '--'}</div>
              </div>
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold flex items-center gap-1"><Clock className="w-3.5 h-3.5 text-clinical-teal" /> QTc / PR</div>
                <div className="text-xl font-extrabold text-white mt-1">{screeningResult.telemetry.qtc_interval_ms} <span className="text-xs font-normal text-slate-400">ms</span></div>
                <div className="text-[11px] text-pacs-300 font-mono">PR: {screeningResult.telemetry.pr_interval_ms} ms</div>
              </div>
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold flex items-center gap-1"><Activity className="w-3.5 h-3.5 text-clinical-cyan" /> ST / Spectral</div>
                <div className="text-xl font-extrabold text-white mt-1">{screeningResult.telemetry.st_segment_mv.toFixed(2)} <span className="text-xs font-normal text-slate-400">mV</span></div>
                <div className="text-[11px] text-pacs-300 font-mono">Spec Entropy: {screeningResult.telemetry.spectral_entropy ?? '--'}</div>
              </div>
            </div>
          )}
        </div>

        {/* Right Differential & Threshold Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300">5-CLASS RHYTHM DIFFERENTIAL</span>
              <span className="text-[11px] font-mono text-clinical-teal">18 BIOMARKERS</span>
            </div>

            {screeningResult && (
              <>
                <div className="space-y-2">
                  {Object.entries(screeningResult.subtype_probabilities || {}).map(([clsName, p]) => (
                    <div key={clsName} className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-200">{clsName}</span>
                        <span className="text-clinical-cyan font-bold">{(p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-pacs-950 rounded-full overflow-hidden border border-pacs-800">
                        <div style={{ width: `${Math.min(100, p * 100)}%` }} className="h-full bg-gradient-to-r from-clinical-teal to-clinical-cyan" />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3.5 space-y-2 text-xs">
                  <div className="flex justify-between font-mono">
                    <span className="text-pacs-300">Autoencoder MSE:</span>
                    <span className={screeningResult.is_anomaly ? 'text-clinical-rose font-bold' : 'text-clinical-green font-bold'}>
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
                    <span>Sensitive (0.005)</span>
                    <span className="text-amber-300 font-bold">&theta; = {threshold.toFixed(4)}</span>
                    <span>Specific (0.060)</span>
                  </div>
                </div>

                <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-1.5 text-xs">
                  <span className="text-slate-300 font-bold block mb-1">Biomarker Flags:</span>
                  {Object.entries(screeningResult.telemetry.clinical_flags).map(([k, flagged]) => (
                    <div key={k} className="flex items-center gap-2 font-mono text-[11px]">
                      <span className={`w-2 h-2 rounded-full ${flagged ? 'bg-clinical-rose' : 'bg-slate-600'}`} />
                      <span className={flagged ? 'text-clinical-rose font-bold' : 'text-pacs-400'}>
                        {k.replace(/_/g, ' ')}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
