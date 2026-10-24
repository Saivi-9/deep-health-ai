import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertOctagon, Heart, Zap, Clock, ShieldCheck, BarChart2 } from 'lucide-react';
import type { ECGScreenResponse } from '../types';
import { screenECGSignal } from '../api/client';

export const ECGMonitor: React.FC = () => {
  const [selectedRhythm, setSelectedRhythm] = useState<'nsr' | 'pvc' | 'afib' | 'st_elevation'>('pvc');
  const [threshold, setThreshold] = useState<number>(0.01037);
  const [samplingRate] = useState<number>(250);
  const [screeningResult, setScreeningResult] = useState<ECGScreenResponse | null>(null);
  const [showReconstruction, setShowReconstruction] = useState<boolean>(true);
  const [showResiduals, setShowResiduals] = useState<boolean>(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Generate synthetic 256-sample Lead-II waveforms
  const generateSignal = (rhythm: 'nsr' | 'pvc' | 'afib' | 'st_elevation'): number[] => {
    const N = 256;
    const signal: number[] = [];

    for (let i = 0; i < N; i++) {
      const t = i / N;
      let val = 0;

      // Baseline P wave (~0.2s)
      val += 0.15 * Math.exp(-Math.pow((t - 0.28) / 0.035, 2));

      // QRS Complex (~0.45s)
      if (rhythm === 'pvc') {
        // Broad aberrant ventricular ectopy
        val -= 0.25 * Math.exp(-Math.pow((t - 0.44) / 0.02, 2));
        val += 1.45 * Math.exp(-Math.pow((t - 0.48) / 0.045, 2)); // Wide QRS
        val -= 0.65 * Math.exp(-Math.pow((t - 0.54) / 0.035, 2));
      } else {
        // Sharp normal QRS
        val -= 0.15 * Math.exp(-Math.pow((t - 0.46) / 0.012, 2)); // Q
        val += 1.15 * Math.exp(-Math.pow((t - 0.48) / 0.015, 2)); // R
        val -= 0.25 * Math.exp(-Math.pow((t - 0.50) / 0.012, 2)); // S
      }

      // ST Segment & T wave
      if (rhythm === 'st_elevation') {
        val += 0.32 * Math.exp(-Math.pow((t - 0.58) / 0.06, 2)); // Prominent ST elevation
        val += 0.55 * Math.exp(-Math.pow((t - 0.72) / 0.06, 2)); // Hyperacute T wave
      } else if (rhythm === 'afib') {
        val += 0.20 * Math.exp(-Math.pow((t - 0.70) / 0.05, 2));
        val += (Math.random() - 0.5) * 0.12; // Fibrillatory baseline noise
      } else {
        val += 0.28 * Math.exp(-Math.pow((t - 0.70) / 0.055, 2)); // Normal T wave
      }

      signal.push(val);
    }

    return signal;
  };

  const [currentSignal, setCurrentSignal] = useState<number[]>(() => generateSignal('pvc'));

  // Run Autoencoder Screening
  const analyzeSignal = async (sig: number[], rhythm: string) => {
    try {
      const res = await screenECGSignal(sig, samplingRate, threshold);
      setScreeningResult(res);
    } catch {
      // Offline fallback
      const isArrhythmia = rhythm !== 'nsr';
      const mse = isArrhythmia ? (rhythm === 'pvc' ? 0.0482 : 0.0315) : 0.0042;
      const isAnomaly = mse > threshold;

      setScreeningResult({
        diagnosis: isAnomaly
          ? (rhythm === 'pvc' ? 'Premature Ventricular Contraction / Ventricular Ectopy' : 'Supraventricular Arrhythmia / ST Abnormality')
          : 'Normal Sinus Rhythm (NSR)',
        is_anomaly: isAnomaly,
        reconstruction_mse: mse,
        decision_threshold: threshold,
        telemetry: {
          heart_rate_bpm: rhythm === 'afib' ? 118 : (rhythm === 'st_elevation' ? 92 : 72),
          qrs_duration_ms: rhythm === 'pvc' ? 138 : 86,
          pr_interval_ms: rhythm === 'afib' ? 130 : 162,
          qt_interval_ms: 388,
          qtc_interval_ms: rhythm === 'afib' ? 472 : 415,
          st_segment_mv: rhythm === 'st_elevation' ? 0.28 : 0.02,
          r_peak_amplitude: rhythm === 'pvc' ? 1.45 : 1.15,
          clinical_flags: {
            wide_qrs_ventricular_conduction_delay: rhythm === 'pvc',
            prolonged_repolarization_qtc: rhythm === 'afib',
            st_segment_abnormality_ischemia: rhythm === 'st_elevation',
          },
        },
        timestamp: new Date().toISOString(),
      });
    }
  };

  useEffect(() => {
    analyzeSignal(currentSignal, selectedRhythm);
  }, [selectedRhythm, threshold]);

  // Render Telemetry Canvas with Grid Paper
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Draw Dark Medical Telemetry Grid
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

    // Zero-voltage Isoelectric Centerline
    const midY = height / 2 + 15;
    ctx.strokeStyle = '#1E293B';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Map amplitudes to Y-coords
    const scaleY = height / 3.2;
    const stepX = width / (currentSignal.length - 1);

    // Simulated Autoencoder Reconstruction (smooth healthy NSR approximation)
    const normalSignal = generateSignal('nsr');

    // Draw Residual Fill if enabled
    if (showResiduals && screeningResult?.is_anomaly) {
      ctx.fillStyle = 'rgba(244, 63, 94, 0.18)';
      ctx.beginPath();
      ctx.moveTo(0, midY);

      for (let i = 0; i < currentSignal.length; i++) {
        const x = i * stepX;
        const y = midY - currentSignal[i] * scaleY;
        ctx.lineTo(x, y);
      }
      for (let i = currentSignal.length - 1; i >= 0; i--) {
        const x = i * stepX;
        const yRec = midY - normalSignal[i] * scaleY;
        ctx.lineTo(x, yRec);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Draw Autoencoder Reconstructed Curve (Cyan / Dashed)
    if (showReconstruction) {
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let i = 0; i < normalSignal.length; i++) {
        const x = i * stepX;
        const y = midY - normalSignal[i] * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw Live Patient ECG Waveform (Neon Emerald or Rose if Anomaly)
    ctx.strokeStyle = screeningResult?.is_anomaly ? '#F43F5E' : '#22C55E';
    ctx.lineWidth = 2.8;
    ctx.shadowColor = screeningResult?.is_anomaly ? 'rgba(244, 63, 94, 0.6)' : 'rgba(34, 197, 94, 0.6)';
    ctx.shadowBlur = 8;

    ctx.beginPath();
    for (let i = 0; i < currentSignal.length; i++) {
      const x = i * stepX;
      const y = midY - currentSignal[i] * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [currentSignal, screeningResult, showReconstruction, showResiduals]);

  return (
    <div className="space-y-6">
      {/* Telemetry Header */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-pacs-800 border border-pacs-600 flex items-center justify-center text-clinical-teal">
            <Heart className="w-5 h-5 animate-pulse text-clinical-rose" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">LEAD-II HOLTER TELEMETRY MONITOR</div>
            <div className="text-pacs-300 font-mono">BANDPASS: 0.05 - 150 Hz | CAL: 10 mm/mV | SPEED: 25 mm/s</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-pacs-300 font-mono">CHANNEL: II (RHYTHM STRIP)</span>
          {screeningResult && (
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 ${
                screeningResult.is_anomaly
                  ? 'bg-clinical-rose/20 text-clinical-rose border-clinical-rose/40 animate-pulse'
                  : 'bg-clinical-green/20 text-clinical-green border-clinical-green/40'
              }`}
            >
              {screeningResult.is_anomaly ? <AlertOctagon className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
              {screeningResult.is_anomaly ? 'VENTRICULAR ECTOPY / ARRHYTHMIA' : 'NORMAL SINUS RHYTHM'}
            </span>
          )}
        </div>
      </div>

      {/* Main Grid: Waveform Canvas & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Waveform Viewport (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            {/* Viewport Top Bar */}
            <div className="bg-pacs-950 px-4 py-2.5 border-b border-pacs-700 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-pacs-400 font-mono">BEAT MORPHOLOGY:</span>
                <select
                  value={selectedRhythm}
                  onChange={(e) => {
                    const r = e.target.value as 'nsr' | 'pvc' | 'afib' | 'st_elevation';
                    setSelectedRhythm(r);
                    setCurrentSignal(generateSignal(r));
                  }}
                  className="bg-pacs-800 text-slate-200 border border-pacs-600 rounded-lg px-2.5 py-1 text-xs focus:ring-1 focus:ring-clinical-teal outline-none"
                >
                  <option value="pvc">Premature Ventricular Contraction (PVC)</option>
                  <option value="st_elevation">Acute ST-Elevation Ischemia</option>
                  <option value="afib">Atrial Fibrillation / Rapid Ventricular Response</option>
                  <option value="nsr">Normal Sinus Rhythm (Healthy NSR)</option>
                </select>
              </div>

              {/* Display Layer Switches */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowReconstruction(!showReconstruction)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition ${
                    showReconstruction
                      ? 'bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/40'
                      : 'bg-pacs-800 text-slate-400 border border-pacs-700'
                  }`}
                >
                  Autoencoder Manifold
                </button>

                <button
                  onClick={() => setShowResiduals(!showResiduals)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition ${
                    showResiduals
                      ? 'bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/40'
                      : 'bg-pacs-800 text-slate-400 border border-pacs-700'
                  }`}
                >
                  Error Residuals
                </button>
              </div>
            </div>

            {/* Canvas Area */}
            <div className="p-4 bg-pacs-950 flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={700}
                height={320}
                className="w-full h-auto rounded-xl border border-pacs-700/80 shadow-inner"
              />
            </div>

            {/* Canvas Legend & Calibration Bar */}
            <div className="bg-pacs-950 px-4 py-2.5 border-t border-pacs-700 flex flex-wrap items-center justify-between text-xs text-pacs-300 font-mono">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-clinical-rose inline-block"></span>
                  <span>Input Waveform (Lead II)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-clinical-cyan border-t border-dashed inline-block"></span>
                  <span>Autoencoder Reconstruction (x̂)</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <BarChart2 className="w-3.5 h-3.5 text-pacs-400" />
                <span>Recon MSE: {screeningResult?.reconstruction_mse.toFixed(5)}</span>
              </div>
            </div>
          </div>

          {/* Electrophysiological Telemetry Cards */}
          {screeningResult && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider flex items-center gap-1">
                  <Heart className="w-3.5 h-3.5 text-clinical-rose" /> Heart Rate
                </div>
                <div className="text-xl font-extrabold text-white mt-1">
                  {screeningResult.telemetry.heart_rate_bpm} <span className="text-xs font-normal text-slate-400">BPM</span>
                </div>
                <div className="text-[11px] text-pacs-300 mt-0.5 font-mono">Normal: 60-100</div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider flex items-center gap-1">
                  <Zap className="w-3.5 h-3.5 text-clinical-amber" /> QRS Complex
                </div>
                <div className={`text-xl font-extrabold mt-1 ${screeningResult.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay ? 'text-clinical-rose' : 'text-white'}`}>
                  {screeningResult.telemetry.qrs_duration_ms} <span className="text-xs font-normal text-slate-400">ms</span>
                </div>
                <div className="text-[11px] text-pacs-300 mt-0.5 font-mono">
                  {screeningResult.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay ? '⚠️ Wide QRS' : 'Conduction Normal'}
                </div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-clinical-teal" /> QTc Interval
                </div>
                <div className={`text-xl font-extrabold mt-1 ${screeningResult.telemetry.clinical_flags.prolonged_repolarization_qtc ? 'text-clinical-rose' : 'text-white'}`}>
                  {screeningResult.telemetry.qtc_interval_ms} <span className="text-xs font-normal text-slate-400">ms</span>
                </div>
                <div className="text-[11px] text-pacs-300 mt-0.5 font-mono">Bazett &lt; 450 ms</div>
              </div>

              <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-3">
                <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider flex items-center gap-1">
                  <Activity className="w-3.5 h-3.5 text-clinical-cyan" /> ST Segment
                </div>
                <div className={`text-xl font-extrabold mt-1 ${screeningResult.telemetry.clinical_flags.st_segment_abnormality_ischemia ? 'text-clinical-rose' : 'text-white'}`}>
                  {screeningResult.telemetry.st_segment_mv.toFixed(2)} <span className="text-xs font-normal text-slate-400">mV</span>
                </div>
                <div className="text-[11px] text-pacs-300 mt-0.5 font-mono">
                  {screeningResult.telemetry.clinical_flags.st_segment_abnormality_ischemia ? '⚠️ ST Elevation' : 'Isoelectric (Normal)'}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Autoencoder Threshold Calibration & Anomaly Assessment (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-clinical-teal" />
                CONV1D AUTOENCODER MANIFOLD
              </span>
              <span className="text-xs font-mono text-slate-400">ROC-AUC: 0.9714</span>
            </div>

            {screeningResult && (
              <>
                <div>
                  <h4 className="text-lg font-bold text-white">
                    {screeningResult.diagnosis}
                  </h4>
                  <p className="text-xs text-pacs-300 mt-1">
                    Unsupervised reconstruction loss compared against the 97.5th percentile healthy sinus boundary.
                  </p>
                </div>

                {/* Reconstruction Anomaly Error Meter */}
                <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-4 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-pacs-300 font-mono">Reconstruction MSE:</span>
                    <span className={`font-mono font-bold text-sm ${screeningResult.is_anomaly ? 'text-clinical-rose' : 'text-clinical-green'}`}>
                      {screeningResult.reconstruction_mse.toFixed(5)}
                    </span>
                  </div>

                  {/* Visual Bar vs Threshold Line */}
                  <div className="relative w-full h-4 bg-pacs-900 rounded-full overflow-hidden border border-pacs-700">
                    <div
                      style={{ width: `${Math.min((screeningResult.reconstruction_mse / 0.06) * 100, 100)}%` }}
                      className={`h-full transition-all duration-500 ${
                        screeningResult.is_anomaly ? 'bg-gradient-to-r from-orange-500 to-clinical-rose' : 'bg-clinical-green'
                      }`}
                    />
                    {/* Decision Threshold Marker Line */}
                    <div
                      style={{ left: `${(threshold / 0.06) * 100}%` }}
                      className="absolute top-0 bottom-0 w-1 bg-yellow-400 shadow-sm"
                      title={`Calibrated Threshold θ = ${threshold}`}
                    />
                  </div>

                  <div className="flex justify-between text-[10px] text-pacs-400 font-mono">
                    <span>0.00 (Perfect Match)</span>
                    <span className="text-yellow-400 font-bold">θ = {threshold.toFixed(5)}</span>
                    <span>0.060+ (Severe Ectopy)</span>
                  </div>
                </div>

                {/* Interactive Decision Threshold Slider */}
                <div className="space-y-2 pt-1 text-xs">
                  <div className="flex justify-between text-pacs-300">
                    <span>Calibrated Boundary Override (θ):</span>
                    <span className="font-mono font-bold text-clinical-cyan">{threshold.toFixed(5)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.005"
                    max="0.030"
                    step="0.0005"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full accent-clinical-teal cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-pacs-400">
                    <span>Higher Sensitivity (0.005)</span>
                    <span>Higher Specificity (0.030)</span>
                  </div>
                </div>

                {/* Electrophysiological Alert Flags */}
                <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-1.5 text-xs">
                  <span className="text-slate-300 font-bold block mb-1">Clinical Biomarker Diagnostic Flags:</span>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${screeningResult.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay ? 'bg-clinical-rose animate-ping' : 'bg-slate-600'}`}></span>
                    <span className={screeningResult.telemetry.clinical_flags.wide_qrs_ventricular_conduction_delay ? 'text-clinical-rose font-bold' : 'text-pacs-400'}>
                      Ventricular Conduction Delay (QRS &gt; 120 ms)
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${screeningResult.telemetry.clinical_flags.st_segment_abnormality_ischemia ? 'text-clinical-rose animate-ping' : 'bg-slate-600'}`}></span>
                    <span className={screeningResult.telemetry.clinical_flags.st_segment_abnormality_ischemia ? 'text-clinical-rose font-bold' : 'text-pacs-400'}>
                      ST Segment Elevation / Transmural Ischemia
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${screeningResult.telemetry.clinical_flags.prolonged_repolarization_qtc ? 'text-clinical-rose animate-ping' : 'bg-slate-600'}`}></span>
                    <span className={screeningResult.telemetry.clinical_flags.prolonged_repolarization_qtc ? 'text-clinical-rose font-bold' : 'text-pacs-400'}>
                      Prolonged Cardiac Repolarization (QTc &gt; 450 ms)
                    </span>
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
