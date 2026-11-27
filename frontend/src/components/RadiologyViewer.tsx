import React, { useState, useRef, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Sliders,
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
  Download,
  FileText,
  Sparkles,
  Upload,
  Activity,
  Columns,
  Crosshair,
  HelpCircle,
  ArrowRight,
  Stethoscope,
} from 'lucide-react';
import type { VisionDiagnosisResponse, VisionSample, TriageUrgency, ExplainMode } from '../types';
import { diagnoseVisionScan } from '../api/client';

interface RadiologyViewerProps {
  samples: VisionSample[];
  availableModels: string[];
  onOpenReport: (result: VisionDiagnosisResponse, imageSrc: string) => void;
  onVisionUpdate?: (res: VisionDiagnosisResponse) => void;
  explainMode?: ExplainMode;
  onNextStep?: () => void;
}

function getAnatomicalLungZone(xNorm: number, yNorm: number): string {
  const side = xNorm < 0.44 ? 'Right Lung Field' : xNorm > 0.56 ? 'Left Lung Field' : 'Central Mediastinum / Heart Shadow';
  if (side.includes('Central')) return side;
  const zone = yNorm < 0.36 ? 'Upper Lobe (Apical)' : yNorm < 0.68 ? 'Middle Lung Zone' : 'Lower Lobe (Basilar)';
  return `${side} • ${zone}`;
}

export const RadiologyViewer: React.FC<RadiologyViewerProps> = ({
  samples,
  availableModels,
  onOpenReport,
  onVisionUpdate,
  explainMode = 'simple',
  onNextStep,
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [modelId, setModelId] = useState<string>('residual_cnn');
  const [colormap, setColormap] = useState<string>('turbo');
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [activeWindow, setActiveWindow] = useState<'standard' | 'lung' | 'bone' | 'invert'>('standard');
  const [showGradcam, setShowGradcam] = useState<boolean>(true);
  const [compareSideBySide, setCompareSideBySide] = useState<boolean>(false);
  const [gradcamOpacity, setGradcamOpacity] = useState<number>(0.62);
  const [mcDropoutSamples, setMcDropoutSamples] = useState<number>(20);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [diagnosisResult, setDiagnosisResult] = useState<VisionDiagnosisResponse | null>(null);
  const [customImageSrc, setCustomImageSrc] = useState<string>('');
  const [customImageTitle, setCustomImageTitle] = useState<string>('');
  const [inspectedPoint, setInspectedPoint] = useState<{ x: number; y: number; zone: string } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeSample = samples.find((s) => s.id === selectedSampleId) || samples[0];
  const effectiveSampleId = selectedSampleId === 'custom_upload' ? 'custom_upload' : (activeSample?.id || '');
  const currentImageSrc = selectedSampleId === 'custom_upload' ? customImageSrc : (activeSample?.image_base64 || '');
  const currentImageTitle = selectedSampleId === 'custom_upload' ? customImageTitle : (activeSample?.title || 'Loading cohort scan...');

  useEffect(() => {
    if (!currentImageSrc) return;
    let active = true;
    const fname = (effectiveSampleId || 'uploaded_cxr') + '.png';
    fetch(currentImageSrc)
      .then((r) => r.blob())
      .then((blob) => {
        const file = new File([blob], fname, { type: 'image/png' });
        return diagnoseVisionScan(file, mcDropoutSamples, modelId, colormap);
      })
      .then((res) => {
        if (!active) return;
        setErrorMsg(null);
        setDiagnosisResult(res);
        setAnalyzing(false);
        onVisionUpdate?.(res);
      })
      .catch((err) => {
        if (active) {
          setErrorMsg(err instanceof Error ? err.message : 'Vision diagnosis failed');
          setAnalyzing(false);
        }
      });
    return () => {
      active = false;
    };
  }, [currentImageSrc, effectiveSampleId, modelId, mcDropoutSamples, colormap, onVisionUpdate]);

  // Composite real Radiograph + real Backend Grad-CAM PNG heatmap onto canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentImageSrc) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const baseImg = new Image();
    baseImg.src = currentImageSrc;
    baseImg.onload = () => {
      canvas.width = 256;
      canvas.height = 256;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();

      if (activeWindow === 'lung') ctx.filter = 'contrast(165%) brightness(108%)';
      else if (activeWindow === 'bone') ctx.filter = 'contrast(210%) brightness(88%)';
      else if (activeWindow === 'invert') ctx.filter = 'invert(100%) contrast(135%)';
      else ctx.filter = 'none';

      ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const hmSrc = diagnosisResult?.gradcam_heatmap_base64;
      if (showGradcam && hmSrc) {
        const hmImg = new Image();
        hmImg.src = hmSrc;
        hmImg.onload = () => {
          ctx.save();
          ctx.globalAlpha = gradcamOpacity;
          ctx.drawImage(hmImg, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          // Draw interactive inspection crosshair if user clicked/hovered
          if (inspectedPoint) {
            const cx = inspectedPoint.x * canvas.width;
            const cy = inspectedPoint.y * canvas.height;
            ctx.save();
            ctx.strokeStyle = '#38BDF8';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx - 18, cy);
            ctx.lineTo(cx + 18, cy);
            ctx.moveTo(cx, cy - 18);
            ctx.lineTo(cx, cy + 18);
            ctx.stroke();
            ctx.restore();
          }
        };
      }
    };
  }, [currentImageSrc, activeWindow, showGradcam, gradcamOpacity, diagnosisResult, inspectedPoint]);

  const handleSampleChange = (id: string) => {
    setSelectedSampleId(id);
    setInspectedPoint(null);
    setAnalyzing(true);
  };

  const handleCustomUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (ev.target?.result) {
        setCustomImageTitle(file.name);
        setCustomImageSrc(ev.target.result as string);
        setSelectedSampleId('custom_upload');
        setInspectedPoint(null);
        setAnalyzing(true);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setInspectedPoint({ x, y, zone: getAnatomicalLungZone(x, y) });
  };

  const getUrgencyBadge = (urgency: TriageUrgency) => {
    const u = urgency.toUpperCase();
    if (u.includes('IMMEDIATE') || u.includes('STAT') || u.includes('CRITICAL')) {
      return (
        <span className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-bold bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/40">
          <AlertTriangle className="w-4 h-4" /> Priority Review: {urgency}
        </span>
      );
    }
    if (u.includes('URGENT') || u.includes('EXPEDITED') || u.includes('FOLLOW')) {
      return (
        <span className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
          <ShieldAlert className="w-4 h-4" /> Follow-Up: {urgency}
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-bold bg-clinical-green/20 text-clinical-green border border-clinical-green/40">
        <CheckCircle className="w-4 h-4" /> All Clear: {urgency}
      </span>
    );
  };

  const isPneumonia = (diagnosisResult?.pneumonia_probability || 0) >= 0.5;
  const hotspotZone = diagnosisResult?.hotspot
    ? getAnatomicalLungZone(diagnosisResult.hotspot.x_norm, diagnosisResult.hotspot.y_norm)
    : 'Lung Parenchyma';

  return (
    <div id="panel-radiology" role="tabpanel" aria-labelledby="tab-radiology" className="space-y-6">
      {/* Interactive Visual Case Picker Strip */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-4 shadow-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-clinical-cyan" />
            <span className="text-xs font-bold uppercase tracking-wider text-white">
              Choose a Patient Chest X-Ray to Inspect (Active: {currentImageTitle})
            </span>
          </div>
          <div>{diagnosisResult && getUrgencyBadge(diagnosisResult.severity.clinical_triage_urgency)}</div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2.5">
          {samples.map((s) => {
            const active = effectiveSampleId === s.id;
            const isNormalCase = s.ground_truth.toLowerCase().includes('normal');
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => handleSampleChange(s.id)}
                className={`group flex items-center gap-2.5 p-2 rounded-xl border text-left transition cursor-pointer ${
                  active
                    ? 'bg-pacs-800 border-clinical-cyan shadow-md ring-1 ring-clinical-cyan/40'
                    : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                }`}
              >
                <img
                  src={s.image_base64}
                  alt={s.title}
                  className="w-11 h-11 rounded-lg object-cover border border-pacs-600 shrink-0"
                />
                <div className="min-w-0">
                  <span
                    className={`inline-block text-[9px] font-bold uppercase px-1.5 py-0.5 rounded mb-0.5 ${
                      isNormalCase
                        ? 'bg-clinical-green/20 text-clinical-green'
                        : 'bg-clinical-rose/20 text-clinical-rose'
                    }`}
                  >
                    {isNormalCase ? 'Healthy Lung' : 'Pneumonia'}
                  </span>
                  <div className="text-[11px] font-semibold text-white truncate">{s.title}</div>
                </div>
              </button>
            );
          })}

          {/* Upload Your Own X-Ray Card */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className={`flex items-center justify-center gap-2 p-2.5 rounded-xl border border-dashed text-xs font-bold transition cursor-pointer ${
              selectedSampleId === 'custom_upload'
                ? 'bg-clinical-cyan/20 border-clinical-cyan text-clinical-cyan'
                : 'bg-pacs-950/50 border-pacs-600 text-pacs-300 hover:text-white hover:border-clinical-cyan'
            }`}
          >
            <Upload className="w-4 h-4 text-clinical-cyan shrink-0" />
            <span>Upload X-Ray</span>
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleCustomUpload}
            accept="image/png,image/jpeg"
            className="hidden"
          />
        </div>
      </div>

      {errorMsg && (
        <div className="bg-clinical-rose/15 border border-clinical-rose/40 text-clinical-rose px-4 py-2.5 rounded-xl text-xs font-mono">
          API Warning: {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Interactive X-Ray Viewport (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            {/* Top Viewer Toolbar */}
            <div className="bg-pacs-950 px-4 py-3 border-b border-pacs-700 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-pacs-400 font-semibold mr-1">Contrast Filter:</span>
                {(
                  [
                    { id: 'standard', label: 'Standard' },
                    { id: 'lung', label: 'Airways & Lung' },
                    { id: 'bone', label: 'Ribs & Bone' },
                    { id: 'invert', label: 'X-Ray Invert' },
                  ] as const
                ).map((w) => (
                  <button
                    key={w.id}
                    type="button"
                    onClick={() => setActiveWindow(w.id)}
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${
                      activeWindow === w.id
                        ? 'bg-clinical-cyan text-slate-950 shadow-sm'
                        : 'bg-pacs-900 text-pacs-300 hover:text-white border border-pacs-700'
                    }`}
                  >
                    {w.label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setCompareSideBySide(!compareSideBySide)}
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition cursor-pointer ${
                    compareSideBySide
                      ? 'bg-clinical-teal/20 text-clinical-teal border-clinical-teal/40'
                      : 'bg-pacs-900 text-pacs-300 border-pacs-700 hover:text-white'
                  }`}
                  title="Compare original X-Ray next to AI Heatmap"
                >
                  <Columns className="w-3.5 h-3.5" />
                  <span>Side-by-Side</span>
                </button>
                <button
                  type="button"
                  onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.2))}
                  className="p-1.5 rounded-lg bg-pacs-900 border border-pacs-700 hover:bg-pacs-800 text-pacs-300 hover:text-white cursor-pointer"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.7))}
                  className="p-1.5 rounded-lg bg-pacs-900 border border-pacs-700 hover:bg-pacs-800 text-pacs-300 hover:text-white cursor-pointer"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setZoomLevel(1.0);
                    setInspectedPoint(null);
                  }}
                  className="p-1.5 rounded-lg bg-pacs-900 border border-pacs-700 hover:bg-pacs-800 text-pacs-300 hover:text-white cursor-pointer"
                  title="Reset View"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Main X-Ray Canvas Stage */}
            <div className="relative min-h-[410px] bg-slate-950 pacs-grid flex flex-col items-center justify-center p-5 overflow-hidden">
              <div
                style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.15s ease-out' }}
                className="flex flex-wrap items-center justify-center gap-4"
              >
                {compareSideBySide && currentImageSrc && (
                  <div className="relative rounded-xl shadow-2xl border border-slate-700 overflow-hidden bg-black">
                    <img
                      src={currentImageSrc}
                      alt="Original Chest Radiograph"
                      className="block w-[260px] h-[260px] object-contain"
                    />
                    <span className="absolute bottom-2 left-2 bg-slate-900/90 text-slate-200 text-[10px] font-bold px-2 py-0.5 rounded">
                      1. Original X-Ray
                    </span>
                  </div>
                )}

                <div className="relative rounded-xl shadow-2xl border border-clinical-cyan/40 overflow-hidden bg-black">
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    title="Click anywhere on the lung scan to inspect that anatomical region"
                    className={`block object-contain cursor-crosshair ${
                      compareSideBySide ? 'w-[260px] h-[260px]' : 'w-[340px] h-[340px]'
                    }`}
                  />
                  {compareSideBySide && (
                    <span className="absolute bottom-2 left-2 bg-slate-900/90 text-clinical-cyan text-[10px] font-bold px-2 py-0.5 rounded">
                      2. AI Attention Heatmap
                    </span>
                  )}
                </div>
              </div>

              {/* Interactive Anatomical Hotspot & Click Inspector Badges */}
              {diagnosisResult?.hotspot && (
                <div className="hidden sm:flex items-center gap-1.5 absolute top-3 right-3 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 text-[11px] text-sky-300">
                  <Crosshair className="w-3.5 h-3.5 text-clinical-rose" />
                  <span>
                    AI Peak Focus: <strong className="text-white">{hotspotZone}</strong>
                  </span>
                </div>
              )}

              <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
                <div className="bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 text-[11px] text-slate-200">
                  {inspectedPoint ? (
                    <span>
                      Inspecting: <strong className="text-sky-300">{inspectedPoint.zone}</strong> (Click scan to move pin)
                    </span>
                  ) : (
                    <span>Tip: Click anywhere on the X-Ray to identify the lung lobe &amp; zone</span>
                  )}
                </div>

                {showGradcam && (
                  <div className="bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 flex items-center gap-2 text-[10px] text-slate-300">
                    <span>Low AI Focus</span>
                    <span className="w-16 h-2 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 to-rose-500 inline-block" />
                    <span>High AI Focus</span>
                  </div>
                )}
              </div>
            </div>

            {/* Bottom Controls Bar */}
            <div className="bg-pacs-950/90 px-4 py-3 border-t border-pacs-700 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => setShowGradcam(!showGradcam)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-semibold transition cursor-pointer ${
                    showGradcam
                      ? 'bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/40'
                      : 'bg-pacs-800 text-pacs-300 border border-pacs-700'
                  }`}
                >
                  {showGradcam ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                  {showGradcam ? 'AI Heatmap: ON' : 'AI Heatmap: OFF'}
                </button>

                {showGradcam && (
                  <div className="flex items-center gap-2">
                    <Sliders className="w-3.5 h-3.5 text-pacs-400" />
                    <span className="text-pacs-300 text-[11px]">Intensity:</span>
                    <input
                      type="range"
                      aria-label="Grad-CAM opacity"
                      min="0.15"
                      max="1.0"
                      step="0.05"
                      value={gradcamOpacity}
                      onChange={(e) => setGradcamOpacity(parseFloat(e.target.value))}
                      className="w-20 cursor-pointer"
                    />
                    <select
                      aria-label="Grad-CAM colormap"
                      value={colormap}
                      onChange={(e) => setColormap(e.target.value)}
                      className="bg-pacs-800 border border-pacs-700 rounded-lg px-2 py-1 text-[11px] text-white"
                    >
                      <option value="turbo">Turbo Rainbow</option>
                      <option value="jet">Thermal Jet</option>
                      <option value="plasma">Violet Plasma</option>
                      <option value="inferno">Warm Inferno</option>
                    </select>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-2 text-pacs-300">
                <span className="text-[11px]">AI Brain:</span>
                <select
                  aria-label="Vision Model Backbone"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="bg-pacs-800 text-clinical-cyan border border-pacs-700 rounded-lg px-2.5 py-1 text-xs font-semibold outline-none"
                >
                  <option value="residual_cnn">ResidualChestCNN (Spatial Attention)</option>
                  {availableModels.includes('resnet18') && <option value="resnet18">ResNet-18 (95.7% Real Acc)</option>}
                  {availableModels.includes('mobilenet_v2') && <option value="mobilenet_v2">MobileNetV2 (96.3% Real Acc)</option>}
                </select>
                {explainMode === 'pro' && (
                  <select
                    aria-label="Monte Carlo Dropout passes"
                    value={mcDropoutSamples}
                    onChange={(e) => setMcDropoutSamples(parseInt(e.target.value, 10))}
                    className="bg-pacs-800 text-white border border-pacs-700 rounded-lg px-2 py-1 text-xs outline-none"
                  >
                    <option value={10}>10 Safety Passes</option>
                    <option value={20}>20 Safety Passes</option>
                    <option value={35}>35 Safety Passes</option>
                  </select>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Friendly Explanation + Visual Biomarkers (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-clinical-cyan" />
                {explainMode === 'simple' ? 'AI LUNG SCAN SUMMARY' : 'NEURAL DIAGNOSTIC READOUT'}
              </span>
              <span className="text-[11px] font-mono text-clinical-cyan">{diagnosisResult?.model_used || modelId}</span>
            </div>

            {analyzing ? (
              <div className="py-12 flex flex-col items-center justify-center gap-3 text-clinical-cyan">
                <div className="w-9 h-9 border-2 border-clinical-cyan border-t-transparent rounded-full animate-spin" />
                <span className="text-xs font-medium">Examining lung zones &amp; running 20 safety checks...</span>
              </div>
            ) : diagnosisResult ? (
              <>
                {/* Main Finding Banner */}
                <div
                  className={`p-4 rounded-2xl border ${
                    isPneumonia
                      ? 'bg-clinical-rose/10 border-clinical-rose/40'
                      : 'bg-clinical-green/10 border-clinical-green/40'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-pacs-300">
                      Primary Finding
                    </span>
                    <span
                      className={`text-xs font-extrabold px-2.5 py-0.5 rounded-full ${
                        isPneumonia
                          ? 'bg-clinical-rose/20 text-clinical-rose'
                          : 'bg-clinical-green/20 text-clinical-green'
                      }`}
                    >
                      {(diagnosisResult.confidence * 100).toFixed(1)}% AI Confidence
                    </span>
                  </div>
                  <h3 className="text-xl font-extrabold tracking-tight text-white">{diagnosisResult.diagnosis}</h3>
                  <p className="text-xs text-pacs-200 mt-2 leading-relaxed">
                    {isPneumonia ? (
                      <>
                        <strong>In plain English:</strong> The AI detected cloudy areas (fluid or inflammation) covering{' '}
                        <strong>{diagnosisResult.severity.infiltrate_burden_percentage.toFixed(1)}%</strong> of the lung scan,
                        most prominent in the <strong>{hotspotZone}</strong>.
                      </>
                    ) : (
                      <>
                        <strong>In plain English:</strong> Both left and right lung fields look clear and well-aerated, with no
                        significant fluid buildup or pneumonia hotspots detected.
                      </>
                    )}
                  </p>
                </div>

                {/* Probability Split Meter */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-clinical-rose">
                      Pneumonia Likelihood: {(diagnosisResult.pneumonia_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-clinical-green">
                      Healthy Lung: {(diagnosisResult.normal_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-3.5 bg-pacs-950 rounded-full overflow-hidden flex border border-pacs-700 p-0.5">
                    <div
                      style={{ width: `${diagnosisResult.pneumonia_probability * 100}%` }}
                      className="bg-gradient-to-r from-orange-500 to-clinical-rose h-full rounded-l-full transition-all"
                    />
                    <div
                      style={{ width: `${diagnosisResult.normal_probability * 100}%` }}
                      className="bg-gradient-to-r from-emerald-500 to-clinical-green h-full rounded-r-full transition-all"
                    />
                  </div>
                </div>

                {/* 2 Key Metric Cards */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5">
                    <span className="text-[11px] text-pacs-300 font-semibold block">
                      {explainMode === 'simple' ? 'Affected Lung Area' : 'Infiltrate Burden'}
                    </span>
                    <span className="text-2xl font-extrabold text-white mt-0.5 block">
                      {diagnosisResult.severity.infiltrate_burden_percentage.toFixed(1)}%
                    </span>
                    <span className="text-[11px] text-pacs-400">
                      {diagnosisResult.severity.infiltrate_burden_percentage < 15
                        ? 'Minimal / Clear (<15%)'
                        : diagnosisResult.severity.infiltrate_burden_percentage < 35
                        ? 'Moderate Consolidation'
                        : 'Extensive Lung Involvement'}
                    </span>
                  </div>
                  <div className="bg-pacs-950/90 border border-pacs-700 rounded-xl p-3.5">
                    <span className="text-[11px] text-pacs-300 font-semibold block">
                      {explainMode === 'simple' ? 'AI Self-Check Certainty' : 'MC-Dropout Reliability'}
                    </span>
                    <span className="text-base font-extrabold text-clinical-teal mt-1 block">
                      {diagnosisResult.uncertainty.uncertainty_risk_level === 'LOW'
                        ? 'High Agreement'
                        : diagnosisResult.uncertainty.uncertainty_risk_level}
                    </span>
                    <span className="text-[11px] text-pacs-400 block">
                      {diagnosisResult.uncertainty.human_referral_mandatory
                        ? 'Doctor Double-Check Advised'
                        : `Consistent across ${diagnosisResult.uncertainty.monte_carlo_samples} passes`}
                    </span>
                  </div>
                </div>

                {/* Visual Lung Biomarker Bars (Easy to understand!) */}
                {diagnosisResult.radiomics && Object.keys(diagnosisResult.radiomics).length > 0 && (
                  <div className="bg-pacs-950/80 border border-pacs-700 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white flex items-center gap-1.5">
                        <Activity className="w-4 h-4 text-clinical-cyan" />
                        {explainMode === 'simple'
                          ? 'Key Lung Measurements Explained'
                          : 'Quantitative Radiomic Biomarkers (14 Features)'}
                      </span>
                      <span title="Measured automatically from pixel density & lung symmetry">
                        <HelpCircle className="w-3.5 h-3.5 text-pacs-400" />
                      </span>
                    </div>

                    <div className="space-y-2.5 text-xs">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-pacs-200 font-medium">Lung Cloudiness (Opacity Fraction)</span>
                          <span className="text-white font-bold font-mono">
                            {(diagnosisResult.radiomics.consolidation_opacity_fraction * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-pacs-900 rounded-full overflow-hidden">
                          <div
                            style={{
                              width: `${Math.min(100, diagnosisResult.radiomics.consolidation_opacity_fraction * 160)}%`,
                            }}
                            className="h-full bg-gradient-to-r from-clinical-cyan to-clinical-rose"
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-pacs-200 font-medium">Left vs. Right Lung Difference</span>
                          <span className="text-white font-bold font-mono">
                            {(diagnosisResult.radiomics.left_right_asymmetry * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-pacs-900 rounded-full overflow-hidden">
                          <div
                            style={{
                              width: `${Math.min(100, diagnosisResult.radiomics.left_right_asymmetry * 350)}%`,
                            }}
                            className="h-full bg-clinical-teal"
                          />
                        </div>
                      </div>

                      {explainMode === 'pro' && (
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1 pt-2 border-t border-pacs-800 text-[11px] font-mono">
                          <div className="flex justify-between text-pacs-300">
                            <span>Basilar/Apical:</span>
                            <span className="text-white font-bold">
                              {diagnosisResult.radiomics.basilar_apical_ratio.toFixed(3)}
                            </span>
                          </div>
                          <div className="flex justify-between text-pacs-300">
                            <span>Edge Detail:</span>
                            <span className="text-white font-bold">
                              {diagnosisResult.radiomics.sobel_edge_density.toFixed(4)}
                            </span>
                          </div>
                          <div className="flex justify-between text-pacs-300">
                            <span>Entropy (Bits):</span>
                            <span className="text-white font-bold">
                              {diagnosisResult.radiomics.shannon_entropy.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between text-pacs-300">
                            <span>MC Variance:</span>
                            <span className="text-white font-bold">
                              {diagnosisResult.uncertainty.predictive_variance.toFixed(5)}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="pt-1 flex flex-wrap items-center gap-2.5">
                  <button
                    type="button"
                    onClick={() => onOpenReport(diagnosisResult, currentImageSrc)}
                    className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-clinical-teal to-clinical-cyan hover:opacity-95 text-slate-950 font-bold text-xs shadow-lg shadow-clinical-cyan/20 flex items-center justify-center gap-2 transition cursor-pointer"
                  >
                    <FileText className="w-4 h-4" />
                    View Printable Doctor Report
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const link = document.createElement('a');
                      link.download = `pacs_gradcam_${Date.now()}.png`;
                      link.href = canvasRef.current?.toDataURL('image/png') || '';
                      link.click();
                    }}
                    className="p-2.5 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-white transition cursor-pointer"
                    title="Download Heatmap Image"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  {onNextStep && (
                    <button
                      type="button"
                      onClick={onNextStep}
                      className="w-full py-2 px-3 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-clinical-cyan font-semibold text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
                    >
                      <span>Continue to Step 2: Check ECG Heartbeat</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};

