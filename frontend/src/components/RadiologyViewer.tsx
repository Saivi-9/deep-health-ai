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
  UserCheck,
  Sparkles,
  Upload,
  Activity,
} from 'lucide-react';
import type { VisionDiagnosisResponse, VisionSample, TriageUrgency } from '../types';
import { diagnoseVisionScan } from '../api/client';

interface RadiologyViewerProps {
  samples: VisionSample[];
  availableModels: string[];
  onOpenReport: (result: VisionDiagnosisResponse, imageSrc: string) => void;
  onVisionUpdate?: (res: VisionDiagnosisResponse) => void;
}

export const RadiologyViewer: React.FC<RadiologyViewerProps> = ({
  samples,
  availableModels,
  onOpenReport,
  onVisionUpdate,
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [modelId, setModelId] = useState<string>('residual_cnn');
  const [colormap, setColormap] = useState<string>('turbo');
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [activeWindow, setActiveWindow] = useState<'standard' | 'lung' | 'bone' | 'invert'>('standard');
  const [showGradcam, setShowGradcam] = useState<boolean>(true);
  const [gradcamOpacity, setGradcamOpacity] = useState<number>(0.65);
  const [mcDropoutSamples, setMcDropoutSamples] = useState<number>(20);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [diagnosisResult, setDiagnosisResult] = useState<VisionDiagnosisResponse | null>(null);
  const [customImageSrc, setCustomImageSrc] = useState<string>('');
  const [customImageTitle, setCustomImageTitle] = useState<string>('');

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
        };
      }
    };
  }, [currentImageSrc, activeWindow, showGradcam, gradcamOpacity, diagnosisResult]);

  const handleSampleChange = (id: string) => {
    setSelectedSampleId(id);
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
        setAnalyzing(true);
      }
    };
    reader.readAsDataURL(file);
  };

  const getUrgencyBadge = (urgency: TriageUrgency) => {
    const u = urgency.toUpperCase();
    if (u.includes('IMMEDIATE') || u.includes('STAT') || u.includes('CRITICAL')) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/40">
          <AlertTriangle className="w-4 h-4" /> {urgency}
        </span>
      );
    }
    if (u.includes('URGENT') || u.includes('EXPEDITED') || u.includes('FOLLOW')) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
          <ShieldAlert className="w-4 h-4" /> {urgency}
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-clinical-green/20 text-clinical-green border border-clinical-green/40">
        <CheckCircle className="w-4 h-4" /> {urgency}
      </span>
    );
  };

  const currentSampleMeta = samples.find((s) => s.id === selectedSampleId);

  return (
    <div id="panel-radiology" role="tabpanel" aria-labelledby="tab-radiology" className="space-y-6">
      {/* Study Bar */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-pacs-800 border border-pacs-600 flex items-center justify-center">
            <UserCheck className="w-5 h-5 text-clinical-cyan" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">{currentImageTitle}</div>
            <div className="text-pacs-300 font-mono">
              SOURCE: {currentSampleMeta?.source || 'Uploaded Radiograph'}
              {currentSampleMeta ? ` | GROUND TRUTH: ${currentSampleMeta.ground_truth.toUpperCase()}` : ''}
            </div>
          </div>
        </div>
        <div>{diagnosisResult && getUrgencyBadge(diagnosisResult.severity.clinical_triage_urgency)}</div>
      </div>

      {errorMsg && (
        <div className="bg-clinical-rose/15 border border-clinical-rose/40 text-clinical-rose px-4 py-2.5 rounded-xl text-xs font-mono">
          API Warning: {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Viewport (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            <div className="bg-pacs-950 px-4 py-2.5 border-b border-pacs-700 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <label className="text-pacs-400 font-mono" htmlFor="cxr-case-select">STUDY:</label>
                <select
                  id="cxr-case-select"
                  value={effectiveSampleId}
                  onChange={(e) => handleSampleChange(e.target.value)}
                  className="bg-pacs-800 text-slate-200 border border-pacs-600 rounded-lg px-2.5 py-1 text-xs outline-none focus:border-clinical-cyan max-w-[230px]"
                >
                  {samples.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.title}
                    </option>
                  ))}
                  {selectedSampleId === 'custom_upload' && <option value="custom_upload">{currentImageTitle}</option>}
                </select>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleCustomUpload}
                  accept="image/png,image/jpeg"
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 text-xs transition"
                >
                  <Upload className="w-3.5 h-3.5 text-clinical-cyan" /> Upload CXR
                </button>
              </div>

              <div className="flex items-center gap-1 bg-pacs-900 p-0.5 rounded-lg border border-pacs-700">
                {(['standard', 'lung', 'bone', 'invert'] as const).map((w) => (
                  <button
                    key={w}
                    type="button"
                    onClick={() => setActiveWindow(w)}
                    className={`px-2 py-0.5 rounded text-[11px] font-semibold transition capitalize ${
                      activeWindow === w ? 'bg-clinical-cyan text-pacs-950' : 'text-pacs-300 hover:text-white'
                    }`}
                  >
                    {w}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-1">
                <button type="button" onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.2))} className="p-1 rounded hover:bg-pacs-800 text-pacs-300 hover:text-white" title="Zoom In">
                  <ZoomIn className="w-4 h-4" />
                </button>
                <button type="button" onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.7))} className="p-1 rounded hover:bg-pacs-800 text-pacs-300 hover:text-white" title="Zoom Out">
                  <ZoomOut className="w-4 h-4" />
                </button>
                <button type="button" onClick={() => setZoomLevel(1.0)} className="p-1 rounded hover:bg-pacs-800 text-pacs-300 hover:text-white" title="Reset Scale">
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="relative min-h-[400px] bg-pacs-950 pacs-grid flex items-center justify-center p-4 overflow-hidden">
              <div style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.15s ease-out' }} className="relative rounded-lg shadow-2xl border border-pacs-700/60 overflow-hidden">
                <canvas ref={canvasRef} className="block w-[340px] h-[340px] object-contain" />
              </div>

              {diagnosisResult?.hotspot && (
                <div className="hidden sm:flex absolute top-3 right-3 bg-pacs-900/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-pacs-700 text-[11px] font-mono text-clinical-cyan">
                  Peak CAM: ({(diagnosisResult.hotspot.x_norm * 100).toFixed(0)}%, {(diagnosisResult.hotspot.y_norm * 100).toFixed(0)}%)
                </div>
              )}

              <div className="absolute bottom-3 left-3 bg-pacs-900/90 backdrop-blur-md px-3 py-1 rounded-lg border border-pacs-700 text-[11px] font-mono text-slate-300">
                SCALE: {(zoomLevel * 100).toFixed(0)}% | WINDOW: {activeWindow.toUpperCase()}
              </div>
            </div>

            <div className="bg-pacs-950/90 px-4 py-3 border-t border-pacs-700 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => setShowGradcam(!showGradcam)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition ${
                    showGradcam ? 'bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/40' : 'bg-pacs-800 text-slate-400 border border-pacs-700'
                  }`}
                >
                  {showGradcam ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                  Real Grad-CAM Overlay
                </button>

                {showGradcam && (
                  <div className="flex items-center gap-2">
                    <Sliders className="w-3.5 h-3.5 text-pacs-400" />
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
                    <span className="font-mono text-slate-300">{(gradcamOpacity * 100).toFixed(0)}%</span>
                    <select
                      aria-label="Grad-CAM colormap"
                      value={colormap}
                      onChange={(e) => setColormap(e.target.value)}
                      className="bg-pacs-800 border border-pacs-700 rounded px-2 py-0.5 text-[11px] text-slate-200"
                    >
                      <option value="turbo">Turbo</option>
                      <option value="jet">Jet</option>
                      <option value="plasma">Plasma</option>
                      <option value="inferno">Inferno</option>
                    </select>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-2 font-mono text-pacs-300">
                <select
                  aria-label="Vision Model Backbone"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="bg-pacs-800 text-clinical-cyan border border-pacs-700 rounded px-2 py-1 text-xs outline-none"
                >
                  <option value="residual_cnn">ResidualChestCNN</option>
                  {availableModels.includes('resnet18') && <option value="resnet18">TransferResNet18</option>}
                  {availableModels.includes('mobilenet_v2') && <option value="mobilenet_v2">TransferMobileNetV2</option>}
                </select>
                <select
                  aria-label="Monte Carlo Dropout passes"
                  value={mcDropoutSamples}
                  onChange={(e) => setMcDropoutSamples(parseInt(e.target.value, 10))}
                  className="bg-pacs-800 text-slate-200 border border-pacs-700 rounded px-2 py-1 text-xs outline-none"
                >
                  <option value={10}>MC=10</option>
                  <option value={20}>MC=20</option>
                  <option value={35}>MC=35</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Diagnostics + 14 Radiomic Biomarkers (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-pacs-700/60 pb-3">
              <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-clinical-cyan" />
                NEURAL DIAGNOSTIC READOUT
              </span>
              <span className="text-[11px] font-mono text-clinical-cyan">{diagnosisResult?.model_used || modelId}</span>
            </div>

            {analyzing ? (
              <div className="py-10 flex flex-col items-center justify-center gap-2 text-clinical-cyan">
                <div className="w-8 h-8 border-2 border-clinical-cyan border-t-transparent rounded-full animate-spin" />
                <span className="text-xs font-mono">Running PyTorch Grad-CAM &amp; MC-Dropout...</span>
              </div>
            ) : diagnosisResult ? (
              <>
                <div>
                  <h3 className="text-xl font-extrabold tracking-tight text-white">{diagnosisResult.diagnosis}</h3>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-clinical-rose font-bold">
                      Pneumonia: {(diagnosisResult.pneumonia_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-clinical-green font-bold">
                      Normal: {(diagnosisResult.normal_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-pacs-950 rounded-full overflow-hidden flex border border-pacs-700">
                    <div style={{ width: `${diagnosisResult.pneumonia_probability * 100}%` }} className="bg-gradient-to-r from-orange-500 to-clinical-rose h-full transition-all" />
                    <div style={{ width: `${diagnosisResult.normal_probability * 100}%` }} className="bg-gradient-to-r from-emerald-500 to-clinical-green h-full transition-all" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3">
                    <span className="text-[10px] text-pacs-400 uppercase font-bold block">Infiltrate Burden</span>
                    <span className="text-xl font-extrabold text-white mt-0.5 block">
                      {diagnosisResult.severity.infiltrate_burden_percentage.toFixed(1)}%
                    </span>
                    <span className="text-[11px] text-pacs-300 font-mono">Grad-CAM Area &ge;0.40</span>
                  </div>
                  <div className="bg-pacs-950 border border-pacs-700/80 rounded-xl p-3">
                    <span className="text-[10px] text-pacs-400 uppercase font-bold block">Severity Grade</span>
                    <span className="text-base font-extrabold text-clinical-cyan mt-0.5 block">
                      {diagnosisResult.severity.radiological_grade}
                    </span>
                    <span className="text-[11px] text-pacs-300 font-mono truncate block">
                      {diagnosisResult.severity.radiological_severity}
                    </span>
                  </div>
                </div>

                <div className="bg-pacs-950/90 border border-pacs-700/80 rounded-xl p-3.5 space-y-1.5 text-xs">
                  <div className="flex items-center justify-between font-bold">
                    <span className="text-slate-300 flex items-center gap-1.5">
                      <ShieldAlert className="w-4 h-4 text-clinical-teal" />
                      Epistemic Uncertainty
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-clinical-teal/15 text-clinical-teal border border-clinical-teal/30">
                      {diagnosisResult.uncertainty.uncertainty_risk_level}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                    <div className="text-pacs-300">Variance: <span className="text-white font-bold">{diagnosisResult.uncertainty.predictive_variance.toFixed(5)}</span></div>
                    <div className="text-pacs-300">Entropy: <span className="text-white font-bold">{diagnosisResult.uncertainty.normalized_entropy.toFixed(4)}</span></div>
                  </div>
                </div>

                {diagnosisResult.radiomics && Object.keys(diagnosisResult.radiomics).length > 0 && (
                  <div className="bg-pacs-950/80 border border-pacs-700/80 rounded-xl p-3.5 space-y-2">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-pacs-400 flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-clinical-cyan" />
                      Quantitative Radiomic Biomarkers (Selected from 14 Features)
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] font-mono">
                      <div className="flex justify-between text-pacs-300"><span>Opacity Frac:</span><span className="text-white font-bold">{(diagnosisResult.radiomics.consolidation_opacity_fraction * 100).toFixed(1)}%</span></div>
                      <div className="flex justify-between text-pacs-300"><span>L/R Asymmetry:</span><span className="text-white font-bold">{diagnosisResult.radiomics.left_right_asymmetry.toFixed(3)}</span></div>
                      <div className="flex justify-between text-pacs-300"><span>Basilar/Apical:</span><span className="text-white font-bold">{diagnosisResult.radiomics.basilar_apical_ratio.toFixed(3)}</span></div>
                      <div className="flex justify-between text-pacs-300"><span>Sobel Density:</span><span className="text-white font-bold">{diagnosisResult.radiomics.sobel_edge_density.toFixed(4)}</span></div>
                      <div className="flex justify-between text-pacs-300"><span>Entropy (Bits):</span><span className="text-white font-bold">{diagnosisResult.radiomics.shannon_entropy.toFixed(2)}</span></div>
                      <div className="flex justify-between text-pacs-300"><span>Kurtosis:</span><span className="text-white font-bold">{diagnosisResult.radiomics.intensity_kurtosis.toFixed(2)}</span></div>
                    </div>
                  </div>
                )}

                <div className="pt-1 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => onOpenReport(diagnosisResult, currentImageSrc)}
                    className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-clinical-cyan to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-pacs-950 font-bold text-xs shadow-lg shadow-clinical-cyan/20 flex items-center justify-center gap-2 transition"
                  >
                    <FileText className="w-4 h-4" />
                    Consultation Report
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const link = document.createElement('a');
                      link.download = `pacs_gradcam_${Date.now()}.png`;
                      link.href = canvasRef.current?.toDataURL('image/png') || '';
                      link.click();
                    }}
                    className="p-2.5 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 transition"
                    title="Export Canvas Snapshot"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};
