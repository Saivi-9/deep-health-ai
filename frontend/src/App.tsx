import React, { useState, useEffect } from 'react';
import {
  Stethoscope,
  Activity,
  HeartPulse,
  CheckCircle2,
  ArrowRight,
  HelpCircle,
  Sparkles,
} from 'lucide-react';
import { Navbar } from './components/Navbar';
import { RadiologyViewer } from './components/RadiologyViewer';
import { ECGMonitor } from './components/ECGMonitor';
import { ClinicalTriageView } from './components/ClinicalTriageView';
import { PretrainedHubView } from './components/PretrainedHubView';
import { CohortView } from './components/CohortView';
import { GovernanceView } from './components/GovernanceView';
import { ReportModal } from './components/ReportModal';
import { fetchHealth, fetchDatasetSamples } from './api/client';
import type {
  HealthResponse,
  VisionDiagnosisResponse,
  ECGScreenResponse,
  DatasetSamplesResponse,
  ActiveTab,
  ThemeMode,
  ExplainMode,
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('radiology');
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem('deep_health_theme') as ThemeMode | null;
    return saved && ['ocean', 'daylight', 'indigo'].includes(saved) ? saved : 'ocean';
  });
  const [explainMode, setExplainMode] = useState<ExplainMode>('simple');
  const [showQuickGuide, setShowQuickGuide] = useState<boolean>(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [samples, setSamples] = useState<DatasetSamplesResponse>({
    vision_samples: [],
    ecg_samples: [],
    clinical_presets: [],
  });

  // Live session findings shared seamlessly with Cardiology & Multimodal Triage
  const [latestVision, setLatestVision] = useState<VisionDiagnosisResponse | null>(null);
  const [latestEcg, setLatestEcg] = useState<ECGScreenResponse | null>(null);

  // Modal state
  const [isReportOpen, setIsReportOpen] = useState<boolean>(false);
  const [reportResult, setReportResult] = useState<VisionDiagnosisResponse | null>(null);
  const [reportImageSrc, setReportImageSrc] = useState<string>('');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('deep_health_theme', theme);
  }, [theme]);

  useEffect(() => {
    fetchHealth().then(setHealth);
    fetchDatasetSamples().then(setSamples).catch(() => {});
    const interval = setInterval(() => {
      fetchHealth().then(setHealth);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleOpenReport = (result: VisionDiagnosisResponse, imageSrc: string) => {
    setReportResult(result);
    setReportImageSrc(imageSrc);
    setIsReportOpen(true);
  };

  const availVisionModels = [
    'residual_cnn',
    ...(health?.models_loaded?.TransferResNet18 ? ['resnet18'] : []),
    ...(health?.models_loaded?.TransferMobileNetV2 ? ['mobilenet_v2'] : []),
  ];
  const availEcgModels = [
    'autoencoder',
    ...(health?.models_loaded?.ECGResNet1D ? ['resnet1d'] : []),
    ...(health?.models_loaded?.ECGResNet1D_MITBIH ? ['mitbih_resnet1d'] : []),
  ];

  return (
    <div className="min-h-screen bg-pacs-950 text-slate-100 flex flex-col font-sans selection:bg-clinical-cyan selection:text-pacs-950">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        theme={theme}
        setTheme={setTheme}
        explainMode={explainMode}
        setExplainMode={setExplainMode}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Interactive 3-Step Clinical Journey & Plain-English Orientation Banner */}
        {showQuickGuide ? (
          <section
            aria-label="Interactive Clinical AI Workflow Guide"
            className="bg-gradient-to-r from-pacs-900 via-pacs-850 to-pacs-900 border border-pacs-700 rounded-2xl p-4 sm:p-5 shadow-xl"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-clinical-teal/20 text-clinical-teal border border-clinical-teal/40 flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5" /> Interactive 3-Step Patient Checkup
                </span>
                <span className="text-xs text-pacs-300">
                  Explore how AI combines Lung X-Rays, ECG Heartbeats, and Vitals into one clear health picture.
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowQuickGuide(false)}
                className="text-xs text-pacs-400 hover:text-white underline cursor-pointer"
              >
                Hide Guide
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* Step 1: Chest X-Ray */}
              <button
                type="button"
                onClick={() => setActiveTab('radiology')}
                className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition cursor-pointer ${
                  activeTab === 'radiology'
                    ? 'bg-pacs-800 border-clinical-cyan shadow-md'
                    : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-xl bg-clinical-cyan/15 border border-clinical-cyan/40 text-clinical-cyan flex items-center justify-center shrink-0">
                    <Stethoscope className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-white flex items-center gap-1.5">
                      <span>1. Lung X-Ray Scan</span>
                      {latestVision && <CheckCircle2 className="w-3.5 h-3.5 text-clinical-green shrink-0" />}
                    </div>
                    <div className="text-[11px] text-pacs-300 truncate">
                      {latestVision
                        ? `${latestVision.diagnosis} (${(latestVision.confidence * 100).toFixed(0)}% sure)`
                        : 'Click to inspect lung scan & AI heatmap'}
                    </div>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 text-pacs-400 shrink-0" />
              </button>

              {/* Step 2: ECG Heart Monitor */}
              <button
                type="button"
                onClick={() => setActiveTab('ecg')}
                className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition cursor-pointer ${
                  activeTab === 'ecg'
                    ? 'bg-pacs-800 border-clinical-teal shadow-md'
                    : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-xl bg-clinical-teal/15 border border-clinical-teal/40 text-clinical-teal flex items-center justify-center shrink-0">
                    <Activity className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-white flex items-center gap-1.5">
                      <span>2. ECG Heart Rhythm</span>
                      {latestEcg && <CheckCircle2 className="w-3.5 h-3.5 text-clinical-green shrink-0" />}
                    </div>
                    <div className="text-[11px] text-pacs-300 truncate">
                      {latestEcg
                        ? `${latestEcg.rhythm_subtype || 'Analyzed'} (${latestEcg.telemetry.heart_rate_bpm} BPM)`
                        : 'Click to test live heartbeat waves'}
                    </div>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 text-pacs-400 shrink-0" />
              </button>

              {/* Step 3: Combined Heart Risk & Vascular Age */}
              <button
                type="button"
                onClick={() => setActiveTab('clinical')}
                className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition cursor-pointer ${
                  activeTab === 'clinical'
                    ? 'bg-pacs-800 border-clinical-rose shadow-md'
                    : 'bg-pacs-950/70 border-pacs-700 hover:bg-pacs-800/60'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-xl bg-clinical-rose/15 border border-clinical-rose/40 text-clinical-rose flex items-center justify-center shrink-0">
                    <HeartPulse className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-white">3. Heart Risk &amp; Heart Age</div>
                    <div className="text-[11px] text-pacs-300 truncate">
                      Combines Steps 1 + 2 with blood pressure &amp; labs
                    </div>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 text-pacs-400 shrink-0" />
              </button>
            </div>
          </section>
        ) : (
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setShowQuickGuide(true)}
              className="flex items-center gap-1.5 text-xs text-clinical-cyan hover:underline cursor-pointer"
            >
              <HelpCircle className="w-3.5 h-3.5" /> Show 3-Step Interactive Checkup Guide
            </button>
          </div>
        )}

        {activeTab === 'radiology' && (
          <RadiologyViewer
            samples={samples.vision_samples}
            availableModels={availVisionModels}
            onOpenReport={handleOpenReport}
            onVisionUpdate={setLatestVision}
            explainMode={explainMode}
            onNextStep={() => setActiveTab('ecg')}
          />
        )}
        {activeTab === 'ecg' && (
          <ECGMonitor
            samples={samples.ecg_samples}
            availableModels={availEcgModels}
            onEcgUpdate={setLatestEcg}
            explainMode={explainMode}
            onNextStep={() => setActiveTab('clinical')}
          />
        )}
        {activeTab === 'clinical' && (
          <ClinicalTriageView
            presets={samples.clinical_presets}
            latestVision={latestVision}
            latestEcg={latestEcg}
            explainMode={explainMode}
          />
        )}
        {activeTab === 'pretrained' && (
          <PretrainedHubView
            visionSamples={samples.vision_samples}
            ecgSamples={samples.ecg_samples}
            clinicalPresets={samples.clinical_presets}
          />
        )}
        {activeTab === 'batch' && <CohortView />}
        {activeTab === 'governance' && <GovernanceView />}
      </main>

      <footer className="border-t border-pacs-700 bg-pacs-900/70 py-4 px-6 text-xs text-pacs-300">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>Deep Health AI Suite v2.0 &bull; Interactive Multi-Modal Clinical Decision Support</div>
          <div className="flex items-center gap-3 font-mono text-[11px]">
            <span>3,148 Chest X-Rays + 17,500 ECG Beats + 1,305 Heart Patients</span>
            <span>&bull;</span>
            <span className="text-clinical-cyan font-semibold">PyTorch + LightGBM + FastAPI + React 19</span>
          </div>
        </div>
      </footer>

      <ReportModal
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
        result={reportResult}
        imageSrc={reportImageSrc}
      />
    </div>
  );
};

export default App;

