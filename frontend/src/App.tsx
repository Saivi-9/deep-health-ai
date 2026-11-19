import React, { useState, useEffect } from 'react';
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
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('radiology');
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
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'radiology' && (
          <RadiologyViewer
            samples={samples.vision_samples}
            availableModels={availVisionModels}
            onOpenReport={handleOpenReport}
            onVisionUpdate={setLatestVision}
          />
        )}
        {activeTab === 'ecg' && (
          <ECGMonitor
            samples={samples.ecg_samples}
            availableModels={availEcgModels}
            onEcgUpdate={setLatestEcg}
          />
        )}
        {activeTab === 'clinical' && (
          <ClinicalTriageView
            presets={samples.clinical_presets}
            latestVision={latestVision}
            latestEcg={latestEcg}
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

      <footer className="border-t border-pacs-800 bg-pacs-900/60 py-4 px-6 text-xs text-pacs-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>Deep Health AI Suite v2.0 &bull; Multi-Modal Clinical Decision Support Workstation</div>
          <div className="flex items-center gap-3 font-mono text-[11px]">
            <span>PneumoniaMNIST + MIT-BIH + UCI Cardiology</span>
            <span>&bull;</span>
            <span className="text-clinical-cyan">PyTorch + LightGBM + FastAPI + HF Hub</span>
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
