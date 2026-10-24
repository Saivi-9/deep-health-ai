import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { RadiologyViewer } from './components/RadiologyViewer';
import { ECGMonitor } from './components/ECGMonitor';
import { CohortView } from './components/CohortView';
import { GovernanceView } from './components/GovernanceView';
import { ReportModal } from './components/ReportModal';
import { fetchHealth } from './api/client';
import type { HealthResponse, VisionDiagnosisResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'radiology' | 'ecg' | 'batch' | 'governance'>('radiology');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Modal state
  const [isReportOpen, setIsReportOpen] = useState<boolean>(false);
  const [reportResult, setReportResult] = useState<VisionDiagnosisResponse | null>(null);
  const [reportImageSrc, setReportImageSrc] = useState<string>('');

  useEffect(() => {
    fetchHealth().then(setHealth);
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

  return (
    <div className="min-h-screen bg-pacs-950 text-slate-100 flex flex-col font-sans selection:bg-clinical-cyan selection:text-pacs-950">
      {/* Top Clinical Navigation Bar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      {/* Main Clinical Diagnostic Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'radiology' && <RadiologyViewer onOpenReport={handleOpenReport} />}
        {activeTab === 'ecg' && <ECGMonitor />}
        {activeTab === 'batch' && <CohortView />}
        {activeTab === 'governance' && <GovernanceView />}
      </main>

      {/* Hospital Footer */}
      <footer className="border-t border-pacs-800 bg-pacs-900/60 py-4 px-6 text-xs text-pacs-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            © 2026 Deep Health AI Suite. Clinical Decision Support System. Developed by Saivi-9.
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px]">
            <span>FDA SaMD Class II Guidance</span>
            <span>•</span>
            <span>ISO 13485 Compliant</span>
            <span>•</span>
            <span className="text-clinical-cyan">Native PyTorch + FastAPI</span>
          </div>
        </div>
      </footer>

      {/* Formal Consultation Report Modal */}
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
