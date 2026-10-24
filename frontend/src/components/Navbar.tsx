import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Cpu, Clock, Stethoscope, Layers, FileText } from 'lucide-react';
import type { HealthResponse } from '../types';

interface NavbarProps {
  activeTab: 'radiology' | 'ecg' | 'batch' | 'governance';
  setActiveTab: (tab: 'radiology' | 'ecg' | 'batch' | 'governance') => void;
  health: HealthResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  const [time, setTime] = useState<string>(new Date().toLocaleTimeString());

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="border-b border-pacs-700 bg-pacs-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & Hospital PACS Identifiers */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-clinical-cyan/20 to-blue-600/30 border border-clinical-cyan/40 flex items-center justify-center text-clinical-cyan shadow-lg shadow-clinical-cyan/10">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-clinical-cyan bg-clip-text text-transparent">
                DEEP HEALTH AI
              </span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-clinical-cyan/10 text-clinical-cyan border border-clinical-cyan/30">
                PACS WORKSTATION v1.3
              </span>
            </div>
            <p className="text-xs text-pacs-300 font-mono">
              Clinical Decision Support & Electrophysiological Diagnostic Suite
            </p>
          </div>
        </div>

        {/* Workstation Modality Navigation Tabs */}
        <nav className="flex items-center gap-1.5 bg-pacs-950/80 p-1 rounded-xl border border-pacs-700">
          <button
            onClick={() => setActiveTab('radiology')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'radiology'
                ? 'bg-pacs-700 text-clinical-cyan shadow-sm border border-clinical-cyan/30'
                : 'text-pacs-300 hover:text-white hover:bg-pacs-800'
            }`}
          >
            <Stethoscope className="w-3.5 h-3.5" />
            Thoracic Radiology
          </button>

          <button
            onClick={() => setActiveTab('ecg')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'ecg'
                ? 'bg-pacs-700 text-clinical-teal shadow-sm border border-clinical-teal/30'
                : 'text-pacs-300 hover:text-white hover:bg-pacs-800'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            1D ECG Telemetry
          </button>

          <button
            onClick={() => setActiveTab('batch')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'batch'
                ? 'bg-pacs-700 text-clinical-purple shadow-sm border border-clinical-purple/30'
                : 'text-pacs-300 hover:text-white hover:bg-pacs-800'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Cohort Processor
          </button>

          <button
            onClick={() => setActiveTab('governance')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'governance'
                ? 'bg-pacs-700 text-clinical-green shadow-sm border border-clinical-green/30'
                : 'text-pacs-300 hover:text-white hover:bg-pacs-800'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            Model Governance
          </button>
        </nav>

        {/* Live System Telemetry Badges */}
        <div className="flex items-center gap-3 text-xs">
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
            <Cpu className="w-3.5 h-3.5 text-clinical-cyan" />
            <span>ACCEL: {health?.device?.toUpperCase() || 'CPU'}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-clinical-green" />
            <span className={health?.status === 'HEALTHY' ? 'text-clinical-green' : 'text-clinical-amber'}>
              {health?.status === 'HEALTHY' ? 'ONLINE (FASTAPI)' : 'STANDBY'}
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span>{time}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
