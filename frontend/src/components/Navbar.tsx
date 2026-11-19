import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  ShieldCheck,
  Cpu,
  Clock,
  Stethoscope,
  Layers,
  FileText,
  HeartPulse,
  Sparkles,
} from 'lucide-react';
import type { HealthResponse, ActiveTab } from '../types';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  health: HealthResponse | null;
}

const TABS: { id: ActiveTab; label: string; icon: React.ReactNode; activeColor: string }[] = [
  { id: 'radiology', label: 'Radiology', icon: <Stethoscope className="w-3.5 h-3.5" />, activeColor: 'text-clinical-cyan' },
  { id: 'ecg', label: '1D ECG', icon: <Activity className="w-3.5 h-3.5" />, activeColor: 'text-clinical-teal' },
  { id: 'clinical', label: 'Cardiology Triage', icon: <HeartPulse className="w-3.5 h-3.5" />, activeColor: 'text-clinical-rose' },
  { id: 'pretrained', label: 'HF Model Hub', icon: <Sparkles className="w-3.5 h-3.5" />, activeColor: 'text-amber-300' },
  { id: 'batch', label: 'Cohort Eval', icon: <Layers className="w-3.5 h-3.5" />, activeColor: 'text-clinical-purple' },
  { id: 'governance', label: 'Benchmarks', icon: <FileText className="w-3.5 h-3.5" />, activeColor: 'text-clinical-green' },
];

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  const [time, setTime] = useState<string>('--:--:--');
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString());
    const initId = window.setTimeout(tick, 0);
    const timer = window.setInterval(tick, 1000);
    return () => {
      window.clearTimeout(initId);
      window.clearInterval(timer);
    };
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent, idx: number) => {
    let next = idx;
    if (e.key === 'ArrowRight') next = (idx + 1) % TABS.length;
    else if (e.key === 'ArrowLeft') next = (idx - 1 + TABS.length) % TABS.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = TABS.length - 1;
    else return;
    e.preventDefault();
    setActiveTab(TABS[next].id);
    tabRefs.current[next]?.focus();
  };

  return (
    <header className="border-b border-pacs-700 bg-pacs-900/95 backdrop-blur-md sticky top-0 z-40 px-4 sm:px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-col xl:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-clinical-cyan/20 to-blue-600/30 border border-clinical-cyan/40 flex items-center justify-center text-clinical-cyan shadow-lg shadow-clinical-cyan/10">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-base sm:text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-clinical-cyan bg-clip-text text-transparent">
                DEEP HEALTH AI
              </span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-clinical-cyan/15 text-clinical-cyan border border-clinical-cyan/30">
                PACS WORKSTATION v2.0
              </span>
            </div>
            <p className="text-[11px] text-pacs-300 font-mono">
              Multi-Modal Radiology • Electrophysiology • Cardiology EHR • HF Hub
            </p>
          </div>
        </div>

        <nav
          role="tablist"
          aria-label="Clinical Workstation Modalities"
          className="pacs-tablist bg-pacs-950/90 p-1 rounded-xl border border-pacs-700"
        >
          {TABS.map((t, idx) => {
            const selected = activeTab === t.id;
            return (
              <button
                key={t.id}
                ref={(el) => {
                  tabRefs.current[idx] = el;
                }}
                type="button"
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={selected}
                aria-controls={`panel-${t.id}`}
                tabIndex={selected ? 0 : -1}
                onClick={() => setActiveTab(t.id)}
                onKeyDown={(e) => handleKeyDown(e, idx)}
                className={`pacs-tab flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  selected
                    ? `bg-pacs-800 ${t.activeColor}`
                    : 'text-pacs-300 hover:text-white hover:bg-pacs-800/60'
                }`}
              >
                {t.icon}
                <span>{t.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-2.5 text-xs">
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
            <Cpu className="w-3.5 h-3.5 text-clinical-cyan" />
            <span>{health?.device?.toUpperCase() || 'CPU'}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-clinical-green" />
            <span className={health?.status === 'HEALTHY' ? 'text-clinical-green' : 'text-amber-300'}>
              {health?.status === 'HEALTHY' ? 'LIVE API' : 'CONNECTING'}
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
