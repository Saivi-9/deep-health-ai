import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  ShieldCheck,
  Cpu,
  Clock,
  Stethoscope,
  Layers,
  Award,
  HeartPulse,
  Sparkles,
  Sun,
  Moon,
  Waves,
  BookOpen,
  Sliders,
} from 'lucide-react';
import type { HealthResponse, ActiveTab, ThemeMode, ExplainMode } from '../types';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  explainMode: ExplainMode;
  setExplainMode: (mode: ExplainMode) => void;
}

const TABS: {
  id: ActiveTab;
  label: string;
  subtitle: string;
  icon: React.ReactNode;
  activeColor: string;
}[] = [
  {
    id: 'radiology',
    label: 'Lung X-Ray AI',
    subtitle: 'Scan & Heatmap',
    icon: <Stethoscope className="w-4 h-4" />,
    activeColor: 'text-clinical-cyan',
  },
  {
    id: 'ecg',
    label: 'ECG Heart Monitor',
    subtitle: 'Live Rhythm Wave',
    icon: <Activity className="w-4 h-4" />,
    activeColor: 'text-clinical-teal',
  },
  {
    id: 'clinical',
    label: 'Heart Risk & Age',
    subtitle: 'Vitals & Triage',
    icon: <HeartPulse className="w-4 h-4" />,
    activeColor: 'text-clinical-rose',
  },
  {
    id: 'pretrained',
    label: 'Open AI Models',
    subtitle: 'Hugging Face Hub',
    icon: <Sparkles className="w-4 h-4" />,
    activeColor: 'text-amber-300',
  },
  {
    id: 'batch',
    label: 'Patient Cohort Lab',
    subtitle: 'Batch Testing',
    icon: <Layers className="w-4 h-4" />,
    activeColor: 'text-clinical-purple',
  },
  {
    id: 'governance',
    label: 'Accuracy & Proof',
    subtitle: 'Model Benchmarks',
    icon: <Award className="w-4 h-4" />,
    activeColor: 'text-clinical-green',
  },
];

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  theme,
  setTheme,
  explainMode,
  setExplainMode,
}) => {
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
    <header className="border-b border-pacs-700 bg-pacs-900/90 backdrop-blur-xl sticky top-0 z-40 px-4 sm:px-6 py-3 shadow-lg">
      <div className="max-w-7xl mx-auto flex flex-col gap-3">
        {/* Top Row: Brand + Theme & Explanation Mode Controls + System Health */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-clinical-teal via-clinical-cyan to-blue-600 p-0.5 shadow-lg shadow-clinical-cyan/20">
              <div className="w-full h-full bg-pacs-950 rounded-[10px] flex items-center justify-center text-clinical-cyan">
                <HeartPulse className="w-5 h-5 animate-heartbeat" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-extrabold text-base sm:text-lg tracking-tight text-white">
                  Deep Health AI
                </span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2.5 py-0.5 rounded-full bg-clinical-teal/15 text-clinical-teal border border-clinical-teal/30">
                  Interactive Clinical Studio
                </span>
              </div>
              <p className="text-xs text-pacs-300">
                Explainable Chest X-Ray, Heart Rhythm &amp; Cardiovascular Health Assistant
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Simple Guide vs Clinician Pro Toggle */}
            <div
              className="flex items-center bg-pacs-950 p-1 rounded-xl border border-pacs-700"
              role="group"
              aria-label="Explanation complexity mode"
            >
              <button
                type="button"
                onClick={() => setExplainMode('simple')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-semibold transition ${
                  explainMode === 'simple'
                    ? 'bg-clinical-teal text-slate-950 shadow-sm'
                    : 'text-pacs-300 hover:text-white'
                }`}
                title="Plain-English explanations for everyone"
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Simple Guide</span>
              </button>
              <button
                type="button"
                onClick={() => setExplainMode('pro')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-semibold transition ${
                  explainMode === 'pro'
                    ? 'bg-clinical-cyan text-slate-950 shadow-sm'
                    : 'text-pacs-300 hover:text-white'
                }`}
                title="Full clinical biomarkers & deep telemetry"
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Clinician Pro</span>
              </button>
            </div>

            {/* 3-Palette Theme Switcher */}
            <div
              className="flex items-center bg-pacs-950 p-1 rounded-xl border border-pacs-700"
              role="group"
              aria-label="Color theme switcher"
            >
              <button
                type="button"
                onClick={() => setTheme('daylight')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-semibold transition ${
                  theme === 'daylight'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-pacs-300 hover:text-white'
                }`}
                title="Clean Daylight Clinic Theme"
              >
                <Sun className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Daylight</span>
              </button>
              <button
                type="button"
                onClick={() => setTheme('ocean')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-semibold transition ${
                  theme === 'ocean'
                    ? 'bg-clinical-teal text-slate-950 shadow-sm'
                    : 'text-pacs-300 hover:text-white'
                }`}
                title="Bioluminescent Ocean PACS Theme"
              >
                <Waves className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Ocean</span>
              </button>
              <button
                type="button"
                onClick={() => setTheme('indigo')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-semibold transition ${
                  theme === 'indigo'
                    ? 'bg-indigo-500 text-white shadow-sm'
                    : 'text-pacs-300 hover:text-white'
                }`}
                title="Soft Midnight Indigo Theme"
              >
                <Moon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Indigo</span>
              </button>
            </div>

            {/* Status Pills */}
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
              <Cpu className="w-3.5 h-3.5 text-clinical-cyan" />
              <span>{health?.device?.toUpperCase() || 'CPU'}</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-pacs-800/80 border border-pacs-700 font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-clinical-green" />
              <span className={health?.status === 'HEALTHY' ? 'text-clinical-green font-bold' : 'text-amber-300'}>
                {health?.status === 'HEALTHY' ? 'AI ONLINE' : 'READY'}
              </span>
            </div>
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-pacs-800/80 border border-pacs-700 text-pacs-300 font-mono">
              <Clock className="w-3.5 h-3.5 text-pacs-400" />
              <span>{time}</span>
            </div>
          </div>
        </div>

        {/* Bottom Row: Navigation Tab Bar with Clear Labels & Subtitles */}
        <nav
          role="tablist"
          aria-label="Clinical Workstation Modalities"
          className="pacs-tablist bg-pacs-950/90 p-1.5 rounded-2xl border border-pacs-700 justify-between"
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
                className={`pacs-tab flex-1 min-w-[135px] flex items-center gap-2.5 px-3 py-2 rounded-xl text-left transition-all ${
                  selected
                    ? 'bg-pacs-800 shadow-md border-pacs-600'
                    : 'text-pacs-300 hover:text-white hover:bg-pacs-800/50'
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                    selected
                      ? `bg-pacs-900 ${t.activeColor} border border-pacs-600`
                      : 'bg-pacs-900/60 text-pacs-400'
                  }`}
                >
                  {t.icon}
                </div>
                <div className="min-w-0">
                  <div className={`text-xs font-bold truncate ${selected ? 'text-white' : ''}`}>
                    {t.label}
                  </div>
                  <div className="text-[10px] text-pacs-400 truncate">{t.subtitle}</div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

