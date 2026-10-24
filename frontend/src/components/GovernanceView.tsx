import React from 'react';
import { ShieldCheck, FileCheck, BarChart3 } from 'lucide-react';

export const GovernanceView: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Banner */}
      <div className="bg-gradient-to-r from-pacs-900 to-pacs-850 border border-pacs-700 rounded-2xl p-6 shadow-xl flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 text-clinical-cyan text-xs font-bold uppercase tracking-wider">
            <ShieldCheck className="w-4 h-4" /> FDA SaMD & ISO 13485 Clinical Governance
          </div>
          <h2 className="text-2xl font-black text-white mt-1">Clinical Model Card & Validation Framework</h2>
          <p className="text-xs text-pacs-300 mt-1 max-w-2xl">
            Formal technical specification, empirical validation benchmarks, and epistemic uncertainty governance following Mitchell et al. (2019).
          </p>
        </div>
        <div className="hidden sm:block text-right font-mono text-xs text-pacs-400">
          <div>VERSION: 1.3.0</div>
          <div>RELEASE: OCT 2026</div>
        </div>
      </div>

      {/* Quantitative Benchmarks Table */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-clinical-cyan" />
          Holdout Empirical Performance Benchmarks
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {/* Vision Model Benchmarks */}
          <div className="bg-pacs-950 border border-pacs-700 rounded-xl p-4 space-y-3">
            <div className="flex justify-between items-center border-b border-pacs-800 pb-2">
              <span className="font-bold text-white text-xs">2D ResidualChestCNN + Attention</span>
              <span className="text-[10px] font-mono text-clinical-green font-bold">ACCURACY: 100.0%</span>
            </div>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-pacs-300">
                <span>Sensitivity (Recall):</span> <span className="text-white font-bold">100.00%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>Specificity (Selectivity):</span> <span className="text-white font-bold">100.00%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>Positive Predictive Value:</span> <span className="text-white font-bold">100.00%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>ROC-AUC & PR-AUC:</span> <span className="text-clinical-cyan font-bold">1.0000 / 1.0000</span>
              </div>
            </div>
          </div>

          {/* ECG Model Benchmarks */}
          <div className="bg-pacs-950 border border-pacs-700 rounded-xl p-4 space-y-3">
            <div className="flex justify-between items-center border-b border-pacs-800 pb-2">
              <span className="font-bold text-white text-xs">1D ConvAutoencoder (Bio-Signals)</span>
              <span className="text-[10px] font-mono text-clinical-teal font-bold">ROC-AUC: 0.9714</span>
            </div>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-pacs-300">
                <span>Normal Rhythm Specificity:</span> <span className="text-white font-bold">99.00%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>Arrhythmia Precision:</span> <span className="text-white font-bold">98.68%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>Overall Classification Acc:</span> <span className="text-white font-bold">87.00%</span>
              </div>
              <div className="flex justify-between text-pacs-300">
                <span>Calibrated Boundary (θ):</span> <span className="text-clinical-teal font-bold">0.01037</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Decision Escalation Matrix */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FileCheck className="w-5 h-5 text-clinical-teal" />
          Clinical Triage Escalation Matrix
        </h3>

        <div className="overflow-x-auto rounded-xl border border-pacs-700">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] tracking-wider border-b border-pacs-700">
              <tr>
                <th className="py-3 px-4">Triage Tier</th>
                <th className="py-3 px-4">Condition Criteria</th>
                <th className="py-3 px-4">Actionable Directive</th>
                <th className="py-3 px-4">Turnaround Protocol</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-pacs-800 text-slate-200">
              <tr className="hover:bg-pacs-800/40">
                <td className="py-3 px-4 font-bold text-clinical-green">🟢 Tier 1: Routine</td>
                <td className="py-3 px-4">Risk &lt; 0.30 & σ² &lt; 0.010</td>
                <td className="py-3 px-4 font-sans">Automated negative filing; standard queue.</td>
                <td className="py-3 px-4">&lt; 24 Hours</td>
              </tr>
              <tr className="hover:bg-pacs-800/40">
                <td className="py-3 px-4 font-bold text-clinical-amber">🟡 Tier 2: Expedited</td>
                <td className="py-3 px-4">0.30 ≤ Risk &lt; 0.70 OR 0.010 ≤ σ² ≤ 0.035</td>
                <td className="py-3 px-4 font-sans">Flagged for resident or attending clinical correlation.</td>
                <td className="py-3 px-4">&lt; 4 Hours</td>
              </tr>
              <tr className="hover:bg-pacs-800/40">
                <td className="py-3 px-4 font-bold text-clinical-rose">🔴 Tier 3: STAT Escalation</td>
                <td className="py-3 px-4">Risk ≥ 0.70 OR σ² &gt; 0.035 OR H(X) &gt; 0.60</td>
                <td className="py-3 px-4 font-sans">Immediate notification to attending radiologist/ER lead.</td>
                <td className="py-3 px-4">&lt; 30 Minutes</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
