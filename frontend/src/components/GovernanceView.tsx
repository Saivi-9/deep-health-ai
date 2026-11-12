import React, { useState, useEffect } from 'react';
import { ShieldCheck, BarChart3, Award } from 'lucide-react';
import { fetchBenchmarks } from '../api/client';

export const GovernanceView: React.FC = () => {
  const [data, setData] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    fetchBenchmarks().then(setData).catch(() => {});
  }, []);

  const visionModels = data?.vision?.models_comparison || {};
  const ecgModels = data?.ecg?.models_comparison || {};
  const mitbih = data?.ecg?.mitbih_benchmark;
  const clinicalModels = data?.clinical?.models_comparison || {};
  const clinicalImp = data?.clinical?.feature_importance || [];

  return (
    <div id="panel-governance" role="tabpanel" aria-labelledby="tab-governance" className="space-y-6 max-w-6xl mx-auto">
      <div className="bg-gradient-to-r from-pacs-900 to-pacs-850 border border-pacs-700 rounded-2xl p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-clinical-cyan text-xs font-bold uppercase tracking-wider">
            <ShieldCheck className="w-4 h-4" /> Empirical Multi-Architecture Benchmark &amp; Governance Registry
          </div>
          <h2 className="text-2xl font-black text-white mt-1">Deep Health AI v2.0 Validation Matrix</h2>
          <p className="text-xs text-pacs-300 mt-1 max-w-3xl">
            Every metric below is loaded dynamically from isolated holdout &amp; stratified cross-validation evaluations stored by `train_vision.py`, `train_ecg.py`, `train_clinical.py`, and `benchmark_all_pretrained()`.
          </p>
        </div>
        <div className="font-mono text-xs text-clinical-green bg-clinical-green/10 border border-clinical-green/30 px-3 py-1.5 rounded-lg">
          3,148 CXRs &bull; 17,500 ECGs &bull; 1,305 Cardiology EHR
        </div>
      </div>

      {/* 1. Vision Tournament */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-6 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-clinical-cyan" />
          Thoracic Radiograph 4-Architecture Tournament
        </h3>
        <div className="table-wrapper rounded-xl border border-pacs-700">
          <table className="pacs-responsive-table text-left text-xs font-mono">
            <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] border-b border-pacs-700">
              <tr>
                <th scope="col" className="py-3 px-4">Model</th>
                <th scope="col" className="py-3 px-4">Holdout Acc</th>
                <th scope="col" className="py-3 px-4">Holdout ROC-AUC</th>
                <th scope="col" className="py-3 px-4">Holdout F1</th>
                <th scope="col" className="py-3 px-4">Real PneumoniaMNIST Acc / AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-pacs-800 text-slate-200">
              {Object.entries(visionModels).map(([name, info]: [string, any]) => {
                const um = info.unified_test_metrics || {};
                const rm = info.real_pneumoniamnist_metrics || {};
                return (
                  <tr key={name} className="hover:bg-pacs-800/40">
                    <td data-label="Model" className="py-3 px-4 font-bold text-clinical-cyan">{name}</td>
                    <td data-label="Holdout Acc" className="py-3 px-4">{um.accuracy !== undefined ? `${(um.accuracy * 100).toFixed(2)}%` : '--'}</td>
                    <td data-label="Holdout ROC-AUC" className="py-3 px-4">{um.roc_auc?.toFixed(4) ?? '--'}</td>
                    <td data-label="Holdout F1" className="py-3 px-4">{um.f1_score?.toFixed(4) ?? '--'}</td>
                    <td data-label="Real PneumoniaMNIST" className="py-3 px-4 text-clinical-green">
                      {rm.accuracy !== undefined ? `${(rm.accuracy * 100).toFixed(2)}% / ${rm.roc_auc?.toFixed(4)}` : '--'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. ECG Tournament */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-clinical-teal" />
            ECG Bio-Signal &amp; 5-Class Arrhythmia Benchmarks
          </h3>
          {mitbih && (
            <span className="text-xs font-mono text-clinical-teal bg-clinical-teal/10 border border-clinical-teal/30 px-3 py-1 rounded-lg">
              MIT-BIH 10k ResNet1D: {(mitbih.multiclass_accuracy * 100).toFixed(2)}% 5-Class Acc (Macro-F1 {mitbih.macro_f1?.toFixed(4)})
            </span>
          )}
        </div>
        <div className="table-wrapper rounded-xl border border-pacs-700">
          <table className="pacs-responsive-table text-left text-xs font-mono">
            <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] border-b border-pacs-700">
              <tr>
                <th scope="col" className="py-3 px-4">Model</th>
                <th scope="col" className="py-3 px-4">Anomaly Acc</th>
                <th scope="col" className="py-3 px-4">Binary ROC-AUC</th>
                <th scope="col" className="py-3 px-4">5-Class Acc</th>
                <th scope="col" className="py-3 px-4">Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-pacs-800 text-slate-200">
              {Object.entries(ecgModels).map(([name, info]: [string, any]) => {
                const bm = info.binary_anomaly_metrics || {};
                const mm = info.multiclass_metrics || {};
                return (
                  <tr key={name} className="hover:bg-pacs-800/40">
                    <td data-label="Model" className="py-3 px-4 font-bold text-clinical-teal">{name}</td>
                    <td data-label="Anomaly Acc" className="py-3 px-4">{bm.accuracy !== undefined ? `${(bm.accuracy * 100).toFixed(2)}%` : '--'}</td>
                    <td data-label="Binary ROC-AUC" className="py-3 px-4">{bm.roc_auc?.toFixed(4) ?? '--'}</td>
                    <td data-label="5-Class Acc" className="py-3 px-4">{mm.multiclass_accuracy !== undefined ? `${(mm.multiclass_accuracy * 100).toFixed(2)}%` : 'Unsupervised Binary'}</td>
                    <td data-label="Macro F1" className="py-3 px-4">{mm.macro_f1?.toFixed(4) ?? bm.f1_score?.toFixed(4) ?? '--'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Cardiology EHR 5-Model Tournament + Top Drivers */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-pacs-900 border border-pacs-700 rounded-2xl p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-clinical-rose" />
              Real Cardiology EHR 5-Model Tournament (n=1,305)
            </h3>
            {data?.clinical?.best_model && (
              <span className="text-xs font-mono text-clinical-green font-bold">Winner: {data.clinical.best_model}</span>
            )}
          </div>
          <div className="table-wrapper rounded-xl border border-pacs-700">
            <table className="pacs-responsive-table text-left text-xs font-mono">
              <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] border-b border-pacs-700">
                <tr>
                  <th scope="col" className="py-3 px-4">Architecture</th>
                  <th scope="col" className="py-3 px-4">5-Fold CV AUC</th>
                  <th scope="col" className="py-3 px-4">Holdout Acc</th>
                  <th scope="col" className="py-3 px-4">Holdout AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-pacs-800 text-slate-200">
                {Object.entries(clinicalModels).map(([name, info]: [string, any]) => {
                  const tm = info.test_metrics || {};
                  return (
                    <tr key={name} className="hover:bg-pacs-800/40">
                      <td data-label="Architecture" className="py-3 px-4 font-bold text-white">{name}</td>
                      <td data-label="5-Fold CV AUC" className="py-3 px-4">{info.cv_roc_auc_mean?.toFixed(4)}</td>
                      <td data-label="Holdout Acc" className="py-3 px-4">{tm.accuracy !== undefined ? `${(tm.accuracy * 100).toFixed(2)}%` : '--'}</td>
                      <td data-label="Holdout AUC" className="py-3 px-4 text-clinical-cyan font-bold">{tm.roc_auc?.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-4 bg-pacs-900 border border-pacs-700 rounded-2xl p-6 space-y-3">
          <h4 className="text-sm font-bold text-white">Top Cardiovascular Risk Features</h4>
          <div className="space-y-2 text-xs font-mono">
            {clinicalImp.slice(0, 8).map((item: any) => (
              <div key={item.feature} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-200">{item.feature}</span>
                  <span className="text-clinical-cyan">{(item.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 bg-pacs-950 rounded-full overflow-hidden">
                  <div style={{ width: `${Math.min(100, item.importance * 400)}%` }} className="h-full bg-clinical-cyan" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
