import React, { useState, useEffect } from 'react';
import { Download, Search, Play, RefreshCw } from 'lucide-react';
import { evaluateLiveCohort } from '../api/client';

export const CohortView: React.FC = () => {
  const [modality, setModality] = useState<'vision_real' | 'vision' | 'ecg' | 'clinical'>('vision_real');
  const [limit, setLimit] = useState<number>(20);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [summary, setSummary] = useState<Record<string, any> | null>(null);
  const [records, setRecords] = useState<Record<string, any>[]>([]);

  const runCohort = async (mod: 'vision_real' | 'vision' | 'ecg' | 'clinical', lim: number) => {
    setLoading(true);
    try {
      const res = await evaluateLiveCohort(mod, lim);
      setSummary(res.summary);
      setRecords(res.records);
    } catch {
      setSummary(null);
      setRecords([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    evaluateLiveCohort(modality, limit)
      .then((res) => {
        if (active) {
          setSummary(res.summary);
          setRecords(res.records);
        }
      })
      .catch(() => {
        if (active) {
          setSummary(null);
          setRecords([]);
        }
      });
    return () => {
      active = false;
    };
  }, [modality, limit]);

  const filtered = records.filter((row) =>
    JSON.stringify(row).toLowerCase().includes(searchTerm.toLowerCase())
  );

  const columns = records.length > 0 ? Object.keys(records[0]).filter((c) => c !== 'file_path').slice(0, 7) : [];

  const exportCsv = () => {
    if (filtered.length === 0) return;
    const cols = Object.keys(filtered[0]);
    const header = cols.join(',') + '\n';
    const body = filtered.map((r) => cols.map((c) => JSON.stringify(r[c] ?? '')).join(',')).join('\n');
    const blob = new Blob([header + body], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `deep_health_cohort_${modality}_${Date.now()}.csv`;
    link.click();
  };

  return (
    <div id="panel-batch" role="tabpanel" aria-labelledby="tab-batch" className="space-y-6">
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-pacs-400">COHORT SOURCE:</span>
          <select
            aria-label="Batch Cohort Modality"
            value={modality}
            onChange={(e) => setModality(e.target.value as any)}
            className="bg-pacs-950 border border-pacs-600 rounded-lg px-3 py-1.5 text-white outline-none"
          >
            <option value="vision_real">Real PneumoniaMNIST Test Set (Hospital CXR)</option>
            <option value="vision">Generated Thoracic Radiography Cohort</option>
            <option value="ecg">Lead-II 5-Class ECG Holdout Beats</option>
            <option value="clinical">Real Multi-Center Cardiology EHR Patients</option>
          </select>

          <select
            aria-label="Cohort Sample Size"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="bg-pacs-950 border border-pacs-600 rounded-lg px-2.5 py-1.5 text-clinical-cyan font-mono"
          >
            <option value={15}>N=15</option>
            <option value={25}>N=25</option>
            <option value={40}>N=40</option>
          </select>
        </div>

        <button
          type="button"
          onClick={() => runCohort(modality, limit)}
          disabled={loading}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-clinical-cyan text-pacs-950 font-bold hover:bg-cyan-400 transition"
        >
          {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {loading ? 'Evaluating Live...' : 'Run Live Batch Evaluation'}
        </button>
      </div>

      {summary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {Object.entries(summary).slice(0, 4).map(([k, v]) => (
            <div key={k} className="bg-pacs-900 border border-pacs-700 rounded-xl p-4">
              <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider">{k.replace(/_/g, ' ')}</div>
              <div className="text-xl font-black text-white mt-1 truncate">{String(v)}</div>
            </div>
          ))}
        </div>
      )}

      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-pacs-400" />
            <input
              type="text"
              placeholder="Filter live evaluation records..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-pacs-950 border border-pacs-700 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-pacs-400 outline-none focus:border-clinical-cyan"
            />
          </div>

          <button
            type="button"
            onClick={exportCsv}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 text-xs font-semibold transition"
          >
            <Download className="w-3.5 h-3.5 text-clinical-cyan" /> Export Cohort CSV
          </button>
        </div>

        <div className="table-wrapper rounded-xl border border-pacs-700">
          <table className="pacs-responsive-table text-left text-xs font-mono">
            <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] tracking-wider border-b border-pacs-700">
              <tr>
                {columns.map((col) => (
                  <th key={col} scope="col" className="py-3 px-4">
                    {col.replace(/_/g, ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-pacs-800 text-slate-200">
              {filtered.map((row, idx) => (
                <tr key={idx} className="hover:bg-pacs-800/40 transition">
                  {columns.map((col) => (
                    <td key={col} data-label={col.replace(/_/g, ' ')} className="py-2.5 px-4">
                      {String(row[col] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
