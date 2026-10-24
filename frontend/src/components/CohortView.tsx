import React, { useState } from 'react';
import { Download, Search } from 'lucide-react';

interface CohortPatient {
  id: string;
  name: string;
  age: number;
  modality: 'Chest X-Ray' | 'Lead-II ECG';
  diagnosis: string;
  confidence: number;
  triage: 'STAT' | 'EXPEDITED' | 'ROUTINE';
  variance: number;
}

export const CohortView: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterModality, setFilterModality] = useState<'all' | 'Chest X-Ray' | 'Lead-II ECG'>('all');

  const cohortData: CohortPatient[] = [
    { id: 'PT-101', name: 'John Doe', age: 54, modality: 'Chest X-Ray', diagnosis: 'Alveolar Consolidation', confidence: 0.998, triage: 'STAT', variance: 0.0018 },
    { id: 'PT-102', name: 'Sarah Connor', age: 38, modality: 'Lead-II ECG', diagnosis: 'Ventricular Ectopy (PVC)', confidence: 0.985, triage: 'EXPEDITED', variance: 0.0031 },
    { id: 'PT-103', name: 'Robert Chen', age: 62, modality: 'Chest X-Ray', diagnosis: 'Normal Healthy Parenchyma', confidence: 0.994, triage: 'ROUTINE', variance: 0.0009 },
    { id: 'PT-104', name: 'Maria Garcia', age: 47, modality: 'Lead-II ECG', diagnosis: 'ST-Elevation Ischemia', confidence: 0.992, triage: 'STAT', variance: 0.0022 },
    { id: 'PT-105', name: 'David Kim', age: 29, modality: 'Chest X-Ray', diagnosis: 'Normal Healthy Parenchyma', confidence: 0.989, triage: 'ROUTINE', variance: 0.0012 },
    { id: 'PT-106', name: 'Elena Rostova', age: 71, modality: 'Chest X-Ray', diagnosis: 'Multi-Focal Consolidation', confidence: 0.997, triage: 'STAT', variance: 0.0025 },
    { id: 'PT-107', name: 'James Wilson', age: 65, modality: 'Lead-II ECG', diagnosis: 'Normal Sinus Rhythm', confidence: 0.991, triage: 'ROUTINE', variance: 0.0007 },
  ];

  const filtered = cohortData.filter((pt) => {
    const matchesSearch = pt.name.toLowerCase().includes(searchTerm.toLowerCase()) || pt.id.toLowerCase().includes(searchTerm.toLowerCase()) || pt.diagnosis.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesMod = filterModality === 'all' || pt.modality === filterModality;
    return matchesSearch && matchesMod;
  });

  const exportCsv = () => {
    const headers = 'ID,Name,Age,Modality,Diagnosis,Confidence,Triage,Predictive_Variance\n';
    const rows = filtered.map(p => `${p.id},"${p.name}",${p.age},"${p.modality}","${p.diagnosis}",${p.confidence},${p.triage},${p.variance}`).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `Cohort_Summary_${Date.now()}.csv`;
    link.click();
  };

  return (
    <div className="space-y-6">
      {/* Metric Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4">
          <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider">Total Evaluated Cohort</div>
          <div className="text-2xl font-black text-white mt-1">7 Patients</div>
          <div className="text-xs text-clinical-cyan font-mono mt-1">100% Ingestion Success</div>
        </div>

        <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4">
          <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider">Thoracic Consolidation</div>
          <div className="text-2xl font-black text-clinical-rose mt-1">50.0%</div>
          <div className="text-xs text-pacs-400 font-mono mt-1">2 Positive of 4 X-Rays</div>
        </div>

        <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4">
          <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider">Cardiac Arrhythmia Burden</div>
          <div className="text-2xl font-black text-clinical-amber mt-1">66.7%</div>
          <div className="text-xs text-pacs-400 font-mono mt-1">2 Anomaly of 3 Waveforms</div>
        </div>

        <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4">
          <div className="text-[10px] text-pacs-400 uppercase font-bold tracking-wider">Mean Inference Latency</div>
          <div className="text-2xl font-black text-clinical-green mt-1">24.5 ms</div>
          <div className="text-xs text-pacs-400 font-mono mt-1">PyTorch Native Engine</div>
        </div>
      </div>

      {/* Cohort Search & Filter Controls */}
      <div className="bg-pacs-900 border border-pacs-700 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-3 flex-1 max-w-md">
            <div className="relative w-full">
              <Search className="w-4 h-4 absolute left-3 top-2.5 text-pacs-400" />
              <input
                type="text"
                placeholder="Search by patient ID, name, or diagnosis..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-pacs-950 border border-pacs-700 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-pacs-400 outline-none focus:border-clinical-cyan"
              />
            </div>

            <select
              value={filterModality}
              onChange={(e) => setFilterModality(e.target.value as any)}
              className="bg-pacs-950 border border-pacs-700 rounded-xl px-3 py-2 text-xs text-slate-200 outline-none"
            >
              <option value="all">All Modalities</option>
              <option value="Chest X-Ray">Chest Radiographs</option>
              <option value="Lead-II ECG">ECG Telemetry</option>
            </select>
          </div>

          <button
            onClick={exportCsv}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 text-xs font-semibold transition"
          >
            <Download className="w-3.5 h-3.5 text-clinical-cyan" />
            Export Cohort CSV
          </button>
        </div>

        {/* Data Table */}
        <div className="overflow-x-auto rounded-xl border border-pacs-700">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-pacs-950 text-pacs-400 uppercase text-[10px] tracking-wider border-b border-pacs-700">
              <tr>
                <th className="py-3 px-4">Patient ID</th>
                <th className="py-3 px-4">Subject</th>
                <th className="py-3 px-4">Modality</th>
                <th className="py-3 px-4">AI Diagnosis</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Triage Priority</th>
                <th className="py-3 px-4">Epistemic Variance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-pacs-800 text-slate-200">
              {filtered.map((pt) => (
                <tr key={pt.id} className="hover:bg-pacs-800/40 transition">
                  <td className="py-3 px-4 font-bold text-clinical-cyan">{pt.id}</td>
                  <td className="py-3 px-4 text-white font-sans">{pt.name} ({pt.age}Y)</td>
                  <td className="py-3 px-4 text-pacs-300">{pt.modality}</td>
                  <td className="py-3 px-4 text-white">{pt.diagnosis}</td>
                  <td className="py-3 px-4">{(pt.confidence * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4">
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                      pt.triage === 'STAT' ? 'bg-clinical-rose/20 text-clinical-rose border border-clinical-rose/30' :
                      (pt.triage === 'EXPEDITED' ? 'bg-clinical-amber/20 text-clinical-amber border border-clinical-amber/30' :
                      'bg-clinical-green/20 text-clinical-green border border-clinical-green/30')
                    }`}>
                      {pt.triage}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-pacs-300">{pt.variance.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
