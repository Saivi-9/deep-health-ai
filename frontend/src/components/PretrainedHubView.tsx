import React, { useState, useEffect } from 'react';
import { Sparkles, Search, ExternalLink, Play, Cpu } from 'lucide-react';
import type {
  PretrainedCatalogEntry,
  HFSearchResult,
  VisionSample,
  ECGSample,
  ClinicalPreset,
} from '../types';
import {
  fetchPretrainedCatalog,
  searchPretrainedModels,
  inferPretrainedVision,
  inferPretrainedECG,
  inferPretrainedClinical,
} from '../api/client';

interface PretrainedHubViewProps {
  visionSamples: VisionSample[];
  ecgSamples: ECGSample[];
  clinicalPresets: ClinicalPreset[];
}

export const PretrainedHubView: React.FC<PretrainedHubViewProps> = ({
  visionSamples,
  ecgSamples,
  clinicalPresets,
}) => {
  const [catalog, setCatalog] = useState<PretrainedCatalogEntry[]>([]);
  const [benchmarks, setBenchmarks] = useState<Record<string, any> | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('chest xray pneumonia');
  const [searchResults, setSearchResults] = useState<HFSearchResult[]>([]);
  const [searching, setSearching] = useState<boolean>(false);

  // Live Pretrained Side-by-Side Test Lab
  const [selectedRepo, setSelectedRepo] = useState<string>('Purino/mobilenetv2-chest-xray-pneumonia-256-fullframe-gray');
  const [labResult, setLabResult] = useState<Record<string, any> | null>(null);
  const [runningLab, setRunningLab] = useState<boolean>(false);

  useEffect(() => {
    fetchPretrainedCatalog()
      .then((res) => {
        setCatalog(res.catalog);
        setBenchmarks(res.benchmarks);
      })
      .catch(() => {});
  }, []);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await searchPretrainedModels(searchQuery.trim());
      setSearchResults(res.results);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const runLiveHFInference = async (repoId: string) => {
    setSelectedRepo(repoId);
    setRunningLab(true);
    setLabResult(null);
    try {
      const entry = catalog.find((c) => c.repo_id === repoId);
      if (entry?.domain === 'vision' && visionSamples.length > 0) {
        const sample = visionSamples[0];
        const blob = await (await fetch(sample.image_base64)).blob();
        const file = new File([blob], `${sample.id}.png`, { type: 'image/png' });
        const out = await inferPretrainedVision(file, repoId);
        setLabResult({ domain: 'vision', sampleTitle: sample.title, groundTruth: sample.ground_truth, ...out });
      } else if (entry?.domain === 'ecg' && ecgSamples.length > 0) {
        const mit = ecgSamples.find((s) => s.id === 'mitbih_V') || ecgSamples[0];
        const out = await inferPretrainedECG(mit.signal, repoId);
        setLabResult({ domain: 'ecg', sampleTitle: mit.title, ...out });
      } else if (entry?.domain === 'clinical' && clinicalPresets.length > 0) {
        const pt = clinicalPresets[clinicalPresets.length - 1];
        const out = await inferPretrainedClinical(pt.features);
        setLabResult({ domain: 'clinical', sampleTitle: pt.title, trueOutcome: pt.true_outcome, ...out });
      }
    } catch (err) {
      setLabResult({ error: err instanceof Error ? err.message : 'HF Inference Error' });
    } finally {
      setRunningLab(false);
    }
  };

  const getBenchmarkBadge = (entry: PretrainedCatalogEntry) => {
    if (!benchmarks) return null;
    if (entry.domain === 'vision' && benchmarks.vision?.[entry.repo_id]) {
      const m = benchmarks.vision[entry.repo_id];
      return `Real PneumoniaMNIST: ${(m.accuracy * 100).toFixed(1)}% Acc • ${m.roc_auc?.toFixed(3)} AUC`;
    }
    if (entry.domain === 'ecg' && benchmarks.ecg?.[entry.repo_id]) {
      const m = benchmarks.ecg[entry.repo_id];
      return `MIT-BIH Holdout: ${(m.multiclass_accuracy * 100).toFixed(1)}% Acc • F1 ${m.macro_f1?.toFixed(3)}`;
    }
    if (entry.domain === 'clinical' && benchmarks.clinical?.[entry.repo_id]) {
      const m = benchmarks.clinical[entry.repo_id];
      return `Real Cardiology Split: ${(m.accuracy * 100).toFixed(1)}% Acc • ${m.roc_auc?.toFixed(3)} AUC`;
    }
    return null;
  };

  return (
    <div id="panel-pretrained" role="tabpanel" aria-labelledby="tab-pretrained" className="space-y-6">
      <div className="bg-pacs-900 border border-pacs-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-pacs-800 border border-pacs-600 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-amber-300" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">HUGGING FACE MEDICAL AI PRE-TRAINED HUB</div>
            <div className="text-pacs-300 font-mono">
              5 Verified Open-Weights Medical Checkpoints + Live Hugging Face Hub Discovery &amp; Execution
            </div>
          </div>
        </div>
        <span className="px-3 py-1 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30 font-mono font-bold">
          5 VERIFIED MODELS + LIVE HF SEARCH
        </span>
      </div>

      {/* Verified Pre-Trained Catalog Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {catalog.map((entry) => {
          const benchText = getBenchmarkBadge(entry);
          const isSelected = selectedRepo === entry.repo_id;
          return (
            <div
              key={entry.repo_id}
              className={`bg-pacs-900 border rounded-2xl p-4 flex flex-col justify-between gap-3 transition ${
                isSelected ? 'border-clinical-cyan' : 'border-pacs-700'
              }`}
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                  <span className="uppercase px-2 py-0.5 rounded bg-pacs-800 text-clinical-cyan border border-pacs-600 font-bold">
                    {entry.domain}
                  </span>
                  <span className="text-pacs-400">{entry.license}</span>
                </div>
                <h3 className="text-sm font-extrabold text-white leading-snug">{entry.display_name}</h3>
                <a
                  href={`https://huggingface.co/${entry.repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-[11px] font-mono text-clinical-cyan hover:underline break-all"
                >
                  {entry.repo_id} <ExternalLink className="w-3 h-3 shrink-0" />
                </a>
                <p className="text-xs text-pacs-300 leading-relaxed">{entry.summary}</p>
              </div>

              <div className="space-y-2 pt-2 border-t border-pacs-800">
                {benchText && (
                  <div className="text-[11px] font-mono text-clinical-green bg-clinical-green/10 border border-clinical-green/30 rounded-lg px-2.5 py-1.5">
                    {benchText}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => runLiveHFInference(entry.repo_id)}
                  className="w-full py-2 px-3 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Play className="w-3.5 h-3.5 text-amber-300" />
                  Run Live Pre-Trained Check
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Live Inference Result Drawer & Live HF Search Side-by-Side */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-pacs-900 border border-pacs-700 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-pacs-700/60 pb-2.5">
            <span className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-clinical-cyan" />
              Live Pre-Trained Execution Output
            </span>
            <span className="text-[11px] font-mono text-pacs-400 truncate max-w-[220px]">{selectedRepo}</span>
          </div>

          {runningLab ? (
            <div className="py-10 flex flex-col items-center gap-2 text-clinical-cyan text-xs font-mono">
              <div className="w-7 h-7 border-2 border-clinical-cyan border-t-transparent rounded-full animate-spin" />
              <span>Running cached Hugging Face checkpoint...</span>
            </div>
          ) : labResult ? (
            <div className="space-y-3 text-xs">
              {labResult.error ? (
                <div className="text-clinical-rose font-mono">{labResult.error}</div>
              ) : (
                <>
                  <div className="bg-pacs-950 p-3 rounded-xl border border-pacs-700 font-mono space-y-1">
                    <div className="text-pacs-400">TEST SAMPLE: <span className="text-white font-bold">{labResult.sampleTitle}</span></div>
                    <div className="text-pacs-400">PREDICTION: <span className="text-clinical-cyan font-bold">{labResult.diagnosis || labResult.predicted_label}</span></div>
                    {labResult.confidence !== undefined && (
                      <div className="text-pacs-400">CONFIDENCE: <span className="text-clinical-green font-bold">{(labResult.confidence * 100).toFixed(2)}%</span></div>
                    )}
                    {labResult.risk_probability !== undefined && (
                      <div className="text-pacs-400">CARDIAC RISK PROB: <span className="text-amber-300 font-bold">{(labResult.risk_probability * 100).toFixed(2)}%</span></div>
                    )}
                  </div>
                  {labResult.gradcam_overlay_base64 && (
                    <div className="flex items-center gap-4 bg-pacs-950 p-3 rounded-xl border border-pacs-700">
                      <img src={labResult.gradcam_overlay_base64} alt="HF Model Grad-CAM" className="w-28 h-28 rounded-lg border border-pacs-600" />
                      <div className="text-[11px] font-mono text-pacs-300 space-y-1">
                        <div className="text-white font-bold">Live HF Grad-CAM Attribution</div>
                        <div>Grade: {labResult.severity?.radiological_grade}</div>
                        <div>Burden: {labResult.severity?.infiltrate_burden_percentage}%</div>
                      </div>
                    </div>
                  )}
                  {labResult.class_probabilities && (
                    <div className="grid grid-cols-5 gap-1.5 font-mono text-[11px]">
                      {Object.entries(labResult.class_probabilities).map(([k, v]: [string, any]) => (
                        <div key={k} className="bg-pacs-950 border border-pacs-700 rounded-lg p-2 text-center">
                          <div className="text-pacs-400">{k}</div>
                          <div className="text-white font-bold">{(Number(v) * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-pacs-400 font-mono">
              Click any verified model card above to execute live inference on real cached clinical data.
            </div>
          )}
        </div>

        {/* Live Hugging Face Hub API Search */}
        <div className="lg:col-span-6 bg-pacs-900 border border-pacs-700 rounded-2xl p-5 space-y-4">
          <div className="text-xs uppercase font-bold tracking-wider text-pacs-300 flex items-center gap-1.5">
            <Search className="w-4 h-4 text-amber-300" />
            Live Hugging Face Medical Model Discovery
          </div>
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search e.g. chest xray pneumonia, ecg arrhythmia, cardiology..."
              className="flex-1 bg-pacs-950 border border-pacs-700 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-clinical-cyan"
            />
            <button
              type="submit"
              disabled={searching}
              className="px-4 py-2 rounded-xl bg-clinical-cyan text-pacs-950 font-bold text-xs hover:bg-cyan-400 transition"
            >
              {searching ? 'Searching...' : 'Search HF'}
            </button>
          </form>

          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
            {searchResults.map((r) => (
              <div key={r.repo_id} className="bg-pacs-950 border border-pacs-800 rounded-xl p-3 flex items-center justify-between gap-2 text-xs">
                <div className="min-w-0">
                  <a href={r.url} target="_blank" rel="noreferrer" className="font-mono font-bold text-clinical-cyan hover:underline truncate block">
                    {r.repo_id}
                  </a>
                  <div className="text-[11px] text-pacs-400 font-mono">
                    {r.pipeline_tag} &bull; {r.downloads.toLocaleString()} downloads &bull; {r.likes} likes
                  </div>
                </div>
                {r.verified_in_studio && (
                  <span className="px-2 py-0.5 rounded bg-clinical-green/20 text-clinical-green border border-clinical-green/30 text-[10px] font-bold shrink-0">
                    INTEGRATED
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
