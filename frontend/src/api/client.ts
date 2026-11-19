import type {
  HealthResponse,
  VisionDiagnosisResponse,
  ECGScreenResponse,
  DatasetSamplesResponse,
  ClinicalFeaturesInput,
  ClinicalTriageResponse,
  PretrainedCatalogEntry,
  HFSearchResult,
} from '../types';

const API_BASE_URL = window.location.port === '5173' ? 'http://localhost:8000' : '';

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return {
      status: 'OFFLINE',
      service: 'Deep Health AI Microservice (Standby)',
      version: '2.0.0',
      device: 'cpu',
      models_loaded: {
        ResidualChestCNN: false,
        Conv1dAutoencoder: false,
      },
    };
  }
}

export async function fetchDatasetSamples(): Promise<DatasetSamplesResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/dataset/samples`);
  if (!res.ok) throw new Error(`Failed to fetch clinical samples (${res.status})`);
  return await res.json();
}

export async function fetchBenchmarks(): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/v1/benchmarks`);
  if (!res.ok) throw new Error(`Failed to fetch benchmarks (${res.status})`);
  return await res.json();
}

export async function diagnoseVisionScan(
  file: File,
  mcSamples: number = 20,
  modelId: string = 'residual_cnn',
  colormap: string = 'turbo'
): Promise<VisionDiagnosisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(
    `${API_BASE_URL}/v1/vision/diagnose?mc_samples=${mcSamples}&model_id=${encodeURIComponent(modelId)}&colormap=${encodeURIComponent(colormap)}`,
    { method: 'POST', body: formData }
  );
  if (!res.ok) {
    const errJson = await res.json().catch(() => ({}));
    throw new Error(errJson.detail || `Server returned ${res.status}`);
  }
  return await res.json();
}

export async function screenECGSignal(
  signal: number[],
  samplingRate: number = 250,
  thresholdOverride?: number,
  modelId: string = 'autoencoder'
): Promise<ECGScreenResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/ecg/screen`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      signal,
      sampling_rate_hz: samplingRate,
      threshold_override: thresholdOverride,
      model_id: modelId,
    }),
  });
  if (!res.ok) {
    const errJson = await res.json().catch(() => ({}));
    throw new Error(errJson.detail || `Server returned ${res.status}`);
  }
  return await res.json();
}

export async function runClinicalTriage(payload: ClinicalFeaturesInput): Promise<ClinicalTriageResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/clinical/triage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errJson = await res.json().catch(() => ({}));
    throw new Error(errJson.detail || `Server returned ${res.status}`);
  }
  return await res.json();
}

export async function evaluateLiveCohort(
  modality: 'vision_real' | 'vision' | 'ecg' | 'clinical',
  limit: number = 20
): Promise<{ summary: Record<string, any>; records: Record<string, any>[] }> {
  const res = await fetch(`${API_BASE_URL}/v1/batch/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modality, limit }),
  });
  if (!res.ok) {
    const errJson = await res.json().catch(() => ({}));
    throw new Error(errJson.detail || `Server returned ${res.status}`);
  }
  return await res.json();
}

export async function fetchPretrainedCatalog(): Promise<{
  catalog: PretrainedCatalogEntry[];
  benchmarks: Record<string, any> | null;
}> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/catalog`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function searchPretrainedModels(query: string): Promise<{ query: string; results: HFSearchResult[] }> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/search?q=${encodeURIComponent(query)}&limit=10`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function inferPretrainedVision(file: File, repoId: string): Promise<Record<string, any>> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/vision?repo_id=${encodeURIComponent(repoId)}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function inferPretrainedECG(signal: number[], repoId: string): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/ecg`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ signal, repo_id: repoId }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function inferPretrainedClinical(payload: ClinicalFeaturesInput): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/clinical`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}
