import type { HealthResponse, VisionDiagnosisResponse, ECGScreenResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend offline, using fallback state:', err);
    return {
      status: 'OFFLINE',
      service: 'Deep Health AI Microservice (Standby)',
      version: '1.3.0',
      device: 'cpu',
      models_loaded: {
        ResidualChestCNN: true,
        Conv1dAutoencoder: true,
      },
    };
  }
}

export async function diagnoseVisionScan(
  file: File,
  mcSamples: number = 20
): Promise<VisionDiagnosisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(
      `${API_BASE_URL}/v1/vision/diagnose?mc_samples=${mcSamples}`,
      {
        method: 'POST',
        body: formData,
      }
    );
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error('Vision diagnosis error:', err);
    throw err;
  }
}

export async function screenECGSignal(
  signal: number[],
  samplingRate: number = 250,
  thresholdOverride?: number
): Promise<ECGScreenResponse> {
  const payload = {
    signal,
    sampling_rate_hz: samplingRate,
    threshold_override: thresholdOverride,
  };

  try {
    const res = await fetch(`${API_BASE_URL}/v1/ecg/screen`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error('ECG screening error:', err);
    throw err;
  }
}
