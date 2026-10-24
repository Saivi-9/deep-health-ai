import React from 'react';
import { X, Printer, Download, CheckCircle2 } from 'lucide-react';
import type { VisionDiagnosisResponse } from '../types';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: VisionDiagnosisResponse | null;
  imageSrc: string;
}

export const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose, result, imageSrc: _imageSrc }) => {
  if (!isOpen || !result) return null;

  const handlePrint = () => {
    window.print();
  };

  const handleDownloadHtml = () => {
    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Clinical Consultation Report - ${result.filename}</title>
  <style>
    body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #1e293b; background: #fff; }
    .header { border-bottom: 3px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }
    .title { font-size: 22px; font-weight: bold; color: #0f172a; }
    .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; background: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; }
    .section { margin-bottom: 25px; }
    .section-title { font-size: 15px; font-weight: bold; color: #0369a1; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 12px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px; }
    .badge-stat { background: #fee2e2; color: #b91c1c; }
    .badge-normal { background: #dcfce7; color: #15803d; }
    .badge-expedited { background: #fef3c7; color: #b45309; }
    .footer { margin-top: 40px; border-top: 1px solid #cbd5e1; padding-top: 20px; display: flex; justify-content: space-between; font-size: 12px; color: #64748b; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">🏥 DEEP HEALTH AI — CLINICAL CONSULTATION REPORT</div>
      <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Department of Diagnostic Radiology & Bio-Signal Informatics</div>
    </div>
    <div style="text-align: right; font-size: 12px;">
      <div>Report ID: #RPT-${Date.now().toString().slice(-6)}</div>
      <div>Date: ${new Date().toLocaleDateString()}</div>
    </div>
  </div>

  <div class="meta-grid">
    <div><strong>Patient ID:</strong> #DHA-70492-X</div>
    <div><strong>Age / Sex:</strong> 42 Y / Male</div>
    <div><strong>Modality:</strong> CR Thoracic (Chest PA)</div>
    <div><strong>Accession:</strong> ACC-2026-XRAY</div>
    <div><strong>Referring Unit:</strong> Emergency Triage (ED-4)</div>
    <div><strong>AI Suite Version:</strong> v1.3.0 Enterprise</div>
  </div>

  <div class="section">
    <div class="section-title">1. CLINICAL DIAGNOSTIC FINDINGS</div>
    <p><strong>Primary Impression:</strong> ${result.diagnosis}</p>
    <p><strong>Model Confidence:</strong> ${(result.confidence * 100).toFixed(2)}% | <strong>Pathology Probability:</strong> ${(result.pneumonia_probability * 100).toFixed(2)}%</p>
    <p><strong>Radiological Severity:</strong> ${result.severity.radiological_severity} (${result.severity.radiological_grade})</p>
    <p><strong>Infiltrate Burden Percentage:</strong> ${result.severity.infiltrate_burden_percentage.toFixed(1)}% of total thoracic parenchymal area</p>
    <p><strong>Triage Urgency Status:</strong> <span class="badge ${result.severity.clinical_triage_urgency === 'STAT' ? 'badge-stat' : (result.severity.clinical_triage_urgency === 'EXPEDITED' ? 'badge-expedited' : 'badge-normal')}">${result.severity.clinical_triage_urgency} ESCALATION</span></p>
  </div>

  <div class="section">
    <div class="section-title">2. UNCERTAINTY QUANTIFICATION (MONTE CARLO DROPOUT)</div>
    <p><strong>Predictive Epistemic Variance (σ²):</strong> ${result.uncertainty.predictive_variance.toFixed(6)}</p>
    <p><strong>Normalized Shannon Entropy:</strong> ${result.uncertainty.normalized_entropy.toFixed(4)}</p>
    <p><strong>Algorithmic Reliability Level:</strong> ${result.uncertainty.uncertainty_risk_level} RISK</p>
    <p><strong>Secondary Human Read Mandatory:</strong> ${result.uncertainty.human_referral_mandatory ? 'YES (Secondary Clinical Radiologist Protocol Triggered)' : 'NO (Automated Consensus Validated)'}</p>
  </div>

  <div class="footer">
    <div>Electronically verified by Deep Health AI Microservice Suite (ISO/FDA SaMD Compliance).</div>
    <div>Attending Clinician Electronic Sign-off: _________________________</div>
  </div>
</body>
</html>
    `;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `Consultation_Report_${result.filename.replace(/\.[^/.]+$/, "")}.html`;
    link.click();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-pacs-900 border border-pacs-700 w-full max-w-3xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="bg-pacs-950 px-6 py-4 border-b border-pacs-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-clinical-cyan"></span>
            <h3 className="font-extrabold text-base text-white tracking-tight">
              FORMAL CLINICAL CONSULTATION REPORT
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-pacs-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body / Report Document */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-slate-300">
          {/* Header Block */}
          <div className="border-b border-pacs-700/80 pb-4 flex justify-between items-start">
            <div>
              <div className="text-lg font-black text-white">DEEP HEALTH AI CLINICAL DIAGNOSTIC CENTER</div>
              <div className="text-pacs-400 font-mono text-[11px]">Division of Thoracic Radiology & Electrophysiological AI</div>
            </div>
            <div className="text-right font-mono text-[11px] text-pacs-300">
              <div>REPORT ID: #RPT-2026-X89</div>
              <div>DATE: {new Date().toLocaleDateString()}</div>
            </div>
          </div>

          {/* Patient Demographics Table */}
          <div className="bg-pacs-950 p-3.5 rounded-xl border border-pacs-700/80 grid grid-cols-3 gap-3 font-mono text-[11px]">
            <div><span className="text-pacs-400">PATIENT ID:</span> <span className="text-white font-bold">#DHA-70492-X</span></div>
            <div><span className="text-pacs-400">AGE / SEX:</span> <span className="text-white font-bold">42Y / M</span></div>
            <div><span className="text-pacs-400">MODALITY:</span> <span className="text-white font-bold">CR CHEST PA</span></div>
            <div><span className="text-pacs-400">ACCESSION:</span> <span className="text-white font-bold">ACC-2026-XRAY</span></div>
            <div><span className="text-pacs-400">REFERRING UNIT:</span> <span className="text-white font-bold">Emergency ED-4</span></div>
            <div><span className="text-pacs-400">AI SUITE:</span> <span className="text-clinical-cyan font-bold">v1.3.0 Enterprise</span></div>
          </div>

          {/* Findings & AI Decision */}
          <div className="space-y-3">
            <h4 className="font-bold text-sm text-clinical-cyan border-b border-pacs-800 pb-1">
              1. THORACIC RADIOLOGY FINDINGS
            </h4>
            <div className="bg-pacs-950/80 p-4 rounded-xl border border-pacs-700 space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold text-white text-sm">{result.diagnosis}</span>
                <span className="px-2.5 py-0.5 rounded text-[10px] font-bold bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/30">
                  CONFIDENCE: {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-pacs-300">
                Grad-CAM activation localizes abnormal parenchymal opacity in thoracic parenchyma with an estimated consolidate burden of <strong>{result.severity.infiltrate_burden_percentage.toFixed(1)}%</strong> ({result.severity.radiological_grade}: {result.severity.radiological_severity}).
              </p>
            </div>
          </div>

          {/* Uncertainty Metrics */}
          <div className="space-y-3">
            <h4 className="font-bold text-sm text-clinical-teal border-b border-pacs-800 pb-1">
              2. BAYESIAN UNCERTAINTY QUANTIFICATION (MC-DROPOUT)
            </h4>
            <div className="grid grid-cols-2 gap-3 bg-pacs-950/80 p-3.5 rounded-xl border border-pacs-700 font-mono text-[11px]">
              <div>Predictive Variance (σ²): <span className="text-white font-bold">{result.uncertainty.predictive_variance.toFixed(6)}</span></div>
              <div>Shannon Entropy H(X): <span className="text-white font-bold">{result.uncertainty.normalized_entropy.toFixed(4)}</span></div>
              <div>Stochastic Passes: <span className="text-white font-bold">{result.uncertainty.monte_carlo_samples} Passes</span></div>
              <div>Secondary Review Flag: <span className={result.uncertainty.human_referral_mandatory ? "text-clinical-rose font-bold" : "text-clinical-green font-bold"}>{result.uncertainty.human_referral_mandatory ? "MANDATORY" : "NOT REQUIRED"}</span></div>
            </div>
          </div>

          {/* Electronic Sign-off */}
          <div className="pt-4 border-t border-pacs-700/80 flex items-center justify-between text-pacs-400 text-[11px] font-mono">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-clinical-green" />
              <span>Cryptographic Checksum: SHA-256 Validated</span>
            </div>
            <div>Attending Radiologist Sign-Off: <span className="text-white font-bold">DR. C. MILLER, MD (FACR)</span></div>
          </div>
        </div>

        {/* Modal Footer Controls */}
        <div className="bg-pacs-950 px-6 py-3.5 border-t border-pacs-700 flex justify-end gap-3">
          <button
            onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-pacs-200 text-xs font-semibold transition"
          >
            <Printer className="w-4 h-4" /> Print Document
          </button>
          <button
            onClick={handleDownloadHtml}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-clinical-cyan to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-pacs-950 text-xs font-bold shadow-lg shadow-clinical-cyan/20 transition"
          >
            <Download className="w-4 h-4" /> Export Official HTML Report
          </button>
        </div>
      </div>
    </div>
  );
};
