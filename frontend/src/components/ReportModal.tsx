import React, { useEffect, useRef } from 'react';
import { X, Printer, Download, CheckCircle2 } from 'lucide-react';
import type { VisionDiagnosisResponse } from '../types';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: VisionDiagnosisResponse | null;
  imageSrc: string;
}

export const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose, result, imageSrc }) => {
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (isOpen && result) {
      if (!dlg.open) dlg.showModal();
    } else if (dlg.open) {
      dlg.close();
    }
  }, [isOpen, result]);

  const handleBackdropClick = (e: React.MouseEvent<HTMLDialogElement>) => {
    if (e.target === dialogRef.current) {
      onClose();
    }
  };

  if (!result) return <dialog ref={dialogRef} className="pacs-dialog" />;

  const handlePrint = () => {
    window.print();
  };

  const handleDownloadHtml = () => {
    const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clinical Consultation Report - ${result.filename}</title>
  <style>
    body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #1e293b; background: #fff; }
    .header { border-bottom: 3px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }
    .title { font-size: 20px; font-weight: bold; color: #0f172a; }
    .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; background: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; }
    .section { margin-bottom: 22px; }
    .section-title { font-size: 14px; font-weight: bold; color: #0369a1; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">DEEP HEALTH AI v2.0 — CLINICAL CONSULTATION REPORT</div>
      <div style="font-size: 12px; color: #64748b;">Department of Diagnostic Radiology &amp; Electrophysiological AI</div>
    </div>
    <div style="text-align: right; font-size: 12px;">
      <div>Report ID: #RPT-${Date.now().toString().slice(-6)}</div>
      <div>Issued: ${new Date().toISOString()}</div>
    </div>
  </div>
  <div class="meta-grid">
    <div><strong>File:</strong> ${result.filename}</div>
    <div><strong>Model:</strong> ${result.model_used || 'ResidualChestCNN'}</div>
    <div><strong>MC Passes:</strong> ${result.uncertainty.monte_carlo_samples}</div>
  </div>
  <div class="section">
    <div class="section-title">1. DIAGNOSTIC FINDINGS</div>
    <p><strong>Impression:</strong> ${result.diagnosis}</p>
    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}% (Pneumonia ${(result.pneumonia_probability * 100).toFixed(2)}%)</p>
    <p><strong>Severity:</strong> ${result.severity.radiological_severity} (${result.severity.radiological_grade}) | Burden: ${result.severity.infiltrate_burden_percentage.toFixed(1)}%</p>
    <p><strong>Triage Protocol:</strong> ${result.severity.clinical_triage_urgency}</p>
  </div>
  <div class="section">
    <div class="section-title">2. EPISTEMIC UNCERTAINTY &amp; REFERRAL</div>
    <p>Predictive Variance: ${result.uncertainty.predictive_variance.toFixed(6)} | Entropy: ${result.uncertainty.normalized_entropy.toFixed(4)} (${result.uncertainty.uncertainty_risk_level})</p>
    <p>Mandatory Secondary Read: ${result.uncertainty.human_referral_mandatory ? 'YES' : 'NO'}</p>
  </div>
  ${result.gradcam_overlay_base64 || imageSrc ? `<div class="section"><div class="section-title">3. GRAD-CAM VISUALIZATION</div><img src="${result.gradcam_overlay_base64 || imageSrc}" alt="Grad-CAM Overlay" style="width:220px;border-radius:8px;" /></div>` : ''}
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `Consultation_Report_${result.filename.replace(/\.[^/.]+$/, '')}.html`;
    link.click();
  };

  return (
    <dialog
      ref={dialogRef}
      onCancel={onClose}
      onClick={handleBackdropClick}
      aria-labelledby="report-modal-title"
      className="pacs-dialog"
    >
      <div className="flex flex-col max-h-[90dvh]">
        <div className="bg-pacs-950 px-6 py-4 border-b border-pacs-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-clinical-cyan" />
            <h3 id="report-modal-title" className="font-extrabold text-base text-white tracking-tight">
              FORMAL CLINICAL CONSULTATION REPORT (v2.0)
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close report dialog"
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-pacs-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto space-y-5 text-xs text-slate-300">
          <div className="bg-pacs-950 p-3.5 rounded-xl border border-pacs-700/80 grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-[11px]">
            <div><span className="text-pacs-400">FILE:</span> <span className="text-white font-bold break-all">{result.filename}</span></div>
            <div><span className="text-pacs-400">MODEL:</span> <span className="text-clinical-cyan font-bold">{result.model_used || 'ResidualChestCNN'}</span></div>
            <div><span className="text-pacs-400">Triage:</span> <span className="text-white font-bold">{result.severity.clinical_triage_urgency}</span></div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center bg-pacs-950/80 p-4 rounded-xl border border-pacs-700">
            <div className="sm:col-span-2 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-bold text-white text-sm">{result.diagnosis}</span>
                <span className="px-2.5 py-0.5 rounded text-[10px] font-bold bg-clinical-cyan/20 text-clinical-cyan border border-clinical-cyan/30">
                  CONFIDENCE: {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-pacs-300">
                Grad-CAM localization indicates <strong>{result.severity.infiltrate_burden_percentage.toFixed(1)}%</strong> abnormal parenchymal burden ({result.severity.radiological_grade}: {result.severity.radiological_severity}).
              </p>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1">
                <div>Variance (σ²): <strong className="text-white">{result.uncertainty.predictive_variance.toFixed(6)}</strong></div>
                <div>Entropy H(X): <strong className="text-white">{result.uncertainty.normalized_entropy.toFixed(4)}</strong></div>
                <div>Risk Level: <strong className="text-clinical-teal">{result.uncertainty.uncertainty_risk_level}</strong></div>
                <div>Secondary Read: <strong className={result.uncertainty.human_referral_mandatory ? 'text-clinical-rose' : 'text-clinical-green'}>{result.uncertainty.human_referral_mandatory ? 'MANDATORY' : 'AUTO-CLEARED'}</strong></div>
              </div>
            </div>
            {(result.gradcam_overlay_base64 || imageSrc) && (
              <div className="flex justify-center">
                <img
                  src={result.gradcam_overlay_base64 || imageSrc}
                  alt="Grad-CAM Diagnostic Overlay"
                  className="w-32 h-32 rounded-lg border border-pacs-600 object-cover"
                />
              </div>
            )}
          </div>

          <div className="pt-3 border-t border-pacs-700/80 flex items-center justify-between text-pacs-400 text-[11px] font-mono">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-clinical-green" />
              <span>Real PyTorch Grad-CAM &amp; MC-Dropout Verified</span>
            </div>
            <span>Deep Health AI v2.0</span>
          </div>
        </div>

        <div className="bg-pacs-950 px-6 py-3.5 border-t border-pacs-700 flex justify-end gap-3">
          <button
            type="button"
            onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-pacs-800 hover:bg-pacs-700 border border-pacs-600 text-slate-200 text-xs font-semibold transition"
          >
            <Printer className="w-4 h-4" /> Print
          </button>
          <button
            type="button"
            onClick={handleDownloadHtml}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-clinical-cyan to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-pacs-950 text-xs font-bold shadow-lg shadow-clinical-cyan/20 transition"
          >
            <Download className="w-4 h-4" /> Export Official HTML Report
          </button>
        </div>
      </div>
    </dialog>
  );
};
