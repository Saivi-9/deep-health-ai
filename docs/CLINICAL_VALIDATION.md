# 🏥 Clinical Validation & Triage Governance Framework

This document outlines the clinical validation methodology, mathematical foundations for uncertainty quantification, threshold calibration, and clinical triage escalation protocols for the **Deep Health AI Suite**.

---

## 1. Mathematical Foundations of Uncertainty Quantification

Standard deep learning classifiers frequently exhibit overconfidence on out-of-distribution (OOD) clinical presentations, severe radiographic artifacts, or atypical pathologies. Deep Health AI deploys **Monte Carlo Dropout (MC-Dropout)** to approximate Bayesian inference over model parameters without requiring computationally prohibitive MCMC sampling.

### 1.1 Bayesian Approximation via MC-Dropout
Let $f_{\hat{W}}(x)$ denote the neural network parameterized by weights $\hat{W}$. At test time, dropout layers remain active with rate $p \in (0, 1)$, generating $T$ independent stochastic forward passes for each input $x$:

$$\hat{y}_t = \sigma(f_{\hat{W}_t}(x)), \quad t = 1, \dots, T$$

where $\hat{W}_t$ represents the sampled subnet realization at step $t$.

### 1.2 Predictive Mean and Epistemic Variance
The predictive consensus probability $\bar{p}$ is computed as:

$$\bar{p} = \frac{1}{T} \sum_{t=1}^T \hat{y}_t$$

The epistemic uncertainty (model uncertainty arising from lack of knowledge or novel data representations) is captured by the empirical variance $\sigma^2$:

$$\sigma^2 = \frac{1}{T} \sum_{t=1}^T (\hat{y}_t - \bar{p})^2$$

### 1.3 Predictive Shannon Entropy
For multi-class or binary triage, normalized Shannon entropy quantifies informational dispersion:

$$H(Y|X) = - \sum_{c \in \{0, 1\}} \bar{p}_c \log_2(\bar{p}_c + \epsilon)$$

When $H(Y|X) \to 1$, predictive disagreement between stochastic passes is maximal, signaling that the specimen borders the decision manifold or contains ambiguous anatomical structures.

---

## 2. Statistical Calibration & Threshold Optimization

### 2.1 Expected Calibration Error (ECE)
Clinical predictions must reflect true posterior risk probabilities. To calibrate confidence estimates, model outputs are evaluated across $M$ probability bins $B_m \subset (0, 1]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where:
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(y_i = \hat{y}_i)$
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$

### 2.2 Triage Operating Point Selection
In emergency chest radiography triage, false negatives (missed pneumonia/consolidation) carry significantly higher clinical liability than false positives. Thus, decision thresholds $\tau$ are chosen by maximizing the $F_2$-score (weighting recall twice as heavily as precision):

$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}, \quad \text{with } \beta = 2.0$$

Alternatively, the optimal Youden's $J$ statistic operating point is calculated on validation ROC curves:

$$J = \max_{\tau} \left( \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \right)$$

### 2.3 1D ECG Autoencoder Anomaly Thresholding
For reconstruction-based cardiac anomaly detection, reconstruction error is computed as:

$$\mathcal{L}_{\text{recon}}(s) = \frac{1}{L} \sum_{j=1}^L (s_j - \hat{s}_j)^2$$

Threshold selection follows a two-tier hybrid protocol:
1. **Parametric Formulation:** $\theta_{\text{param}} = \mu_{\text{normal}} + 3.0 \cdot \sigma_{\text{normal}}$
2. **Empirical Quantile Calibration:** $\theta_{\text{emp}} = \text{Percentile}_{99.0}(\mathcal{L}_{\text{recon}}(S_{\text{train, normal}}))$

Any telemetry window satisfying $\mathcal{L}_{\text{recon}}(s) > \theta$ triggers an immediate arrhythmia annotation.

---

## 3. Clinical Decision Escalation Matrix

Clinical systems require determinism in escalation paths. The tri-color triage routing protocol maps epistemic variance and risk score into actionable clinician directives:

| Triage Tier | Condition Criteria | Actionable Protocol | Expected Turnaround |
| :--- | :--- | :--- | :--- |
| **🟢 Level 1: Low Priority** | Risk $< 0.30$ AND $\sigma^2 < 0.010$ | Automated normal filing. Included in routine diagnostic worklist. | Standard (within 24 hours) |
| **🟡 Level 2: Expedited Review** | $0.30 \le \text{Risk} < 0.70$ OR $0.010 \le \sigma^2 \le 0.035$ | Flagged for clinical correlation by attending physician or resident. | Expedited (within 4 hours) |
| **🔴 Level 3: STAT Escalation** | Risk $\ge 0.70$ OR $\sigma^2 > 0.035$ OR $H(Y\|X) > 0.60$ | High acute probability OR high diagnostic uncertainty requiring mandatory secondary radiologist read. | Immediate / STAT (< 30 minutes) |

---

## 4. Quality Assurance & Continuous Post-Market Surveillance

1. **Drift Detection:** Input covariate drift is monitored using Kolmogorov-Smirnov (KS) tests on image intensity distributions and ECG power spectral densities (PSD).
2. **Auditing & Traceability:** Every inference request records:
   - Git commit hash & model weight checksum (`SHA-256`)
   - Input metadata (dimensions, sample rate, client timestamp)
   - Grad-CAM heatmap bounding box coordinates
   - Full array of $T=25$ MC-Dropout iterations
3. **Hardware Fail-safes:** In the event of GPU compute degradation or out-of-memory errors, the inference microservice falls back gracefully to deterministic single-pass CPU evaluation and downgrades the triage confidence badge to `"DEGRADED_COMPUTE"`.
