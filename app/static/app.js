/**
 * PhishLens Neo-Brutalist Client Application
 */

let activeMode = 'text';
let currentSelectedFile = null;
let defaultOperatingThreshold = 0.42;

const PRESETS = {
  sbi_kyc: "URGENT: Your SBI account has been blocked due to incomplete KYC. Update your PAN card immediately at http://sbi-kyc-update.top or service will stop.",
  fedex: "Your FedEx package #US98421 could not be delivered due to incomplete address. Please update details and pay $1.99 redelivery fee at http://fedx-tracking-portal.com",
  crypto: "Crypto Giveaway: Elon Musk is giving away 5,000 ETH! Send 0.1 ETH to verify wallet and receive 1.0 ETH back instantly at http://tesla-eth-promo.org",
  otp: "Your one-time password (OTP) for logging into your HDFC Bank account is 482910. Do not share this OTP with anyone, including bank officials.",
  uber: "Uber: Your driver Rajesh (Swift Dzire - MH02AB1234) is arriving in 3 mins. OTP for this ride is 7731."
};

document.addEventListener('DOMContentLoaded', () => {
  initApp();
  setupTextListeners();
  setupDropzone();
});

async function initApp() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('api-status-text').innerText = 'RADAR ONLINE';
      if (data.operating_threshold) {
        defaultOperatingThreshold = data.operating_threshold;
        document.getElementById('metric-threshold').innerText = defaultOperatingThreshold.toFixed(2);
        document.getElementById('threshold-slider').value = defaultOperatingThreshold;
        document.getElementById('threshold-val-display').innerText = `DEFAULT (${defaultOperatingThreshold.toFixed(2)})`;
      }
    } else {
      document.getElementById('api-status-text').innerText = 'MODEL OFFLINE';
    }
  } catch (err) {
    document.getElementById('api-status-text').innerText = 'API UNREACHABLE';
  }

  // Load metrics
  try {
    const mRes = await fetch('/metrics');
    if (mRes.ok) {
      const mData = await mRes.json();
      if (mData.metrics) {
        document.getElementById('metric-precision').innerText = `${(mData.metrics.scam_precision * 100).toFixed(1)}%`;
        document.getElementById('metric-recall').innerText = `${(mData.metrics.scam_recall * 100).toFixed(1)}%`;
        document.getElementById('metric-accuracy').innerText = `${(mData.metrics.accuracy * 100).toFixed(1)}%`;
      }
    }
  } catch (err) {
    console.warn('Metrics endpoint not ready:', err);
  }
}

function switchTab(mode) {
  activeMode = mode;
  document.getElementById('tab-text').classList.toggle('active', mode === 'text');
  document.getElementById('tab-image').classList.toggle('active', mode === 'image');
  document.getElementById('panel-text').classList.toggle('active', mode === 'text');
  document.getElementById('panel-image').classList.toggle('active', mode === 'image');
}

function setupTextListeners() {
  const textarea = document.getElementById('input-text');
  const countEl = document.getElementById('char-count');
  textarea.addEventListener('input', () => {
    countEl.innerText = `${textarea.value.length.toLocaleString()} / 10,000`;
  });
}

function loadPreset(presetKey) {
  switchTab('text');
  const txt = PRESETS[presetKey] || '';
  const textarea = document.getElementById('input-text');
  textarea.value = txt;
  document.getElementById('char-count').innerText = `${txt.length} / 10,000`;
}

function updateThresholdDisplay(val) {
  const num = parseFloat(val).toFixed(2);
  const isDefault = Math.abs(parseFloat(val) - defaultOperatingThreshold) < 0.005;
  document.getElementById('threshold-val-display').innerText = isDefault ? `DEFAULT (${num})` : `${num} (CUSTOM)`;
}

// Drag & Drop Handling
function setupDropzone() {
  const dz = document.getElementById('dropzone');
  ['dragenter', 'dragover'].forEach(eventName => {
    dz.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dz.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.remove('dragover');
    }, false);
  });

  dz.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processSelectedFile(files[0]);
    }
  }, false);
}

function triggerFileInput() {
  document.getElementById('file-input').click();
}

function handleFileSelected(e) {
  if (e.target.files && e.target.files.length > 0) {
    processSelectedFile(e.target.files[0]);
  }
}

function processSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file (PNG, JPG, WEBP).');
    return;
  }
  currentSelectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('image-preview').src = e.target.result;
    document.getElementById('preview-filename').innerText = file.name;
    document.getElementById('dropzone-empty').style.display = 'none';
    document.getElementById('dropzone-preview').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function removeFile(e) {
  e.stopPropagation();
  currentSelectedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('image-preview').src = '';
  document.getElementById('dropzone-empty').style.display = 'block';
  document.getElementById('dropzone-preview').style.display = 'none';
}

// Prediction Analysis Invoker
async function runAnalysis() {
  const thresholdVal = parseFloat(document.getElementById('threshold-slider').value);
  const btnAnalyze = document.getElementById('btn-analyze');
  const emptyState = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultContent = document.getElementById('result-content');

  emptyState.style.display = 'none';
  resultContent.style.display = 'none';
  loadingState.style.display = 'block';
  btnAnalyze.disabled = true;

  try {
    let result = null;

    if (activeMode === 'text') {
      const text = document.getElementById('input-text').value.trim();
      if (!text) {
        alert('Please paste or type a text message before analyzing.');
        loadingState.style.display = 'none';
        emptyState.style.display = 'block';
        btnAnalyze.disabled = false;
        return;
      }
      document.getElementById('loading-msg').innerText = 'ANALYZING TEXT TOKENS & TF-IDF FEATURES...';
      const res = await fetch('/predict/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, threshold: thresholdVal })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }
      result = await res.json();
    } else {
      if (!currentSelectedFile) {
        alert('Please select or drop a screenshot image first.');
        loadingState.style.display = 'none';
        emptyState.style.display = 'block';
        btnAnalyze.disabled = false;
        return;
      }
      document.getElementById('loading-msg').innerText = 'RUNNING OCR EXTRACTION & CLASSIFICATION...';
      const formData = new FormData();
      formData.append('file', currentSelectedFile);
      formData.append('threshold', thresholdVal);

      const res = await fetch('/predict/image', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'OCR Prediction failed');
      }
      result = await res.json();
    }

    renderResults(result);
  } catch (err) {
    alert(`Error: ${err.message}`);
    emptyState.style.display = 'block';
  } finally {
    loadingState.style.display = 'none';
    btnAnalyze.disabled = false;
  }
}

function renderResults(data) {
  const resultContent = document.getElementById('result-content');
  const banner = document.getElementById('verdict-banner');
  const verdictLabel = document.getElementById('verdict-label');
  const riskBadge = document.getElementById('risk-badge');
  const thresholdMeta = document.getElementById('threshold-meta');
  const probValue = document.getElementById('probability-value');
  const meterFill = document.getElementById('meter-bar-fill');
  const thresholdMarker = document.getElementById('threshold-marker');
  const thresholdMarkerLabel = document.getElementById('threshold-marker-label');
  const ocrSection = document.getElementById('ocr-section');
  const ocrTextBox = document.getElementById('ocr-text-box');
  const ocrConfBadge = document.getElementById('ocr-conf-badge');
  const signalsList = document.getElementById('signals-list');
  const signalsCount = document.getElementById('signals-count');
  const jsonPre = document.getElementById('json-pre');

  const isScam = data.prediction === 'SCAM';
  const probPercent = (data.scam_probability * 100).toFixed(1);
  const threshPercent = (data.threshold * 100).toFixed(1);

  // Verdict Banner
  verdictLabel.innerText = data.prediction;
  banner.className = `verdict-banner ${isScam ? 'verdict-scam' : 'verdict-safe'}`;
  riskBadge.innerText = data.risk_level;
  thresholdMeta.innerText = `Decision Cutoff: ${data.threshold}`;

  // Meter Bar
  probValue.innerText = `${probPercent}%`;
  meterFill.style.width = `${probPercent}%`;
  thresholdMarker.style.left = `${threshPercent}%`;
  thresholdMarkerLabel.innerText = `Th: ${threshPercent}%`;

  // OCR Section
  if (data.extracted_text) {
    ocrSection.style.display = 'block';
    ocrTextBox.innerText = data.extracted_text;
    ocrConfBadge.innerText = `CONF: ${(data.ocr_confidence * 100).toFixed(1)}%`;
  } else {
    ocrSection.style.display = 'none';
  }

  // Risk Signals
  signalsList.innerHTML = '';
  if (data.signals && data.signals.length > 0) {
    signalsCount.innerText = `${data.signals.length} Risk Signal${data.signals.length > 1 ? 's' : ''}`;
    data.signals.forEach(sig => {
      const item = document.createElement('div');
      item.className = 'signal-item';
      item.innerHTML = `<span class="signal-icon">⚠️</span> <span>${escapeHtml(sig)}</span>`;
      signalsList.appendChild(item);
    });
  } else {
    signalsCount.innerText = '0 Risk Signals';
    const item = document.createElement('div');
    item.className = 'signal-item';
    item.innerHTML = `<span class="signal-icon">✅</span> <span>No prominent phishing keywords or malicious markers identified.</span>`;
    signalsList.appendChild(item);
  }

  // JSON Drawer
  jsonPre.innerText = JSON.stringify(data, null, 2);

  resultContent.style.display = 'flex';
}

function toggleJsonDrawer() {
  const jsonPre = document.getElementById('json-pre');
  const icon = document.getElementById('json-toggle-icon');
  const isHidden = jsonPre.style.display === 'none';
  jsonPre.style.display = isHidden ? 'block' : 'none';
  icon.innerText = isHidden ? '▲' : '▼';
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
