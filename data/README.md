# PhishLens Datasets

This directory contains the raw and processed datasets utilized for training and evaluating PhishLens scam detection models.

## Dataset Structure

```text
data/
├── raw/
│   ├── sms_dataset.csv       # Unified SMS dataset with text and label
│   └── modern_phishing.csv   # Modern phishing & benign message instances
├── processed/
│   ├── train.csv             # 70% Stratified training set
│   ├── val.csv               # 15% Stratified validation set (for threshold tuning & model comparison)
│   └── test.csv              # 15% Stratified held-out test set (unseen evaluation)
└── README.md
```

## Data Sources & Provenance

1. **SMS Spam Collection v.1**
   - **Source**: UCI Machine Learning Repository / Almeida et al.
   - **Original Samples**: 5,574 tagged SMS messages (747 spam, 4,827 ham).
   - **License**: Public for research purposes.

2. **Modern Phishing & Smishing Synthetic/Curated Dataset**
   - **Source**: Curated real-world smishing templates (2023-2026 threats: UPI/Paytm scams, package delivery FedEx/USPS, KYC account suspension, crypto giveaways, OTP interception, toll road smishing).
   - **Purpose**: Augment historical SMS dataset with modern attack vectors and modern legitimate transactional alerts.

## Label Schema

- `scam`: Malicious, phishing, fraud, credential harvesting, unauthorized transaction prompts.
- `not_scam`: Legitimate messages, personal texts, transactional alerts, verified system OTPs.

## Splitting Methodology

- **Split Ratios**: 70% Train, 15% Validation, 15% Test.
- **Stratification**: Stratified by class label to preserve class balance across all splits.
- **Data Leakage Safeguards**: Preprocessing and vectorizer fitting are performed strictly on the Training set. Threshold tuning is performed exclusively on the Validation set. The Test set is evaluated strictly once at the end.
