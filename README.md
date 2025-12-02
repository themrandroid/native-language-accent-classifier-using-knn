# Native Language Accent Classifier (Hausa, Igbo, Yoruba)

A compact, reproducible pipeline that classifies a speaker's native language (Hausa, Igbo, Yoruba) from short audio samples using engineered audio features (MFCCs, pitch, ZCR, spectral contrast) and a K-Nearest Neighbors (KNN) model. The repository contains audio collection utilities, preprocessing, feature extraction, training/evaluation code in a notebook, saved feature CSVs, and a Streamlit demo app.

Key goals
- Reliable feature extraction from short audio samples.
- Reproducible preprocessing and augmentation / bootstrapping.
- Simple, explainable baseline model for rapid iteration and demo.


Primary code symbols (open these for details)
- [`app.extract_features`](app.py): feature extraction used in the Streamlit app. See [app.py](app.py).
- [`native_language_accent.preprocess_audio`](native_language_accent.ipynb): audio conversion, normalization, trimming routine. See [native_language_accent.ipynb](native_language_accent.ipynb).
- [`native_language_accent.extract_features`](native_language_accent.ipynb): dataset feature extraction loop (13 MFCC means, pitch mean/std, zcr, 7 spectral contrast means). See [native_language_accent.ipynb](native_language_accent.ipynb).
- [`native_language_accent.download_audio`](native_language_accent.ipynb): YouTube/shorts download helper used to populate [data/](data/). See [native_language_accent.ipynb](native_language_accent.ipynb).

Quick start

1) Prerequisites
- Python 3.8+ (tested).
- System dependency: ffmpeg (required by pydub).
- Recommended: create a virtual environment.

2) Install Python dependencies
Run:
```bash
python -m pip install -U pip
python -m pip install numpy pandas librosa scikit-learn matplotlib seaborn pydub yt_dlp streamlit joblib
```
Or:
```bash
python -m pip install -r requirements.txt
```
(If you add a requirements file, put it at repo root.)

3) Reproduce feature extraction and training (recommended)
- Open and run [native_language_accent.ipynb](native_language_accent.ipynb).
  - Use [`native_language_accent.preprocess_audio`](native_language_accent.ipynb) to convert files from [data/](data/) into [processed_data/](processed_data/).
  - Use [`native_language_accent.extract_features`](native_language_accent.ipynb) to build `accent_features.csv`.
  - The notebook contains a bootstrapping step (resample per-class to fixed size) and writes `accent_features_bootstrapped.csv`.

4) Run the demo Streamlit app
- Ensure the trained artifacts `knn_accent_model.pkl` and `scaler.pkl` are present at the repository root (the app expects them).
- Launch:
```bash
streamlit run app.py
```
- Use the app to upload a short audio file (.wav/.mp3). The app calls [`app.extract_features`](app.py), scales with `scaler.pkl`, and predicts with `knn_accent_model.pkl`.

Feature schema
Both CSV files ([accent_features.csv](accent_features.csv), [accent_features_bootstrapped.csv](accent_features_bootstrapped.csv)) use this column order (header present in files):
mfcc_1, mfcc_2, ..., mfcc_13, pitch_mean, pitch_std, zcr, contrast_1, ..., contrast_7, native_language

Notes and implementation points
- Preprocessing: audio is converted to mono and 16 kHz, normalized, silence-trimmed, and clipped/padded to a fixed duration. See [`native_language_accent.preprocess_audio`](native_language_accent.ipynb).
- Features: MFCC mean across frames (13 coeffs), pitch mean/std from librosa.yin, zero-crossing rate mean, and spectral contrast means (7 bands). See [`native_language_accent.extract_features`](native_language_accent.ipynb) and [`app.extract_features`](app.py).
- Bootstrapping: the notebook uses sklearn.utils.resample to balance classes and produce `accent_features_bootstrapped.csv`.
- Model: baseline KNN trained with StandardScaler and k selected by cross-validation (notebook). The demo loads `knn_accent_model.pkl` and `scaler.pkl` in [app.py](app.py).

Reproducibility checklist
- Install system ffmpeg for audio conversions.
- Use the notebook's fixed random_state values for deterministic resampling and splits.
- Keep saved model artifacts next to [app.py](app.py) for the demo.

Evaluation pointers
- The notebook computes train/test splits, cross-validation for k, classification report, and confusion matrix. Use these cells in [native_language_accent.ipynb](native_language_accent.ipynb) to reproduce metrics.
- If you want a stronger model: try tree-based ensembles, logistic regression, or simple neural nets; consider per-frame or sequence models for longer audio.

Data & licensing
- The repository contains feature CSVs and scripts to download audio (YouTube links are in the notebook). Confirm license/permission for any redistributed audio.
- Add a LICENSE file (MIT or similar) to make reuse terms explicit.

Contributing
- Fork, create a branch, and open a concise pull request describing changes and tests or reproducible steps.

Contact and authorship
- The notebook and app include author metadata. If you want the README to display a different author line, update it here.