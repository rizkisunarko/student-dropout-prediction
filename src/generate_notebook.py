import json
from pathlib import Path
import nbformat as nbf

from src.config import ROOT

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# PROYEK SAINS DATA (UAS)
## Prediksi Status Akademik Mahasiswa (Student Dropout & Academic Success)
**Metode: Random Forest Classifier (Multi-Class: Dropout, Enrolled, Graduate)**

---

### Informasi Proyek & Dataset
- **Mata Kuliah:** Proyek Sains Data — Ujian Akhir Semester (UAS)
- **Topik:** Early Warning System Prediksi Status Akademik Mahasiswa
- **Dataset:** [UCI Machine Learning Repository - Predict Students Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (Realinho et al., 2021)
- **Pipeline:** Preprocessing (Ordinal & Passthrough) + SMOTENC Balancing (Train Fold Only) + Tuned Random Forest (5-Fold Stratified CV)
"""),

    nbf.v4.new_markdown_cell("""## 1. Setup Environment & Konfigurasi Anti-Leakage
Sesuai perencanaan anti-kebocoran data (*data leakage prevention*):
1. Split **80:20 Stratified** dilakukan terlebih dahulu pada data mentah.
2. Evaluasi hyperparameter dan teknik penanganan *class imbalance* dilakukan **hanya pada data training** menggunakan **5-Fold Stratified Cross-Validation**.
3. Resampling **SMOTENC** dipasang di dalam `imblearn.pipeline.Pipeline`, sehingga SMOTENC **hanya di-fit pada fold training** dan tidak menyentuh fold validasi maupun test set.
4. Preprocessing encoder (`OrdinalEncoder`) di-fit hanya pada data training di dalam `ColumnTransformer`.
5. Data test set dipegang terisolasi (*held-out*) dan hanya disentuh satu kali saat evaluasi final E1–E4.
"""),

    nbf.v4.new_code_cell("""import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.config import ROOT, DATA_PATH, MODEL_PATH, FEATURES_PATH, MODEL_INFO_PATH, SEED, FEATURE_SET, CLASSES
from src.data import load_data
from src.schema import FEATURES, TARGET, get_feature_columns
from src.pipeline import build_model
from src.train import SCORING, _columns

print(f'Feature Set Terpilih: {FEATURE_SET}')
print(f'Random Seed: {SEED}')
print(f'Target Classes: {CLASSES}')
"""),

    nbf.v4.new_markdown_cell("""## 2. Pemuatan dan Validasi Dataset
Dataset memiliki 4.424 baris dan 37 kolom (36 fitur kandidat + 1 target `Target`).
Tidak ada *missing values* dan tidak ada baris duplikat.
"""),

    nbf.v4.new_code_cell("""df = load_data(DATA_PATH)
print('Dimensi data:', df.shape)
print('\\nDistribusi Target:')
target_counts = df[TARGET].value_counts()
target_pct = df[TARGET].value_counts(normalize=True) * 100
for cls in CLASSES:
    print(f' - {cls:10s}: {target_counts[cls]:4d} ({target_pct[cls]:.2f}%)')
"""),

    nbf.v4.new_markdown_cell("""## 3. Hasil 5-Fold Stratified Cross-Validation (E1 vs E2 vs E3)
Perbandingan 3 strategi penanganan ketidakseimbangan kelas pada data latih:
- **E1 (Baseline):** Random Forest parameter default scikit-learn tanpa penyeimbangan.
- **E2 (Class Weight):** Random Forest dengan `class_weight='balanced'`.
- **E3 (SMOTENC):** Synthetic Minority Over-sampling Technique for Nominal and Continuous features di dalam pipeline.
"""),

    nbf.v4.new_code_cell("""cv_metrics = pd.read_csv(ROOT / 'reports' / 'metrics' / 'cv_E1_E3.csv')
cv_pivot = cv_metrics.pivot(index='experiment', columns='metric', values='mean')
cv_std = cv_metrics.pivot(index='experiment', columns='metric', values='std')

print('Tabel Rata-rata Skor 5-Fold CV (Training Only):')
display_cols = ['macro_f1', 'balanced_acc', 'accuracy', 'recall_enrolled', 'recall_dropout']
print(cv_pivot[display_cols].to_markdown())
"""),

    nbf.v4.new_markdown_cell("""**Analisis CV:**
Strategi **E3 (SMOTENC)** menghasilkan nilai **Macro F1 tertinggi (0.7066)** dan secara signifikan meningkatkan recall kelas minoritas **Enrolled** dari 34.0% (E1) menjadi **46.9%** (E3). Berdasarkan protokol plan, strategi terbaik dipilih murni dari hasil CV training, sehingga SMOTENC dipilih untuk dilanjutkan ke tahap hyperparameter tuning (E4).
"""),

    nbf.v4.new_markdown_cell("""## 4. Hyperparameter Tuning (E4)
Tuning dilakukan menggunakan `RandomizedSearchCV` (40 iterasi) dengan 5-Fold Stratified CV dan metrik evaluasi `f1_macro`.
Parameter yang dituning meliputi: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, dan `max_features`.
"""),

    nbf.v4.new_code_cell("""info = json.loads(MODEL_INFO_PATH.read_text(encoding='utf-8'))
print('Best Hyperparameters terpilih:')
for k, v in info.get('best_params', {}).items():
    print(f'  {k}: {v}')
"""),

    nbf.v4.new_markdown_cell("""## 5. Evaluasi Final pada Hold-out Test Set (E1–E4)
Data test (885 mahasiswa) dievaluasi pertama kali untuk membandingkan model final terhadap baseline.
"""),

    nbf.v4.new_code_cell("""test_metrics = pd.read_csv(ROOT / 'reports' / 'metrics' / 'test_E1_E4.csv')
overall_test = test_metrics[test_metrics['class'] == 'overall'][['experiment', 'accuracy', 'balanced_accuracy', 'macro_f1']]
print('Performa Keseluruhan Test Set:')
print(overall_test.to_markdown(index=False))

print('\\nRecall per Kelas pada Test Set:')
class_recall = test_metrics[test_metrics['class'] != 'overall'].pivot(index='experiment', columns='class', values='recall')
print(class_recall.to_markdown())
"""),

    nbf.v4.new_markdown_cell("""## 6. Confusion Matrix Model Terpilih (E4)
Visualisasi performa klasifikasi pada ketiga kelas target.
"""),

    nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img1 = plt.imread(ROOT / 'reports' / 'figures' / 'confusion_E4_counts.png')
img2 = plt.imread(ROOT / 'reports' / 'figures' / 'confusion_E4_normalized.png')
ax1.imshow(img1); ax1.axis('off'); ax1.set_title('E4 Confusion Matrix (Counts)')
ax2.imshow(img2); ax2.axis('off'); ax2.set_title('E4 Confusion Matrix (Normalized)')
plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("""## 7. Analisis Kepentingan Fitur (Feature Importance)
Tingkat pengaruh fitur dianalisis menggunakan dua metode:
1. **MDI (Mean Decrease in Impurity / Gini Importance):** Dihitung dari penurunan impuritas pohon keputusan dalam Random Forest.
2. **Permutation Importance:** Dihitung dari penurunan nilai Macro F1 saat nilai fitur diacak pada test set.
"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 6))
img_imp = plt.imread(ROOT / 'reports' / 'figures' / 'feature_importance.png')
ax.imshow(img_imp); ax.axis('off')
plt.title('Analisis Kepentingan Fitur Top 15', fontsize=14)
plt.tight_layout()
plt.show()

imp_df = pd.read_csv(ROOT / 'reports' / 'metrics' / 'feature_importance.csv')
print('Top 10 Fitur Paling Berpengaruh:')
print(imp_df.head(10)[['feature', 'mdi_importance', 'perm_importance_mean']].to_markdown(index=False))
"""),

    nbf.v4.new_markdown_cell("""## 8. Ekspor Pipeline & Kesimpulan Deployment
Pipeline inference final diekspor ke `models/rf_pipeline.joblib` dengan kontrak fitur `models/features.json`.
Langkah SMOTENC dihilangkan saat inferensi agar proses prediksi murni transformasi deterministik.

Aplikasi web Flask menyajikan antarmuka visual (hasil konversi desain Stitch) dan endpoint API `/predict` serta `/api/schema`.
"""),

    nbf.v4.new_code_cell("""pipeline = joblib.load(MODEL_PATH)
print('Struktur Pipeline Tersimpan:', [name for name, _ in pipeline.steps])
print('Jumlah Fitur Input:', len(pipeline.feature_names_in_))
print('Kelas Output:', list(pipeline.classes_))
""")
]

nb.cells = cells
dest = ROOT / 'Notebook' / '04_uas_rf_pipeline_utama.ipynb'
with open(dest, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook created successfully at {dest}!')
