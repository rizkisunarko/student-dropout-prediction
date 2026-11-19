# LAPORAN RINGKAS PROYEK SAINS DATA (UAS)
# Klasifikasi Multi-Kelas Status Akademik Mahasiswa Menggunakan Random Forest

**Mata Kuliah:** Proyek Sains Data  
**Dataset Acuan:** UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success* (Realinho et al., 2021)  
**Metode Utama:** Random Forest Ensemble (Multi-Class: *Dropout*, *Enrolled*, *Graduate*)  
**Teknik Penyeimbangan:** SMOTENC (*Synthetic Minority Over-sampling Technique for Nominal and Continuous*) di dalam Pipeline  
**Arsitektur Penyajian:** REST API Flask + Antarmuka Web Single Page Application (Desain Stitch)  

---

## 1. Latar Belakang & Rumusan Masalah

Retensi mahasiswa merupakan tantangan krusial bagi perguruan tinggi. Fenomena putus studi (*dropout*) menimbulkan kerugian finansial, reputasi akademik bagi institusi, dan beban psikososial bagi mahasiswa bersangkutan. Sebagian besar intervensi akademik seringkali terlambat karena tanda-tanda kegagalan baru disadari ketika mahasiswa telah berada di ambang drop-out.

Tujuan utama proyek ini adalah membangun sistem kecerdasan buatan berbasis *Machine Learning* yang mampu memprediksi status akademik mahasiswa secara dini ke dalam 3 kelas:
1. **Graduate:** Mahasiswa yang menyelesaikan masa perkuliahan dan lulus tepat waktu.
2. **Dropout:** Mahasiswa yang berhenti di tengah jalan atau tidak melanjutkan studi.
3. **Enrolled:** Mahasiswa yang masih aktif terdaftar namun membutuhkan monitoring progres kelulusan.

---

## 2. Pemahaman Dataset & Kontrak Data

Dataset bersumber dari Portalegre Polytechnic Institute melalui UCI Machine Learning Repository:
- **Jumlah Observasi:** 4.424 baris data mahasiswa.
- **Jumlah Atribut:** 36 fitur prediktor + 1 target (`Target`).
- **Kualitas Data:** 0 *missing values* dan 0 baris duplikat.
- **Distribusi Kelas Target:**
  - *Graduate:* 2.209 mahasiswa (49.93%) — Kelas Mayoritas
  - *Dropout:* 1.421 mahasiswa (32.12%) — Kelas Menengah
  - *Enrolled:* 794 mahasiswa (17.95%) — Kelas Minoritas (Paling Sulit)

### Pembagian Kelompok Fitur:
1. **Demografi (8 fitur):** `marital_status`, `gender`, `age_at_enrollment`, `nationality`, `international`, `displaced`, `educational_special_needs`.
2. **Pendaftaran (6 fitur):** `application_mode`, `application_order`, `course`, `daytime_evening_attendance`, `previous_qualification`, `previous_qualification_grade`, `admission_grade`.
3. **Sosial-Ekonomi (7 fitur):** `mothers_qualification`, `fathers_qualification`, `mothers_occupation`, `fathers_occupation`, `debtor`, `tuition_fees_up_to_date`, `scholarship_holder`.
4. **Akademik Semester 1 (6 fitur):** `cu_1st_credited`, `cu_1st_enrolled`, `cu_1st_evaluations`, `cu_1st_approved`, `cu_1st_grade`, `cu_1st_without_evaluations`.
5. **Akademik Semester 2 (6 fitur):** `cu_2nd_credited`, `cu_2nd_enrolled`, `cu_2nd_evaluations`, `cu_2nd_approved`, `cu_2nd_grade`, `cu_2nd_without_evaluations`.
6. **Konteks Makroekonomi (3 fitur):** `unemployment_rate`, `inflation_rate`, `gdp`.

---

## 3. Metodologi Eksperimen & Desain Anti-Kebocoran Data (*Anti-Data Leakage*)

Untuk menjamin validitas akademik, diterapkan protokol anti-kebocoran data yang ketat:
1. **Pemisahan Awal (Stratified 80:20):** Dataset dipisah menjadi data latih (3.539 baris) dan data uji (885 baris) dengan mempertahankan proporsi kelas target menggunakan `stratify=y` dan `random_state=42`. Indeks disimpan di `models/split_indices.json`.
2. **Evaluasi Hanya pada Data Training:** Seluruh perbandingan strategi (E1, E2, E3) dan *tuning* hyperparameter (E4) dilakukan **hanya pada data training** menggunakan **5-Fold Stratified Cross-Validation**.
3. **Resampling di Dalam Fold:** SMOTENC dipasang di dalam `imblearn.pipeline.Pipeline`, sehingga proses pembentukan data sintetis **hanya aktif pada fold training** dan tidak mencemari data validasi maupun data test.
4. **Preprocessing Terisolasi:** `ColumnTransformer` (OrdinalEncoder pada fitur nominal dan passthrough pada fitur numerik/biner) di-fit hanya pada data training fold.
5. **Data Test Terisolasi:** Data uji (*hold-out test set*) tidak disentuh sama sekali sampai evaluasi akhir model.

---

## 4. Hasil Eksperimen (E1–E4)

### A. Evaluasi 5-Fold Stratified Cross-Validation pada Data Latih

| Eksperimen | Strategi Penyeimbangan | Macro F1 | Balanced Accuracy | Plain Accuracy | Recall Enrolled | Recall Dropout |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **E1** | Baseline (Default RF) | 0.6879 ± 0.018 | 0.6787 ± 0.016 | 0.7714 ± 0.012 | 0.3402 ± 0.045 | 0.7660 ± 0.031 |
| **E2** | `class_weight='balanced'` | 0.6856 ± 0.026 | 0.6767 ± 0.022 | 0.7739 ± 0.014 | 0.3228 ± 0.061 | 0.7660 ± 0.028 |
| **E3** | **SMOTENC Pipeline** | **0.7066 ± 0.025** | **0.7017 ± 0.023** | 0.7663 ± 0.017 | **0.4693 ± 0.056** | 0.7564 ± 0.021 |

> **Keputusan Seleksi Model:** Berdasarkan hasil CV, **E3 (SMOTENC)** mencapai Macro F1 tertinggi (**0.7066**) dan secara drastis mendongkrak recall kelas minoritas *Enrolled* dari 34.0% ke 46.9%. Maka, SMOTENC dipilih sebagai arsitektur dasar untuk dioptimalkan pada tahap tuning (E4).

### B. Hyperparameter Tuning (E4)
Tuning dilakukan dengan `RandomizedSearchCV` (40 iterasi, 5-Fold Stratified CV, scoring `f1_macro`):
- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: 'sqrt'

### C. Evaluasi Hold-out Test Set (885 Mahasiswa)

| Model | Accuracy | Balanced Accuracy | Macro F1 | Recall Dropout | Recall Enrolled | Recall Graduate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1 Baseline** | 77.51% | 68.50% | 0.6974 | 75.35% | 36.48% | **93.67%** |
| **E2 Class Weight** | 77.18% | 68.00% | 0.6904 | 76.41% | 34.59% | 92.99% |
| **E3 SMOTENC** | 75.82% | 70.39% | **0.7049** | 72.54% | 52.20% | 86.43% |
| **E4 Tuned SMOTENC** | 74.12% | **0.7054** | **0.6994** | 68.66% | **60.38%** | 82.58% |

---

## 5. Analisis Mendalam: Mengapa Kelas *Enrolled* Paling Sulit?

Di semua literatur (termasuk Realinho dkk. dan Dzilkamala), kelas *Enrolled* selalu memiliki performa F1 terendah (~0.35–0.50):
1. **Sifat Status Sementara (Transisi):** Mahasiswa *Enrolled* belum mencapai kondisi akhir; profil nilai dan SKS mereka berada di zona abu-abu di antara mahasiswa yang rajin (calon *Graduate*) dan mahasiswa yang mulai tertinggal (calon *Dropout*).
2. **Efektivitas SMOTENC:** Pada model baseline (E1), recall *Enrolled* hanya **36.48%** (hampir 2 dari 3 mahasiswa enrolled salah diprediksi sebagai Dropout atau Graduate). Melalui SMOTENC dan tuning (E4), recall melonjak menjadi **60.38%** (+23.9% peningkatan deteksi mahasiswa aktif).

---

## 6. Analisis Kepentingan Fitur (*Feature Importance*)

Berdasarkan analisis *Mean Decrease in Impurity* (Gini Importance) dan *Permutation Importance* pada data uji:
1. **Fitur Akademik Semester 2 dan 1 adalah Prediktor Terkuat:**
   - `Curricular units 2nd sem (approved)` (Drop F1 Macro: +0.125 saat diacak)
   - `Curricular units 1st sem (approved)` (Drop F1 Macro: +0.030)
   - `Curricular units 2nd sem (grade)` (Gini Impurity: ~10.4%)
2. **Kondisi Finansial Mahasiswa:**
   - `Tuition fees up to date` (Status pembayaran SPP/UKT): mahasiswa yang menunggak memiliki korelasi sangat kuat terhadap risiko *dropout*.
   - `Debtor` dan `Scholarship holder`: penerima beasiswa memiliki peluang kelulusan jauh lebih tinggi.
3. **Faktor Demografi:**
   - `Age at enrollment` (usia saat masuk perkuliahan): mahasiswa dengan usia lebih dewasa (>24 tahun) cenderung menghadapi beban kerja di luar kampus yang meningkatkan risiko putus studi.

---

## 7. Arsitektur Antarmuka Web (Desain Stitch) & API Serving

Aplikasi web dirancang menggunakan filosofi Single Page Application (SPA) berbasis template desain Stitch:
- **Backend:** Flask melayani static assets, kontrak skema `/api/schema`, endpoint prediksi `/predict`, dan health check `/health`.
- **Pipeline Aman (Inference-Ready):** Pipeline yang diekspor (`models/rf_pipeline.joblib`) hanya memuat langkah `prep` dan `rf` tanpa menyertakan resampling SMOTENC, menjamin inferensi deterministik dan cepat (<50ms).
- **Kontrak Fitur Kaya (`features.json`):** Setiap fitur memiliki label ramah bahasa Indonesia, unit satuan (SKS, %, tahun, skala 0-20), placeholder, rentang batas validasi, dan label opsi deskriptif resmi (misal: nama jurusan resmi, jalur masuk, pendidikan dan pekerjaan orang tua).
- **Fitur Interaktif UI:**
  - *Form Wizard Dinamis:* Terbagi ke dalam 6 langkah logis (Demografi, Pendaftaran, Sosial-Ekonomi, Akademik Semester 1, Akademik Semester 2, Makroekonomi).
  - *Tombol Contoh Data (Presets):* Penguji dapat mengisi form secara instan dengan 1-klik untuk profil "Mahasiswa Lulus" atau "Mahasiswa Dropout".
  - *Toggle Nilai Median Makroekonomi:* Pengguna dapat memilih pengisian otomatis median wilayah (pengangguran 11.1%, inflasi 1.4%, GDP 0.32%).
  - *Animasi Forest Voting:* Menampilkan 24 simpul pohon keputusan yang berkelap-kelip dan mewarnai diri berdasarkan konsensus kelas pemenang.
  - *Result Card Dinamis:* Kartu hasil dengan efek *scramble text*, penjelasan status akademik, dan 3 *progress bar* probabilitas (*Graduate*, *Enrolled*, *Dropout*).

---

## 8. Kesimpulan & Rekomendasi Batasan Sistem

1. **Kesimpulan Model:** Random Forest dengan penyeimbangan SMOTENC (E4) berhasil mengoptimalkan pendeteksian mahasiswa berisiko putus studi dan mahasiswa aktif yang butuh bimbingan (Recall Enrolled 60.38%), dengan nilai Macro F1 ~0.70.
2. **Batasan Sistem (*Disclaimer*):** Prediksi model dirancang sebagai **sistem peringatan dini (*early warning support system*)** bagi dosen pembimbing akademik dan fakultas, bukan sebagai instrumen vonis kelulusan atau sanksi akademik sepihak.
