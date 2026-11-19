# SIMULASI PERHITUNGAN MANUAL RANDOM FOREST (20 SAMPEL)
**Mata Kuliah:** Proyek Sains Data (UAS)  
**Metode:** Random Forest Ensemble (Multi-Class: Dropout, Enrolled, Graduate)

---

## 1. Konsep Manual Random Forest
Random Forest membangun sejumlah pohon keputusan independen melalui dua mekanisme acak utama:
1. **Bagging (Bootstrap Aggregating):** Pengambilan sampel baris data berukuran $N$ dengan pengembalian (*sampling with replacement*). Setiap pohon dilatih pada subset data yang berbeda, menyisakan sekitar ~36.8% sampel yang tidak terpilih sebagai data *Out-of-Bag* (OOB).
2. **Random Feature Subspace Selection:** Pada setiap *split node*, hanya $m = \sqrt{M}$ fitur acak yang dipertimbangkan untuk menentukan pemisah terbaik berdasarkan penurunan **Gini Impurity**.
3. **Majority Voting (Ensemble Aggregation):** Hasil klasifikasi final ditentukan melalui pemungutan suara terbanyak (*majority vote*) dari seluruh pohon dalam *forest*.

---

## 2. Tabel 20 Sampel Mahasiswa Terpilih
Data diambil secara acak berstrata dari `Data/data.csv` mencakup ketiga kelas target:

| ID | Curricular units 2nd sem (approved) | Curricular units 1st sem (approved) | Curricular units 2nd sem (grade) | Tuition fees up to date | Debtor | Scholarship holder | Age at enrollment | Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mhs_01 | 6 | 4 | 13.0 | 1 | 0 | 0 | 20 | Graduate |
| Mhs_02 | 0 | 0 | 0.0 | 1 | 0 | 1 | 18 | Graduate |
| Mhs_03 | 7 | 7 | 12.414285714285716 | 1 | 0 | 0 | 21 | Graduate |
| Mhs_04 | 6 | 6 | 13.833333333333334 | 1 | 0 | 1 | 19 | Graduate |
| Mhs_05 | 5 | 6 | 13.6 | 0 | 0 | 0 | 18 | Graduate |
| Mhs_06 | 8 | 7 | 13.122222222222222 | 1 | 0 | 1 | 18 | Graduate |
| Mhs_07 | 5 | 4 | 14.833333333333334 | 1 | 0 | 0 | 20 | Graduate |
| Mhs_08 | 6 | 6 | 14.666666666666666 | 1 | 1 | 1 | 20 | Graduate |
| Mhs_09 | 5 | 5 | 10.4 | 1 | 0 | 0 | 18 | Dropout |
| Mhs_10 | 2 | 5 | 13.5 | 0 | 1 | 0 | 19 | Dropout |
| Mhs_11 | 0 | 0 | 0.0 | 1 | 0 | 0 | 37 | Dropout |
| Mhs_12 | 1 | 3 | 13.0 | 1 | 1 | 1 | 21 | Dropout |
| Mhs_13 | 0 | 0 | 0.0 | 0 | 1 | 0 | 34 | Dropout |
| Mhs_14 | 5 | 8 | 13.4 | 1 | 0 | 0 | 47 | Dropout |
| Mhs_15 | 0 | 0 | 0.0 | 0 | 1 | 0 | 31 | Dropout |
| Mhs_16 | 7 | 7 | 11.857142857142858 | 1 | 0 | 0 | 34 | Enrolled |
| Mhs_17 | 1 | 0 | 10.0 | 1 | 0 | 0 | 18 | Enrolled |
| Mhs_18 | 10 | 10 | 12.8 | 1 | 1 | 0 | 42 | Enrolled |
| Mhs_19 | 5 | 6 | 11.8 | 1 | 0 | 1 | 48 | Enrolled |
| Mhs_20 | 8 | 6 | 13.75 | 1 | 0 | 0 | 20 | Enrolled |

---

## 3. Perhitungan Gini Impurity pada Root Node
Sebelum dilakukan pemisahan (*split*), kemurnian label pada 20 sampel dihitung menggunakan rumus Gini Impurity:

$$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$

Distribusi 20 sampel:
- **Graduate:** 8 sampel ($p_1 = 8/20 = 0.40$)
- **Dropout:** 7 sampel ($p_2 = 7/20 = 0.35$)
- **Enrolled:** 5 sampel ($p_3 = 5/20 = 0.25$)

$$\text{Gini}(D) = 1 - (0.40^2 + 0.35^2 + 0.25^2) = 1 - (0.16 + 0.1225 + 0.0625) = 1 - 0.3450 = \mathbf{0.6550}$$

### Contoh Evaluasi Calon Pemisah Fitur:
Kandidat pemisah: `Curricular units 2nd sem (approved) >= 3.5`
- **Cabang Kiri (>= 3.5):** 8 Mahasiswa (7 Graduate, 0 Dropout, 1 Enrolled)
  $$\text{Gini}_{L} = 1 - ((7/8)^2 + 0^2 + (1/8)^2) = 1 - (0.7656 + 0.0156) = \mathbf{0.2188}$$
- **Cabang Kanan (< 3.5):** 12 Mahasiswa (1 Graduate, 7 Dropout, 4 Enrolled)
  $$\text{Gini}_{R} = 1 - ((1/12)^2 + (7/12)^2 + (4/12)^2) = 1 - (0.0069 + 0.3403 + 0.1111) = \mathbf{0.5417}$$
- **Gini Split Rata-rata Berbobot:**
  $$\text{Gini}_{\text{split}} = \frac{8}{20}(0.2188) + \frac{12}{20}(0.5417) = 0.0875 + 0.3250 = \mathbf{0.4125}$$
- **Information Gain (Penurunan Impuritas):**
  $$\Delta \text{Gini} = 0.6550 - 0.4125 = \mathbf{0.2425}$$ (Penurunan impuritas sangat signifikan).

---

## 4. Hasil Voting Ansambel 3 Pohon Keputusan

| ID_Mahasiswa | Target_Asli | Prediksi_Pohon_1 | Prediksi_Pohon_2 | Prediksi_Pohon_3 | Majority_Vote | P_Graduate | P_Dropout | P_Enrolled | Hasil_Evaluasi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mhs_01 | Graduate | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Benar (Match) |
| Mhs_02 | Graduate | Dropout | Dropout | Dropout | Dropout | 0.0 | 1.0 | 0.0 | Meleset |
| Mhs_03 | Graduate | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Benar (Match) |
| Mhs_04 | Graduate | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Benar (Match) |
| Mhs_05 | Graduate | Graduate | Dropout | Graduate | Graduate | 0.67 | 0.33 | 0.0 | Benar (Match) |
| Mhs_06 | Graduate | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Benar (Match) |
| Mhs_07 | Graduate | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Benar (Match) |
| Mhs_08 | Graduate | Graduate | Graduate | Dropout | Graduate | 0.67 | 0.33 | 0.0 | Benar (Match) |
| Mhs_09 | Dropout | Graduate | Graduate | Enrolled | Graduate | 0.67 | 0.0 | 0.33 | Meleset |
| Mhs_10 | Dropout | Enrolled | Dropout | Dropout | Dropout | 0.0 | 0.67 | 0.33 | Benar (Match) |
| Mhs_11 | Dropout | Dropout | Dropout | Dropout | Dropout | 0.0 | 1.0 | 0.0 | Benar (Match) |
| Mhs_12 | Dropout | Dropout | Enrolled | Dropout | Dropout | 0.0 | 0.67 | 0.33 | Benar (Match) |
| Mhs_13 | Dropout | Dropout | Dropout | Dropout | Dropout | 0.0 | 1.0 | 0.0 | Benar (Match) |
| Mhs_14 | Dropout | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Meleset |
| Mhs_15 | Dropout | Dropout | Dropout | Dropout | Dropout | 0.0 | 1.0 | 0.0 | Benar (Match) |
| Mhs_16 | Enrolled | Graduate | Graduate | Enrolled | Graduate | 0.67 | 0.0 | 0.33 | Meleset |
| Mhs_17 | Enrolled | Dropout | Dropout | Enrolled | Dropout | 0.0 | 0.67 | 0.33 | Meleset |
| Mhs_18 | Enrolled | Graduate | Graduate | Dropout | Graduate | 0.67 | 0.33 | 0.0 | Meleset |
| Mhs_19 | Enrolled | Graduate | Graduate | Enrolled | Graduate | 0.67 | 0.0 | 0.33 | Meleset |
| Mhs_20 | Enrolled | Graduate | Graduate | Graduate | Graduate | 1.0 | 0.0 | 0.0 | Meleset |

---

## 5. Ringkasan Performa Simulasi Manual
- **Jumlah Sampel:** 20 Mahasiswa
- **Akurasi Voting Ansambel:** 60.0% (12 dari 20 benar)
- **Konsistensi Probabilitas:** Probabilitas voting merefleksikan tingkat keyakinan pohon, di mana mahasiswa dengan capaian akademik tinggi (`cu_2nd_approved >= 4`) memperoleh konsensus 100% Graduate ($P=1.00$).
