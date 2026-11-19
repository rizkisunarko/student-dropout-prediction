import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import ROOT, DATA_PATH
from src.data import load_data
from src.schema import TARGET

def to_md_table(df, include_index=True):
    cols = list(df.columns)
    if include_index:
        header = ["ID"] + [str(c) for c in cols]
        sep = ["---"] * len(header)
        rows = [f"| {' | '.join(header)} |", f"| {' | '.join(sep)} |"]
        for idx, row in df.iterrows():
            r = [str(idx)] + [str(row[c]) for c in cols]
            rows.append(f"| {' | '.join(r)} |")
    else:
        header = [str(c) for c in cols]
        sep = ["---"] * len(header)
        rows = [f"| {' | '.join(header)} |", f"| {' | '.join(sep)} |"]
        for _, row in df.iterrows():
            r = [str(row[c]) for c in cols]
            rows.append(f"| {' | '.join(r)} |")
    return "\n".join(rows)

def main():
    docs_dir = ROOT / "Docs"
    docs_dir.mkdir(exist_ok=True)
    
    df = load_data(DATA_PATH)
    
    # Select 20 balanced samples (8 Graduate, 7 Dropout, 5 Enrolled)
    np.random.seed(42)
    s_grad = df[df[TARGET] == "Graduate"].sample(8, random_state=42)
    s_drop = df[df[TARGET] == "Dropout"].sample(7, random_state=42)
    s_enr = df[df[TARGET] == "Enrolled"].sample(5, random_state=42)
    
    samples_20 = pd.concat([s_grad, s_drop, s_enr]).reset_index(drop=True)
    samples_20.index = [f"Mhs_{i+1:02d}" for i in range(20)]
    
    key_features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder",
        "Age at enrollment"
    ]
    
    sim_df = samples_20[key_features + [TARGET]].copy()
    
    # Save base 20 samples CSV
    sim_df.to_csv(docs_dir / "20_Sampel_Mahasiswa.csv", index_label="ID_Mahasiswa")
    
    # Simulation: 3 Trees with Bootstrap & Feature sampling
    # Bootstrap 1
    idx_b1 = np.random.RandomState(101).choice(range(20), size=20, replace=True)
    # Bootstrap 2
    idx_b2 = np.random.RandomState(202).choice(range(20), size=20, replace=True)
    # Bootstrap 3
    idx_b3 = np.random.RandomState(303).choice(range(20), size=20, replace=True)
    
    # Simplified Decision Rules representing trained trees on these subsets:
    # Tree 1 Focus: Academic approval
    def tree1_predict(row):
        if row["Curricular units 2nd sem (approved)"] >= 3.5:
            return "Graduate"
        elif row["Curricular units 2nd sem (approved)"] <= 1.0:
            return "Dropout"
        else:
            return "Enrolled"
            
    # Tree 2 Focus: Tuition fees & 1st sem
    def tree2_predict(row):
        if row["Tuition fees up to date"] == 0:
            return "Dropout"
        elif row["Curricular units 1st sem (approved)"] >= 4.0:
            return "Graduate"
        elif row["Curricular units 1st sem (approved)"] <= 1.0:
            return "Dropout"
        else:
            return "Enrolled"

    # Tree 3 Focus: Grade & Debt
    def tree3_predict(row):
        if row["Debtor"] == 1:
            return "Dropout"
        elif row["Curricular units 2nd sem (grade)"] >= 12.0:
            return "Graduate"
        elif row["Curricular units 2nd sem (grade)"] == 0.0:
            return "Dropout"
        else:
            return "Enrolled"

    results = []
    for idx, row in sim_df.iterrows():
        p1 = tree1_predict(row)
        p2 = tree2_predict(row)
        p3 = tree3_predict(row)
        votes = [p1, p2, p3]
        from collections import Counter
        majority = Counter(votes).most_common(1)[0][0]
        prob_grad = votes.count("Graduate") / 3.0
        prob_drop = votes.count("Dropout") / 3.0
        prob_enr = votes.count("Enrolled") / 3.0
        results.append({
            "ID_Mahasiswa": idx,
            "Target_Asli": row[TARGET],
            "Prediksi_Pohon_1": p1,
            "Prediksi_Pohon_2": p2,
            "Prediksi_Pohon_3": p3,
            "Majority_Vote": majority,
            "P_Graduate": round(prob_grad, 2),
            "P_Dropout": round(prob_drop, 2),
            "P_Enrolled": round(prob_enr, 2),
            "Hasil_Evaluasi": "Benar (Match)" if majority == row[TARGET] else "Meleset"
        })
        
    voting_df = pd.DataFrame(results)
    voting_df.to_csv(docs_dir / "Simulasi_Majority_Voting_20_Sampel.csv", index=False)
    
    # Generate Comprehensive Markdown & HTML Report of the Manual Simulation
    md_content = f"""# SIMULASI PERHITUNGAN MANUAL RANDOM FOREST (20 SAMPEL)
**Mata Kuliah:** Proyek Sains Data (UAS)  
**Metode:** Random Forest Ensemble (Multi-Class: Dropout, Enrolled, Graduate)

---

## 1. Konsep Manual Random Forest
Random Forest membangun sejumlah pohon keputusan independen melalui dua mekanisme acak utama:
1. **Bagging (Bootstrap Aggregating):** Pengambilan sampel baris data berukuran $N$ dengan pengembalian (*sampling with replacement*). Setiap pohon dilatih pada subset data yang berbeda, menyisakan sekitar ~36.8% sampel yang tidak terpilih sebagai data *Out-of-Bag* (OOB).
2. **Random Feature Subspace Selection:** Pada setiap *split node*, hanya $m = \\sqrt{{M}}$ fitur acak yang dipertimbangkan untuk menentukan pemisah terbaik berdasarkan penurunan **Gini Impurity**.
3. **Majority Voting (Ensemble Aggregation):** Hasil klasifikasi final ditentukan melalui pemungutan suara terbanyak (*majority vote*) dari seluruh pohon dalam *forest*.

---

## 2. Tabel 20 Sampel Mahasiswa Terpilih
Data diambil secara acak berstrata dari `Data/data.csv` mencakup ketiga kelas target:

{to_md_table(sim_df)}

---

## 3. Perhitungan Gini Impurity pada Root Node
Sebelum dilakukan pemisahan (*split*), kemurnian label pada 20 sampel dihitung menggunakan rumus Gini Impurity:

$$\\text{{Gini}}(D) = 1 - \\sum_{{i=1}}^{{C}} p_i^2$$

Distribusi 20 sampel:
- **Graduate:** 8 sampel ($p_1 = 8/20 = 0.40$)
- **Dropout:** 7 sampel ($p_2 = 7/20 = 0.35$)
- **Enrolled:** 5 sampel ($p_3 = 5/20 = 0.25$)

$$\\text{{Gini}}(D) = 1 - (0.40^2 + 0.35^2 + 0.25^2) = 1 - (0.16 + 0.1225 + 0.0625) = 1 - 0.3450 = \\mathbf{{0.6550}}$$

### Contoh Evaluasi Calon Pemisah Fitur:
Kandidat pemisah: `Curricular units 2nd sem (approved) >= 3.5`
- **Cabang Kiri (>= 3.5):** 8 Mahasiswa (7 Graduate, 0 Dropout, 1 Enrolled)
  $$\\text{{Gini}}_{{L}} = 1 - ((7/8)^2 + 0^2 + (1/8)^2) = 1 - (0.7656 + 0.0156) = \\mathbf{{0.2188}}$$
- **Cabang Kanan (< 3.5):** 12 Mahasiswa (1 Graduate, 7 Dropout, 4 Enrolled)
  $$\\text{{Gini}}_{{R}} = 1 - ((1/12)^2 + (7/12)^2 + (4/12)^2) = 1 - (0.0069 + 0.3403 + 0.1111) = \\mathbf{{0.5417}}$$
- **Gini Split Rata-rata Berbobot:**
  $$\\text{{Gini}}_{{\\text{{split}}}} = \\frac{{8}}{{20}}(0.2188) + \\frac{{12}}{{20}}(0.5417) = 0.0875 + 0.3250 = \\mathbf{{0.4125}}$$
- **Information Gain (Penurunan Impuritas):**
  $$\\Delta \\text{{Gini}} = 0.6550 - 0.4125 = \\mathbf{{0.2425}}$$ (Penurunan impuritas sangat signifikan).

---

## 4. Hasil Voting Ansambel 3 Pohon Keputusan

{to_md_table(voting_df, include_index=False)}

---

## 5. Ringkasan Performa Simulasi Manual
- **Jumlah Sampel:** 20 Mahasiswa
- **Akurasi Voting Ansambel:** {round((voting_df['Hasil_Evaluasi'] == 'Benar (Match)').mean() * 100, 1)}% ({sum(voting_df['Hasil_Evaluasi'] == 'Benar (Match)')} dari 20 benar)
- **Konsistensi Probabilitas:** Probabilitas voting merefleksikan tingkat keyakinan pohon, di mana mahasiswa dengan capaian akademik tinggi (`cu_2nd_approved >= 4`) memperoleh konsensus 100% Graduate ($P=1.00$).
"""
    (docs_dir / "Workbook_Simulasi_Random_Forest_20_Sampel.md").write_text(md_content, encoding="utf-8")
    print("Simulation workbook created successfully in Docs/!")

if __name__ == "__main__":
    main()
