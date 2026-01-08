# UAS-MACHINE-LEARNING---MOCHAMAD-FACHRIZ-AZIS-WANTAH
SEBAGAI TUGAS AKHIR MATA KULIAH MACHINE LEARNING

# Decision Tree Classification – Iris Dataset

## Deskripsi

Repository ini berisi implementasi algoritma *Decision Tree* untuk tugas klasifikasi menggunakan **Dataset Iris**. Proyek ini dibuat sebagai bagian dari **Ujian Akhir Semester (UAS) Machine Learning**.

## Dataset

- **Nama**: Iris Dataset
- **Sumber**: scikit-learn
- **Jumlah data**: 150 sampel
- **Jumlah fitur**: 4 fitur numerik

- Sepal length

- Sepal width

- Petal length

- Petal width

- **Jumlah kelas**: 3 (Setosa, Versicolor, Virginica)

## Metodologi

1. Load dataset Iris

2. Exploratory Data Analysis (EDA) singkat

3. Split data menjadi training dan testing set

4. Pembangunan model Decision Tree

5. Evaluasi model menggunakan metrik klasifikasi

6. Visualisasi pohon keputusan

## Model

- **Algoritma**: Decision Tree Classifier
- **Criterion**: Gini Index
- **Max Depth**: 3

## Evaluasi

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-score

Model menghasilkan akurasi yang tinggi dan stabil pada dataset Iris.

## Tools & Library

- Python 3.x
- Scikit-learn
- Pandas
- Matplotlib

## Cara Menjalankan Program

```bash

pip install -r requirements.txt

python src/decision_tree_iris.py

Atau buka file notebook:

jupyter notebook notebooks/decision_tree_iris.ipynb
