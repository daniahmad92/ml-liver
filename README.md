# Integrasi Model K-Nearest Neighbors (KNN) dengan Hyperparameter Tunning Optuna Untuk Memprediksi Penyakit Liver - Dadan Ahmad Dani

## Domain Proyek
Hati adalah salah satu organ penting yang perlu dijaga kesehatannya [[1]](https://www.halodoc.com/artikel/fungsi-hati-perlu-dijaga-ini-8-caranya). Pasalnya, organ terbesar dalam tubuh ini berperan besar dalam semua proses pencernaan dan penyerapan zat gizi dalam tubuh serta membuang racun.

Untuk mengetahui kondisi kesehatan liver atau hati , salah satunya dengan cara melakukan Tes Fungsi Hati(TFH) [[2]](https://www.halodoc.com/kesehatan/pemeriksaan-fungsi-hati).Tes Fungsi Hati (TFH) adalah tes darah yang digunakan untuk menilai kondisi kesehatan organ hati yang bisa dilakukan baik secara rutin maupun ketika terjadi penyakit pada hati. Tes ini dilakukan dengan mengukur kadar senyawa kimia tertentu dalam darah, lalu membandingkannya dengan nilai normal senyawa kimia tersebut. Bila hasil pengukuran zat kimia menunjukkan kadar yang tidak normal, maka kemungkinan besar terdapat penyakit hati atau kerusakan hati.

Namun, Hasil normal tes fungsi hati dapat sedikit berbeda antara satu laboratorium dengan laboratorium lainnya. Perbedaan juga bisa muncul dan tergantung pada jenis kelamin maupun usia pasien, serta kondisi hamil atau tidak [[3]](https://www.sehatq.com/tindakan-medis/tes-fungsi-hati).Adapun yang diperiksa dari sampel darah tersebut yaitu SGPT,SGOT,ALP, Bilibrubin,Albumin dan lainnya

Berdasarkan latar belakang diatas bahwa untuk menentukan pasien mengalami penyakit liver atau tidak itu dibutuhkan pembacaan yang akurat dari berbagai parameter hasil tes darah seperti kadar SGPT,SGOT,ALP, Bilibrubin,Albumin dan lainnya.

Klasifikasi merupakan salah satu tugas utama dalam pembelajaran mesin yang bertujuan untuk mengelompokkan data ke dalam kelas atau label yang tepat berdasarkan fitur-fitur tertentu. Salah satu algoritma klasifikasi yang sederhana namun cukup efektif adalah K-Nearest Neighbors (KNN). KNN adalah metode pembelajaran berbasis instan (instance-based) yang memprediksi label suatu data berdasarkan mayoritas label tetangga terdekat dari data tersebut.

Model KNN sangat bergantung pada parameter n_neighbors, weight, dan metric . Parameter n_neighbors menentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Parameter weight mengatur bagaimana bobot (weight) akan diberikan pada tetangga terdekat saat melakukan prediksi, sedangkan parameter metric menentukan metrik jarak yang digunakan untuk mengukur kedekatan antara titik data dalam ruang fitur[[4]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

Pemilihan parameter n_neighbors, weight, dan metric yang optimal dapat sangat mempengaruhi performa model KNN. Namun, menemukan kombinasi parameter yang tepat secara manual dapat menjadi proses yang memakan waktu dan rumit, terutama jika ada banyak opsi parameter yang harus dijelajahi. Oleh karena itu, penting untuk mencari pendekatan otomatis yang dapat mengoptimalkan parameter-parameter ini untuk meningkatkan kinerja model KNN dalam tugas klasifikasi.

Di sinilah peran Hyperparameter Tunning Optuna menjadi relevan. Optuna merupakan salah satu pustaka (library) Python yang menerapkan teknik optimasi hiperparameter dengan menggunakan algoritma cerdas dan efisien[[5]](https://optuna.org/). Dengan mengintegrasikan Optuna pada Model KNN, kemungkinan untuk mencari kombinasi hiperparameter yang optimal dapat dilakukan dengan lebih efisien dan akurat.

## Business Understanding

### Problem Statements

Bagaimana integrasi Model K-Nearest Neighbors (KNN) dengan Hyperparameter Tunning Optuna dapat digunakan secara efektif untuk memprediksi penyakit liver dengan akurasi yang lebih tinggi?

### Goals

Mengoptimalkan hiperparameter Model K-Nearest Neighbors (KNN) menggunakan Optuna untuk meningkatkan akurasi deteksi penyakit liver.

### Solution

Membandingkan tingkat akurasi menggunakan pendekatan model K-Nearest Neighbors (KNN) dengan parameter set default dan parameter hasil dari tunning hyperparameter dengan menggunakan Optuna.

## Data Understanding

### Variabel dalam dataset

Data yang digunakan dalam proyek ini adalah data sekunder ILPD (Indian Liver Patient Dataset).yang diambil dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)).Dataset ini berisi 583 data catatan tentang pasien liver (416 pasien yg terkena penyakit liver dan 167 pasien tidak terkena penyakit liver).

| Variabel | Keterangan | Kadar Normal|
| ---------- | -------------- |-------------- |
| *Age* | Umur pasien | - |
| *Gender* | Jenis kelamin pasien |
| *TB* | Total Bilirubin| 0,3 - 1,2 mg/dL|
| *DB* | Direct Bilirubin |0 - 0,4 mg/dL|
| *Alkphos* |  Alkaline Phosphotas |40-129 U/L|
| *Sgpt* | Alamine Aminotransferase|7 - 55 U/L|
| *Sgot* | Aspartate Aminotransferase|5 - 40 u/L|
| *TP* | Total Protiens|6.3-7.9 g/dL|
| *ALB* | Albumin|3.5-5.0 g/dL|
| *A/G* | Albumin and Globulin Ratio|-|
| *Class* | Kategori kelas pasien|1: Liver dan 2:Non-Liver|


### Exploratory Data Analysis (EDA)- Univariate Analysis

![Gambar 1](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/perbandingan%20pasien%20liver%20dan%20non%20liver.png)

Gambar 1. perbandingan pasien liver dan non-liver

berdasarkan pie chart diatas,bahwa data pasien penyakit liver (71,4%) lebih banyak dibandingkan dengan data pasien non-liver (28,6%)

![Gamber 2](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/perbandingan%20pasien%20liver%20berdasarkan%20gender.png)

Gambar 2. Pie Chart Perbandingan Pasien berdasarkan Gender

jika dilihat berdasarkan jenis kelamin, bahwa pasien liver kebanyakan laki-laki (77,9%) dibandingkan perempuan (22,1%)

![Gambar 3 ](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/distribusi%20pasien%20berdasarkan%20usia.png)

Gambar 3. Distribusi Pasien Liver Berdasarkan Usia

Sedangkan jika dilihat dari rentang jenis usia, kebanyakan yang menjadi pasien liver merupakan di rentang usia 30-60 yang merupakan usia Dewasa dengan rata-rata diusia 45 seperti yang terlihat dari tabel analisis deskriptif di bawah ini

![Gambar 4](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/deskripsi%20fitur%20numerik.JPG)

Gambar 4. Statistik Deskriptif Fitur

### Exploratory Data Analysis (EDA)--Multivariate Analysis

Multivariate Analysis adalah analisis yang melibatkan dua atau lebih variabel secara bersamaan untuk memahami hubungan antara variabel-variabel tersebut dan mengidentifikasi pola yang lebih kompleks dalam data.

Adapun Teknik yang saya gunakan yaitu Correlation Matrix

Correlation matrix adalah tabel yang menampilkan koefisien korelasi antara semua pasangan variabel numerik.

![Gambar 5 ](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/korelasi%20fitur.png)

Gambar 5. Tabel Matriks Korelasi Fitur

berdasarkan gambar matrik korelasi diatas ada 4 kelompok yang memiliki korelasi yang tinggi (diatas 0,6) diantaranya:

1.DB dan TB (nilai korelasi=0,87)

2.SGot dan Sgpt (nilai korelasi=0,79)

3 ALB dan TP (nilai korelasi=0,78)

4.ALB dan A/G (nilai korelasi=0,69)


### Exploratory Data Analysis (EDA)- EDA-Menangani Missing Value

![Gambar 6](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/jumlah%20missing%20value.JPG)

Gambar 6. Tabel jumlah missing value

pada gambar diatas, ada 4 observasi yang mengalami kasus missing value pada variabel Ratio Albumin and Globuline (A/G). Masalah ini selanjutnya akan diatasi dengan mengimputasikan nilai rata-rata (mean) ke dalam 4 observasi tersebut karena variabel ini mempunyai skala numerik

### Exploratory Data Analysis (EDA)- Visualisasi dan penanganan Outlier

Outlier adalah nilai yang berada jauh dari mayoritas data dalam distribusi. Metode IQR (Interquartile Range) adalah salah satu cara untuk mendeteksi dan menangani outlier dalam analisis data. IQR dihitung sebagai selisih antara kuartil pertama (Q1) dan kuartil ketiga (Q3) dari data.

Berikut adalah langkah-langkah untuk mendeteksi outlier menggunakan metode IQR:

1. Urutkan data secara ascending (pengurutan dari nilai terkecil ke nilai terbesar).
2. Hitung kuartil pertama (Q1) dan kuartil ketiga (Q3):
   - Kuartil pertama (Q1) adalah nilai median dari setengah bagian bawah data (setengah data dengan nilai lebih rendah dari Q2, median).
   - Kuartil ketiga (Q3) adalah nilai median dari setengah bagian atas data (setengah data dengan nilai lebih tinggi dari Q2, median).

3. Hitung Interquartile Range (IQR):
   - IQR dihitung sebagai selisih antara Q3 dan Q1: IQR = Q3 - Q1.

4. Tentukan Batas Atas (Upper Fence) dan Batas Bawah (Lower Fence):
   - Batas Atas (Upper Fence) adalah nilai maksimum yang masih dianggap tidak outlier: Upper Fence = Q3 + (1.5 * IQR).
   - Batas Bawah (Lower Fence) adalah nilai minimum yang masih dianggap tidak outlier: Lower Fence = Q1 - (1.5 * IQR).

5. Identifikasi dan Tangani Outlier:
   - Identifikasi nilai-nilai yang berada di luar Batas Atas dan Batas Bawah sebagai outlier.

6. Tangani outlier dengan memilih salah satu dari tiga opsi berikut:

     a. Menghapus Outlier: Hapus outlier dari dataset jika jumlah dan dampaknya terhadap analisis tidak signifikan.

     b. Imputasi: Gantikan nilai outlier dengan nilai lain yang lebih masuk akal, misalnya nilai median atau rata-rata dari data yang tidak outlier.

     c. Binning: Kelompokkan nilai outlier ke dalam kategori tertentu untuk mengurangi efeknya.

![Gambar 7](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/iqr.JPG)

Gambar 7. Script untuk menghitung jumlah outlier

![Gambar 8](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/total%20outlier.JPG)

Gambar 8. Jumlah Outlier Tiap Fitur

Berdasarkan tabel diatas, fitur yang tidak memiliki outlier adalah Fitur Age dan ALB. sedangkan untuk TP dan A/G jumlahnya sedikit yaitu hanya 8 dan 10 data saja.

sedangkan pada fitur TB,DB, Alkphos,Sgpt,Sgot memiliki jumlah outlier yang cukup banyak

#### Visualisasi Outlier

![Gambar 9](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/boxplot%20outlier.png)

Gambar 9. Histogram dan Boxplot visualisasi outlier

#### Penangan Outlier

jika outlier diatas upper diganti nilainya dengan nilai upper dan outlier dibawah lower diganti nilainya dengan lower.
outputnya dapat dilihat dari histogram dan boxplot di bawah ini

![Gambar 10](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/data%20bersih.png)

Gambar 10. Boxplot dan Histogram setelah penanganan outlier


## Data Preparation

### Label Encoding

adapun varibel kategori yang diubah yaitu Gender dan Class

- pada data Gender,{"Male":1,"Female":0}
- pada data Class, {"Liver":1 , "Non-Liver":2}

### Split data training dan testing

***train_test_split*** adalah fungsi dari pustaka scikit-learn yang digunakan untuk membagi dataset menjadi data training dan data testing.Pada penelitian ini saya membagi data tersebut menjadi **80% sebagai training dan 20% sebagai testing**.

### Normalisasi Data

Normalisasi data adalah salah satu tahap penting dalam pra-pemrosesan (preprocessing) data sebelum melatih model pembelajaran mesin. Normalisasi bertujuan untuk mengubah skala nilai fitur-fitur dalam dataset sehingga memiliki mean (rata-rata) 0 dan standar deviasi (standard deviation) 1. Salah satu teknik normalisasi yang umum digunakan adalah menggunakan **StandardScaler** dari pustaka scikit-learn.


Langkah-langkah yang dilakukan oleh **StandardScaler** adalah sebagai berikut:

1. Hitung Mean dan Standar Deviasi:
   - Pertama, hitung nilai mean dan standar deviasi dari setiap variabel numerik di dataset.

2. Transformasi Data:
   - Selanjutnya, setiap nilai pada variabel numerik diubah menjadi nilai standar menggunakan rumus z-score:
     z = (x - mean) / std_dev
   di mana z adalah nilai standar, x adalah nilai asli, mean adalah rata-rata, dan std_dev adalah standar deviasi.

3. Skala Data:
   - Dalam tahap ini, data telah discaling sehingga rata-rata variabel adalah 0 dan standar deviasi adalah 1.
   - 

### Resample Data

Untuk penanganan ketidakseimbangan kelas, pada penelitian ini saya menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE digunakan untuk menangani ketidakseimbangan kelas dengan mengoversampling sampel dari kelas minoritas agar jumlah sampel kelas mayoritas dan kelas minoritas menjadi lebih seimbang.

Teknik ini bekerja dengan cara berikut:

1. SMOTE mengidentifikasi sampel individu pada kelas minoritas.

2. Untuk setiap sampel minoritas, SMOTE memilih beberapa tetangga terdekat dari kelas minoritas menggunakan algoritma K-Nearest Neighbors (KNN).

3. SMOTE kemudian membuat sampel sintetis baru dengan menggabungkan sampel minoritas asli dengan tetangga terdekat dan menambahkan variasi pada fitur-fitur ini.

4. Sampel sintetis ini ditambahkan ke dataset, sehingga meningkatkan jumlah sampel pada kelas minoritas.


## Modeling - Parameter KNN Default

Algoritma K-Nearest Neighbors (KNN) adalah salah satu algoritma yang bekerja dengan cara mencari K tetangga terdekat dari suatu data uji dan kemudian melakukan klasifikasi dari tetangga tersebut untuk menentukan label atau nilai prediksi dari data uji.

adapun parameter input model knn secara default seperti pada tabel di bawah ini

| parameter | Opsi | Default|
| ---------- | -------------- |-------------- |
| *n_neighbors* | int |5|
| *weights* | uniform,distance|uniform|
| *algorithm* | auto,ball_tree,kd_tree,brute|auto|
| *leaf_size* |int|30|
| *p* |int|2|
| *metric* |euclidean,manhattan,minkowski|minkowski|
| *metric_params* |dict|None|
| *n_jobs* |int|None|


![Gambar 11](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/model_awal.JPG)

Gambar 11. Script KNN parameter default


1. **n_neighbors**:

  Parameter **n_neighbors** adalah jumlah tetangga terdekat yang akan digunakan untuk melakukan prediksi pada setiap data baru

2. **weigth**

  Parameter **weight** digunakan untuk memberikan bobot pada tetangga berdasarkan jaraknya dari data baru yang akan diprediksi. Terdapat dua opsi umum untuk weight

  - **uniform**: Setiap tetangga diberi bobot yang sama. Ini berarti tetangga terdekat dan terjauh memiliki kontribusi yang sama dalam proses prediksi.
  
  - **distance:** Bobot diberikan berdasarkan jaraknya. Tetangga yang lebih dekat akan memiliki kontribusi yang lebih besar dalam proses prediksi dibandingkan tetangga yang lebih jauh. Ini memungkinkan model untuk memberikan perhatian lebih pada tetangga yang lebih dekat, yang dapat meningkatkan akurasi prediksi

3. **metric:**

	Parameter **metric** digunakan untuk menentukan metrik jarak yang akan digunakan untuk mengukur kedekatan antara data. Metrik ini penting karena akan mempengaruhi bagaimana KNN menghitung jarak antara data poin. Contoh metrik umum yang dapat digunakan adalah:
  
  - **euclidean**: Menggunakan jarak Euclidean biasa untuk mengukur kedekatan antara data poin. Ini cocok untuk data yang memiliki skala numerik.
  
  - **manhattan**: Menggunakan jarak Manhattan (juga dikenal sebagai jarak L1) yang mengukur jarak horizontal dan vertikal antara data poin. Cocok untuk data yang memiliki fitur diskrit atau data ordinal.
  
  - **minkowski**: Metrik umum yang menggeneralisasi jarak Euclidean dan Manhattan dengan memperkenalkan parameter p. Ketika p=1, itu menjadi jarak Manhattan, dan ketika p=2, itu menjadi jarak Euclidean.



## Tunning Hyperparameter Optuna

**Optuna** adalah sebuah library Python yang digunakan untuk optimasi hyperparameter secara otomatis


### Fungsi Evaluasi (Objective Function)

**Fungsi objektif (Objective Function)** adalah sebuah fungsi yang akan dievaluasi selama proses optimasi hyperparameter.Tujuan utama dari fungsi objektif adalah untuk memberikan nilai skor atau evaluasi berdasarkan performa model. Nilai skor ini akan digunakan oleh algoritma optimasi untuk menentukan kombinasi hyperparameter mana yang menghasilkan hasil terbaik

adapun parameter KNN yang akan ditunning yaitu **n_neighbors,weight,dan metric**

| parameter | Opsi |
| ---------- | --------------|
| *n_neighbors* | 1 s.d 10|
| *weights* | uniform,distance|
| *metric* |euclidean,manhattan,minkowski|

![Gambar 12](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/tunning-optuna.JPG)

Gambar 12. Script Tunning Hyperparameter Optuna

### Optimization History Plot

Grafik Optimization History Plot pada Optuna merupakan alat yang berguna untuk memvisualisasikan progres optimasi hyperparameter selama proses pencarian kombinasi hyperparameter terbaik. Plot dapat memvisualisasikan bagaimana nilai fungsi objektif (akurasi) berubah selama iterasi dari algoritma optimasi

Pada plot "Optimization History", sumbu-x mewakili iterasi (trial) yang dilakukan oleh algoritma optimasi, sedangkan sumbu-y mewakili nilai fungsi objektif pada setiap iterasi. Nilai fungsi objektif ini menunjukkan performa model dengan kombinasi hyperparameter yang diuji pada iterasi tersebut

Pada Penelitian ini, saya membuat 100 iterasi

![Gambar 13](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/optuna-histori-plot.JPG)

Gambar 13. Optimazation History Plot

### Hyperparameter Importances Optuna

![Gambar 14](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/optuna-hyperparameter.JPG)

Gambar 14. Hyperparameter Importances Optuna

### Best Parameter Optuna

![Gambar 15 ](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/optuna-best-params.JPG)

Gambar 15. Best Parameter Optuna

### Create Model best Parameter

![Gambar 16](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/model-best.JPG)

Gambar 16. Script Model KNN dengan menggunakan Best Parameter

## Nilai Akurasi Model

### Nilai Akurasi Model KNN Default
berikut adalah classification Report dari model KNN dengan parameter:
- n_neighbors =5
- weight = uniform
- metric = minkowski

![Gambar 17](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/klasifikasi-report-default.JPG)

Gambar 17. Hasil Classification Report Dengan parameter Default KNN

berdasarkan gambar diatas, bahwa nilai akurasi model KNN Default adalah 0,68

### Nilai Akurasi Model KNN menggunakan Optuna

berikut adalah classification Report dari model KNN setalah dilaukan hyperparameter tuning dengan best parameter:

- n_neighbors =8
- weight = distance
- metric = manhattan

![Gambar 18](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/klasifikasi-report-optuna.JPG)

Gambar 18. Hasil Classification Report Dengan parameter Hasil Tunning Optuna

setalah dilakukan tunning hyperparameter menggunakan optuna, didapatkan nilai akurasinya menjadi 0,72


### Perbandingan NIlai AKurasi model KNN dengan parameter Default dan Best Parameter Optuna

setalah dilakukan tuning hyperparameter dengan optuna, didapatkan adanya peningkatkan nilai akurasi dari 0,68 menjadi 0,72 atau dengan peningkatan 5,9%

![Gambar 19](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/akurasi.JPG)

Gambar 19. Grafik Perbandingan NIlai Akurasi Default dan Optuna


## Kesimpulan

Setelah dilakukan optimasi hiperparameter dengan Optuna,nilai akurasi deteksi penyakit liver meningkat dari 0,67 menjadi 0,72 atau dengan peningkatan sebesar 5,9%.Dengan demikian, dapat disimpulkan bahwa integrasi Hyperparameter Tunning Optuna dalam Model KNN berhasil meningkatkan akurasi deteksi penyakit liver

## Referensi

[[1]](https://www.halodoc.com/artikel/fungsi-hati-perlu-dijaga-ini-8-caranya) Halodoc. (2019). *Penyakit Liver*. Diakses pada 22 Juli 2023.https://www.halodoc.com/artikel/fungsi-hati-perlu-dijaga-ini-8-caranya

[[2]](https://www.halodoc.com/kesehatan/pemeriksaan-fungsi-hati) Halodoc.(2022).*Pemeriksaan Fungsi Hati*.Diakses pada 22 Juli 2023.https://www.halodoc.com/kesehatan/pemeriksaan-fungsi-hati

[[3]](https://www.sehatq.com/tindakan-medis/tes-fungsi-hati) SehatQ.(2023).*Tes Fungsi Hati*.Diakses pada 22 Juli 2023.https://www.sehatq.com/tindakan-medis/tes-fungsi-hati

[[4]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) Scikit-Learn.(2023).*KNeighborsClassifier*.Diakses pada 22 Juli 2023.https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[[5]](https://optuna.org/) Optuna.(2023).*Optuna*.Diakses pada 22 Juli 2023.https://optuna.org



















