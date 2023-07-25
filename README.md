# Integrasi Model K-Nearest Neighbors (KNN) dengan Hyperparameter Tunning Optuna Untuk Mendeteksi Penyakit Liver Sejak Dini - Dadan Ahmad Dani

## Domain Proyek
Hati adalah salah satu organ penting yang perlu dijaga kesehatannya.Pasalnya,organ terbesar dalam tubuh ini berperan besar dalam semua proses pencernaan dan penyerapan zat gizi dalam tubuh serta membuang racun.

Untuk mengetahui kondisi kesehatan liver atau hati , salah satunya dengan cara melakukan Tes Fungsi Hati(TFH).Tes Fungsi Hati (TFH) adalah tes darah yang digunakan untuk menilai kondisi kesehatan organ hati yang bisa dilakukan baik secara rutin maupun ketika terjadi penyakit pada hati. Tes ini dilakukan dengan mengukur kadar senyawa kimia tertentu dalam darah, lalu membandingkannya dengan nilai normal senyawa kimia tersebut. Bila hasil pengukuran zat kimia menunjukkan kadar yang tidak normal, maka kemungkinan besar terdapat penyakit hati atau kerusakan hati [[1]](https://www.halodoc.com/artikel/cek-kesehatan-hati-dengan-tes-fungsi-hati-ini).

Berdasarkan penelitian yang dilakukan oleh British Liver Trust mengungkapkan bahwa penyakit hati atau Liver merupakan penyebab kematian terbesar pada orang yang berusia antara 35-49 tahun, khususnya di Inggris. Penelitian tersebut juga menyatakan bahwa penyakit hati diperkirakan akan menggeser penyakit jantung sebagai penyebab terbesar kematian dini dalam beberapa tahun mendatang [[2]](https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/)

## Business Understanding

Kematian akibat penyakit hati atau liver telah menjadi masalah kesehatan masyarakat yang signifikan di seluruh dunia. Penyakit hati, seperti sirosis dan kanker hati, dapat berkembang secara perlahan tanpa gejala yang jelas pada tahap awal, sehingga seringkali sulit untuk dideteksi sejak dini. Oleh karena itu, penting untuk memiliki alat deteksi dini yang efektif agar dapat mendiagnosis dan mengobati penyakit hati sebelum mencapai tahap lanjut yang lebih serius dan berpotensi fatal.

Ada beberapa metode yang dapat digunakan untuk mendeteksi penyakit hati sejak dini salah satunya yaitu menggunakan Teknologi Kecerdasan Buatan. Dengan kemajuan dalam bidang kecerdasan buatan, teknologi seperti machine learning dapat digunakan untuk menganalisis data medis dan mencari pola yang menunjukkan potensi masalah hati. Ini dapat membantu dalam deteksi dini dan memberikan peringatan lebih awal kepada pasien dan dokter

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan sebelumnya, maka projek ini dikembangkan untuk menjawab permasalahan berikut: 
   1. Bagaimana cara menerapkan model *Machine Learning* sehingga dapat mendeteksi penyakit liver sejak dini?
   2. Berapa tingkat akurasi dari model *Machine Learning* yg dibuat?

Rumusan masalah ini berfokus pada penilaian kinerja model *Machine Learning* dalam tugas klasifikasi, yaitu membedakan antara dua kelas, yaitu pasien dengan penyakit liver (positif) dan pasien tanpa penyakit liver (negatif).

### Goals

Tujuan proyek ini diataranya:
   1. Membuat model *Machine Learning* untuk mendeteksi pasien penyakit liver
   2. Mengukur tingkat akurasi dari model *Machine Learning* yang telah dibuat

### Solution

Solusi yang ditawarkan untuk mendeteksi penyakit liver sejak dini adalah dengan menggunakan model klasifikasi, seperti K-Nearest Neighbors (KNN). Model ini akan memanfaatkan informasi rekam medis dari pasien-pasien sebelumnya untuk membedakan antara pasien dengan penyakit liver (positif) dan pasien tanpa penyakit liver (negatif).

Model KNN sangat bergantung pada parameter n_neighbors, weight, dan metric . Parameter n_neighbors menentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Parameter weight mengatur bagaimana bobot (weight) akan diberikan pada tetangga terdekat saat melakukan prediksi, sedangkan parameter metric menentukan metrik jarak yang digunakan untuk mengukur kedekatan antara titik data dalam ruang fitur[[3]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

Pemilihan parameter n_neighbors, weight, dan metric yang optimal dapat sangat mempengaruhi performa model KNN. Namun, menemukan kombinasi parameter yang tepat secara manual dapat menjadi proses yang memakan waktu dan rumit, terutama jika ada banyak opsi parameter yang harus dijelajahi. Oleh karena itu, penting untuk mencari pendekatan otomatis yang dapat mengoptimalkan parameter-parameter ini untuk meningkatkan kinerja model KNN dalam tugas klasifikasi.

Di sinilah peran Hyperparameter Tunning Optuna menjadi relevan. Optuna merupakan salah satu pustaka (library) Python yang menerapkan teknik optimasi hiperparameter dengan menggunakan algoritma cerdas dan efisien[[4]](https://optuna.org/). Dengan mengintegrasikan Optuna pada Model KNN, kemungkinan untuk mencari kombinasi hiperparameter yang optimal dapat dilakukan dengan lebih efisien dan akurat.

Dengan menggunakan solusi ini, diharapkan dapat deteksi dini penyakit liver dapat ditingkatkan sehingga penanganan lebih lanjut bisa dilakukan dengan lebih tepat dan efektif.

## Data Understanding

Data yang digunakan dalam proyek ini adalah data sekunder ILPD (Indian Liver Patient Dataset).yang diambil dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)).Dataset ini berisi 583 data catatan tentang pasien liver.Dari data tersebut, terdapat 416 pasien yang terkena penyakit liver dan 167 pasien yang tidak terkena penyakit liver.

### Struktur Dataset

Dataset ini memiliki 583 baris and 11 kolom. Dari 11 kolom tersebut, 1 kolom memiliki tipe data object, 5 kolom integer,dan 5 kolom lagi bertipe data float. Adapun deskripsi dan tipe datanya dapat dilihat pada Tabel 1 dibawah ini.

Tabel 1. Variabel dalam dataset

| Variabel | Deskripsi|Tipe Data|
| ---------- | -------------- |-------------- |
| *Age* | Usia pasien |int64|
| *Gender* | Jenis kelamin pasien |object|
| *TB* | Kadar total bilirubin dalam darah pasien|float64|
| *DB* | Kadar bilirubin langsung dalam darah pasien|float64|
| *Alkphos* |Kadar fosfatase alkali dalam darah pasien|int64|
| *Sgpt* | Kadar enzim Sgpt dalam darah pasien|int64|
| *Sgot* | Kadar enzim Sgot dalam darah pasien|int64|
| *TP* | Kadar total protein dalam darah pasien|float64|
| *ALB* |Kadar albumin dalam darah pasien|float64|
| *A/G* | Rasio albumin dan globulin dalam darah pasien|float64|
| *Class* |Variabel target yang menunjukkan apakah pasien menderita penyakit liver atau tidak (1 untuk Liver, 2 untuk Non-Liver).|int64|

### Inisialisasi Variabel Fitur dan Target

Variabel fitur adalah kolom-kolom dalam dataset yang digunakan sebagai input atau prediktor untuk mengembangkan model deteksi penyakit liver. Sedangkan Variabel target adalah kolom dalam dataset yang menjadi output atau label yang ingin diprediksi oleh model.

Dari 11 Variabel yang terdapat dalam dataset,ada 10 variabel yang menjadi *Variabel Fitur* dan 1 variabel yang menjadi *Variabel Target*

- Variabel Fitur : Age,Gender,TB,DB,Alkhpos,Sgpt,Sgot,TP,ALB dan A/G

- Variabel Target: Class

Dalam pengembangan model deteksi penyakit liver menggunakan dataset ini, variabel fitur akan digunakan sebagai input untuk memprediksi nilai target (penyakit liver atau tidak). Model akan belajar dari pola yang terdapat pada data untuk melakukan klasifikasi pasien menjadi dua kelas berdasarkan nilai dari variabel-fitur yang telah diberikan.

### Deteksi Missing Value

*Missing Value* (nilai yang hilang) adalah kondisi di mana data atau nilai pada suatu kolom dalam dataset tidak ada atau kosong
Untuk mendeteksi ada atau tidaknya *Missing value* ,dapat mengggunakan fungsi yang ada pada library Pandas yaitu .isnull().

Melalui deteksi missing value menggunakan .isnull(), didapatkan ada 4 nilai yang kosong pada kolom *A/G*.Terdapat tiga cara untuk mengatasi *missing value* yaitu dibiarkan, dihilangkan dan mensubtitusi nilai yang hilang menggunakan nilai mean / median / modus. Cara yang digunakan untuk mengatasi *missing value* pada proyek ini yaitu dengan cara mensubtitusikan nilai *mean*  kedalam data kosong tersebut.

### Deteksi Imbalance pada Variabel Target

Imbalance (ketidakseimbangan) adalah kondisi di mana terdapat perbedaan yang signifikan antara jumlah data pada masing-masing kelas dalam dataset. Dalam konteks klasifikasi, imbalance terjadi ketika jumlah sampel atau data pada satu kelas jauh lebih banyak atau jauh lebih sedikit dibandingkan dengan kelas lainnya.Ketidakseimbangan kelas dapat menjadi masalah dalam pengembangan model klasifikasi, terutama ketika model lebih cenderung memprediksi ke kelas mayoritas dan mengabaikan kelas minoritas. Hal ini dapat menyebabkan akurasi model yang tinggi secara keseluruhan, tetapi performa yang buruk dalam memprediksi kelas minoritas yang sebenarnya lebih penting dalam aplikasi tertentu (misalnya, deteksi penyakit langka).


![Gambar 1](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/perbandingan%20pasien%20liver%20dan%20non%20liver.png)

Gambar 1. perbandingan pasien liver dan non-liver

Berdasarkan informasi yang tertera pada Gambar diatas, terdapat dua kelas pada variabel target:

   1. Kelas "Liver" (positive class) dengan persentase 71,4%.
   2. Kelas "Non-Liver" (negative class) dengan persentase 28,6%.

Perbedaan persentase yang cukup besar antara dua kelas tersebut menunjukkan **adanya ketidakseimbangan data pada variabel target**. Jumlah data pasien dengan penyakit liver (kelas positif) lebih banyak daripada pasien tanpa penyakit liver (kelas negatif). Kondisi ini dapat mempengaruhi performa model klasifikasi, terutama jika model cenderung memprediksi ke kelas mayoritas (penyakit liver) dan mengabaikan kelas minoritas (non-liver).

### Exploratory Data Analysis (EDA)- Univariate Analysis

   - Visualisasi Variabel Fitur (Kategori Gender)

      ![Gender](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/Gambar_gender.png)

   - Viusalisasi Variabel Fitur (Numerik)

      ![Histogram](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/Gambar_histogram.png)

   - Statistik Deskriptif Variabel Fitur Numerik

      ![stat_des](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/deskripsi%20fitur%20numerik.JPG)


### Exploratory Data Analysis (EDA)-Multivariate Analysis

   - Korelasi Variabel Fitur Numerik

   ![Korelasi](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/korelasi%20fitur.png)




### Outlier
   
   - Visualisasi Outlier mengguakan Boxplot

     ![outlier_box](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/Gambar_boxplot.png)

   - Deteksi Jumlah Outlier menggunakan IQR

```
def count_Outliers(X_num):
    indices = [x for x in X_num.index]
    out_indexlist = []
    outlier_tbl=[]
    for col in feature_numeric:
        q1 = np.percentile(X_num[col], 25)
        q3 = np.percentile(X_num[col], 75)
        iqr = q3 - q1
        lower = q1 - (iqr*1.5)
        upper = q3 + (iqr*1.5)
        outliers_index = X_num[col][(X_num[col] < lower) | (X_num[col] > upper)].index.tolist()
        outliers = X_num[col][(X_num[col] < lower) | (X_num[col] > upper)].values
        out_indexlist.extend(outliers_index)
        outlier_tbl.append({
            'Fitur': col,
            'Jml Outlier': len(outliers),
        })
    print('\nTotal outliers: ', len(out_indexlist))
    out_df = pd.DataFrame(outlier_tbl,columns=['Fitur','Jml Outlier'])
    return out_df

```


   ![outlier_total](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/total%20outlier.JPG)



## Data Preparation

### Label Encoding

Tahapan ini bertujuan untuk merubah data kategorik menjadi data numerik.Terdapat 1 variabel yang bertipe data kategori yaitu variabel Gender.Pada variabel Gender berisi kategori "Male" dan "Female".

Untuk melakukan Label Encoding, pada penelitian ini akan menggunakan kelas LabelEncoder dari library scikit-learn dengan tahapan sebagai berikut:

1. Buat objek LabelEncoder yang akan digunakan untuk melakukan Label Encoding
2. Selanjutnya, lakukan proses fit_transform pada data Gender untuk mengubah nilai kategorikal menjadi nilai numerik.

Output yang dihasilkan adalah sebagai berikut:

| Kategori  | Numerik  |
| ----------| ---------|
|  Female   |   0      |
|  Male     |   1      |

### Split data training dan testing

Pada tahap ini, dataset akan dibagi menjadi data training dan data uji data testing. Data training akan digunakan untuk melatih model,sedangkan data testing akan digunakan untuk menguji kinerja model yang telah dilatih pada data training.

***train_test_split*** adalah fungsi dari pustaka scikit-learn yang digunakan untuk membagi dataset menjadi data training dan data testing.Pada penelitian ini , data tersebut dibagi menjadi **80% sebagai data training dan 20% sebagai data testing**.

```
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=123,stratify= y)

```

Perubahan jumlah data awal menjadi data training dan data testing dapat dilihat dari tabel di bawah ini.

| data awal | data training| data testing |
| ----------| ------------ |--------------|
|  583  rows|   466 rows   |  117  rows   |

Sedangkan jumlah pasien liver dan Non-liver pada data training dan testing dapat dilihat dari tabel di bawah ini.

| Class     | data awal    | data training| data testing |
| ----------| ------------ |--------------|--------------|
|  Liver    |   416 rows   |  333  rows   |  83  rows    |
|  Non-Liver|   167 rows   |  133  rows   |  34  rows    |


### Normalisasi Data

Dalam klasifikasi, normalisasi data merujuk pada proses mengubah skala data sehingga setiap fitur memiliki rentang yang serupa atau sama, tanpa mengubah distribusi relatif antara fitur-fitur tersebut.

K-Nearest Neighbors (KNN)  menghitung jarak antara data untuk menentukan tetangga terdekat. Jika skala fitur berbeda-beda, fitur dengan skala besar akan memiliki pengaruh lebih besar pada perhitungan jarak daripada fitur dengan skala kecil. Normalisasi membantu KNN memberikan klasifikasi yang lebih adil antara fitur-fitur yang berbeda skala.

Salah satu teknik normalisasi yang umum digunakan adalah menggunakan **StandardScaler** dari pustaka scikit-learn.Adapun untuk menormalisasi data dengan menggunakan StandardScaler dengan cara sebagai berikut:

1. Buat objek StandardScaler untuk melakukan normalisasi data.
2. Lakukan normalisasi pada data menggunakan metode fit_transform dari objek StandardScaler.

```
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

```

### Resample Data

Berdasarkan pemaparan sebelumnya, bahwa terdapat ketidakseimbangan kelas (imblance) pada data target.Jumlah data kelas liver (71,4%) lebih banyak daripada kelas Non-Liver (28,6%).Dalam Hal ini kelas mayoritas adalah kelas Liver dan Kelas minoritas adalah kelas Non-Liver.

Untuk penanganan ketidakseimbangan kelas, teknik yang digunakan pada penelitian ini yaitu dengan cara mengoversampling sampel dari kelas minoritas. Dengan Tujuan supaya jumlah sampel kelas mayoritas dan kelas minoritas menjadi lebih seimbang.
Teknik oversampling yang akan digunakan yaitu teknik SMOTE (Synthetic Minority Over-sampling Technique).

Teknik SMOTE diilustrasikan seperti gambar dibawah ini

![smote](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/SMOTE.JPG)


Teknik SMOTE bekerja dengan cara berikut:

1. SMOTE mengidentifikasi sampel individu pada kelas minoritas.

2. Untuk setiap sampel minoritas, SMOTE memilih beberapa tetangga terdekat dari kelas minoritas menggunakan algoritma K-Nearest Neighbors (KNN).

3. SMOTE kemudian membuat sampel sintetis baru dengan menggabungkan sampel minoritas asli dengan tetangga terdekat dan menambahkan variasi pada fitur-fitur ini.

4. Sampel sintetis ini ditambahkan ke dataset, sehingga meningkatkan jumlah sampel pada kelas minoritas.

Untuk menggunakan resampling data menggunakan SMOTE, bisa memakai pustaka imblearn.over_sampling .Adapun langkah penggunaanya yaitu:

1. Buat objek SMOTE
2. Gunakan metode fit_resample() dari objek SMOTE untuk melakukan resampling data

Adapun contoh scriptnya dapat dilihat dibawah ini:
```
smote = SMOTE(sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(X, y)
```

## Modeling

Pada tahap ini, model dibangun dengan menggunakan Algoritma ***K-Nearest Neighbors (KNN)*** dan dioptimasi dengan menggunakan ***Optuna*** sebagai tuning hyperparameternya.

Untuk mendapatkan performa model KNN yang optimal, dalam pembuatan model ini dilakukan 2 pendekatan:

1. Melihat pengaruh data training terhadap performa model

   Dalam tahap ini peneliti membandingkan 2 buah data training yang diterapkan ke model KNN dengan parameter default.
   
   Adapun data yang dibandingkan diantarnya:

   - Data training hasil dari proses normalisasi menggunakan StandarScaler (x_train_scaler,y_train_scaler)
   - Data training hasil dari proses StandarScaler dan Resample SMOTE (x_train_resampled, y_train_resampled)
   
   Sedangkan parameter default yang diterapkan ke dalam model KNN diataranya:

   - n_neighbors= 5 
   - weights: 'distance'
   - metric: 'manhattan'
   
2. Mencari kombinasi hyperparameter untuk menghasilkan performa model yang optimal

   Dalam tahap ini, peneliti menggunakan Optuna sebagai tunning hyperparameternya.Sedangkan data yang digunakan sebagai data latih adalah data yang memiliki nilai akurasi dan recall model yang paling tinggi dari tahap sebelumnya. 
   
   Adapun opsi dari parameter yang diset kedalam optuna diantaranya:

   - n_neighbors: nilainya dari 1 sampai 10
   - weight: ['uniform', 'distance']
   - metric : ['euclidean', 'manhattan', 'minkowski']

### K-Nearest Neighbors (KNN)

#### Prinsip Kerja Algoritma KNN

K-Nearest Neighbors (KNN) adalah algoritma klasifikasi yang bekerja berdasarkan prinsip kedekatan data. Ketika diberikan data uji, KNN mencari k tetangga terdekat dari data uji di dalam data latih. Kelas yang paling sering muncul dari k tetangga terdekat akan diambil sebagai prediksi untuk data uji. KNN mengasumsikan bahwa data dengan fitur yang mirip cenderung memiliki label kelas yang sama.

#### Parameter KNN

KNN diimplementasikan menggunakan library scikit-learn, yang menyediakan algoritma KNN dengan berbagai pilihan hyperparameter, 
Pada library scikit-learn, terdapat beberapa parameter default yang digunakan dalam model KNN,diantaranya:

1. ***n_neighbors (int, default=5):*** Parameter ini menentukan jumlah tetangga terdekat yang akan digunakan untuk melakukan prediksi kelas pada data uji. Nilai defaultnya adalah 5, yang berarti KNN akan mencari 5 tetangga terdekat dari data uji dan memilih mayoritas kelas dari kelima tetangga tersebut sebagai prediksi.

2. ***weights (str or callable, default='uniform'):*** Parameter ini menentukan jenis bobot yang akan digunakan dalam perhitungan jarak antara data uji dengan data latih. Nilai 'uniform' berarti semua tetangga memiliki bobot yang sama, sedangkan nilai 'distance' berarti bobot tetangga sebanding dengan kebalikannya dari jaraknya. Selain itu, kita juga dapat menggunakan fungsi callable yang akan memberikan bobot kustom berdasarkan jarak.

3. ***metric (str or callable, default='minkowski'):*** Parameter ini menentukan metrik jarak yang akan digunakan dalam perhitungan jarak antara data uji dan data latih. Nilai 'minkowski' mengindikasikan penggunaan metrik Minkowski dengan nilai p=2, yang sama dengan metrik Euclidean. Nilai lain yang umum digunakan adalah 'manhattan' untuk metrik Manhattan dan 'euclidean' untuk metrik Euclidean.

4. ***algorithm (str, default='auto'):*** Parameter ini menentukan algoritma yang akan digunakan untuk mencari tetangga terdekat. Nilai 'auto' akan memilih algoritma yang paling sesuai berdasarkan ukuran dataset dan jenis metrik. Nilai 'ball_tree', 'kd_tree', dan 'brute' adalah pilihan algoritma yang dapat ditentukan secara eksplisit.

5. ***leaf_size (int, default=30):*** Parameter ini menentukan ukuran daun (leaf size) yang digunakan dalam algoritma BallTree atau KDTree. Nilai defaultnya adalah 30.

6. ***p (int, default=2):*** Parameter ini digunakan jika metrik Minkowski digunakan. Nilai defaultnya adalah 2, yang mengindikasikan penggunaan metrik Euclidean. Jika p=1, maka metrik akan menjadi Manhattan distance.

7. ***n_jobs (int, default=None):*** Parameter ini menentukan jumlah pekerjaan yang akan dijalankan secara paralel. Nilai defaultnya adalah None, yang berarti semua CPU yang tersedia akan digunakan.

### Optuna untuk Tuning Hyperparameter

Tuning hyperparameter merupakan salah satu tahap penting dalam pengembangan model machine learning. Tujuan dari tuning hyperparameter menggunakan library Optuna adalah untuk mencari kombinasi hyperparameter yang menghasilkan performa model yang optimal. Hyperparameter adalah parameter yang harus diatur sebelum melatih model. Oleh karena itu, penentuan hyperparameter yang tepat sangat mempengaruhi kualitas dan kinerja model.

Berikut adalah langkah-langkah yang diperlukan untuk melakukan tuning hyperparameter pada algoritma k-Nearest Neighbors (KNN) menggunakan Optuna:

***Langkah 1:*** Definisikan fungsi objektif untuk optimisasi dengan Optuna. Fungsi ini akan mengevaluasi model KNN berdasarkan kombinasi hyperparameter yang diuji dan mengembalikan skor akurasi sebagai objektif optimisasi.

***Langkah 2:*** Buat variabel studi dengan ***'optuna.create_study()'*** dan mulai proses optimisasi dengan ***'study.optimize()'***. Optuna akan mencoba berbagai kombinasi hyperparameter dan mencari yang menghasilkan skor akurasi tertinggi.

***Langkah 3:*** Setelah proses optimisasi selesai, visualisasikan riwayat optimisasi dan pentingnya masing-masing hyperparameter menggunakan fungsi visualisasi Optuna.

***Langkah 4:*** Dapatkan nilai hyperparameter terbaik yang dihasilkan oleh Optuna dengan mengakses atribut ***best_params*** dari objek studi.

***Langkah 5:*** Gunakan nilai hyperparameter terbaik untuk membuat model KNN yang optimal. Latih model tersebut menggunakan seluruh data latih untuk menghasilkan model yang efisien dan akurat.

***Langkah 6:*** Evaluasi kinerja model KNN pada data uji untuk mendapatkan estimasi akurasi dan kemampuan generalisasi model dalam memprediksi penyakit liver


### Hasil Optimasi Optuna

Setelah proses optimasi selesai, diperoleh hyperparameter terbaik untuk model KNN. Hasil dari eksperimen ini adalah sebagai berikut:

1. Hyperparameter Terbaik:

   - n_neighbors= 5 
   - weights: 'distance'
   - metric: 'manhattan'

2. Nilai Akurasi Training Model : 0,77

3. Bobot parameter yang berpengaruh terhadap nilai akurasi

   - n_neighbors= 0,50
   - weights: 0,03
   - metric: 0,48

   Dari ketiga paremeter yang disetting, parameter n_neighbors dan metric yang mempunyai pengaruh yang tinggi dalam menentukan nilai akurasi model.


## Evaluasi Model


Pada tahap ini, model KNN dievaluasi pada data uji untuk mendapatkan estimasi performa model dalam deteksi penyakit liver. Adapun metode yang digunakan dalam evaluasi ini yaitu menggunakan Confusion Matrix dan Classification Report

### Confusion Matrix

Confusion matrix adalah alat yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan menggambarkan hasil prediksi model terhadap data yang sebenarnya


|            | Predicted Positive  | Predicted Negative |
| ---------- | -------------- |-------------- |
|Actual Positive| True Positive (TP)   |   False Negative (FN)  |
|Actual Negative |   False Positive (FP)  |   True Negative (TN)   |


Berikut adalah penjelasan singkat untuk masing-masing sel dalam confusion matrix:

- True Positive (TP):

Ini adalah jumlah kasus positif yang benar diidentifikasi oleh model sebagai positif. Model dengan nilai TP yang tinggi menunjukkan bahwa model dengan baik dalam mengidentifikasi kasus positif.

- True Negative (TN):

Ini adalah jumlah kasus negatif yang benar diidentifikasi oleh model sebagai negatif. Model dengan nilai TN yang tinggi menunjukkan bahwa model dengan baik dalam mengidentifikasi kasus negatif.

- False Positive (FP):

Ini adalah jumlah kasus negatif yang salah diidentifikasi oleh model sebagai positif. Jika FP tinggi, ini menunjukkan bahwa model cenderung memberikan kesalahan dengan mengklasifikasikan data negatif sebagai positif

- False Negative (FN):

Ini adalah jumlah kasus positif yang salah diidentifikasi oleh model sebagai negatif. Jika FN tinggi, ini menunjukkan bahwa model cenderung memberikan kesalahan dengan mengklasifikasikan data positif sebagai negatif.

### Classification Report

Classification report adalah laporan yang memberikan informasi rinci tentang kinerja model klasifikasi. Laporan ini berisi beberapa metrik evaluasi yang dihitung berdasarkan confusion matrix dan memberikan insight tentang seberapa baik model dapat melakukan klasifikasi pada setiap kelas yang ada dalam data.

Classification report biasanya mencakup beberapa metrik berikut:

1. Precision (Presisi):

Precision mengukur seberapa banyak dari kasus yang diidentifikasi sebagai positif oleh model yang sebenarnya benar positif. Precision dihitung dengan rumus:
   
   ```
   Precision = (True Positives) / (True Positives + False Positives)
   ```

2. Recall (Sensitivity):

Recall mengukur seberapa banyak dari seluruh kasus positif yang berhasil diidentifikasi oleh model. Recall dihitung dengan rumus:
   
   ```
   Recall = (True Positives) / (True Positives + False Negatives)
   ```
3. F1-Score:

F1-score adalah rata-rata harmonik antara presisi dan recall. F1-score memberikan keseimbangan antara presisi dan recall, dan berguna ketika ada ketidakseimbangan kelas. F1-score dihitung dengan rumus:

   ```
   F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
   ```
4. Accuracy (Akurasi):

Akurasi mengukur seberapa banyak dari seluruh kasus (positif dan negatif) yang berhasil diidentifikasi dengan benar oleh model. Akurasi dihitung dengan rumus:
   ```
   Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
```

### Hasil Prediksi terhadap data uji (data testing)

   Pada tahap ini, model diuji dengan menggunakan data testing sebanyak 117 data yang terdiri dari 83 Class Liver dan 34 Class Non-Liver

   #### Hasil Prediksi Eksperimen Ke-1:

   Pada eksperimen ke-1, model dilatih dengan menggunakan data latih hasil nomralisasi dari StandardScaler, adapun hasil prediksinya dapat dilihat pada gambar dibawah ini
   
   Tabel Confusion Matriks Eksperimen Ke-1

   |                 | Predicted Liver  | Predicted Non-Liver |
   | ----------      | --------------   |---------------------|
   |Actual Liver     |  65              | 18                  |
   |Actual Non-Liver |  22              | 12                  |

   
   Tabel Classification Report Eksperimen Ke-1

   |Class     | Precision | recall  | f1-score| support |
   | ---------| ----------|---------|---------|---------|
   |Liver     |  0,75     | 0,78    | 0,76    | 83      |
   |Non-Liver |  0,40     | 0,35    | 0,38    | 34      |
   |Accuracy = 0,66|||                        | 117     |

   Bila dilihat dari Tabel Classification Report Eksperimen Ke-1, ***Recall*** untuk Non-liver nilainya sangat kecil yaitu 0,35 .Ini hanya 35% pasien Non-liver yang dapat diprediksi benar oleh model. Hal ini disebabkan karena ada keteidakseimbangan kelas (imblanace) pada data target yaitu jumlah data kelas liver (71,4%) lebih banyak daripada kelas Non-Liver (28,6%).Dengan demikian bahwa kelemahan model ini belum bisa memprediksi dengan benar untuk kelas minoritas (NOn-Liver). Selanjutnya model pertama ini akan diperbaiki melalui eksperimen ke-2. 

   #### Hasil Prediksi Eksperimen Ke-2:

   Eksperimen Ke 2 dilakukan untuk memperbaiki kinerja model pada eksperimen pertama, yaitu menggunakan data latih hasil resample SMOTE untuk mengatasi ketidakseimbangan kelas (imbalance).Adapun hasil prediksinya sebagai berikut:

   Tabel Confusion Matriks Eksperimen Ke-2

   |                 | Predicted Liver  | Predicted Non-Liver |
   | ----------      | --------------   |---------------------|
   |Actual Liver     |  52              | 31                  |
   |Actual Non-Liver |  8               | 26                  |


   Tabel Classification Report Eksperimen Ke-2

   |Class     | Precision | recall  | f1-score| support |
   | ---------| ----------|---------|---------|---------|
   |Liver     |  0,87     | 0,63    | 0,73    | 83      |
   |Non-Liver |  0,46     | 0,76    | 0,57    | 34      |
   |Accuracy = 0,67|||                         | 117     |

   Bila dilihat dari Tabel Classification Report Eksperimen Ke-2, nilai Recall untuk Non-Liver berhasil meningkat menjadi 0,76 (sebelumnya 0,35).Dengan demikian bahwa, diterapkannya resample SMOTE pada data training bisa mengatasi kasus imbalance data.Dalam Hal ini model bisa memprediksi kelas minoritas (NOn-liver) dengan cukup baik.Namun, Akurasi yang dihasilkan oleh model ini masih kecil yaitu 0,67.Untuk mengatasi hal ini,dilakukanlah eksperimen ke 3 untuk mengoptimalkan nilai akurasi dari model tersebut.


   #### Hasil Prediksi Eksperimen Ke-3:

   Eksperimen Ke-3  merupakan hasil optimalisasi model dengan melalukan tunning hyperparameter menggunakan optuna. Adapun hasil prediksinya sebagai berikut:

   Tabel Confusion Matriks Eksperimen Ke-3

   |                 | Predicted Liver  | Predicted Non-Liver |
   | ----------      | --------------   |---------------------|
   |Actual Liver     |  56              | 27                  |
   |Actual Non-Liver |  7               | 27                  |


   Tabel Classification Report Eksperimen Ke-3

   |Class     | Precision | recall  | f1-score| support |
   | ---------| ----------|---------|---------|---------|
   |Liver     |  0,89     | 0,67    | 0,77    | 83      |
   |Non-Liver |  0,50     | 0,79    | 0,61    | 34      |
   |Accuracy = 0,71 |||                       | 117     |

   Berdasarkan Tabel Classification Report Eksperimen Ke-3, Nilai Akurasi model meningkat dari 0,67 menjadi 0,71 setelah dilakukan tunning hyperparameter optuna.
   
   #### Perbandingan ***Accuracy,Precision,Recall,F1-Score*** dari Eksperimen ke 1,2,3

   |Class           | Accuracy  | Precision  | Recall  |F1-Score |
   | ---------------| ----------|------------|---------|---------|
   |Eksperiman Ke-1 |  0,65     | 0,78       | 0,74    | 0,76    |
   |Eksperiman Ke-2 |  0,66     | 0,62       | 0,86    | 0,72    |
   |Eksperiman Ke-3 |  0,71     | 0,67       | 0,88    | 0,76    |

   Bila dilihat dari tabel diatas, bahwa Eksperimen ke-3 (mengintegrasikan model KNN dan Optuna) berhasil meningkatkan nilai AKurasi,Presisi,Recall,dan F1-Score.

   Dengan demikian bahwa mengintegrasikan tunning hyperparameter Optuna ke dalam model KNN dapat meningkatkan performa model


## Kesimpulan



## Referensi

[[1]](https://www.halodoc.com/artikel/cek-kesehatan-hati-dengan-tes-fungsi-hati-ini) Halodoc.(2023).*Cek Kesehatan Hati dengan Tes Fungsi Hati*.Diakses pada 22 Juli 2023. https://www.halodoc.com/artikel/cek-kesehatan-hati-dengan-tes-fungsi-hati-ini

[[2]](https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/) Kemendagri. (2019). *Liver Disebut Penyebab Kematian Terbesar di Usia 35-49 Tahun)*. Diakses pada 22 Juli 2023 https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/

[[3]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) Scikit-Learn.(2023).*KNeighborsClassifier*.Diakses pada 22 Juli 2023.https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[[4]](https://optuna.org/) Optuna.(2023).*Optuna*.Diakses pada 22 Juli 2023.https://optuna.org
