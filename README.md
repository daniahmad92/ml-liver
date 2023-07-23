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


## Data Preparation

### Label Encoding

Tahapan ini bertujuan untuk merubah data kategorik menjadi data numerik.Terdapat 1 variabel yang bertipe data kategori yaitu variabel Gender.Pada variabel Gender berisi kategori "Male" dan "Female".

Untuk melakukan Label Encoding, pada penelitian ini akan menggunakan kelas LabelEncoder dari library scikit-learn dengan tahapan sebagai berikut:

1. Buat objek LabelEncoder yang akan digunakan untuk melakukan Label Encoding
2. Selanjutnya, lakukan proses fit_transform pada data Gender untuk mengubah nilai kategorikal menjadi nilai numerik.

Output yang dihasilkan adalah sebagai berikut:

| Kategori  | Numerik  |
| ----------| ---------|
|  Male     |   0      |
|  Female   |   1      |

### Split data training dan testing

Pada tahap ini, dataset akan dibagi menjadi data latih (data training) dan data uji (data testing).Data training akan digunakan untuk melatih model,sedangkan data testing akan digunakan untuk menguji kinerja model yang telah dilatih pada data training.

***train_test_split*** adalah fungsi dari pustaka scikit-learn yang digunakan untuk membagi dataset menjadi data training dan data testing.Pada penelitian ini , data tersebut dibagi menjadi **80% sebagai data training dan 20% sebagai data testing**.

```
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=123,stratify= y)

```
Output yang dihasilkan adalah sebagai berikut:

| data awal | data training| data testing |
| ----------| ------------ |--------------|
|  583  rows|   466 rows   |  117  rows   |


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

## Modeling - Parameter KNN Default

Algoritma K-Nearest Neighbors (KNN) adalah salah satu algoritma yang bekerja dengan cara mencari K tetangga terdekat dari suatu data uji dan kemudian melakukan klasifikasi dari tetangga tersebut untuk menentukan label atau nilai prediksi dari data uji.

Adapun parameter input model knn secara default seperti pada tabel di bawah ini

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


```
knn_model_default = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')

```

Pada script diatas parameter yang diset diantaranya n_neighbors,weight,dan metric.berikut penjelasannya:

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



## Evaluasi Model dengan Tunning Hyperparameter Optuna

**Optuna** adalah sebuah library Python yang digunakan untuk optimasi hyperparameter secara otomatis

Berikut adalah langkah-langkah untuk melakukan tuning hyperparameter K-Nearest Neighbors (KNN) menggunakan library Optuna

#### 1. Definisikan Objective Function

Fungsi ini akan menerima objek trial yang berisi nilai hyperparameter yang akan diuji.adapun parameter KNN yang akan ditunning yaitu **n_neighbors,weight,dan metric**

| parameter | Opsi |
| ---------- | --------------|
| *n_neighbors* | 1 s.d 10|
| *weights* | uniform,distance|
| *metric* |euclidean,manhattan,minkowski|


```
def objective(trial):
    
    # Definisikan hyperparameter yang akan dioptimasi dan rentang pencariannya

    n_neighbors = trial.suggest_int("n_neighbors", 1,10)
    weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
    metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])

    # Inisialisasi model KNN
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Latih model dengan data latih
    knn_model.fit(x_train_resampled, y_train_resampled)

    # Prediksi label kelas pada data uji
    y_pred = knn_model.predict(x_test_resampled)

    # Hitung akurasi prediksi
    accuracy = accuracy_score(y_test_resampled, y_pred)

   # Kembalikan nilai akurasi sebagai nilai objektif yang akan dioptimasi
    return accuracy


```

#### 2. Buat dan Mulai Studi Optuna

- buat  objek Studi Optuna
```
study = optuna.create_study(direction="maximize")
```
Bagian ini digunakan untuk membuat objek Studi Optuna. Studi adalah entitas utama dalam Optuna yang merepresentasikan ruang pencarian hyperparameter. Parameter direction menentukan arah optimasi yang ingin dilakukan. Nilai "maximize" berarti mencari nilai hyperparameter yang memaksimalkan nilai objektif (dalam kasus ini, akurasi)

- jalankan proses optimasi
```
study.optimize(objective, n_trials=100)
```
Bagian ini adalah saat dimana proses optimasi dilakukan. Fungsi optimize() dari objek Studi digunakan untuk memulai proses optimasi. Fungsi ini menerima dua argumen, yaitu objective yang merupakan fungsi tujuan (objective function) yang ingin dioptimasi, dan n_trials yang merupakan jumlah iterasi atau percobaan yang akan dilakukan oleh Optuna untuk mencari nilai hyperparameter yang optimal.



#### 3. Hasil Tunning Hyperparameter Optuna

- ***Optimization History Plot***

Grafik Optimization History Plot pada Optuna merupakan alat yang berguna untuk memvisualisasikan progres optimasi hyperparameter selama proses pencarian kombinasi hyperparameter terbaik. Plot dapat memvisualisasikan bagaimana nilai fungsi objektif (akurasi) berubah selama iterasi dari algoritma optimasi.

untuk melihat grafiknya, dapat menggunakan script berikut:

```
optuna.visualization.plot_optimization_history(study)

```


![Gambar 13](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/optuna-histori-plot.JPG)

Gambar 13. Optimazation History Plot

Pada plot "Optimization History", sumbu-x mewakili iterasi (trial) yang dilakukan oleh algoritma optimasi, sedangkan sumbu-y mewakili nilai fungsi objektif pada setiap iterasi. Nilai fungsi objektif ini menunjukkan performa model dengan kombinasi hyperparameter yang diuji pada iterasi tersebut


Jika kita lihat dari history plot gambar di atas, pada iterasi awal sampai 20 terjadi kenaikan nilai akurasi,dan mulai konstan diiterasi ke 40 sampai 100.



- ***Hyperparameter Importances***

untuk melihat hyperparameter importance, jalankan script berikut:

```
optuna.visualization.plot_param_importances(study)

```

adapun outputnya sebagai berikut:

![Gambar 14](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/optuna-hyperparameter.JPG)

Gambar 14. Hyperparameter Importances Optuna

Berdasarkan grafik diatas, parameter knn yang paling berpengaruh terhadap nilai akurasi yaitu pada penentuan nilai n_neighbors (0,86), kemudian diikuti oleh parameter weights(0,08),dan yang terakhir adalah metrics0,06)


- ***Best Parameter***

Untuk mandapatkan nilai Best Parameter yang telah ditemukan selama proses optimasi,jalankan script berikut:

```
study.best_params

```
output yang didapatkan dalam penelitian ini,diantaranya:

```
- n_neighbors: 8
- weights : distance
- metrics : manhattan

```
### 4. Buat Model baru dengan parameter input dari Best Parameter Optuna

Setalah mendapatkan best parameter dari proses sebelumnya,kemudian parameter tersebut digunakan untuk membuat model baru hasil tunning seperti script yang dituliskan dibawah ini

```
def create_model_best_params(best_params):
    best_n_neighbors =best_params['n_neighbors']
    best_weights =best_params['weights']
    best_metric =best_params['metric']
    knn_optuna=KNeighborsClassifier(n_neighbors=best_n_neighbors,weights=best_weights,metric=best_metric)
    return knn_optuna

knn_model_optuna=create_model_best_params(best_params)

```


### Perbandingan NIlai AKurasi model KNN dengan parameter Default dan Best Parameter Optuna

setalah dilakukan tuning hyperparameter dengan optuna, didapatkan adanya peningkatkan nilai akurasi dari 0,68 menjadi 0,72 atau dengan peningkatan 5,9%

![Gambar 19](https://raw.githubusercontent.com/daniahmad92/ml-liver/main/akurasi.JPG)

Gambar 19. Grafik Perbandingan NIlai Akurasi Default dan Optuna

## Evaluasi Model

### Confusion matrix
Confusion matrix (matriks kebingungan) adalah alat yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan menggambarkan hasil prediksi model terhadap data yang sebenarnya

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




## Kesimpulan

Setelah dilakukan optimasi hiperparameter dengan Optuna,nilai akurasi deteksi penyakit liver meningkat dari 0,67 menjadi 0,72 atau dengan peningkatan sebesar 5,9%.Dengan demikian, dapat disimpulkan bahwa integrasi Hyperparameter Tunning Optuna dalam Model KNN berhasil meningkatkan akurasi deteksi penyakit liver

## Referensi

[[1]](https://www.halodoc.com/artikel/cek-kesehatan-hati-dengan-tes-fungsi-hati-ini) Halodoc.(2023).*Cek Kesehatan Hati dengan Tes Fungsi Hati*.Diakses pada 22 Juli 2023. https://www.halodoc.com/artikel/cek-kesehatan-hati-dengan-tes-fungsi-hati-ini

[[2]](https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/) Kemendagri. (2019). *Liver Disebut Penyebab Kematian Terbesar di Usia 35-49 Tahun)*. Diakses pada 22 Juli 2023 https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/

[[3]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) Scikit-Learn.(2023).*KNeighborsClassifier*.Diakses pada 22 Juli 2023.https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[[4]](https://optuna.org/) Optuna.(2023).*Optuna*.Diakses pada 22 Juli 2023.https://optuna.org



















