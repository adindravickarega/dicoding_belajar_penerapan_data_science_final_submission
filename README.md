# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan di Jaya Jaya Institut

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

Nah, sebagai calon data scientist masa depan Anda diminta untuk membantu Jaya Jaya Institut dalam menyelesaikan permasalahannya. Mereka telah menyediakan dataset yang dapat Anda unduh melalui tautan berikut: students' performance. Selain itu, mereka juga meminta Anda untuk membuatkan dashboard agar mereka mudah dalam memahami data dan memonitor performa siswa. 

### Permasalahan Bisnis
Permasalahan bisnis yang akan diselesaikan adalah sebagai berikut :
1. Seberapa tinggi tingkat persentase mahasiswa yang Dropout (DO) di Jaya Jaya Institut ?
2. Faktor-faktor apa saja yang berpengaruh terhadap tingginya persentase mahasiswa yang memutuskan untuk Dropout dari kampusnya ? 

### Cakupan Proyek
1. Data Preparation
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Visualization di Jupyter Notebook
5. Data Preprocessing, Train Test Split
6. Pemodelan Machine Learning untuk prediksi probabilitas mahasiswa Dropout
7. Evaluasi Model Machine Learning
8. Simpan Model, Deploy model secara lokal dengan User Interface Streamlit (app.py)
9. Pembuatan Business Dashboard menggunakan Metabase
10. Deploy Streamlit App di Streamlit Community Cloud 

### Persiapan

Sumber data: Sumber data berasal dari Dataset Dicoding yang dapat diakses melalui link berikut : 
https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

Setup environment:
```
1. Buka Anaconda pada CMD
2. Buat virtual environment  : 
    ```
    conda create --name student_dropout_final_submission python=3.9
    ```
3. Aktifkan virtual environment :
    ```
    conda activate student_dropout_final_submission
    ```
4. Arahkan virtual environment ke directory submission : 
    ```
    cd <letak directory submission>
    ```
5. Install module yang dibutuhkan :
    ```
    pip install -r requirements.txt
    ```
```

## Business Dashboard
Dashboard dibuat dengan program Metabase.
https://github.com/adindravickarega/dicoding_belajar_penerapan_data_science_final_submission/blob/main/mahega_0107-dashboard.jpg

## Menjalankan Sistem Machine Learning

**Cara menjalankan prototype**:
1. Secara online :
    Streamlit App versi Cloud dapat diakses melalui link berikut :
    https://adindravickarega-student-dropout-predict.streamlit.app/

2. Secara offline : 
    1. Clone github repository :
        ```
        git clone https://github.com/adindravickarega/dicoding_belajar_penerapan_data_science_final_submission.git
        ```
    2. Pindah direktori :
        ```
        cd dicoding_belajar_penerapan_data_science_final_submission
        ``` 
    3. Jalankan command "streamlit run app.py"
    4. Halaman web localhost akan muncul dari browser
    5. Isi informasi yang dibutuhkan untuk melakukan prediksi status mahasiswa Dropout atau tidak

## Conclusion
1. Persentase mahasiswa Dropout (DO) di Jaya Jaya Institut adalah 32.1%, yang dimana angka tersebut relatif tinggi dibandingkan dengan persentase kelulusan 49.9%.

2. Mahasiswa yang Dropout memiliki:
   - Nilai lebih rendah (Curricular_units_1st/2nd_sem_grade) dibandingkan yang Graduate/Enrolled.
   - Jumlah mata kuliah yang disetujui (approved) lebih sedikit.
   - Partisipasi evaluasi (evaluations) lebih rendah.
3. Faktor Non-Akademik yang Berpengaruh :
   - Tuition_fees_up_to_date: Mahasiswa yang tidak membayar tepat waktu cenderung memiliki risiko Dropout lebih tinggi.
   - Scholarship_holder: Penerima beasiswa memiliki tingkat kelulusan lebih tinggi.
   - Demografi:
     - Usia (Age_at_enrollment): Mahasiswa yang lebih muda cenderung lebih sukses. Mahasiswa yang lebih tua lebih beresiko Dropout (DO)
     - Gender: Tidak ada perbedaan signifikan dalam dropout rate.
4. Perbedaan antar Program Studi (course_category) Beberapa program studi memiliki tingkat Dropout lebih tinggi (misalnya, Sosial & Teknologi), sementara lainnya (misalnya, Bisnis & Kesehatan) memiliki tingkat kelulusan lebih baik.

### Rekomendasi Action Items
Berdasarkan temuan EDA, berikut rekomendasi untuk institusi pendidikan Jaya Jaya Institut:

1. Intervensi Akademik
    Program Bimbingan Akademik: Fokus pada mahasiswa dengan nilai rendah di semester 1 karena mereka berisiko tinggi Dropout.
    Berikan mentoring tambahan untuk mata kuliah dengan tingkat kegagalan tinggi.

2. Sistem Peringatan Dini (Early Warning System):
    Gunakan machine learning untuk memprediksi mahasiswa berisiko Dropout berdasarkan kinerja semester 1.
    Berikan notifikasi kepada dosen/wali jika mahasiswa memiliki nilai di bawah ambang batas.

3. Dukungan Finansial (Beasiswa & Bantuan Biaya Kuliah):
    Prioritaskan mahasiswa dari keluarga kurang mampu (Debtor = 1).
    Berikan penyelesaian biaya kuliah fleksibel untuk mengurangi tekanan finansial.
    Program Kerja Sambil Kuliah: Kolaborasi dengan industri untuk memberikan part-time job bagi mahasiswa yang membutuhkan.

4. Peningkatan Keterlibatan Mahasiswa
    Tingkatkan Partisipasi Evaluasi: Mahasiswa yang tidak mengikuti evaluasi cenderung Dropout.
    Berikan insentif (poin tambahan, sertifikat) untuk meningkatkan kehadiran ujian.

5. Program peer mentoring dan komunitas belajar untuk meningkatkan motivasi.

6. Segmentasi Mahasiswa:
    Kelompokkan mahasiswa berdasarkan risiko Dropout (rendah, sedang, tinggi) dan berikan pendekatan berbeda.
    Dengan strategi ini, Jaya Jaya Institut dapat mengurangi tingkat Dropout dan meningkatkan keberhasilan mahasiswa.
