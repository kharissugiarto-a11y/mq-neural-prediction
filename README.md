# MQ Neural Prediction

Proyek terpisah untuk membangun dataset electronic nose dari Arduino Nano, melatih jaringan saraf klasifikasi gas, dan memprediksi konsentrasi gas dalam PPM.

Proyek ini tidak mengubah dashboard monitoring sebelumnya. Halaman kolektor bersifat statis, tanpa database atau server aplikasi. Python hanya digunakan secara lokal untuk pelatihan dan prediksi model.

## Struktur proyek

```text
mq-neural-prediction/
├── firmware/mq_neural_collector/mq_neural_collector.ino
├── web/index.html
├── web/styles.css
├── web/app.js
├── ml/mqml.py
├── ml/train.py
├── ml/predict.py
├── ml/make_demo_dataset.py
├── ml/requirements.txt
├── data/
└── model/
```

## Prinsip data

- ADC disimpan sebagai data mentah dan untuk audit.
- Input neural network adalah `log(Rs/R0)` dari MQ-6, MQ-2, MQ-135, MQ-3, dan MQ-131.
- `gas_label` adalah nama gas yang benar-benar diberikan saat eksperimen.
- `reference_ppm` hanya boleh diisi dari konsentrasi gas standar atau alat referensi. Jangan mengisinya dengan PPM ekuivalen dari rumus dashboard lama.
- Model klasifikasi memilih jenis gas. Model regresi terpisah untuk setiap kelas gas memprediksi PPM setelah kelas dipilih.

## 1. Unggah firmware Arduino

1. Buka `firmware/mq_neural_collector/mq_neural_collector.ino` di Arduino IDE.
2. Pilih **Arduino Nano**, prosesor yang sesuai, dan port USB.
3. Unggah sketch dengan baud rate serial `115200`.
4. Panaskan sensor baru minimal 48 jam. Untuk pemakaian berikutnya, tunggu hingga respons stabil.
5. Lakukan kalibrasi udara bersih dari halaman web sebelum merekam dataset.

Firmware mengirim JSON berisi ADC, nilai `R0`, dan rasio `Rs/R0`. Firmware ini tidak menghitung PPM agar dataset tidak tercampur dengan estimasi kurva nominal.

## 2. Gunakan halaman kolektor

Web Serial membutuhkan HTTPS atau localhost. Untuk publikasi, isi folder `web/` dapat ditempatkan di GitHub Pages.

1. Buka `web/index.html` melalui GitHub Pages menggunakan Chrome desktop.
2. Hubungkan Arduino.
3. Pilih label gas eksperimen.
4. Isi PPM referensi jika nilainya benar-benar diketahui. Kosongkan jika hanya membuat dataset klasifikasi.
5. Klik **Mulai rekam**, lakukan paparan gas, lalu klik **Stop rekam**.
6. Unduh CSV dan simpan di folder `data/`.

Lakukan beberapa sesi terpisah untuk setiap gas dan konsentrasi. Sediakan fase udara bersih, paparan, dan pemulihan. Jangan mencampur beberapa gas dalam satu label kecuali kelas `MIXTURE` memang dirancang dan komposisinya dicatat.

## 3. Siapkan Python

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```

Dependensi model hanya NumPy; tidak membutuhkan TensorFlow atau scikit-learn.

## 4. Latih model

Masukkan satu atau beberapa CSV hasil eksperimen ke folder `data/`, kemudian jalankan:

```powershell
python ml\train.py --data data\*.csv --out model
```

Hasil pelatihan:

- `model/classifier.npz`: jaringan saraf klasifikasi.
- `model/regressor_<gas>.npz`: jaringan saraf PPM per kelas yang memenuhi data minimum.
- `model/metadata.json`: normalisasi fitur, daftar kelas, dan konfigurasi model.
- `model/training_report.json`: akurasi klasifikasi dan galat regresi.
- `model/confusion_matrix.csv`: matriks kebingungan data uji.

Regresi dibuat hanya bila suatu kelas mempunyai minimal 30 sampel berlabel dan sedikitnya 3 tingkat PPM berbeda. Untuk penelitian yang layak, gunakan jauh lebih banyak data dan beberapa sesi/hari pengambilan.

## 5. Prediksi data baru

```powershell
python ml\predict.py --model model --input data_baru.csv --output hasil_prediksi.csv
```

Kolom tambahan pada hasil:

- `predicted_gas`
- `classification_confidence`
- `predicted_ppm`
- `ppm_model_available`

## Uji alur tanpa data eksperimen

Generator berikut hanya membuat data sintetis untuk memastikan program berjalan. Data ini tidak boleh digunakan sebagai hasil penelitian.

```powershell
python ml\make_demo_dataset.py --output data\demo_synthetic.csv
python ml\train.py --data data\demo_synthetic.csv --out model --epochs 80
python ml\predict.py --model model --input data\demo_synthetic.csv --output demo_predictions.csv
```

## Rekomendasi eksperimen

- Gunakan gas referensi dengan konsentrasi diketahui untuk target PPM.
- Ambil minimal 3–5 tingkat konsentrasi per gas dan beberapa pengulangan terpisah.
- Usahakan jumlah sampel per kelas seimbang.
- Pisahkan pengujian berdasarkan sesi/hari, bukan mengacak baris berurutan dari paparan yang sama.
- Catat suhu, kelembapan, waktu pemanasan, aliran gas, volume ruang uji, dan nomor unit sensor.
- Model hanya mengenali kelas yang dilatih. Gas atau campuran yang tidak dikenal tetap dapat dipaksa menjadi salah satu kelas, sehingga hasil tidak boleh dipakai sebagai alat keselamatan.
