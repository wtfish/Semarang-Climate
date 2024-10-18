# Semarang-Climate

# Informasi Dataset: Data Iklim Kota Semarang

Dataset ini berisi pengukuran iklim untuk kota **Semarang**, yang diambil dari platform [BMKG Data Online](https://dataonline.bmkg.go.id/home) dan diolah oleh Muhammad Rizqi melalui platform [Kaggle](https://www.kaggle.com/kingki19).

- **Kota yang diukur**: Semarang, Jawa Tengah
- **Sumber**: [BMKG Data Online](https://dataonline.bmkg.go.id/home)
- **Stasiun Pengambilan Data**: Stasiun Klimatologi Jawa Tengah
- **Rentang Tanggal**: 1 Februari 2017 - 31 Desember 2023

## Penjelasan Kolom:

- **Tanggal**: Tanggal observasi dalam format (bulan/hari/tahun [jam, opsional]). Kolom ini disimpan sebagai string, sehingga perlu dikonversi ke tipe datetime untuk analisis lebih lanjut.
- **Tn**: Suhu minimum yang tercatat pada hari tersebut (°C).
- **Tx**: Suhu maksimum yang tercatat pada hari tersebut (°C).
- **Tavg**: Suhu rata-rata yang tercatat pada hari tersebut (°C).
- **RH_avg**: Kelembapan relatif rata-rata selama hari tersebut (%).
- **RR**: Jumlah curah hujan pada hari tersebut (mm).
- **ss**: Durasi penyinaran matahari selama hari tersebut (jam).
- **ff_x**: Kecepatan angin maksimum yang tercatat pada hari tersebut (m/s).
- **ddd_x**: Arah angin pada kecepatan angin maksimum (°).
- **ff_avg**: Kecepatan angin rata-rata selama hari tersebut (m/s).
- **ddd_car**: Arah angin yang paling sering terjadi selama hari tersebut (°).

Dataset ini memberikan wawasan berharga tentang pola iklim di Semarang, termasuk fluktuasi suhu, curah hujan, kondisi angin, dan durasi penyinaran matahari selama periode yang diamati.

