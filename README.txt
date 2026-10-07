SISTEM FUZZY MAMDANI KEMISKINAN BANGKA BELITUNG

1. Letakkan app.py, requirements.txt, dan file Excel penelitian dalam satu repository GitHub.
2. File Excel harus mempunyai sheet bernama: Data Lengkap.
3. Buka https://share.streamlit.io/
4. Hubungkan akun GitHub.
5. Pilih Create app -> Yup, I have an app.
6. Pilih repository, branch main, dan file app.py.
7. Klik Deploy.
8. Setelah selesai, Streamlit memberikan URL ...streamlit.app yang dapat dibagikan.

Catatan:
- Model menggunakan 7 kabupaten/kota dan menyingkirkan agregat provinsi.
- 27 rule tetap.
- α-predikat = MIN tiga derajat keanggotaan input.
- Inferensi menggunakan MIN, agregasi MAX, defuzzifikasi centroid.
- Membership parameter dibentuk dari data Excel yang diunggah, sama seperti notebook.
