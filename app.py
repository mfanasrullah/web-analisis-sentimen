import streamlit as st
import joblib
import re

# 1. Muat Model dan Vectorizer
model = joblib.load('model_sentimen.pkl')
vectorizer = joblib.load('vectorizer_sentimen.pkl')

# Fungsi pembersih teks (sesuaikan dengan milik Anda)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# 2. Desain Antarmuka Website
st.title("🚢 Sistem Analisis Sentimen Pelabuhan")
st.write("Masukkan ulasan terkait layanan imigrasi atau pelabuhan di bawah ini:")

# Input dari pengguna
user_input = st.text_area("Ketik ulasan di sini...", height=150)

if st.button("Analisis Sentimen"):
    if user_input:
        # Bersihkan dan prediksi
        teks_bersih = clean_text(user_input)
        if teks_bersih:
            vektor_input = vectorizer.transform([teks_bersih])
            prediksi = model.predict(vektor_input)[0]
            probabilitas = model.predict_proba(vektor_input)[0]
            
            # Tampilkan Hasil
            st.subheader(f"Hasil: **{prediksi.upper()}**")
            
            # Tampilkan detail keyakinan model
            st.write("**Tingkat Keyakinan:**")
            for i, cls in enumerate(model.classes_):
                st.write(f"- {cls}: {probabilitas[i]*100:.1f}%")
        else:
            st.warning("Teks terlalu pendek atau tidak terdeteksi setelah dibersihkan.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
