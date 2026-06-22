import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide")
st.title("🚢 Dashboard Analisis Sentimen & Kinerja Pelabuhan")

# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA & MODEL
# ==========================================
# Gunakan st.cache_data agar data tidak dimuat ulang setiap kali ada interaksi
@st.cache_data
def load_data():
    # GANTI 'data_pelabuhan.csv' DENGAN NAMA FILE DATASET ANDA YANG SEBENARNYA
    df = pd.read_csv('data_pelabuhan.csv') 
    
    # Pastikan ada kolom tanggal dan ubah formatnya ke datetime
    # Sesuaikan 'tanggal' dengan nama kolom tanggal di dataset Anda
    if 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    return df

@st.cache_resource
def load_model():
    model = joblib.load('model_sentimen.pkl')
    vectorizer = joblib.load('vectorizer_sentimen.pkl')
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# ==========================================
# 3. PEMBUATAN TABS (NAVIGASI)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data & Tren", "☁️ WordCloud & Heatmap", "🤖 Prediksi Sentimen AI"])

# ------------------------------------------
# TAB 1: VISUALISASI DATA & TREN
# ------------------------------------------
with tab1:
    st.header("Analisis Kinerja dan Popularitas Pelabuhan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Popularitas (Volume Aktivitas)")
        # Menghitung jumlah ulasan per pelabuhan
        pop_df = df['pelabuhan'].value_counts().reset_index()
        pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
        
        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='viridis', ax=ax_pop)
        ax_pop.set_xlabel("Total Ulasan")
        ax_pop.set_ylabel("")
        st.pyplot(fig_pop)

    with col2:
        st.subheader("Kualitas (Rating) vs Kuantitas (Volume)")
        # Agregasi rating rata-rata dan jumlah ulasan
        scatter_df = df.groupby('pelabuhan').agg(
            Rata_Rating=('review_rating', 'mean'),
            Volume=('review_rating', 'count')
        ).reset_index()
        
        fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=150, ax=ax_scat)
        ax_scat.set_xlabel("Kuantitas (Jumlah Ulasan)")
        ax_scat.set_ylabel("Kualitas (Rata-rata Rating)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_scat)

    st.markdown("---")
    
    st.subheader("Tren Volume Ulasan per Bulan di Setiap Pelabuhan")
    if 'bulan_tahun' in df.columns:
        # Menghitung tren waktu
        trend_df = df.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
        
        fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=trend_df, x='bulan_tahun', y='Jumlah', hue='pelabuhan', marker='o', ax=ax_trend)
        plt.xticks(rotation=45)
        ax_trend.set_xlabel("Bulan")
        ax_trend.set_ylabel("Volume Ulasan")
        ax_trend.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_trend)
    else:
        st.warning("Kolom tanggal tidak ditemukan di dataset untuk membuat tren waktu.")

# ------------------------------------------
# TAB 2: WORDCLOUD & HEATMAP
# ------------------------------------------
with tab2:
    col_wc, col_hm = st.columns(2)
    
    with col_wc:
        st.subheader("Visualisasi WordCloud")
        st.write("Kata yang paling sering muncul dalam ulasan.")
        
        # Gabungkan semua teks
        semua_teks = " ".join(df['review_text'].dropna().astype(str))
        wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
        
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

    with col_hm:
        st.subheader("Heatmap Keluhan (Rating 1 & 2)")
        st.write("Intensitas ulasan negatif per pelabuhan dan bulan.")
        
        if 'bulan_tahun' in df.columns:
            # Saring hanya ulasan negatif (rating <= 2)
            df_negatif = df[df['review_rating'] <= 2]
            
            # Buat Pivot Table
            pivot_keluhan = df_negatif.pivot_table(
                index='pelabuhan', 
                columns='bulan_tahun', 
                values='review_rating', 
                aggfunc='count', 
                fill_value=0
            )
            
            fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, ax=ax_hm)
            ax_hm.set_xlabel("Bulan")
            ax_hm.set_ylabel("Pelabuhan")
            st.pyplot(fig_hm)
        else:
            st.warning("Data waktu tidak tersedia untuk Heatmap.")

# ------------------------------------------
# TAB 3: PREDIKSI SENTIMEN (AI)
# ------------------------------------------
with tab3:
    st.header("Sistem Uji Sentimen Real-Time")
    st.write("Masukkan teks ulasan baru untuk dianalisis oleh model Machine Learning.")
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()
    
    user_input = st.text_area("Ketik ulasan terkait layanan imigrasi atau pelabuhan:", height=120)
    
    if st.button("Analisis Ulasan", type="primary"):
        if user_input:
            teks_bersih = clean_text(user_input)
            if teks_bersih:
                vektor_input = vectorizer.transform([teks_bersih])
                prediksi = model.predict(vektor_input)[0]
                probabilitas = model.predict_proba(vektor_input)[0]
                
                # Tentukan warna berdasarkan sentimen
                warna = "green" if prediksi == "Positif" else "red" if prediksi == "Negatif" else "gray"
                
                st.markdown(f"### Hasil Deteksi: <span style='color:{warna}'>{prediksi.upper()}</span>", unsafe_allow_html=True)
                
                st.write("**Keyakinan Model:**")
                # Menampilkan progress bar untuk tingkat keyakinan
                for i, cls in enumerate(model.classes_):
                    st.write(f"{cls}")
                    st.progress(float(probabilitas[i]))
            else:
                st.warning("Teks tidak valid setelah dibersihkan.")
        else:
            st.error("Harap masukkan ulasan terlebih dahulu!")
