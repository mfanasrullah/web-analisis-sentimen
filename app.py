import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re
import os
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide", initial_sidebar_state="expanded")

# Menerapkan tema seaborn untuk semua grafik agar terlihat lebih korporat
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    file_path = 'data_pelabuhan.csv'
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', encoding_errors='replace')
    except Exception:
        df = pd.read_csv(file_path, sep=';', encoding='latin1') 
    
    df = df.rename(columns={
        'name': 'pelabuhan',
        'review_datetime_utc': 'tanggal'
    })
    
    df = df.dropna(subset=['pelabuhan', 'tanggal'])
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    
    df = df.dropna(subset=['tanggal'])
    df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_sentimen.pkl')
        vectorizer = joblib.load('vectorizer_sentimen.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

df = load_data()
model, vectorizer = load_model()

# ==========================================
# 3. HEADER & LOGO
# ==========================================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    try:
        # Memuat logo Polibatam
        logo = Image.open('02_Logo_4_W_Polibatam_Horizontal@2x.png')
        st.image(logo, use_column_width=True)
    except FileNotFoundError:
        st.error("Logo tidak ditemukan. Pastikan nama file sesuai.")

with col_title:
    st.title("Dashboard Analisis Sentimen & Kinerja Pelabuhan")
    st.markdown("<p style='color: #666666; font-size: 18px;'>Sistem Monitoring dan Evaluasi Kualitas Layanan Publik</p>", unsafe_allow_html=True)

st.markdown("---")

if df is None:
    st.error("File 'data_pelabuhan.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# ==========================================
# 4. RINGKASAN EKSEKUTIF (KPI METRICS)
# ==========================================
st.subheader("Ringkasan Kinerja Keseluruhan")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_ulasan = len(df)
total_pelabuhan = df['pelabuhan'].nunique()

kpi1.metric(label="Total Ulasan", value=f"{total_ulasan:,}")
kpi2.metric(label="Total Pelabuhan", value=f"{total_pelabuhan}")

if 'review_rating' in df.columns:
    rata_rating = df['review_rating'].mean()
    kpi3.metric(label="Rating Rata-rata", value=f"{rata_rating:.2f} / 5.0")
else:
    kpi3.metric(label="Rating Rata-rata", value="N/A")

pelabuhan_teraktif = df['pelabuhan'].value_counts().idxmax()
kpi4.metric(label="Pelabuhan Teraktif", value=pelabuhan_teraktif)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. PEMBUATAN TABS (NAVIGASI)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data & Tren", "☁️ Analisis Teks (WordCloud & Heatmap)", "🤖 Prediksi Sentimen AI"])

# ------------------------------------------
# TAB 1: VISUALISASI DATA & TREN
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Volume Ulasan per Pelabuhan**")
        pop_df = df['pelabuhan'].value_counts().reset_index()
        pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
        
        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_r', ax=ax_pop)
        ax_pop.set_xlabel("Total Ulasan", fontweight='bold')
        ax_pop.set_ylabel("")
        sns.despine() # Menghilangkan border atas dan kanan agar terlihat rapi
        st.pyplot(fig_pop)

    with col2:
        st.markdown("**Kualitas vs Kuantitas Ulasan**")
        if 'review_rating' in df.columns:
            scatter_df = df.groupby('pelabuhan').agg(
                Rata_Rating=('review_rating', 'mean'),
                Volume=('review_rating', 'count')
            ).reset_index()
            
            fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=150, palette='tab10', ax=ax_scat)
            ax_scat.set_xlabel("Kuantitas (Jumlah Ulasan)", fontweight='bold')
            ax_scat.set_ylabel("Kualitas (Rata-rata Rating)", fontweight='bold')
            ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            sns.despine()
            st.pyplot(fig_scat)
        else:
            st.warning("Kolom 'review_rating' tidak ditemukan.")

    st.markdown("---")
    
    st.markdown("**Tren Volume Ulasan per Bulan**")
    if 'bulan_tahun' in df.columns:
        trend_df = df.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
        
        fig_trend, ax_trend = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=trend_df, x='bulan_tahun', y='Jumlah', hue='pelabuhan', marker='o', linewidth=2, ax=ax_trend)
        ax_trend.tick_params(axis='x', rotation=45)
        ax_trend.set_xlabel("Bulan", fontweight='bold')
        ax_trend.set_ylabel("Volume Ulasan", fontweight='bold')
        ax_trend.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
        sns.despine()
        st.pyplot(fig_trend)

# ------------------------------------------
# TAB 2: WORDCLOUD & HEATMAP
# ------------------------------------------
with tab2:
    col_wc, col_hm = st.columns(2)
    
    with col_wc:
        st.markdown("**Pemetaan Topik (WordCloud)**")
        st.caption("Visualisasi kata kunci yang paling sering muncul dalam ulasan masyarakat.")
        
        if 'review_text' in df.columns:
            semua_teks = " ".join(df['review_text'].dropna().astype(str))
            if semua_teks.strip():
                wordcloud = WordCloud(width=600, height=450, background_color='#f8f9fa', colormap='Blues_r', max_words=100).generate(semua_teks)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4.5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada data teks ulasan yang cukup.")
        else:
            st.warning("Kolom 'review_text' tidak ditemukan.")

    with col_hm:
        st.markdown("**Peta Panas Keluhan (Rating 1 & 2)**")
        st.caption("Memantau intensitas ulasan negatif untuk evaluasi perbaikan layanan.")
        
        if 'bulan_tahun' in df.columns and 'review_rating' in df.columns:
            df_negatif = df[df['review_rating'] <= 2]
            
            if not df_negatif.empty:
                pivot_keluhan = df_negatif.pivot_table(
                    index='pelabuhan', 
                    columns='bulan_tahun', 
                    values='review_rating', 
                    aggfunc='count', 
                    fill_value=0
                )
                
                fig_hm, ax_hm = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, cbar_kws={'label': 'Jumlah Keluhan'}, ax=ax_hm)
                ax_hm.set_xlabel("Bulan", fontweight='bold')
                ax_hm.set_ylabel("Pelabuhan", fontweight='bold')
                st.pyplot(fig_hm)
            else:
                st.success("Tidak ada anomali ulasan negatif signifikan (Rating 1 & 2) yang ditemukan.")

# ------------------------------------------
# TAB 3: PREDIKSI SENTIMEN (AI)
# ------------------------------------------
with tab3:
    st.markdown("### Modul Uji Sentimen Real-Time")
    st.caption("Gunakan modul Machine Learning ini untuk mengklasifikasikan sentimen teks ulasan baru secara otomatis.")
    
    if model is None or vectorizer is None:
        st.error("Model Machine Learning belum tersedia. Pastikan file 'model_sentimen.pkl' dan 'vectorizer_sentimen.pkl' ada di direktori kerja.")
    else:
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text.strip()
        
        with st.container():
            user_input = st.text_area("Masukkan teks ulasan pengunjung:", height=100, placeholder="Contoh: Antrean terlalu panjang dan ruang tunggu kurang nyaman.")
            
            if st.button("Analisis Ulasan", type="primary"):
                if user_input:
                    teks_bersih = clean_text(user_input)
                    if teks_bersih:
                        vektor_input = vectorizer.transform([teks_bersih])
                        prediksi = model.predict(vektor_input)[0]
                        probabilitas = model.predict_proba(vektor_input)[0]
                        
                        warna = "#28a745" if prediksi.lower() == "positif" else "#dc3545" if prediksi.lower() == "negatif" else "#6c757d"
                        
                        st.markdown(f"#### Hasil Analisis Sentimen: <span style='color:{warna}; font-weight:bold;'>{prediksi.upper()}</span>", unsafe_allow_html=True)
                        st.markdown("**Tingkat Keyakinan Model (Confidence Score):**")
                        
                        for i, cls in enumerate(model.classes_):
                            st.write(f"{cls.capitalize()}")
                            st.progress(float(probabilitas[i]))
                    else:
                        st.warning("Teks tidak valid setelah dibersihkan (hanya memuat angka/simbol khusus).")
                else:
                    st.error("Silakan masukkan teks ulasan terlebih dahulu.")
