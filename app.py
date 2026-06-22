import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re
import os
import PIL.Image as PILImage # Digunakan untuk memuat dan memproses logo dengan lebih baik

# ==========================================
# 0. KONFIGURASI TEMA & PALETTE WARNA CORPORATE
# ==========================================
# Mengatur tema global Seaborn untuk visualisasi yang bersih
sns.set_theme(style="whitegrid")
# Gunakan palette warna yang tenang untuk corporate feel (contoh: Blue palette)
sns.set_palette("Blues_d")

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide")

# Fungsi untuk memuat logo dengan penanganan error
def load_logo():
    logo_path = 'image_0.png' # Asumsi file yang Anda unggah disimpan dengan nama ini
    try:
        image = PILImage.open(logo_path)
        return image
    except FileNotFoundError:
        # Cadangan jika file lokal tidak ditemukan, gunakan URL yang disediakan
        logo_url = "https://www.polibatam.ac.id/wp-content/uploads/2024/01/01_Logo_4_W_Polibatam_Vertikal@2x-1-768x714.png"
        return logo_url
    except Exception as e:
        return None

logo_polibatam = load_logo()

# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    file_path = 'data_pelabuhan.csv'
    if not os.path.exists(file_path):
        return None
        
    # PERBAIKAN: Menangani error encoding saat membaca CSV
    try:
        # Coba baca dengan utf-8, jika ada karakter aneh, ganti dengan karakter pengganti (?)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', encoding_errors='replace')
    except Exception:
        # Fallback jika masih gagal: gunakan encoding standar Windows/Latin
        df = pd.read_csv(file_path, sep=';', encoding='latin1') 
    
    df = df.rename(columns={
        'name': 'pelabuhan',
        'review_datetime_utc': 'tanggal'
    })
    
    df = df.dropna(subset=['pelabuhan', 'tanggal'])
    
    # Konversi tanggal dengan penanganan error
    if df['tanggal'].dtype == 'object':
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    
    # Drop baris yang tanggalnya menjadi NaT setelah dikonversi
    df = df.dropna(subset=['tanggal'])
    df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    
    return df

@st.cache_resource
def load_model():
    # Menambahkan error handling untuk model
    model, vectorizer = None, None
    try:
        if os.path.exists('model_sentimen.pkl'):
            model = joblib.load('model_sentimen.pkl')
        if os.path.exists('vectorizer_sentimen.pkl'):
            vectorizer = joblib.load('vectorizer_sentimen.pkl')
        return model, vectorizer
    except (FileNotFoundError, Exception):
        return None, None

# Inisialisasi data dan model
df_full = load_data()
model, vectorizer = load_model()

# Cek ketersediaan data sebelum merender UI
if df_full is None:
    st.error("File 'data_pelabuhan.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTERS & BRANDING)
# ==========================================
with st.sidebar:
    # Branding di Sidebar
    if logo_polibatam:
        # Logo polibatam di sidebar, kecil, rapi
        if isinstance(logo_polibatam, PILImage.Image):
             # Resize agar rapi di sidebar
            logo_resized = logo_polibatam.resize((150, int(150 * logo_polibatam.height / logo_polibatam.width)))
            st.image(logo_resized)
        else:
            st.image(logo_polibatam, width=150)
    else:
        st.write("**[POLIBATAM]**")
    
    st.markdown("## Pusat Data Pelabuhan")
    st.markdown("---")
    
    # FILTER PELABUHAN
    st.markdown("#### 🏢 Pilih Pelabuhan")
    all_ports = df_full['pelabuhan'].unique().tolist()
    selected_ports = st.multiselect(
        "Pilih Pelabuhan:",
        options=all_ports,
        default=all_ports # Default pilih semua
    )
    
    # FILTER RENTANG TANGGAL
    st.markdown("#### 📅 Pilih Rentang Tanggal")
    min_date = df_full['tanggal'].min().date()
    max_date = df_full['tanggal'].max().date()
    
    date_range = st.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    st.markdown("---")
    st.markdown("<small>Dikembangkan oleh Tim Analitik Polibatam</small>", unsafe_allow_html=True)

# ==========================================
# 4. MEMPROSES DATA BERDASARKAN FILTER
# ==========================================
df_working = df_full[df_full['pelabuhan'].isin(selected_ports)]

if len(date_range) == 2:
    start_date, end_date = date_range
    # Filter tanggal dengan hati-hati
    df_working = df_working[
        (df_working['tanggal'].dt.date >= start_date) & 
        (df_working['tanggal'].dt.date <= end_date)
    ]
else:
    # Jika user baru memilih tanggal mulai, dashboard belum menampilkan data
    st.info("Pilih tanggal akhir di sidebar untuk memperbarui dashboard.")
    df_working = pd.DataFrame(columns=df_full.columns) # Data kosong

# Cek ketersediaan data setelah filter
if df_working.empty:
    st.warning("Tidak ada data yang sesuai dengan filter di sidebar.")
    st.stop()

# ==========================================
# 5. BODY - HEADER SECTION (LOGO & TITLE)
# ==========================================
# Menggunakan columns agar logo dan judul sejajar secara profesional
header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    if logo_polibatam:
        # Gunakan logo vertikal di sidebar, di header gunakan URL cadangan atau file lokal
        # Agar di header logo terlihat lebih rapi
        st.image(logo_polibatam, width=100) # Ukuran yang tepat untuk header
    else:
        st.write("**[POLIBATAM]**")
with header_col2:
    # Judul yang bersih tanpa emoji untuk corporate appeal
    st.title("Dashboard Analisis Sentimen Pelabuhan")
    st.markdown("<p style='font-size: 18px; color: gray; margin-top:-15px;'>powered by Tim Analitik Polibatam</p>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 6. BODY - KPI ROW (RINGKASAN METRIK UTAMA)
# ==========================================
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

# Menghitung Metrik Utama
total_reviews = len(df_working)
avg_rating = df_working['review_rating'].mean() if 'review_rating' in df_working.columns else 0
ports_counted = df_working['pelabuhan'].nunique()

with kpi_col1:
    st.metric(label="Total Volume Ulasan", value=f"{total_reviews:,}")
with kpi_col2:
    st.metric(label="Rata-rata Rating", value=f"{avg_rating:.2f} ⭐")
with kpi_col3:
    st.metric(label="Jumlah Pelabuhan Teranalisis", value=f"{ports_counted}")

st.markdown("---")

# ==========================================
# 7. BODY - PEMBUATAN TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data & Tren", "☁️ WordCloud & Heatmap Keluhan", "🤖 Prediksi Sentimen AI"])

# ------------------------------------------
# TAB 1: VISUALISASI DATA & TREN
# ------------------------------------------
with tab1:
    #st.header("Analisis Kinerja dan Popularitas Pelabuhan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribusi Popularitas (Volume Aktivitas)")
        st.markdown("<small>Pelabuhan dengan volume aktivitas ulasan tertinggi.</small>", unsafe_allow_html=True)
        pop_df = df_working['pelabuhan'].value_counts().reset_index()
        pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
        
        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_d', ax=ax_pop) # Palette biru corporate
        ax_pop.set_xlabel("Total Ulasan (Volume)", fontsize=10)
        ax_pop.set_ylabel("", fontsize=10)
        ax_pop.tick_params(axis='both', which='major', labelsize=9)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_pop)

    with col2:
        st.markdown("#### Kualitas (Rating) vs Kuantitas (Volume)")
        st.markdown("<small>Menganalisis keseimbangan antara rating rata-rata dan volume ulasan.</small>", unsafe_allow_html=True)
        if 'review_rating' in df_working.columns:
            scatter_df = df_working.groupby('pelabuhan').agg(
                Rata_Rating=('review_rating', 'mean'),
                Volume=('review_rating', 'count')
            ).reset_index()
            
            fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=200, palette='deep', ax=ax_scat)
            ax_scat.set_xlabel("Volume (Jumlah Ulasan)", fontsize=10)
            ax_scat.set_ylabel("Kualitas (Rata-rata Rating)", fontsize=10)
            ax_scat.tick_params(axis='both', which='major', labelsize=9)
            ax_scat.set_ylim(1, 5) # Rating 1-5
            ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Pelabuhan')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig_scat)
        else:
            st.warning("Kolom 'review_rating' tidak ditemukan.")

    st.markdown("---")
    
    st.markdown("#### Tren Volume Ulasan per Bulan di Setiap Pelabuhan")
    if 'bulan_tahun' in df_working.columns:
        trend_df = df_working.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
        
        fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
        # Gunakan seaborn explicit setting theme
        sns.lineplot(data=trend_df, x='bulan_tahun', y='Jumlah', hue='pelabuhan', marker='o', palette='muted', ax=ax_trend)
        ax_trend.tick_params(axis='x', rotation=45, labelsize=9) # Object-oriented approach
        ax_trend.tick_params(axis='y', labelsize=9)
        ax_trend.set_xlabel("Periode (Bulan)", fontsize=10)
        ax_trend.set_ylabel("Volume Ulasan", fontsize=10)
        ax_trend.grid(True, linestyle='--', alpha=0.6)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_trend)

# ------------------------------------------
# TAB 2: WORDCLOUD & HEATMAP
# ------------------------------------------
with tab2:
    col_wc, col_hm = st.columns(2)
    
    with col_wc:
        st.markdown("#### Visualisasi WordCloud")
        st.write("<small>Kata kunci yang paling sering muncul dalam ulasan.</small>", unsafe_allow_html=True)
        
        if 'review_text' in df_working.columns:
            semua_teks = " ".join(df_working['review_text'].dropna().astype(str))
            if semua_teks.strip(): # Memastikan teks tidak kosong
                # Gunakan palet corporate (contoh: Blue)
                wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada data teks ulasan yang cukup untuk membuat WordCloud.")
        else:
            st.warning("Kolom 'review_text' tidak ditemukan.")

    with col_hm:
        st.markdown("#### Heatmap Keluhan Konsumen (Rating 1 & 2)")
        st.write("<small>Fokus pada lokasi dan waktu puncak keluhan konsumen.</small>", unsafe_allow_html=True)
        
        if 'bulan_tahun' in df_working.columns and 'review_rating' in df_working.columns:
            df_negatif = df_working[df_working['review_rating'] <= 2]
            
            if not df_negatif.empty:
                pivot_keluhan = df_negatif.pivot_table(
                    index='pelabuhan', 
                    columns='bulan_tahun', 
                    values='review_rating', 
                    aggfunc='count', 
                    fill_value=0
                )
                
                fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                # Palette merah untuk fokus pada isu
                sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, ax=ax_hm, annot_kws={"size": 8})
                ax_hm.set_xlabel("Periode (Bulan)", fontsize=9)
                ax_hm.set_ylabel("", fontsize=9)
                ax_hm.tick_params(axis='both', which='major', labelsize=8)
                sns.despine(left=True, bottom=True)
                st.pyplot(fig_hm)
            else:
                st.success("Luar biasa! Tidak ada ulasan negatif (Rating 1 & 2) yang ditemukan dalam rentang waktu terfilter.")

# ------------------------------------------
# TAB 3: PREDIKSI SENTIMEN (AI)
# ------------------------------------------
with tab3:
    st.header("Sistem Uji Sentimen Real-Time")
    
    if model is None or vectorizer is None:
        st.error("Model Machine Learning belum tersedia (file .pkl tidak ditemukan). Fitur prediksi dinonaktifkan.")
    else:
        st.write("<small>Masukkan teks ulasan baru untuk dianalisis oleh model Machine Learning.</small>", unsafe_allow_html=True)
        
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text.strip()
        
        user_input = st.text_area("Ketik ulasan terkait layanan pelabuhan (maks. 500 kata):", height=120)
        
        # Primary button di streamlit lebih menonjol
        if st.button("Analisis Ulasan", type="primary"):
            if user_input:
                teks_bersih = clean_text(user_input)
                if teks_bersih:
                    vektor_input = vectorizer.transform([teks_bersih])
                    prediksi = model.predict(vektor_input)[0]
                    probabilitas = model.predict_proba(vektor_input)[0]
                    
                    # Tampilan hasil yang corporate-style (menggunakan kolom untuk rapi)
                    result_col1, result_col2 = st.columns(2)
                    
                    warna = "green" if prediksi.lower() == "positif" else "red" if prediksi.lower() == "negatif" else "gray"
                    
                    with result_col1:
                        st.markdown(f"<div style='border: 1px solid lightgray; padding: 10px; border-radius: 5px;'>Hasil Deteksi Model: <b style='color:{warna}; font-size: 20px;'>{prediksi.upper()}</b></div>", unsafe_allow_html=True)
                    
                    with result_col2:
                        st.write("**Keyakinan Model (Probabilitas):**")
                        # Mengubah list model.classes_ menjadi list probabilitas secara langsung
                        prob_positive = probabilitas[list(model.classes_).index('positif')] if 'positif' in model.classes_ else 0
                        prob_negative = probabilitas[list(model.classes_).index('negatif')] if 'negatif' in model.classes_ else 0
                        
                        st.progress(float(prob_positive), text=f"Positif: {prob_positive:.1%}")
                        st.progress(float(prob_negative), text=f"Negatif: {prob_negative:.1%}")
                else:
                    st.warning("Teks tidak valid setelah dibersihkan (mungkin hanya berisi angka/simbol).")
            else:
                st.error("Harap masukkan ulasan terlebih dahulu!")
