import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re
import os
import PIL.Image as PILImage

# ==========================================
# 0. KONFIGURASI TEMA & PALETTE WARNA CORPORATE
# ==========================================
sns.set_theme(style="whitegrid")
sns.set_palette("Blues_d")

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide")

def load_logo():
    logo_path = 'image_0.png' 
    try:
        image = PILImage.open(logo_path)
        return image
    except FileNotFoundError:
        logo_url = "https://www.polibatam.ac.id/wp-content/uploads/2024/01/cropped-cropped-cropped-02_Logo_1_Utama_Polibatam_Horizontal@2x.png"
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
    
    try:
        df['tanggal'] = df['tanggal'].dt.tz_localize(None)
    except TypeError:
        pass 
    
    df = df.dropna(subset=['tanggal'])
    df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    
    # Memastikan rating terbaca sebagai angka
    if 'review_rating' in df.columns:
        df['review_rating'] = pd.to_numeric(df['review_rating'], errors='coerce')
    
    return df

@st.cache_resource
def load_model():
    model, vectorizer = None, None
    try:
        if os.path.exists('model_sentimen.pkl'):
            model = joblib.load('model_sentimen.pkl')
        if os.path.exists('vectorizer_sentimen.pkl'):
            vectorizer = joblib.load('vectorizer_sentimen.pkl')
        return model, vectorizer
    except (FileNotFoundError, Exception):
        return None, None

df_full = load_data()
model, vectorizer = load_model()

if df_full is None:
    st.error("File 'data_pelabuhan.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTERS & BRANDING)
# ==========================================
with st.sidebar:
    # Logo Polibatam hanya muncul di Sidebar
    if logo_polibatam:
        if isinstance(logo_polibatam, PILImage.Image):
            logo_resized = logo_polibatam.resize((150, int(150 * logo_polibatam.height / logo_polibatam.width)))
            st.image(logo_resized)
        else:
            st.image(logo_polibatam, width=150)
    else:
        st.write("**[POLIBATAM]**")
    
    st.markdown("## Pusat Data Pelabuhan")
    st.markdown("---")
    
    st.markdown("#### 🏢 Pilih Pelabuhan")
    all_ports = df_full['pelabuhan'].unique().tolist()
    selected_ports = st.multiselect(
        "Pilih Pelabuhan:",
        options=all_ports,
        default=all_ports 
    )
    
    st.markdown("#### 📅 Pilih Rentang Tanggal")
    min_date = df_full['tanggal'].min().date()
    max_date = df_full['tanggal'].max().date()
    
    # PERBAIKAN: Membagi input tanggal menjadi 2 kolom terpisah agar lebih intuitif
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Tanggal Mulai", value=min_date, min_value=min_date, max_value=max_date)
    with col_date2:
        end_date = st.date_input("📅 Tanggal Akhir", value=max_date, min_value=min_date, max_value=max_date)
        
    # Validasi jika pengguna salah memasukkan urutan tanggal
    if start_date > end_date:
        st.error("⚠️ Tgl Mulai tidak boleh melewati Tgl Akhir!")

    st.markdown("---")
    st.markdown("<small>Dikembangkan oleh Tim Analitik Polibatam</small>", unsafe_allow_html=True)

# ==========================================
# 4. MEMPROSES DATA BERDASARKAN FILTER
# ==========================================
df_working = df_full[df_full['pelabuhan'].isin(selected_ports)]

# PERBAIKAN: Logika filter menggunakan variabel start_date dan end_date yang baru
if start_date <= end_date:
    df_working = df_working[
        (df_working['tanggal'].dt.date >= start_date) & 
        (df_working['tanggal'].dt.date <= end_date)
    ]
else:
    # Kosongkan dataframe jika tanggal tidak valid
    df_working = pd.DataFrame(columns=df_full.columns) 

if df_working.empty:
    st.warning("⚠️ Tidak ada data pelabuhan yang dipilih atau sesuai rentang waktu. Pilih pelabuhan di sidebar untuk melihat visualisasi data.")

# ==========================================
# 5. BODY - HEADER SECTION (TITLE)
# ==========================================
st.title("Dashboard Analisis Sentimen Pelabuhan")
st.markdown("<p style='font-size: 18px; color: gray; margin-top:-15px;'>powered by Tim Analitik Polibatam</p>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 6. BODY - KPI ROW (RINGKASAN METRIK UTAMA)
# ==========================================
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

total_reviews = len(df_working)
avg_rating = df_working['review_rating'].mean() if ('review_rating' in df_working.columns and not df_working.empty) else 0
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

with tab1:
    if df_working.empty:
        st.info("Pilih minimal satu pelabuhan di sidebar untuk menampilkan visualisasi grafik dan tren.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribusi Popularitas (Volume Aktivitas)")
            st.markdown("<small>Pelabuhan dengan volume aktivitas ulasan tertinggi.</small>", unsafe_allow_html=True)
            pop_df = df_working['pelabuhan'].value_counts().reset_index()
            pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
            
            fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
            sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_d', ax=ax_pop) 
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
                
                nan_ports = scatter_df[scatter_df['Rata_Rating'].isna() | (scatter_df['Volume'] == 0)]['pelabuhan'].tolist()
                if nan_ports:
                    st.warning(f"⚠️ Pelabuhan berikut tidak muncul di grafik karena tidak memiliki data rating/ulasan yang valid: {', '.join(nan_ports)}")
                
                fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=200, palette='deep', ax=ax_scat)
                ax_scat.set_xlabel("Volume (Jumlah Ulasan)", fontsize=10)
                ax_scat
