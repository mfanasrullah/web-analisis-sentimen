import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re
import os
import PIL.Image as PILImage
from collections import Counter

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
    file_path = 'data_pelabuhan_cleaned.csv' # Instruksi 1: Menggunakan file ini
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', encoding_errors='replace')
    except Exception:
        df = pd.read_csv(file_path, sep=';', encoding='latin1') 
    
    # Menstandarkan nama kolom sesuai instruksi
    df = df.rename(columns={
        'name': 'pelabuhan',
        'review_datetime_utc': 'tanggal' # Instruksi 4: Digunakan untuk informasi kapan ulasan diberikan
    })
    
    df = df.dropna(subset=['pelabuhan', 'tanggal'])
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    
    try:
        df['tanggal'] = df['tanggal'].dt.tz_localize(None)
    except TypeError:
        pass 
    
    df = df.dropna(subset=['tanggal'])
    df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    
    # Instruksi 3: Memastikan rating (review_rating) terbaca sebagai angka
    if 'review_rating' in df.columns:
        df['review_rating'] = pd.to_numeric(df['review_rating'], errors='coerce')
        
    # Pastikan label berformat string
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower()
    
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

# Fungsi bantuan untuk mendapatkan kata terbanyak (Instruksi 2 & 5)
def get_top_words(text_series, top_n=5):
    all_text = " ".join(text_series.dropna().astype(str).tolist()).lower()
    all_text = re.sub(r'[^a-z\s]', '', all_text)
    words = all_text.split()
    
    # Stopwords sederhana bahasa Indonesia (bisa disesuaikan)
    stopwords = {'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'ini', 'itu', 'dengan', 'pada', 'saya', 'ada', 'pelabuhan'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not words: 
        return "-"
    common = Counter(words).most_common(top_n)
    return ", ".join([w[0] for w in common])

df_full = load_data()
model, vectorizer = load_model()

if df_full is None:
    st.error("File 'data_pelabuhan_cleaned.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTERS & BRANDING)
# ==========================================
with st.sidebar:
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
    
    st.markdown("#### Pilih Rentang Tanggal")
    min_date = df_full['tanggal'].min().date()
    max_date = df_full['tanggal'].max().date()
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Tanggal Mulai", value=min_date, min_value=min_date, max_value=max_date)
    with col_date2:
        end_date = st.date_input("📅 Tanggal Akhir", value=max_date, min_value=min_date, max_value=max_date)
        
    if start_date > end_date:
        st.error("⚠️ Tgl Mulai tidak boleh melewati Tgl Akhir!")

    st.markdown("---")
    st.markdown("<small>Dikembangkan oleh Tim Analitik Polibatam</small>", unsafe_allow_html=True)

# ==========================================
# 4. MEMPROSES DATA BERDASARKAN FILTER
# ==========================================
df_working = df_full[df_full['pelabuhan'].isin(selected_ports)]

if start_date <= end_date:
    df_working = df_working[
        (df_working['tanggal'].dt.date >= start_date) & 
        (df_working['tanggal'].dt.date <= end_date)
    ]
else:
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
# Penambahan tab "Ringkasan Eksekutif" untuk menampilkan Tabel sesuai instruksi 2, 3, 4, 5
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Ringkasan Eksekutif", 
    "📊 Visualisasi Data & Tren", 
    "☁️ WordCloud & Heatmap Keluhan", 
    "🤖 Prediksi Sentimen AI"
])

with tab1:
    st.markdown("### Ringkasan Data Tiap Pelabuhan")
    st.markdown("Tabel ini memberikan rangkuman kualitas rating, tingkat kesibukan, dan kata kunci (umum, positif, dan negatif) yang paling sering disebutkan oleh pengunjung berdasarkan data yang sudah dibersihkan.")
    
    if df_working.empty:
        st.info("Tidak ada data untuk direkap.")
    else:
        summary_data = []
        for port in sorted(df_working['pelabuhan'].unique()):
            port_df = df_working[df_working['pelabuhan'] == port]
            
            # Instruksi 3: review_rating untuk rating setiap pelabuhan
            avg_rate = round(port_df['review_rating'].mean(), 2) if 'review_rating' in port_df.columns else "N/A"
            tot_rev = len(port_df)
            
            # Instruksi 4: review_datetime_utc (tanggal) untuk melacak tren bulan tersibuk
            monthly = port_df.groupby('bulan_tahun').size()
            busy_month = monthly.idxmax() if not monthly.empty else "-"
            
            # Instruksi 2: review_text untuk menentukan kata sering muncul
            top_umum = get_top_words(port_df['review_text'], 5) if 'review_text' in port_df.columns else "-"
            
            # Instruksi 5: label untuk memetakan kata positif dan negatif
            if 'label' in port_df.columns and 'review_text' in port_df.columns:
                df_pos = port_df[port_df['label'] == 'positif']
                df_neg = port_df[port_df['label'] == 'negatif']
                top_pos = get_top_words(df_pos['review_text'], 5)
                top_neg = get_top_words(df_neg['review_text'], 5)
            else:
                top_pos, top_neg = "-", "-"
                
            summary_data.append([port, avg_rate, tot_rev, busy_month, top_umum, top_pos, top_neg])
            
        summary_df = pd.DataFrame(summary_data, columns=[
            "Nama Pelabuhan", "Rata-rata Rating", "Total Ulasan", 
            "Bulan Tersibuk", "Top 5 Kata (Umum)", "Top 5 Kata Positif", "Top 5 Kata Negatif"
        ])
        
        st.dataframe(summary_df, use_container_width=True)

with tab2:
    if df_working.empty:
        st.info("Pilih minimal satu pelabuhan di sidebar untuk menampilkan visualisasi grafik dan tren.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribusi Popularitas (Volume Aktivitas)")
            pop_df = df_working['pelabuhan'].value_counts().reset_index()
            pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
            
            fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
            sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_d', ax=ax_pop) 
            ax_pop.set_xlabel("Total Ulasan (Volume)", fontsize=10)
            ax_pop.set_ylabel("", fontsize=10)
            sns.despine(left=True, bottom=True)
            st.pyplot(fig_pop)

        with col2:
            st.markdown("#### Kualitas (Rating) vs Kuantitas (Volume)")
            if 'review_rating' in df_working.columns:
                scatter_df = df_working.groupby('pelabuhan').agg(
                    Rata_Rating=('review_rating', 'mean'),
                    Volume=('review_rating', 'count')
                ).reset_index()
                
                fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=200, palette='deep', ax=ax_scat)
                ax_scat.set_xlabel("Volume (Jumlah Ulasan)", fontsize=10)
                ax_scat.set_ylabel("Kualitas (Rata-rata Rating)", fontsize=10)
                ax_scat.set_ylim(1, 5) 
                ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Pelabuhan')
                sns.despine(left=True, bottom=True)
                st.pyplot(fig_scat)

        st.markdown("---")
        
        st.markdown("#### Tren Volume Ulasan per Bulan di Setiap Pelabuhan")
        if 'bulan_tahun' in df_working.columns:
            trend_df = df_working.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
            
            fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=trend_df, x='bulan_tahun', y='Jumlah', hue='pelabuhan', marker='o', palette='muted', ax=ax_trend)
            ax_trend.tick_params(axis='x', rotation=45, labelsize=9) 
            ax_trend.set_xlabel("Periode (Bulan)", fontsize=10)
            ax_trend.set_ylabel("Volume Ulasan", fontsize=10)
            ax_trend.grid(True, linestyle='--', alpha=0.6)
            sns.despine(left=True, bottom=True)
            st.pyplot(fig_trend)

with tab3:
    if df_working.empty:
         st.info("Pilih minimal satu pelabuhan di sidebar untuk menampilkan analisis kata kunci dan heatmap.")
    else:
        col_wc, col_hm = st.columns(2)
        
        with col_wc:
            st.markdown("#### Visualisasi WordCloud")
            
            if 'review_text' in df_working.columns:
                semua_teks = " ".join(df_working['review_text'].dropna().astype(str))
                if semua_teks.strip(): 
                    wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.info("Tidak ada data teks ulasan yang cukup untuk membuat WordCloud.")

        with col_hm:
            st.markdown("#### Heatmap Keluhan Konsumen")
            
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
                    sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, ax=ax_hm, annot_kws={"size": 8})
                    ax_hm.set_xlabel("Periode (Bulan)", fontsize=9)
                    ax_hm.set_ylabel("", fontsize=9)
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig_hm)
                else:
                    st.success("Luar biasa! Tidak ada ulasan negatif (Rating 1 & 2) yang ditemukan dalam rentang waktu terfilter.")

with tab4:
    st.header("Sistem Uji Sentimen Real-Time")
    
    if model is None or vectorizer is None:
        st.error("Model Machine Learning belum tersedia (file .pkl tidak ditemukan). Fitur prediksi dinonaktifkan.")
    else:
        kamus_positif = {'bagus', 'baik', 'cepat', 'bersih', 'ramah', 'nyaman', 'keren', 'mantap', 'memuaskan', 'mudah', 'rapi', 'aman', 'lancar', 'terbaik', 'puas', 'indah', 'luas', 'modern', 'sip', 'jos'}
        kamus_negatif = {'buruk', 'lambat', 'kotor', 'mahal', 'antri', 'jelek', 'kecewa', 'sulit', 'lama', 'ribet', 'bising', 'bau', 'rusak', 'berantakan', 'parah', 'kurang', 'sempit', 'macet', 'panas', 'kacau'}
        
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text.strip()
        
        user_input = st.text_area("Ketik ulasan terkait layanan pelabuhan (maks. 500 kata):", height=120)
        
        if st.button("Analisis Ulasan", type="primary"):
            if user_input:
                teks_bersih = clean_text(user_input)
                if teks_bersih:
                    vektor_input = vectorizer.transform([teks_bersih])
                    prediksi = model.predict(vektor_input)[0]
                    probabilitas = model.predict_proba(vektor_input)[0]
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    warna = "green" if prediksi.lower() == "positif" else "red" if prediksi.lower() == "negatif" else "gray"
                    
                    with result_col1:
                        st.markdown(f"<div style='border: 1px solid lightgray; padding: 10px; border-radius: 5px;'>Hasil Deteksi Model: <b style='color:{warna}; font-size: 20px;'>{prediksi.upper()}</b></div>", unsafe_allow_html=True)
                    
                    with result_col2:
                        st.write("**Keyakinan Model (Probabilitas):**")
                        classes_lower = [str(c).lower() for c in model.classes_]
                        prob_positive = probabilitas[classes_lower.index('positif')] if 'positif' in classes_lower else 0
                        prob_negative = probabilitas[classes_lower.index('negatif')] if 'negatif' in classes_lower else 0
                        
                        st.progress(float(prob_positive), text=f"Positif: {prob_positive:.1%}")
                        st.progress(float(prob_negative), text=f"Negatif: {prob_negative:.1%}")
                        
                    st.markdown("---")
                    st.markdown("#### 🔍 Analisis Kata Kunci dalam Ulasan")
                    
                    kata_dalam_teks = set(teks_bersih.split())
                    kata_positif_ditemukan = kata_dalam_teks.intersection(kamus_positif)
                    kata_negatif_ditemukan = kata_dalam_teks.intersection(kamus_negatif)
                    
                    jml_pos = len(kata_positif_ditemukan)
                    jml_neg = len(kata_negatif_ditemukan)
                    
                    col_word1, col_word2 = st.columns(2)
                    
                    with col_word1:
                        if kata_positif_ditemukan:
                            kata_pos_str = ", ".join([f"`{k}`" for k in kata_positif_ditemukan])
                            st.success(f"**Kata Positif Terdeteksi ({jml_pos}):**\n\n{kata_pos_str}")
                        else:
                            st.info("**Kata Positif Terdeteksi:**\n\n0 kata.")
                            
                    with col_word2:
                        if kata_negatif_ditemukan:
                            kata_neg_str = ", ".join([f"`{k}`" for k in kata_negatif_ditemukan])
                            st.error(f"**Kata Negatif Terdeteksi ({jml_neg}):**\n\n{kata_neg_str}")
                        else:
                            st.info("**Kata Negatif Terdeteksi:**\n\n0 kata.")
                            
                    if prediksi.lower() == "positif" and (jml_neg > jml_pos):
                        st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Positif**, meskipun terdapat lebih banyak kata bernada negatif. Hal ini biasanya terjadi karena model mendeteksi adanya kata penyangkalan (contoh: *'tidak buruk'*), atau kata positif yang ada memiliki bobot konteks yang jauh lebih kuat.")
                        
                    elif prediksi.lower() == "negatif" and (jml_pos > jml_neg):
                        st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Negatif**, meskipun terdapat lebih banyak kata bernada positif. Harap perhatikan konteks kalimat (misal: sarkasme atau keluhan tersembunyi).")
