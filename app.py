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
    
    # Membagi input tanggal menjadi 2 kolom terpisah agar lebih intuitif
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

# Logika filter menggunakan variabel start_date dan end_date
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
# 7. BODY - PEMBUATAN TABS (VERSI DIPERBARUI)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan & Tren Kinerja", 
    "☁️ Analisis Kata & Heatmap", 
    "📝 Suara Konsumen (Daftar Ulasan)", 
    "🤖 Uji AI Real-Time"
])

# ==== TAB 1: RINGKASAN & TREN KINERJA ====
with tab1:
    if df_working.empty:
        st.info("Pilih minimal satu pelabuhan di sidebar.")
    else:
        st.markdown("#### Papan Peringkat Pelabuhan (Leaderboard)")
        st.markdown("<small>Matriks perbandingan performa antar pelabuhan.</small>", unsafe_allow_html=True)
        
        # Membuat tabel summary (Leaderboard)
        if 'label' in df_working.columns and 'review_rating' in df_working.columns:
            leaderboard = df_working.groupby('pelabuhan').agg(
                Total_Ulasan=('review_text', 'count'),
                Rata_Rating=('review_rating', 'mean')
            ).reset_index()
            
            # Hitung persentase sentimen positif
            positif_count = df_working[df_working['label'].str.lower() == 'positif'].groupby('pelabuhan').size()
            leaderboard = leaderboard.set_index('pelabuhan')
            leaderboard['% Sentimen Positif'] = (positif_count / leaderboard['Total_Ulasan'] * 100).fillna(0).round(1).astype(str) + '%'
            leaderboard['Rata_Rating'] = leaderboard['Rata_Rating'].round(2)
            
            st.dataframe(leaderboard.reset_index(), use_container_width=True)

        st.markdown("---")
        
        col1_t1, col2_t1 = st.columns(2)
        
        with col1_t1:
            st.markdown("#### Proporsi Sentimen Pengunjung")
            st.markdown("<small>Persentase ulasan Positif, Netral, dan Negatif.</small>", unsafe_allow_html=True)
            if 'label' in df_working.columns:
                sentimen_counts = df_working['label'].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                # Menggunakan warna khusus: Positif(Hijau), Negatif(Merah), Netral(Abu-abu)
                colors = ['#2ca02c' if label.lower()=='positif' else '#d62728' if label.lower()=='negatif' else '#7f7f7f' for label in sentimen_counts.index]
                
                ax_pie.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10})
                ax_pie.axis('equal')
                st.pyplot(fig_pie)
            else:
                st.warning("Kolom 'label' sentimen tidak ditemukan dalam data.")

        with col2_t1:
            st.markdown("#### Tren Kepuasan (Rata-rata Rating) per Bulan")
            st.markdown("<small>Melihat fluktuasi kepuasan pelanggan dari waktu ke waktu.</small>", unsafe_allow_html=True)
            if 'bulan_tahun' in df_working.columns and 'review_rating' in df_working.columns:
                trend_rating_df = df_working.groupby(['bulan_tahun', 'pelabuhan'])['review_rating'].mean().reset_index()
                
                fig_trend_rating, ax_trend_rating = plt.subplots(figsize=(8, 5))
                sns.lineplot(data=trend_rating_df, x='bulan_tahun', y='review_rating', hue='pelabuhan', marker='s', ax=ax_trend_rating)
                ax_trend_rating.tick_params(axis='x', rotation=45)
                ax_trend_rating.set_xlabel("Bulan", fontsize=10)
                ax_trend_rating.set_ylabel("Rata-rata Rating", fontsize=10)
                ax_trend_rating.set_ylim(1, 5) # Rating selalu 1-5
                sns.despine(left=True, bottom=True)
                st.pyplot(fig_trend_rating)

# ==== TAB 2: WORDCLOUD & HEATMAP ====
with tab2:
    if df_working.empty:
         st.info("Pilih minimal satu pelabuhan di sidebar.")
    else:
        col_wc, col_hm = st.columns(2)
        with col_wc:
            st.markdown("#### Visualisasi WordCloud")
            st.markdown("<small>Kata kunci yang paling sering muncul dalam ulasan.</small>", unsafe_allow_html=True)
            if 'review_text' in df_working.columns:
                semua_teks = " ".join(df_working['review_text'].dropna().astype(str))
                if semua_teks.strip(): 
                    wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.info("Data ulasan tidak cukup.")
                    
        with col_hm:
            st.markdown("#### Heatmap Keluhan Konsumen (Rating 1 & 2)")
            st.markdown("<small>Fokus pada lokasi dan waktu puncak keluhan konsumen.</small>", unsafe_allow_html=True)
            if 'bulan_tahun' in df_working.columns and 'review_rating' in df_working.columns:
                df_negatif = df_working[df_working['review_rating'] <= 2]
                if not df_negatif.empty:
                    pivot_keluhan = df_negatif.pivot_table(index='pelabuhan', columns='bulan_tahun', values='review_rating', aggfunc='count', fill_value=0)
                    fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, ax=ax_hm, annot_kws={"size": 8})
                    ax_hm.set_xlabel("Periode (Bulan)", fontsize=9)
                    ax_hm.set_ylabel("", fontsize=9)
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig_hm)
                else:
                    st.success("Tidak ada ulasan negatif di rentang ini.")

# ==== TAB 3: DAFTAR ULASAN ASLI ====
with tab3:
    st.markdown("### 📝 Daftar Ulasan (Voice of Customer)")
    st.markdown("Telusuri ulasan asli dari pelanggan. Gunakan filter di bawah ini untuk mencari keluhan atau pujian spesifik.")
    
    if not df_working.empty and 'review_text' in df_working.columns:
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_sentimen = st.selectbox("Filter Sentimen:", ["Semua", "Negatif", "Positif", "Netral"])
        with col_filter2:
            cari_kata = st.text_input("Cari kata kunci (Contoh: kotor, lama, calo):", "")
        
        # Aplikasikan filter
        df_tampil = df_working[['tanggal', 'pelabuhan', 'review_rating', 'label', 'review_text']].copy()
        
        if filter_sentimen != "Semua":
            df_tampil = df_tampil[df_tampil['label'].str.lower() == filter_sentimen.lower()]
            
        if cari_kata:
            df_tampil = df_tampil[df_tampil['review_text'].str.contains(cari_kata, case=False, na=False)]
            
        # Format tabel agar rapi
        df_tampil['tanggal'] = df_tampil['tanggal'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(df_tampil.sort_values('tanggal', ascending=False), use_container_width=True, height=400)
    else:
        st.warning("Data ulasan teks tidak tersedia.")

# ==== TAB 4: PREDIKSI AI (SISTEM UJI REAL-TIME) ====
with tab4:
    st.header("Sistem Uji Sentimen Real-Time")
    
    if model is None or vectorizer is None:
        st.error("Model Machine Learning belum tersedia (file .pkl tidak ditemukan). Fitur prediksi dinonaktifkan.")
    else:
        st.write("<small>Masukkan teks ulasan baru untuk dianalisis oleh model Machine Learning.</small>", unsafe_allow_html=True)
        
        # KAMUS KATA POSITIF & NEGATIF 
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
                    # ML Prediction
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
                        
                    # EKSTRAKSI KATA POSITIF DAN NEGATIF
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
                            
                    # LOGIKA DETEKSI ANOMALI (MISMATCH WARNING)
                    if prediksi.lower() == "positif" and (jml_neg > jml_pos):
                        st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Positif**, meskipun terdapat lebih banyak kata bernada negatif. Hal ini biasanya terjadi karena model mendeteksi adanya kata penyangkalan (contoh: *'tidak buruk'*), atau kata positif yang ada memiliki bobot konteks yang jauh lebih kuat.")
                        
                    elif prediksi.lower() == "negatif" and (jml_pos > jml_neg):
                        st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Negatif**, meskipun terdapat lebih banyak kata bernada positif. Harap perhatikan konteks kalimat (misal: sarkasme atau keluhan tersembunyi).")
