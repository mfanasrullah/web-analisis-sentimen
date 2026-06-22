import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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
@st.cache_data
def load_data():
    df = pd.read_csv('data_pelabuhan.csv', sep=';') 
    df = df.rename(columns={
        'name': 'pelabuhan',
        'review_datetime_utc': 'tanggal'
    })
    df = df.dropna(subset=['pelabuhan', 'tanggal'])
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)
    return df

@st.cache_resource
def load_model():
    # Pastikan file model dan vectorizer tersedia, gunakan dummy try-except jika error untuk testing UI
    try:
        model = joblib.load('model_sentimen.pkl')
        vectorizer = joblib.load('vectorizer_sentimen.pkl')
        return model, vectorizer
    except:
        return None, None

df = load_data()
model, vectorizer = load_model()

# ==========================================
# 3. SIDEBAR (FILTER DINAMIS)
# ==========================================
st.sidebar.header("🔍 Filter Data")
st.sidebar.write("Sesuaikan data yang ingin ditampilkan:")

daftar_pelabuhan = df['pelabuhan'].unique()
pilihan_pelabuhan = st.sidebar.multiselect(
    "Pilih Pelabuhan:", 
    options=daftar_pelabuhan, 
    default=daftar_pelabuhan # Tampilkan semua secara default
)

# Terapkan filter ke dataframe
df_filter = df[df['pelabuhan'].isin(pilihan_pelabuhan)]

# ==========================================
# 4. PEMBUATAN TABS (NAVIGASI)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Data & Tren", "☁️ WordCloud & Heatmap", "🤖 Prediksi Sentimen AI"])

# ------------------------------------------
# TAB 1: VISUALISASI DATA & TREN
# ------------------------------------------
with tab1:
    st.header("Analisis Kinerja dan Popularitas Pelabuhan")
    
    # Tambahkan Metrik Angka Utama (KPI)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Ulasan (Terfilter)", f"{len(df_filter):,}")
    if not df_filter.empty and 'review_rating' in df_filter.columns:
        rata_rating = round(df_filter['review_rating'].mean(), 2)
        col_m2.metric("Rata-rata Rating", f"⭐ {rata_rating}")
    col_m3.metric("Pelabuhan Ditampilkan", len(pilihan_pelabuhan))
    
    st.markdown("---")
    
    if df_filter.empty:
        st.warning("Silakan pilih minimal satu pelabuhan di menu Sidebar.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Popularitas (Volume Aktivitas)")
            pop_df = df_filter['pelabuhan'].value_counts().reset_index()
            pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
            
            # Ganti Seaborn dengan Plotly Bar Chart interaktif
            fig_pop = px.bar(
                pop_df, x='Jumlah Ulasan', y='Pelabuhan', 
                orientation='h', color='Jumlah Ulasan', 
                color_continuous_scale='Viridis',
                text='Jumlah Ulasan'
            )
            fig_pop.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_pop, use_container_width=True)

        with col2:
            st.subheader("Kualitas (Rating) vs Kuantitas (Volume)")
            if 'review_rating' in df_filter.columns:
                scatter_df = df_filter.groupby('pelabuhan').agg(
                    Rata_Rating=('review_rating', 'mean'),
                    Volume=('review_rating', 'count')
                ).reset_index()
                
                # Ganti Seaborn dengan Plotly Scatter Plot (Bisa di-hover)
                fig_scat = px.scatter(
                    scatter_df, x='Volume', y='Rata_Rating', 
                    color='pelabuhan', size='Volume', 
                    hover_name='pelabuhan', size_max=40
                )
                st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("---")
        
        st.subheader("Tren Volume Ulasan per Bulan di Setiap Pelabuhan")
        if 'bulan_tahun' in df_filter.columns:
            trend_df = df_filter.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
            
            # Ganti Seaborn dengan Plotly Line Chart (Bisa zoom dan filter legenda)
            fig_trend = px.line(
                trend_df, x='bulan_tahun', y='Jumlah', 
                color='pelabuhan', markers=True
            )
            fig_trend.update_layout(xaxis_title="Bulan & Tahun", yaxis_title="Volume Ulasan")
            st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------
# TAB 2: WORDCLOUD & HEATMAP
# ------------------------------------------
with tab2:
    col_wc, col_hm = st.columns([1, 1.2]) # Sedikit penyesuaian lebar kolom
    
    with col_wc:
        st.subheader("Visualisasi WordCloud")
        st.write("Kata yang paling sering muncul dalam ulasan (berdasarkan filter).")
        
        if not df_filter.empty and 'review_text' in df_filter.columns:
            semua_teks = " ".join(df_filter['review_text'].dropna().astype(str))
            if semua_teks.strip():
                wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Tidak ada teks ulasan untuk ditampilkan.")
        
    with col_hm:
        st.subheader("Heatmap Keluhan (Rating 1 & 2)")
        st.write("Intensitas ulasan negatif per pelabuhan dan bulan.")
        
        if not df_filter.empty and 'bulan_tahun' in df_filter.columns and 'review_rating' in df_filter.columns:
            df_negatif = df_filter[df_filter['review_rating'] <= 2]
            
            if not df_negatif.empty:
                pivot_keluhan = df_negatif.pivot_table(
                    index='pelabuhan', 
                    columns='bulan_tahun', 
                    values='review_rating', 
                    aggfunc='count', 
                    fill_value=0
                )
                
                # Ganti Seaborn dengan Plotly Heatmap
                fig_hm = px.imshow(
                    pivot_keluhan, 
                    text_auto=True, 
                    color_continuous_scale='Reds',
                    aspect="auto",
                    labels=dict(x="Bulan", y="Pelabuhan", color="Jml Keluhan")
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.success("Tidak ada keluhan (Rating 1 & 2) pada data yang difilter!")

# ------------------------------------------
# TAB 3: PREDIKSI SENTIMEN (AI)
# ------------------------------------------
with tab3:
    st.header("Sistem Uji Sentimen Real-Time")
    st.write("Masukkan teks ulasan baru untuk dianalisis oleh model Machine Learning.")
    
    if model and vectorizer:
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
                    
                    warna = "green" if prediksi == "Positif" else "red" if prediksi == "Negatif" else "gray"
                    st.markdown(f"### Hasil Deteksi: <span style='color:{warna}'>{prediksi.upper()}</span>", unsafe_allow_html=True)
                    
                    st.write("**Keyakinan Model:**")
                    for i, cls in enumerate(model.classes_):
                        st.write(f"{cls}")
                        st.progress(float(probabilitas[i]))
                else:
                    st.warning("Teks tidak valid setelah dibersihkan.")
            else:
                st.error("Harap masukkan ulasan terlebih dahulu!")
    else:
        st.error("❌ Model atau Vectorizer tidak ditemukan. Pastikan file .pkl tersedia di folder yang sama.")
