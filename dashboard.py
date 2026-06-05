# ============================================
# DASHBOARD STUDENT LIFESTYLE RISK & PRODUCTIVITY
# Versi 2.0 - Perbaikan Notifikasi, Filter, Visualisasi
# Tim CC26-PRU475 | Sri Hartati
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Student Lifestyle Risk Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# INISIALISASI SESSION STATE UNTUK NOTIFIKASI
# ============================================
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_accuracy" not in st.session_state:
    st.session_state.model_accuracy = None

# ============================================
# CUSTOM CSS (sama seperti sebelumnya, tapi saya persingkat di sini)
# ============================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%); }
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        padding: 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 800; }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border-bottom: 4px solid #6C63FF;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #6C63FF; }
    .section-header {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        padding: 0.5rem 1.5rem;
        border-radius: 40px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .section-header h2 { color: white; margin: 0; font-size: 1.3rem; }
    .insight-box {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid #6C63FF;
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
    [data-testid="stSidebar"] * { color: white; }
    .sidebar-header { text-align: center; padding: 1rem 0; border-bottom: 2px solid #6C63FF; }
    .footer { text-align: center; padding: 1.5rem; margin-top: 2rem; background: white; border-radius: 15px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNGSI LOAD ATAU TRAIN MODEL (dengan notifikasi sekali saja)
# ============================================
@st.cache_resource
def load_or_train_model():
    """Load model jika ada dan kompatibel, atau latih ulang. Notifikasi disimpan ke session state."""
    model_path = Path("models/stress_level_model.joblib")
    metadata_path = Path("models/model_metadata.json")
    data_path = Path("data/StressLevelDataset.csv")
    
    if not data_path.exists():
        st.error("❌ Dataset tidak ditemukan di 'data/StressLevelDataset.csv'")
        return None, None, None
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']
    
    # Coba load model jika ada
    if model_path.exists() and metadata_path.exists():
        try:
            model = joblib.load(model_path)
            # Test prediksi
            test_sample = X.iloc[:1]
            _ = model.predict(test_sample)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Simpan status ke session state (tidak tampil sebagai notifikasi)
            st.session_state.model_loaded = True
            st.session_state.model_accuracy = metadata.get("test_accuracy", "unknown")
            return model, metadata, df
        except Exception as e:
            # Jika error, lanjut ke training ulang
            pass
    
    # Training ulang
    with st.spinner("🔄 Melatih model baru dengan dataset... (hanya sekali)"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', HistGradientBoostingClassifier(random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Simpan model dan metadata
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        
        feature_schema = []
        for col in X.columns:
            feature_schema.append({
                "name": col,
                "label": col.replace("_", " ").title(),
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "step": 1.0,
                "default": float(X[col].median()),
                "description": f"Skor {col.replace('_', ' ')} ({X[col].min():.0f}-{X[col].max():.0f})"
            })
        metadata = {
            "model_name": "HistGradientBoostingClassifier",
            "features": list(X.columns),
            "stress_labels": {"0": "Low Stress", "1": "Medium Stress", "2": "High Stress"},
            "feature_schema": feature_schema,
            "test_accuracy": acc
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        st.session_state.model_loaded = True
        st.session_state.model_accuracy = acc
        # Tampilkan notifikasi hanya sekali dengan toast (tidak mengganggu)
        st.toast(f"✅ Model berhasil dilatih! Akurasi: {acc:.1%}", icon="🎉")
        return pipeline, metadata, df

# ============================================
# FUNGSI VISUALISASI YANG LEBIH BAIK
# ============================================
def plot_stress_distribution(df):
    """Bar chart distribusi stres dengan persentase"""
    counts = df['stress_level'].value_counts().sort_index()
    labels = ['Low Stress', 'Medium Stress', 'High Stress']
    values = counts.values
    percentages = values / values.sum() * 100
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    bars = ax.bar(labels, values, color=['#00D09C', '#FFB347', '#FF4B4B'], edgecolor='white', linewidth=1.5)
    ax.set_title('Distribusi Tingkat Stres Mahasiswa', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tingkat Stres', fontsize=12)
    ax.set_ylabel('Jumlah Mahasiswa', fontsize=12)
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val} ({pct:.1f}%)', 
                ha='center', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def plot_sleep_vs_stress(df):
    """Rata-rata stres per sleep_quality"""
    sleep_agg = df.groupby('sleep_quality')['stress_level'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    bars = ax.bar(sleep_agg['sleep_quality'], sleep_agg['stress_level'], color='#6C63FF', edgecolor='white')
    ax.set_title('Rata-rata Tingkat Stres berdasarkan Kualitas Tidur', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kualitas Tidur (1 = sangat buruk, 5 = sangat baik)', fontsize=11)
    ax.set_ylabel('Rata-rata Tingkat Stres (0-2)', fontsize=11)
    ax.set_ylim(0, 2.2)
    for bar, val in zip(bars, sleep_agg['stress_level']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', 
                ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def plot_anxiety_depression_scatter(df):
    """Scatter plot dengan regresi garis"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    colors = {0: '#00D09C', 1: '#FFB347', 2: '#FF4B4B'}
    for stress in [0,1,2]:
        subset = df[df['stress_level'] == stress]
        label = ['Low Stress', 'Medium Stress', 'High Stress'][stress]
        ax.scatter(subset['anxiety_level'], subset['depression'], c=colors[stress], 
                   label=label, alpha=0.7, edgecolors='white', s=60)
    ax.set_title('Hubungan Kecemasan (Anxiety) dan Depresi', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tingkat Kecemasan', fontsize=12)
    ax.set_ylabel('Tingkat Depresi', fontsize=12)
    ax.legend(title='Tingkat Stres')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def plot_correlation_heatmap(df):
    """Heatmap korelasi Spearman dengan seaborn"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Korelasi Spearman antar Fitur', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return fig

# ============================================
# LOAD MODEL DAN DATA
# ============================================
model, metadata, df = load_or_train_model()

if df is None:
    st.stop()

# Tampilkan notifikasi sukses sekali (jika belum ditampilkan)
if st.session_state.model_loaded and st.session_state.model_accuracy is not None:
    if 'notif_shown' not in st.session_state:
        st.success(f"✅ Model siap digunakan! Akurasi: {st.session_state.model_accuracy:.1%}")
        st.session_state.notif_shown = True

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🧠 Student Lifestyle Risk & Productivity Monitor</h1>
    <p>Analisis Gaya Hidup, Prediksi Stres, dan Rekomendasi Kesehatan Mental Mahasiswa</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGASI
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🧑‍🎓 Menu</h3></div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Beranda", "📊 Eksplorasi Data", "🔮 Prediksi Stres"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Tim CC26-PRU475 | Dataset: Kaggle")

# ============================================
# HALAMAN BERANDA
# ============================================
if page == "🏠 Beranda":
    st.markdown('<div class="section-header"><h2>📌 Tentang Dashboard</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        Dashboard ini membantu mahasiswa dan dosen memahami faktor-faktor yang mempengaruhi stres akademik.
        
        **Tingkat Stres:**
        - 🟢 **Low Stress** (0) – Kondisi mental baik
        - 🟡 **Medium Stress** (1) – Perlu perhatian
        - 🔴 **High Stress** (2) – Risiko tinggi, segera konsultasi
        
        **Fitur:**
        - Eksplorasi data dengan filter interaktif
        - Prediksi tingkat stres berdasarkan gaya hidup
        - Rekomendasi penanganan
        """)
    with col2:
        total = df.shape[0]
        high = (df['stress_level']==2).sum()
        st.metric("Total Responden", total)
        st.metric("Stres Tinggi", high, delta=f"{high/total:.1%}")
    
    st.markdown('<div class="insight-box"><b>💡 Sumber Data:</b> Dataset dari Kaggle dengan lisensi MIT. Model machine learning menggunakan HistGradientBoostingClassifier dengan akurasi test {:.1%}.</div>'.format(st.session_state.model_accuracy or 0), unsafe_allow_html=True)

# ============================================
# HALAMAN EKSPLORASI DATA (FILTER DIPERBAIKI)
# ============================================
elif page == "📊 Eksplorasi Data":
    st.markdown('<div class="section-header"><h2>📊 Eksplorasi Data Gaya Hidup</h2></div>', unsafe_allow_html=True)
    
    # FILTER DI SIDEBAR (dengan nilai default mencakup semua data)
    st.sidebar.markdown("### 🔍 Filter Data")
    
    # Ambil min/max dari dataset untuk default slider yang benar
    min_sleep = int(df['sleep_quality'].min())
    max_sleep = int(df['sleep_quality'].max())
    min_study = int(df['study_load'].min())
    max_study = int(df['study_load'].max())
    min_academic = int(df['academic_performance'].min())
    max_academic = int(df['academic_performance'].max())
    
    # Filter stress level (multiselect dengan semua opsi terpilih default)
    stress_options = ['Low Stress', 'Medium Stress', 'High Stress']
    selected_stress = st.sidebar.multiselect(
        "Tingkat Stres", 
        options=stress_options,
        default=stress_options  # ← semua terpilih, sehingga total data 100%
    )
    stress_map = {'Low Stress':0, 'Medium Stress':1, 'High Stress':2}
    stress_values = [stress_map[s] for s in selected_stress]
    
    # Slider dengan rentang penuh dataset
    sleep_range = st.sidebar.slider("Kualitas Tidur (sleep_quality)", min_sleep, max_sleep, (min_sleep, max_sleep))
    study_range = st.sidebar.slider("Beban Belajar (study_load)", min_study, max_study, (min_study, max_study))
    academic_range = st.sidebar.slider("Performa Akademik (academic_performance)", min_academic, max_academic, (min_academic, max_academic))
    
    # Terapkan filter
    filtered_df = df[
        (df['stress_level'].isin(stress_values)) &
        (df['sleep_quality'] >= sleep_range[0]) & (df['sleep_quality'] <= sleep_range[1]) &
        (df['study_load'] >= study_range[0]) & (df['study_load'] <= study_range[1]) &
        (df['academic_performance'] >= academic_range[0]) & (df['academic_performance'] <= academic_range[1])
    ]
    
    st.markdown(f"**📈 Menampilkan {len(filtered_df)} dari {len(df)} data** (filter aktif)")
    
    if len(filtered_df) == 0:
        st.warning("Tidak ada data yang cocok dengan filter. Coba perluas rentang filter.")
    else:
        # METRIK SINGKAT
        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Stres", f"{filtered_df['stress_level'].mean():.2f} / 2")
        col2.metric("Rata-rata Kualitas Tidur", f"{filtered_df['sleep_quality'].mean():.1f} / {max_sleep}")
        col3.metric("Rata-rata Depresi", f"{filtered_df['depression'].mean():.1f}")
        
        # VISUALISASI (2 baris, masing-masing 2 kolom)
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_stress_distribution(filtered_df))
        with col_b:
            st.pyplot(plot_sleep_vs_stress(filtered_df))
        
        col_c, col_d = st.columns(2)
        with col_c:
            st.pyplot(plot_anxiety_depression_scatter(filtered_df))
        with col_d:
            st.pyplot(plot_correlation_heatmap(filtered_df))
        
        # INSIGHT DINAMIS
        st.markdown('<div class="insight-box"><b>📈 Insight:</b><br>', unsafe_allow_html=True)
        # Korelasi anxiety-stress
        corr_anx = filtered_df[['anxiety_level','stress_level']].corr().iloc[0,1]
        st.markdown(f"- Kecemasan (anxiety) berkorelasi **{corr_anx:.2f}** dengan tingkat stres. Semakin cemas, cenderung stres lebih tinggi.")
        # Perbandingan sleep quality
        high_stress_group = filtered_df[filtered_df['stress_level']==2]
        if len(high_stress_group) > 0:
            avg_sleep_high = high_stress_group['sleep_quality'].mean()
            avg_sleep_all = filtered_df['sleep_quality'].mean()
            st.markdown(f"- Mahasiswa dengan stres tinggi memiliki kualitas tidur rata-rata **{avg_sleep_high:.1f}**, lebih rendah dari rata-rata umum ({avg_sleep_all:.1f}).")
        # Study load
        avg_study = filtered_df['study_load'].mean()
        st.markdown(f"- Beban belajar rata-rata responden adalah **{avg_study:.1f} / {max_study}**. Beban tinggi berkontribusi pada stres.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabel data (collapsible)
        with st.expander("📋 Lihat Data Lengkap"):
            st.dataframe(filtered_df, use_container_width=True, height=400)

# ============================================
# HALAMAN PREDIKSI (sama seperti sebelumnya, sudah cukup baik)
# ============================================
elif page == "🔮 Prediksi Stres":
    st.markdown('<div class="section-header"><h2>🔮 Prediksi Tingkat Stres</h2></div>', unsafe_allow_html=True)
    st.markdown("Masukkan data gaya hidup Anda untuk mendapatkan prediksi tingkat stres.")
    
    if model is None:
        st.error("Model tidak tersedia. Silakan periksa koneksi atau file dataset.")
        st.stop()
    
    feature_schema = metadata.get("feature_schema", [])
    if not feature_schema:
        # fallback
        X_cols = [c for c in df.columns if c != 'stress_level']
        for col in X_cols:
            feature_schema.append({"name": col, "label": col.replace("_"," ").title(), 
                                   "min": float(df[col].min()), "max": float(df[col].max()), 
                                   "step": 1.0, "default": float(df[col].median())})
    
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        input_vals = {}
        for i, feat in enumerate(feature_schema):
            target = col1 if i%2==0 else col2
            val = target.slider(
                feat["label"], 
                min_value=feat["min"], max_value=feat["max"], 
                value=feat["default"], step=feat["step"],
                help=feat.get("description", "")
            )
            input_vals[feat["name"]] = val
        submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)
    
    if submitted:
        features_order = metadata.get("features", [f["name"] for f in feature_schema])
        input_df = pd.DataFrame([input_vals])[features_order]
        pred = model.predict(input_df)[0]
        pred_label = metadata["stress_labels"].get(str(pred), "Unknown")
        
        color_map = {"Low Stress": "#00D09C", "Medium Stress": "#FFB347", "High Stress": "#FF4B4B"}
        advice = {
            "Low Stress": "✅ Pertahankan gaya hidup sehat, terus jaga keseimbangan akademik dan istirahat.",
            "Medium Stress": "⚠️ Coba tingkatkan kualitas tidur dan kurangi beban belajar yang tidak perlu. Lakukan relaksasi.",
            "High Stress": "🚨 Segera konsultasi dengan konselor kampus. Prioritaskan kesehatan mental, olahraga teratur, dan istirahat cukup."
        }
        st.markdown(f"""
        <div style="background:{color_map[pred_label]}20; padding:1.5rem; border-radius:20px; text-align:center; border:2px solid {color_map[pred_label]};">
            <h2 style="color:{color_map[pred_label]};">{pred_label}</h2>
            <p style="font-size:1.1rem;">{advice[pred_label]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            proba_df = pd.DataFrame({"Tingkat Stres": ["Low","Medium","High"], "Probabilitas": proba})
            fig = px.bar(proba_df, x="Tingkat Stres", y="Probabilitas", text_auto='.2f', 
                         color="Tingkat Stres", color_discrete_map={"Low":"#00D09C","Medium":"#FFB347","High":"#FF4B4B"})
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown('<div class="footer">© 2026 Coding Camp powered by DBS Foundation | Tim CC26-PRU475</div>', unsafe_allow_html=True)