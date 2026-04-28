import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from io import StringIO
from pulp import *
import base64

# Konfigurasi Halaman
st.set_page_config(
    page_title="OR-Agent | Operation Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .bab-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .feature-box {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
    }
    .result-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4caf50;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA & KONTEN MATERI ====================

BAB_DATA = {
    "Bab 1": {
        "judul": "Pendahuluan Operation Research",
        "topik": ["Definisi & Sejarah OR", "Karakteristik Pendekatan OR", "Peran OR dalam Pengambilan Keputusan", 
                 "Aplikasi OR di Berbagai Bidang", "Etika & Tanggung Jawab Profesional"],
        "cpmk": "CPMK-1, CPMK-2, CPMK-3, CPMK-4",
        "contoh": None,
        "tugas": "Diskusi: Mengapa pendekatan interdisipliner penting dalam OR?",
        "pdf_link": "pdf/materi_bab_1.pdf"
    },
    "Bab 2": {
        "judul": "Sejarah, Metodologi & Konsep Dasar",
        "topik": ["Evolusi Historis OR", "Fase Penyelesaian Masalah OR", "Konsep Matematika Dasar",
                 "Pendekatan Analitis vs Heuristik vs Simulasi", "Peta Teknik Utama OR", "Perangkat Lunak OR"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": None,
        "tugas": "Identifikasi masalah di sekitar kampus yang dapat diselesaikan dengan OR",
        "pdf_link": "pdf/materi_bab_2.pdf"
    },
    "Bab 3": {
        "judul": "Linear Programming - Bagian 1 (Grafis)",
        "topik": ["Formulasi Model LP", "Asumsi LP", "Konsep Optimasi (Max/Min)", 
                 "Metode Grafis", "Masalah Transportasi"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": {
            "nama": "Produksi Mebel 'Kayu Jati'",
            "deskripsi": "Memaksimalkan keuntungan dari produksi meja dan kursi dengan keterbatasan kayu dan jam kerja",
            "tipe": "maximasi",
            "variabel": {"x1": "Jumlah Meja", "x2": "Jumlah Kursi"},
            "fungsi_tujuan": {"x1": 800, "x2": 600, "tipe": "max"},
            "kendala": [
                {"nama": "Kayu Jati", "x1": 2, "x2": 3, "rhs": 60, "tipe": "<="},
                {"nama": "Jam Kerja", "x1": 5, "x2": 3, "rhs": 100, "tipe": "<="}
            ]
        },
        "tugas": "Tugas 1: Analisis Kasus Optimasi (Minimasi Pakan Ternak & Maksimasi Kedai Kopi)",
        "pdf_link": "pdf/materi_bab_3.pdf"
    },
    "Bab 4": {
        "judul": "Linear Programming - Bagian 2 (Simpleks & Software)",
        "topik": ["Metode Simpleks", "Analisis Sensitivitas", "Shadow Price", 
                 "Range of Feasibility", "Range of Optimality", "PuLP Python"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": {
            "nama": "PT. Agro Maju (Produksi Pupuk)",
            "deskripsi": "Memaksimalkan keuntungan produksi Pupuk Premium dan Standar dengan 3 bahan baku",
            "tipe": "maximasi",
            "variabel": {"x1": "Pupuk Premium", "x2": "Pupuk Standar"},
            "fungsi_tujuan": {"x1": 25000, "x2": 15000, "tipe": "max"},
            "kendala": [
                {"nama": "Kompos", "x1": 2, "x2": 1, "rhs": 120, "tipe": "<="},
                {"nama": "Kandang Ayam", "x1": 3, "x2": 1, "rhs": 150, "tipe": "<="},
                {"nama": "Dedak", "x1": 1, "x2": 3, "rhs": 180, "tipe": "<="},
                {"nama": "Kontrak", "x1": 0, "x2": 1, "rhs": 30, "tipe": ">="}
            ]
        },
        "tugas": "Tugas 2: Studi Kasus LP - Perusahaan Elektronik 'GadgetPro'",
        "pdf_link": "pdf/materi_bab_4.pdf"
    },
    "Bab 5": {
        "judul": "Metode Penyelesaian Lanjutan",
        "topik": ["Masalah Penugasan (Assignment)", "Teori Antrian (Queuing)", 
                 "Prediksi & Simulasi", "Klasifikasi & Clustering", "Supply Chain Management"],
        "cpmk": "CPMK-2, CPMK-3, CPMK-4",
        "contoh": {
            "nama": "Penugasan Programmer (Assignment Problem)",
            "deskripsi": "Menugaskan 3 programmer ke 3 proyek dengan biaya minimum",
            "tipe": "assignment",
            "biaya": [[12, 15, 18], [10, 13, 16], [11, 14, 17]],
            "programmer": ["Ani", "Budi", "Cinta"],
            "proyek": ["Proyek X", "Proyek Y", "Proyek Z"]
        },
        "tugas": "Tugas: Identifikasi masalah Assignment di sekitar Anda",
        "pdf_link": "pdf/materi_bab_5.pdf"
    },
    "Bab 6": {
        "judul": "Masalah Transportasi & Penugasan",
        "topik": ["Masalah Transportasi", "Northwest Corner Rule", "Least Cost Method", 
                 "Vogel's Approximation", "Stepping Stone", "Metode Hungarian"],
        "cpmk": "CPMK-4",
        "contoh": {
            "nama": "Distributor 'Maju Bersama'",
            "deskripsi": "Minimasi biaya transportasi dari 2 pabrik ke 3 kota",
            "tipe": "transportasi",
            "biaya": [[800, 600, 700], [500, 800, 900]],
            "supply": [300, 400],
            "demand": [200, 300, 200],
            "sumber": ["Pabrik A", "Pabrik B"],
            "tujuan": ["Kota 1", "Kota 2", "Kota 3"]
        },
        "tugas": "Tugas 3: Studi Kasus Distribusi Logistik 'Nusantara Furniture'",
        "pdf_link": "pdf/materi_bab_6.pdf"
    },
    "Bab 7": {
        "judul": "Teori Antrian (Queuing Theory)",
        "topik": ["Struktur Sistem Antrian", "Notasi Kendall (M/M/1, M/M/s)", 
                 "Metrik Kinerja Antrian", "Analisis Biaya Total", "Model M/M/s"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": {
            "nama": "Pusat Layanan Gudang",
            "deskripsi": "Analisis antrian truk di gudang dengan 1 mesin pengangkut",
            "tipe": "antrian",
            "lambda": 4,  # kedatangan per jam
            "mu": 6,      # pelayanan per jam
            "s": 1        # jumlah server
        },
        "tugas": "Tugas 4: Analisis & Perbaikan Sistem Antrian Kasus Nyata",
        "pdf_link": "pdf/materi_bab_7.pdf"
    },
    "Bab 8": {
        "judul": "Simulasi & Prediksi",
        "topik": ["Kapan Menggunakan Simulasi", "Simulasi Monte Carlo", 
                 "Simulasi Sistem Diskrit", "Verifikasi & Validasi", "Prediksi dengan Python"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": {
            "nama": "Prediksi Penjualan Monte Carlo",
            "deskripsi": "Simulasi permintaan sepeda motor menggunakan bilangan acak",
            "tipe": "simulasi",
            "data": [0, 1, 2, 3, 4, 5],
            "frekuensi": [6, 8, 8, 9, 11, 10]
        },
        "tugas": "Tugas 5: Proyek Simulasi - Kedai Kopi",
        "pdf_link": "pdf/materi_bab_8.pdf"
    },
    "Bab 9": {
        "judul": "Aplikasi Lanjutan: Classification & Clustering",
        "topik": ["Data Mining & OR", "Klasifikasi (Decision Tree)", 
                 "Clustering (K-Means)", "Prediksi Churn", "Segmentasi Pelanggan"],
        "cpmk": "CPMK-3, CPMK-4",
        "contoh": {
            "nama": "Segmentasi Pelanggan Mall (K-Means)",
            "deskripsi": "Mengelompokkan pelanggan berdasarkan pendapatan dan skor pengeluaran",
            "tipe": "clustering",
            "k": 3,
            "data_preview": "20 data pelanggan dengan fitur: Usia, Jenis Kelamin, Pendapatan, Pengeluaran"
        },
        "tugas": "Tugas 6: Proyek Klasifikasi, Prediksi, Klaster & Churn",
        "pdf_link": "pdf/materi_bab_9.pdf"
    },
    "Bab 10": {
        "judul": "Supply Chain Management (SCM)",
        "topik": ["Manajemen Persediaan (EOQ)", "Reorder Point & Safety Stock", 
                 "Lokasi Fasilitas", "Optimasi Rute (VRP)", "Integrasi Python"],
        "cpmk": "CPMK-4",
        "contoh": {
            "nama": "Optimasi Rantai Pasok 'FreshCatch'",
            "deskripsi": "Memilih pusat pengolahan dan merencanakan distribusi ikan segar",
            "tipe": "scm",
            "biaya_transport": [[5000, 6000, 8000, 10000, 4000], [9000, 3000, 7000, 5000, 8000]],
            "biaya_operasional": [5000000, 7000000],
            "permintaan": [30, 25, 40, 20, 35]
        },
        "tugas": "Diskusi: Mata Rantai Pasok Beras",
        "pdf_link": "pdf/materi_bab_10.pdf"
    },
    "Bab 11": {
        "judul": "Topik Lanjutan & Integrasi Teknologi",
        "topik": ["Integer Programming", "Goal Programming", "Non-Linear Programming", 
                 "IoT & Blockchain dalam OR", "Prescriptive Analytics"],
        "cpmk": "CPMK-2, CPMK-4",
        "contoh": {
            "nama": "Lokasi Fasilitas (Integer Programming)",
            "deskripsi": "Memilih 2 gudang dari 5 lokasi kandidat untuk melayani 8 kota",
            "tipe": "integer",
            "fixed_costs": [5000, 7000, 6000, 8000, 5500],
            "kapasitas": [450, 500, 400, 600, 450],
            "demand": [80, 120, 150, 100, 90, 110, 130, 70]
        },
        "tugas": "Diskusi: Implementasi IoT dan Blockchain dalam SCM",
        "pdf_link": "pdf/materi_bab_11.pdf"
    },
    "Bab 12": {
        "judul": "Workshop: Studi Kasus Komprehensif",
        "topik": ["Fleet Assignment", "Crew Scheduling", "Network Revenue Management", 
                 "Flight Recovery", "Brainstorming OR di Kampus"],
        "cpmk": "CPMK-1, CPMK-2, CPMK-3, CPMK-4",
        "contoh": {
            "nama": "Penjadwalan Maskapai Penerbangan",
            "deskripsi": "Fleet Assignment dan Crew Scheduling untuk biaya minimum",
            "tipe": "comprehensive",
            "rute": ["CGK-SIN", "CGK-DPS", "CGK-KUL"],
            "pesawat": ["Boeing 737", "Airbus A320"],
            "biaya": [[5000, 5500], [4500, 4800], [5200, 5600]],
            "demand": [140, 160, 145],
            "kapasitas": [150, 180]
        },
        "tugas": "Tugas 7: Tugas Kelompok - Pilih Topik & Analisis dengan OR",
        "pdf_link": "pdf/materi_bab_12.pdf"
    },
    "Bab 13": {
        "judul": "Kesimpulan dan Prospek Operation Research",
        "topik": ["Ringkasan Teknik OR", "Tren Masa Depan OR", "Integrasi AI & Machine Learning", 
                 "OR dalam Era Digital", "Pengembangan Karir di OR"],
        "cpmk": "CPMK-1, CPMK-2, CPMK-3, CPMK-4",
        "contoh": None,
        "tugas": "Diskusi: Bagaimana OR akan berkembang di masa depan?",
        "pdf_link": "pdf/materi_bab_13.pdf"
    }
}

# ==================== FUNGSI SOLVER PYTHON ====================

def solve_lp_graphical(c, A_ub, b_ub, A_eq=None, b_eq=None, bounds=None, method='highs'):
    """Solver LP menggunakan scipy.optimize.linprog"""
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method=method)
    return result

def solve_assignment(cost_matrix):
    """Solver Assignment Problem menggunakan Hungarian Algorithm"""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost

def solve_facility_location(fixed_costs, transport_costs, demands, max_facilities):
    """Solver Facility Location Problem menggunakan PuLP"""
    n_locations = len(fixed_costs)
    n_cities = len(demands)
    
    # Inisialisasi model
    model = LpProblem("Facility_Location", LpMinimize)
    
    # Variabel biner: 1 jika gudang j dibuka
    y = LpVariable.dicts("Open", range(n_locations), cat='Binary')
    
    # Variabel kontinu: jumlah dikirim dari i ke j  
    x = LpVariable.dicts("Ship", (range(n_locations), range(n_cities)), 
                         lowBound=0, cat='Continuous')
    
    # Fungsi tujuan: Minimasi biaya tetap + transportasi
    model += lpSum([fixed_costs[j] * y[j] for j in range(n_locations)]) + \
             lpSum([transport_costs[i][j] * x[i][j] 
                    for i in range(n_locations) 
                    for j in range(n_cities)])
    
    # Kendala: setiap kota harus terpenuhi permintaannya
    for j in range(n_cities):
        model += lpSum([x[i][j] for i in range(n_locations)]) == demands[j]
    
    # Kendala: hanya kirim dari gudang yang dibuka
    for i in range(n_locations):
        for j in range(n_cities):
            model += x[i][j] <= demands[j] * y[i]
    
    # Kendala: batas maksimal gudang yang dibuka
    model += lpSum([y[i] for i in range(n_locations)]) <= max_facilities
    
    # Solve
    status = model.solve()
    
    # Ekstrak hasil
    opened_facilities = [i for i in range(n_locations) if y[i].value() == 1]
    shipments = [[x[i][j].value() for j in range(n_cities)] for i in range(n_locations)]
    total_cost = value(model.objective)
    
    return {
        "status": LpStatus[status],
        "opened_facilities": opened_facilities,
        "shipments": shipments,
        "total_cost": total_cost
    }

def calculate_queue_metrics(lam, mu, s=1):
    """Kalkulator metrik antrian M/M/s"""
    rho = lam / (s * mu)
    
    if rho >= 1:
        return {"error": "Sistem tidak stabil (rho >= 1)"}
    
    if s == 1:
        L = lam / (mu - lam)
        Lq = (lam ** 2) / (mu * (mu - lam))
        W = 1 / (mu - lam)
        Wq = lam / (mu * (mu - lam))
        P0 = 1 - rho
    else:
        # M/M/s - menggunakan rumus yang lebih kompleks
        P0 = 1 / (sum([(lam/mu)**n / math.factorial(n) for n in range(s)]) + 
                   ((lam/mu)**s / math.factorial(s)) * (1 / (1 - rho)))
        Lq = (P0 * (lam/mu)**s * rho) / (math.factorial(s) * (1 - rho)**2)
        L = Lq + lam/mu
        Wq = Lq / lam
        W = Wq + 1/mu
    
    return {
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "P0": P0
    }

def monte_carlo_simulation(data, frekuensi, n_simulations=1000, n_weeks=15):
    """Simulasi Monte Carlo untuk prediksi permintaan"""
    total_freq = sum(frekuensi)
    prob = [f/total_freq for f in frekuensi]
    cum_prob = np.cumsum(prob)
    
    results = []
    for _ in range(n_simulations):
        weekly_demand = []
        for _ in range(n_weeks):
            rand = np.random.random()
            for i, cp in enumerate(cum_prob):
                if rand <= cp:
                    weekly_demand.append(data[i])
                    break
        results.append(sum(weekly_demand))
    
    return {
        "mean": np.mean(results),
        "median": np.median(results),
        "std": np.std(results),
        "min": np.min(results),
        "max": np.max(results),
        "all_results": results
    }

# ==================== KOMPONEN UI ====================

def render_header():
    """Render header aplikasi"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1 class="main-header">📊 OR-Agent</h1>', unsafe_allow_html=True)
        st.markdown("""
        <h3 style="text-align: center; color: #666;">
            Mata Kuliah Operation Research bidang Ekonomi Manajemen
        </h3>
        <h6 style="text-align: center; color: #666;">
            Penyusun: Ir.M Nasri AW, M.Eng.Sc, M.Kom / Dosen STIEIMA
        </h6>
        <p style="text-align: center; color: #888;">
            Dashboard Interaktif dengan Solver Python | Buku Ajar OR
        </p>
        """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigasi"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
        st.title("Navigasi")
        
        menu = st.radio(
            "Pilih Menu:",
            ["🏠 Beranda", "📚 Materi Per Bab", "🧮 Solver Python", "📋 Daftar Tugas", "📖 Tentang Aplikasi"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Progress Belajar")
        
        # Progress tracker
        bab_completed = st.session_state.get('bab_completed', [])
        progress = len(bab_completed) / 13 * 100
        st.progress(progress / 100)
        st.caption(f"Progress: {progress:.0f}% ({len(bab_completed)}/13 Bab)")
        
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        st.info("""
        **Tools yang digunakan:**
        - Python 3.x
        - Streamlit
        - SciPy Optimize
        - NumPy & Pandas
        - Plotly
        """)
        
        return menu

def render_beranda():
    """Render halaman beranda"""
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Selamat Datang di OR-Agent!
        
        **OR-Agent** adalah dashboard interaktif untuk mempelajari dan menyelesaikan 
        masalah **Operation Research** dalam bidang **Ekonomi dan Manajemen**.
        
        #### ✨ Fitur Utama:
        """)
        
        features = [
            ("📚", "Materi Lengkap 13 Bab", "Akses materi dari Pendahuluan hingga Studi Kasus Komprehensif"),
            ("🧮", "Solver Python Built-in", "Selesaikan contoh & tugas langsung dengan kode Python"),
            ("📊", "Visualisasi Interaktif", "Grafik dan chart untuk memahami konsep OR"),
            ("📋", "Tracker Tugas", "Pantau progress penyelesaian tugas per bab"),
            ("🎓", "CPMK Mapping", "Pemetaan Capaian Pembelajaran Mata Kuliah")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Statistik dashboard
        st.markdown("### 📈 Statistik Dashboard")
        
        stats = {
            "Total Bab": 13,
            "Contoh Soal": 10,
            "Tugas": 7,
            "Teknik OR": 8
        }
        
        for label, value in stats.items():
            st.metric(label, value)
        
        # CPMK Coverage
        st.markdown("### 🎯 CPMK Coverage")
        cpmk_data = {
            'CPMK': ['CPMK-1\n(Sikap)', 'CPMK-2\n(Pengetahuan)', 'CPMK-3\n(Keterampilan Umum)', 'CPMK-4\n(Keterampilan Khusus)'],
            'Bab': [5, 12, 6, 11]
        }
        fig = px.bar(cpmk_data, x='CPMK', y='Bab', color='Bab',
                    title='Distribusi Pemetaan CPMK per Bab',
                    color_continuous_scale='Viridis')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

def render_materi_bab():
    """Render halaman materi per bab"""
    st.markdown("---")
    st.header("📚 Materi Per Bab")
    
    # Pilih bab
    bab_selected = st.selectbox(
        "Pilih Bab:",
        list(BAB_DATA.keys()),
        format_func=lambda x: f"{x}: {BAB_DATA[x]['judul']}"
    )
    
    bab = BAB_DATA[bab_selected]
    
    # Layout 2 kolom
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="bab-card">
            <h2>{bab_selected}: {bab['judul']}</h2>
            <p><strong>CPMK:</strong> {bab['cpmk']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Link PDF
        if 'pdf_link' in bab and bab['pdf_link']:
            try:
                with open(bab['pdf_link'], "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="📄 Download PDF Materi",
                    data=pdf_data,
                    file_name=f"materi_bab_{bab_selected.split()[-1]}.pdf",
                    mime="application/pdf"
                )
                
                # Tampilkan PDF di halaman
                with st.expander("📖 Lihat PDF Materi", expanded=False):
                    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                    st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)
                    
            except FileNotFoundError:
                st.warning("File PDF belum tersedia")
        
        # Topik pembahasan
        st.subheader("📋 Topik Pembahasan")
        for i, topik in enumerate(bab['topik'], 1):
            st.markdown(f"{i}. {topik}")
        
        # Contoh soal
        if bab['contoh']:
            st.markdown("---")
            st.subheader("💡 Contoh Kasus")
            contoh = bab['contoh']
            
            with st.expander(f"📖 {contoh['nama']}", expanded=True):
                st.write(f"**Deskripsi:** {contoh['deskripsi']}")
                st.write(f"**Tipe:** {contoh['tipe'].upper()}")
                
                if contoh['tipe'] in ['maximasi', 'minimasi']:
                    st.write("**Variabel Keputusan:**")
                    for var, desc in contoh['variabel'].items():
                        st.write(f"- {var}: {desc}")
                    
                    st.write("**Fungsi Tujuan:**")
                    tujuan = contoh['fungsi_tujuan']
                    tipe_str = "Maksimalkan" if tujuan['tipe'] == 'max' else "Minimalkan"
                    eq = " + ".join([f"{tujuan[v]}{v}" for v in contoh['variabel'].keys()])
                    st.latex(f"{tipe_str} \\ Z = {eq}")
                    
                    st.write("**Kendala:**")
                    for k in contoh['kendala']:
                        eq = f"{k['x1']}x_1 + {k['x2']}x_2 {k['tipe']} {k['rhs']}"
                        st.latex(eq + f" \\quad ({k['nama']})")
    
    with col2:
        # Panel info
        st.info(f"""
        **💡 Tips Belajar:**
        1. Baca materi teori terlebih dahulu
        2. Pelajari contoh soal
        3. Coba selesaikan dengan Solver Python
        4. Kerjakan tugas mandiri
        """)
        
        # Tugas
        st.warning(f"""
        **📝 Tugas:**
        {bab['tugas']}
        """)
        
        # Tombol tandai selesai
        if st.button("✅ Tandai Bab Selesai"):
            if 'bab_completed' not in st.session_state:
                st.session_state.bab_completed = []
            if bab_selected not in st.session_state.bab_completed:
                st.session_state.bab_completed.append(bab_selected)
                st.success(f"{bab_selected} ditandai selesai!")
            else:
                st.info("Bab sudah ditandai selesai sebelumnya")

def render_solver():
    """Render halaman solver Python"""
    st.markdown("---")
    st.header("🧮 Solver Python")
    
    solver_type = st.selectbox(
        "Pilih Jenis Solver:",
        ["Linear Programming (Grafis/Simpleks)", "Masalah Transportasi", 
         "Masalah Penugasan (Assignment)", "Teori Antrian (Queuing)",
         "Simulasi Monte Carlo", "Integer Programming"]
    )
    
    if solver_type == "Linear Programming (Grafis/Simpleks)":
        render_solver_lp()
    elif solver_type == "Masalah Transportasi":
        render_solver_transportasi()
    elif solver_type == "Masalah Penugasan (Assignment)":
        render_solver_assignment()
    elif solver_type == "Teori Antrian (Queuing)":
        render_solver_antrian()
    elif solver_type == "Simulasi Monte Carlo":
        render_solver_simulasi()
    elif solver_type == "Integer Programming":
        render_solver_ip()

def render_solver_lp():
    """Solver untuk Linear Programming"""

    st.subheader("🎯 Linear Programming Solver")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Parameter")
    
        # Input jumlah variabel dan kendala
        st.markdown("**Konfigurasi Masalah:**")
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            num_vars = st.number_input("Jumlah Variabel (x):", min_value=2, max_value=10, value=2)
        with col_config2:
            num_constraints = st.number_input("Jumlah Kendala:", min_value=1, max_value=10, value=2)
    
        tipe = st.radio("Tipe:", ["Maximasi", "Minimasi"])
    
        # Input Fungsi Tujuan
        st.markdown("**Fungsi Tujuan (Koefisien):**")
        c = []
        cols_obj = st.columns(int(num_vars))
        for i, col in enumerate(cols_obj):
            with col:
                val = st.number_input(f"c_{i+1} (x_{i+1})", value=800.0 if i == 0 else 600.0, key=f'c_{i}')
                c.append(val)
    
        # Input Kendala
        st.markdown("**Matriks Kendala (A):**")
        A_ub = []
        b_ub = []
        for i in range(int(num_constraints)):
            st.markdown(f"**Kendala {i+1}:**")
            row = []
            cols_constraint = st.columns(int(num_vars) + 1)
        
            for j in range(int(num_vars)):
                with cols_constraint[j]:
                    default_val = 2.0 if (i == 0 and j == 0) else (3.0 if (i == 0 and j == 1) else (5.0 if (i == 1 and j == 0) else 3.0))
                    val = st.number_input(f"a_{i+1}{j+1} (x_{j+1})", value=default_val, key=f'a_{i}_{j}')
                    row.append(val)
        
            with cols_constraint[int(num_vars)]:
                default_rhs = 60.0 if i == 0 else 100.0
                rhs = st.number_input(f"b_{i+1} (RHS)", value=default_rhs, key=f'b_{i}')
                b_ub.append(rhs)
        
            A_ub.append(row)

    with col2:
        st.markdown("### Hasil Perhitungan")
    
        if st.button("🔢 Hitung Solusi Optimal"):
            # Setup LP
            c_obj = [-val for val in c] if tipe == "Maximasi" else c
            bounds = [(0, None)] * int(num_vars)
        
            try:
                result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
                if result.success:
                    x_opt = result.x
                    z_opt = sum(c[i] * x_opt[i] for i in range(int(num_vars)))
                
                    # Tampilkan solusi
                    st.success("✅ Solusi Optimal Ditemukan!")
                
                    results_data = []
                    for i, val in enumerate(x_opt):
                        results_data.append({"Variabel": f"x_{i+1}", "Nilai": f"{val:.4f}"})
                
                    df_result = pd.DataFrame(results_data)
                    st.dataframe(df_result)
                
                    st.metric(f"Z {'Maksimum' if tipe == 'Maximasi' else 'Minimum'}", f"Rp {z_opt:,.2f}")
                
                    # Visualisasi untuk 2D (jika ada 2 variabel)
                    if int(num_vars) == 2:
                        fig = go.Figure()
                    
                        # Garis kendala
                        x_range = np.linspace(0, 100, 100)
                    
                        for i, (a_row, b_val) in enumerate(zip(A_ub, b_ub)):
                            if len(a_row) >= 2 and a_row[1] != 0:
                                y_vals = (b_val - a_row[0] * x_range) / a_row[1]
                                fig.add_trace(go.Scatter(x=x_range, y=np.maximum(y_vals, 0), 
                                                   mode='lines', name=f'Kendala {i+1}'))
                    
                        # Titik optimal
                        fig.add_trace(go.Scatter(x=[x_opt[0]], y=[x_opt[1]], 
                                           mode='markers+text', 
                                           marker=dict(size=15, color='red'),
                                           text=[f"Optimal<br>({x_opt[0]:.2f}, {x_opt[1]:.2f})"],
                                           textposition="top center",
                                           name='Titik Optimal'))
                    
                        fig.update_layout(
                            title="Visualisasi Grafis LP (2D)",
                            xaxis_title="x₁",
                            yaxis_title="x₂",
                            showlegend=True,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                    # Kode Python
                    with st.expander("🐍 Lihat Kode Python"):
                        c_str = str(c_obj)
                        A_str = str(A_ub)
                        b_str = str(b_ub)
                        st.code(f"""
    from scipy.optimize import linprog

    # Fungsi tujuan
    c = {c_str}

    # Matriks kendala (<=)
    A_ub = {A_str}
    b_ub = {b_str}

    # Bounds (variabel >= 0)
    bounds = [(0, None) for _ in range({int(num_vars)})]

    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Hasil
    print(f"Solusi optimal: {{result.x}}")
    print(f"Nilai optimal: {{-result.fun if {tipe == 'Maximasi'} else result.fun:.2f}}")
                    """, language='python')
                else:
                    st.error("❌ Tidak ditemukan solusi optimal")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def render_solver_transportasi():
    """Solver untuk Masalah Transportasi"""
    st.subheader("🚚 Masalah Transportasi Solver")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Data")
        
        n_sumber = st.number_input("Jumlah Sumber (Pabrik):", min_value=2, max_value=5, value=2)
        n_tujuan = st.number_input("Jumlah Tujuan (Kota):", min_value=2, max_value=5, value=3)
        
        st.markdown("**Biaya Transportasi per Unit:**")
        cost_matrix = []
        for i in range(int(n_sumber)):
            row = []
            cols = st.columns(int(n_tujuan))
            for j, col in enumerate(cols):
                with col:
                    val = st.number_input(f"C[{i}][{j}]", value=500.0, key=f'c_{i}_{j}')
                    row.append(val)
            cost_matrix.append(row)
        
        st.markdown("**Supply (Persediaan):**")
        supply = []
        cols = st.columns(int(n_sumber))
        for i, col in enumerate(cols):
            with col:
                val = st.number_input(f"S[{i}]", value=300.0, key=f's_{i}')
                supply.append(val)
        
        st.markdown("**Demand (Permintaan):**")
        demand = []
        cols = st.columns(int(n_tujuan))
        for j, col in enumerate(cols):
            with col:
                val = st.number_input(f"D[{j}]", value=200.0, key=f'd_{j}')
                demand.append(val)
    
    with col2:
        st.markdown("### Hasil Optimasi")
        
        if st.button("🔢 Hitung Solusi Transportasi"):
            # Cek balance
            total_supply = sum(supply)
            total_demand = sum(demand)
            
            if abs(total_supply - total_demand) > 0.01:
                st.warning(f"⚠️ Tidak seimbang! Supply: {total_supply}, Demand: {total_demand}")
                if total_supply > total_demand:
                    st.info("Menambah dummy destination...")
                else:
                    st.info("Menambah dummy source...")
            
            # Solver menggunakan linprog (formulasi LP transportasi)
            # Flatten cost matrix
            c = np.array(cost_matrix).flatten()
            
            # Constraints: supply
            A_eq = []
            b_eq = []
            
            # Supply constraints
            for i in range(int(n_sumber)):
                row = [0] * (int(n_sumber) * int(n_tujuan))
                for j in range(int(n_tujuan)):
                    row[i * int(n_tujuan) + j] = 1
                A_eq.append(row)
                b_eq.append(supply[i])
            
            # Demand constraints
            for j in range(int(n_tujuan)):
                row = [0] * (int(n_sumber) * int(n_tujuan))
                for i in range(int(n_sumber)):
                    row[i * int(n_tujuan) + j] = 1
                A_eq.append(row)
                b_eq.append(demand[j])
            
            bounds = [(0, None)] * (int(n_sumber) * int(n_tujuan))
            
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                st.success("✅ Solusi Optimal Ditemukan!")
                
                # Tampilkan alokasi
                allocation = np.array(result.x).reshape(int(n_sumber), int(n_tujuan))
                
                # DataFrame untuk tampilan
                df_alloc = pd.DataFrame(
                    allocation,
                    index=[f"Pabrik {chr(65+i)}" for i in range(int(n_sumber))],
                    columns=[f"Kota {j+1}" for j in range(int(n_tujuan))]
                )
                
                st.write("**Alokasi Optimal:**")
                st.dataframe(df_alloc.style.format("{:.2f}"))
                
                total_cost = np.sum(allocation * np.array(cost_matrix))
                st.metric("Total Biaya Minimum", f"Rp {total_cost:,.2f}")
                
                # Heatmap
                fig = px.imshow(allocation, 
                              labels=dict(x="Tujuan", y="Sumber", color="Alokasi"),
                              x=[f"Kota {j+1}" for j in range(int(n_tujuan))],
                              y=[f"Pabrik {chr(65+i)}" for i in range(int(n_sumber))],
                              title="Heatmap Alokasi Transportasi")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Gagal menemukan solusi")

def render_solver_assignment():
    """Solver untuk Masalah Penugasan"""
    st.subheader("👥 Masalah Penugasan (Assignment) Solver")
    
    n = st.number_input("Jumlah Agen/Tugas:", min_value=2, max_value=10, value=3)
    
    st.markdown("**Matriks Biaya:**")
    cost_matrix = []
    for i in range(int(n)):
        row = []
        cols = st.columns(int(n))
        for j, col in enumerate(cols):
            with col:
                default_val = [[12, 15, 18], [10, 13, 16], [11, 14, 17]][i][j] if n == 3 and i < 3 and j < 3 else 10
                val = st.number_input(f"C[{i}][{j}]", value=float(default_val), key=f'assign_{i}_{j}')
                row.append(val)
        cost_matrix.append(row)
    
    if st.button("🔢 Hitung Penugasan Optimal"):
        cost_array = np.array(cost_matrix)
        row_ind, col_ind, total_cost = solve_assignment(cost_array)
        
        st.success("✅ Penugasan Optimal Ditemukan!")
        
        # Tampilkan hasil
        results = []
        for i, j in zip(row_ind, col_ind):
            results.append({
                "Agen": f"Programmer {chr(65+i)}",
                "Tugas": f"Proyek {chr(88+j)}",
                "Biaya": cost_array[i, j]
            })
        
        df_result = pd.DataFrame(results)
        st.dataframe(df_result)
        
        st.metric("Total Biaya Minimum", f"Rp {total_cost:,.0f}")
        
        # Visualisasi matriks dengan penugasan
        fig = go.Figure(data=go.Heatmap(
            z=cost_array,
            text=cost_array,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='RdYlGn_r'
        ))
        
        # Tambahkan marker untuk penugasan optimal
        for i, j in zip(row_ind, col_ind):
            fig.add_annotation(
                x=j, y=i,
                text="✓",
                showarrow=False,
                font=dict(size=30, color="white")
            )
        
        fig.update_layout(
            title="Matriks Biaya dengan Penugasan Optimal (✓)",
            xaxis_title="Tugas",
            yaxis_title="Agen"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_solver_antrian():
    """Solver untuk Teori Antrian"""
    st.subheader("⏱️ Teori Antrian (M/M/s) Solver")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lam = st.number_input("λ (Laju Kedatangan per jam):", value=4.0, min_value=0.1)
        mu = st.number_input("μ (Laju Pelayanan per jam):", value=6.0, min_value=0.1)
        s = st.number_input("s (Jumlah Server):", value=1, min_value=1, max_value=10)
        
        cs = st.number_input("Biaya Pelayanan per Server per jam (Cs):", value=150000.0)
        cw = st.number_input("Biaya Tunggu per Pelanggan per jam (Cw):", value=100000.0)
    
    with col2:
        if st.button("🔢 Hitung Metrik Antrian"):
            metrics = calculate_queue_metrics(lam, mu, int(s))
            
            if "error" in metrics:
                st.error(f"❌ {metrics['error']}")
            else:
                st.markdown(f"""
                <div class="result-box">
                    <h4>📊 Metrik Kinerja Sistem</h4>
                    <p><strong>ρ (Utilisasi):</strong> {metrics['rho']:.2%}</p>
                    <p><strong>L (Jumlah dalam sistem):</strong> {metrics['L']:.2f} pelanggan</p>
                    <p><strong>Lq (Jumlah dalam antrian):</strong> {metrics['Lq']:.2f} pelanggan</p>
                    <p><strong>W (Waktu dalam sistem):</strong> {metrics['W']:.2f} jam ({metrics['W']*60:.1f} menit)</p>
                    <p><strong>Wq (Waktu mengantri):</strong> {metrics['Wq']:.2f} jam ({metrics['Wq']*60:.1f} menit)</p>
                    <p><strong>P₀ (Sistem kosong):</strong> {metrics['P0']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Analisis biaya
                total_cost = s * cs + metrics['L'] * cw
                st.metric("Total Biaya per Jam", f"Rp {total_cost:,.0f}")
                
                # Grafik perbandingan server
                if s == 1:
                    st.info("💡 Coba tambah server untuk melihat perbandingan biaya!")
                    
                    # Simulasi untuk 1-3 server
                    server_range = range(1, 4)
                    costs = []
                    wqs = []
                    
                    for s_test in server_range:
                        if lam < s_test * mu:
                            m = calculate_queue_metrics(lam, mu, s_test)
                            costs.append(s_test * cs + m['L'] * cw)
                            wqs.append(m['Wq'] * 60)
                        else:
                            costs.append(None)
                            wqs.append(None)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(server_range), y=costs, 
                                           mode='lines+markers', name='Total Biaya'))
                    fig.update_layout(title="Analisis Biaya vs Jumlah Server",
                                    xaxis_title="Jumlah Server",
                                    yaxis_title="Total Biaya (Rp)")
                    st.plotly_chart(fig, use_container_width=True)

def render_solver_simulasi():
    """Solver untuk Simulasi Monte Carlo"""
    st.subheader("🎲 Simulasi Monte Carlo")
    
    st.markdown("**Data Permintaan Historis:**")
    
    # Default data dari buku
    default_data = [0, 1, 2, 3, 4, 5]
    default_freq = [6, 8, 8, 9, 11, 10]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Permintaan (ribuan unit):**")
        data_input = st.text_input("Data (pisahkan koma):", value=",".join(map(str, default_data)))
        data = [float(x.strip()) for x in data_input.split(",")]
    
    with col2:
        st.markdown("**Frekuensi (minggu):**")
        freq_input = st.text_input("Frekuensi (pisahkan koma):", value=",".join(map(str, default_freq)))
        frekuensi = [int(x.strip()) for x in freq_input.split(",")]
    
    n_sim = st.number_input("Jumlah Simulasi:", value=1000, min_value=100)
    n_weeks = st.number_input("Jumlah Minggu Prediksi:", value=15, min_value=1)
    
    if st.button("🔢 Jalankan Simulasi"):
        if len(data) != len(frekuensi):
            st.error("❌ Jumlah data dan frekuensi harus sama!")
        else:
            result = monte_carlo_simulation(data, frekuensi, int(n_sim), int(n_weeks))
            
            st.success("✅ Simulasi Selesai!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rata-rata Total", f"{result['mean']:.0f} unit")
            col2.metric("Median", f"{result['median']:.0f} unit")
            col3.metric("Rata-rata/Minggu", f"{result['mean']/n_weeks:.1f} unit")
            
            # Histogram
            fig = px.histogram(x=result['all_results'], nbins=30,
                             labels={'x': 'Total Permintaan (unit)'},
                             title=f"Distribusi Hasil Simulasi ({n_sim} iterasi)")
            fig.add_vline(x=result['mean'], line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {result['mean']:.0f}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik
            st.markdown("### 📊 Statistik Detail")
            stats_df = pd.DataFrame({
                'Metrik': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Nilai': [result['mean'], result['median'], result['std'], 
                         result['min'], result['max']]
            })
            st.dataframe(stats_df)

def render_solver_ip():
    """Solver untuk Integer Programming"""
    st.subheader("🔢 Integer Programming (0-1) Solver")
    
    st.markdown("""
    **Contoh: Masalah Lokasi Fasilitas**
    
    Memilih gudang mana yang akan dibuka (ya/tidak) untuk meminimasi total biaya.
    """)
    
    n_locations = st.number_input("Jumlah Lokasi Kandidat:", value=3, min_value=2, max_value=5)
    n_cities = st.number_input("Jumlah Kota:", value=4, min_value=2, max_value=8)
    
    st.markdown("**Biaya Tetap Pembukaan Gudang:**")
    fixed_costs = []
    cols = st.columns(int(n_locations))
    for i, col in enumerate(cols):
        with col:
            val = st.number_input(f"F[{i+1}]", value=5000.0, key=f'fix_{i}')
            fixed_costs.append(val)
    
    st.markdown("**Biaya Transportasi per Unit:**")
    transport_costs = []
    for i in range(int(n_locations)):
        row = []
        cols = st.columns(int(n_cities))
        for j, col in enumerate(cols):
            with col:
                val = st.number_input(f"T[{i+1}][{j+1}]", value=100.0, key=f'trans_{i}_{j}')
                row.append(val)
        transport_costs.append(row)
    
    st.markdown("**Permintaan Kota:**")
    demands = []
    cols = st.columns(int(n_cities))
    for j, col in enumerate(cols):
        with col:
            val = st.number_input(f"D[{j+1}]", value=100.0, key=f'dem_{j}')
            demands.append(val)
    
    max_facilities = st.number_input("Maksimal Gudang Dibuka:", value=2, min_value=1)
    
    if st.button("🔢 Hitung Lokasi Optimal"):
        try:
            result = solve_facility_location(fixed_costs, transport_costs, demands, max_facilities)
            
            if result["status"] == "Optimal":
                st.success("✅ Solusi Optimal Ditemukan!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Biaya", f"Rp {result['total_cost']:,.0f}")
                
                with col2:
                    st.metric("Gudang Dibuka", f"{len(result['opened_facilities'])} dari {len(fixed_costs)}")
                
                st.markdown("### 📍 Gudang yang Dibuka:")
                opened_list = [f"Gudang {i+1}" for i in result['opened_facilities']]
                st.write(", ".join(opened_list))
                
                st.markdown("### 🚚 Matriks Pengiriman:")
                shipments_df = pd.DataFrame(
                    result['shipments'],
                    index=[f"Gudang {i+1}" for i in range(len(fixed_costs))],
                    columns=[f"Kota {j+1}" for j in range(len(demands))]
                )
                st.dataframe(shipments_df.style.format("{:.1f}"))
                
                # Visualisasi
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"G{i+1}" for i in range(len(fixed_costs))],
                    y=[sum(row) for row in result['shipments']],
                    name="Total Pengiriman",
                    marker_color=['red' if i in result['opened_facilities'] else 'lightgray' 
                                for i in range(len(fixed_costs))]
                ))
                fig.update_layout(
                    title="Total Pengiriman per Gudang",
                    xaxis_title="Gudang",
                    yaxis_title="Jumlah Unit"
                )
                st.plotly_chart(fig)
                
            else:
                st.error(f"❌ Tidak dapat menemukan solusi optimal. Status: {result['status']}")
                
        except Exception as e:
            st.error(f"❌ Error dalam penyelesaian: {str(e)}")
            st.info("Pastikan PuLP terinstall dengan: pip install pulp")

def render_daftar_tugas():
    """Render halaman daftar tugas"""
    st.markdown("---")
    st.header("📋 Daftar Tugas Operation Research")
    
    tugas_data = [
        ("Tugas 1", "Analisis Kasus Optimasi", "Bab 3", "Minimasi Pakan Ternak & Maksimasi Kedai Kopi", "LP Grafis & Analitis"),
        ("Tugas 2", "Studi Kasus LP", "Bab 4", "Perusahaan Elektronik 'GadgetPro'", "LPSolve/PuLP"),
        ("Tugas 3", "Distribusi Logistik", "Bab 6", "Nusantara Furniture", "Transportasi"),
        ("Tugas 4", "Analisis Antrian", "Bab 7", "Sistem Antrian Kasus Nyata", "Queuing Theory"),
        ("Tugas 5", "Proyek Simulasi", "Bab 8", "Kedai Kopi", "Monte Carlo"),
        ("Tugas 6", "Klasifikasi & Clustering", "Bab 9", "Prediksi Churn Pelanggan", "K-Means/Decision Tree"),
        ("Tugas 7", "Tugas Kelompok", "Bab 12", "Pilih Topik & Analisis Komprehensif", "Bebas")
    ]
    
    for i, (kode, judul, bab, kasus, teknik) in enumerate(tugas_data, 1):
        with st.expander(f"**{kode}**: {judul} ({bab})"):
            st.write(f"**Studi Kasus:** {kasus}")
            st.write(f"**Teknik OR:** {teknik}")
            
            # Status penyelesaian
            status_key = f"tugas_{i}_done"
            if status_key not in st.session_state:
                st.session_state[status_key] = False
            
            col1, col2 = st.columns([1, 3])
            with col1:
                done = st.checkbox("Selesai", key=status_key)
            with col2:
                if done:
                    st.success("✅ Tugas selesai!")
                else:
                    st.warning("⏳ Belum dikerjakan")

def render_tentang():
    """Render halaman tentang aplikasi"""
    st.markdown("---")
    st.header("📖 Tentang OR-Agent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 AppFlow OR-Agent
        
        **OR-Agent** dirancang sebagai asisten pembelajaran digital untuk mahasiswa 
        Operation Research bidang Ekonomi Manajemen.
        
        #### Alur Aplikasi (AppFlow):
        
        ```
        ┌─────────────────┐
        │   🏠 BERANDA    │ ← Overview & Progress Belajar
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  📚 MATERI BAB  │ ← Akses 12 Bab Lengkap
        │  • Teori        │
        │  • Contoh Soal  │
        │  • CPMK Mapping │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  🧮 SOLVER      │ ← Praktik Langsung
        │  • Input Data   │
        │  • Hitung       │
        │  • Visualisasi  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  📋 TUGAS       │ ← Tracker & Panduan
        │  • Daftar Tugas │
        │  • Progress     │
        └─────────────────┘
        ```
        
        #### Core Features:
        
        1. **📚 Materi Interaktif Per Bab**
           - Pemetaan CPMK (Capaian Pembelajaran Mata Kuliah)
           - Topik pembahasan terstruktur
           - Contoh kasus lengkap dengan rumus matematis
        
        2. **🧮 Python Solver Built-in**
           - Linear Programming (Grafis & Simpleks)
           - Masalah Transportasi
           - Masalah Penugasan (Hungarian Algorithm)
           - Teori Antrian (M/M/s)
           - Simulasi Monte Carlo
           - Integer Programming
        
        3. **📊 Visualisasi Real-time**
           - Grafik feasible region
           - Heatmap alokasi transportasi
           - Histogram simulasi
           - Metrik antrian
        
        4. **📋 Progress Tracker**
           - Tandai bab selesai
           - Track tugas
           - Progress bar visual
        
        5. **🐍 Kode Python Export**
           - Generate kode untuk setiap solver
           - Copy-paste ke editor lokal
           - Kompatibel dengan PuLP, SciPy
        """)
    
    with col2:
        st.info("""
        **👨‍💻 Developer Info**
        
        **Nama Aplikasi:** OR-Agent
        **Versi:** 1.0.0
        **Framework:** Streamlit
        **Bahasa:** Python 3.x
        **Penyusun:** Ir.M Nasri AW, M.Eng.Sc, M.Kom / Dosen STIEIMA
        
        **Library Utama:**
        - scipy.optimize
        - numpy & pandas
        - plotly
        - streamlit
        
        **Sumber Materi:**
        Buku Ajar Operation Research
        (Bidang Ekonomi Manajemen)
        """)
        
        st.success("""
        **🎓 CPMK yang Dicapai:**
        
        ✅ CPMK-1: Sikap Etis
        ✅ CPMK-2: Pengetahuan OR
        ✅ CPMK-3: Komunikasi
        ✅ CPMK-4: Keterampilan Teknis
        """)

# ==================== MAIN APP ====================

def main():
    render_header()
    menu = render_sidebar()
    
    if menu == "🏠 Beranda":
        render_beranda()
    elif menu == "📚 Materi Per Bab":
        render_materi_bab()
    elif menu == "🧮 Solver Python":
        render_solver()
    elif menu == "📋 Daftar Tugas":
        render_daftar_tugas()
    elif menu == "📖 Tentang Aplikasi":
        render_tentang()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>OR-Agent v1.0 | Operation Research Dashboard | 2025</p>
        <p>Mata Kuliah Operation Research bidang Ekonomi Manajemen</p>
        <p>@M Nasri AW, 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
