
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistem Fuzzy Mamdani Kemiskinan Babel",
    page_icon="📊",
    layout="wide"
)

CATEGORIES = ["Rendah", "Sedang", "Tinggi"]
VARS_MODEL = ["X1_TPT", "X2_RLS", "X3_PDRB_Per_Kapita"]
LABELS = {
    "X1_TPT": "TPT",
    "X2_RLS": "RLS",
    "X3_PDRB_Per_Kapita": "PDRB per Kapita",
    "Y_Kemiskinan": "Persentase Penduduk Miskin"
}

# ============================================================
# MEMBERSHIP FUNCTION — SAMA DENGAN NOTEBOOK
# ============================================================
def trapmf(x, a, b, c, d):
    x = float(x)
    y = 0.0
    if b == a:
        if x <= c:
            y = 1.0
    else:
        if a < x < b:
            y = (x-a)/(b-a)
        elif b <= x <= c:
            y = 1.0

    if c == d:
        if x >= b:
            y = 1.0
    else:
        if c < x < d:
            y = (d-x)/(d-c)

    return float(np.clip(y, 0, 1))

def trimf(x, a, b, c):
    x = float(x)
    y = 0.0
    if a < x < b:
        y = (x-a)/(b-a)
    elif x == b:
        y = 1.0
    elif b < x < c:
        y = (c-x)/(c-b)
    return float(np.clip(y, 0, 1))

# ============================================================
# RULE BASE 27 RULE — SAMA DENGAN NOTEBOOK
# ============================================================
rules = [
    {"Rule":"R1","X1_TPT":"Rendah","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Sedang"},
    {"Rule":"R2","X1_TPT":"Rendah","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Sedang"},
    {"Rule":"R3","X1_TPT":"Rendah","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},
    {"Rule":"R4","X1_TPT":"Rendah","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Sedang"},
    {"Rule":"R5","X1_TPT":"Rendah","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Rendah"},
    {"Rule":"R6","X1_TPT":"Rendah","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},
    {"Rule":"R7","X1_TPT":"Rendah","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Sedang"},
    {"Rule":"R8","X1_TPT":"Rendah","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Rendah"},
    {"Rule":"R9","X1_TPT":"Rendah","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},

    {"Rule":"R10","X1_TPT":"Sedang","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Tinggi"},
    {"Rule":"R11","X1_TPT":"Sedang","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Sedang"},
    {"Rule":"R12","X1_TPT":"Sedang","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},
    {"Rule":"R13","X1_TPT":"Sedang","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Sedang"},
    {"Rule":"R14","X1_TPT":"Sedang","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Sedang"},
    {"Rule":"R15","X1_TPT":"Sedang","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},
    {"Rule":"R16","X1_TPT":"Sedang","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Sedang"},
    {"Rule":"R17","X1_TPT":"Sedang","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Rendah"},
    {"Rule":"R18","X1_TPT":"Sedang","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},

    {"Rule":"R19","X1_TPT":"Tinggi","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Tinggi"},
    {"Rule":"R20","X1_TPT":"Tinggi","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Tinggi"},
    {"Rule":"R21","X1_TPT":"Tinggi","X2_RLS":"Rendah","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Sedang"},
    {"Rule":"R22","X1_TPT":"Tinggi","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Tinggi"},
    {"Rule":"R23","X1_TPT":"Tinggi","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Sedang"},
    {"Rule":"R24","X1_TPT":"Tinggi","X2_RLS":"Sedang","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Sedang"},
    {"Rule":"R25","X1_TPT":"Tinggi","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Rendah","Y_Kemiskinan":"Tinggi"},
    {"Rule":"R26","X1_TPT":"Tinggi","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Sedang","Y_Kemiskinan":"Sedang"},
    {"Rule":"R27","X1_TPT":"Tinggi","X2_RLS":"Tinggi","X3_PDRB_Per_Kapita":"Tinggi","Y_Kemiskinan":"Rendah"},
]

@st.cache_data
def load_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    if "Data Lengkap" not in xls.sheet_names:
        raise ValueError("Sheet 'Data Lengkap' tidak ditemukan.")

    df = pd.read_excel(uploaded_file, sheet_name="Data Lengkap", engine="openpyxl")

    def norm_col(x):
        x = str(x).strip().lower().replace("\n", " ")
        x = re.sub(r"[%()\[\]/\\-]", " ", x)
        x = re.sub(r"[^a-z0-9\s]", " ", x)
        return re.sub(r"\s+", " ", x).strip()

    normalized = {c: norm_col(c) for c in df.columns}

    def find_col(patterns):
        for original, n in normalized.items():
            for p in patterns:
                if p in n:
                    return original
        raise ValueError(f"Kolom tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    region_col = find_col(["kabupaten kota", "kabupaten", "kota", "wilayah", "provinsi"])
    year_col = find_col(["tahun", "year"])
    tpt_col = find_col(["tingkat pengangguran terbuka", "tpt"])
    rls_col = find_col(["rata rata lama sekolah", "rata-rata lama sekolah", "lama sekolah", "rls"])
    pdrb_col = find_col(["pdrb per kapita", "pdrb perkapita", "pdrb per kap"])
    poverty_col = find_col(["persentase penduduk miskin", "penduduk miskin", "kemiskinan", "persentase kemiskinan"])

    data = df[[region_col, year_col, tpt_col, rls_col, pdrb_col, poverty_col]].copy()
    data.columns = ["Wilayah","Tahun","X1_TPT","X2_RLS","X3_PDRB_Per_Kapita","Y_Kemiskinan"]

    data["Wilayah"] = data["Wilayah"].astype(str).str.strip()
    data["Tahun"] = pd.to_numeric(
        data["Tahun"].astype(str).str.extract(r"(20\d{2})")[0], errors="coerce"
    )

    def to_numeric_clean(series):
        s = series.astype(str).str.strip()
        s = s.str.replace("%", "", regex=False)
        s = s.str.replace(r"[^0-9,.-]", "", regex=True)
        s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    for c in ["X1_TPT","X2_RLS","X3_PDRB_Per_Kapita","Y_Kemiskinan"]:
        data[c] = to_numeric_clean(data[c])

    def clean_region_key(x):
        x = str(x).upper().strip()
        for s in ["KABUPATEN ","KAB. ","KAB.","KOTA "]:
            x = x.replace(s, "")
        return x.replace(" ","").replace("-","").replace("_","")

    data["REGION_KEY"] = data["Wilayah"].apply(clean_region_key)

    target_regions = {
        "BANGKA","BELITUNG","BANGKABARAT","BANGKATENGAH",
        "BANGKASELATAN","BELITUNGTIMUR","PANGKALPINANG"
    }
    data = data[data["REGION_KEY"].isin(target_regions)].copy()

    data = data.dropna(subset=[
        "Wilayah","Tahun","X1_TPT","X2_RLS",
        "X3_PDRB_Per_Kapita","Y_Kemiskinan"
    ]).drop_duplicates(["Wilayah","Tahun"], keep="first")

    data["Tahun"] = data["Tahun"].astype(int)
    return data.sort_values(["Wilayah","Tahun"]).reset_index(drop=True)

def build_membership_params(data):
    params = {}
    for col in VARS_MODEL + ["Y_Kemiskinan"]:
        mn = float(data[col].min())
        q1 = float(data[col].quantile(.25))
        med = float(data[col].quantile(.50))
        q3 = float(data[col].quantile(.75))
        mx = float(data[col].max())
        params[col] = {
            "Rendah": ("trapmf",[mn,mn,q1,med]),
            "Sedang": ("trimf",[q1,med,q3]),
            "Tinggi": ("trapmf",[med,q3,mx,mx])
        }
    return params

def membership_value(x, col, cat, params):
    kind, p = params[col][cat]
    return trapmf(x,*p) if kind=="trapmf" else trimf(x,*p)

def predict_fuzzy(tpt, rls, pdrb, params, data):
    vals = {"X1_TPT":float(tpt),"X2_RLS":float(rls),"X3_PDRB_Per_Kapita":float(pdrb)}
    fuzzy = {}
    for col in VARS_MODEL:
        for cat in CATEGORIES:
            fuzzy[(col,cat)] = membership_value(vals[col],col,cat,params)

    zmin = float(data["Y_Kemiskinan"].min())
    zmax = float(data["Y_Kemiskinan"].max())
    z = np.linspace(zmin,zmax,2001)

    output_mfs = {
        cat: np.array([membership_value(v,"Y_Kemiskinan",cat,params) for v in z])
        for cat in CATEGORIES
    }

    aggregated = np.zeros_like(z)
    active = []

    for rule in rules:
        alpha = min(
            fuzzy[("X1_TPT",rule["X1_TPT"])],
            fuzzy[("X2_RLS",rule["X2_RLS"])],
            fuzzy[("X3_PDRB_Per_Kapita",rule["X3_PDRB_Per_Kapita"])]
        )
        if alpha > 0:
            clipped = np.minimum(alpha,output_mfs[rule["Y_Kemiskinan"]])
            aggregated = np.maximum(aggregated,clipped)
            active.append({
                "Rule":rule["Rule"],
                "TPT":rule["X1_TPT"],
                "RLS":rule["X2_RLS"],
                "PDRB":rule["X3_PDRB_Per_Kapita"],
                "α-predikat":alpha,
                "Output":rule["Y_Kemiskinan"]
            })

    denominator = aggregated.sum()
    pred = (zmin+zmax)/2 if denominator == 0 else float((z*aggregated).sum()/denominator)

    output_mu = {
        cat: membership_value(pred,"Y_Kemiskinan",cat,params)
        for cat in CATEGORIES
    }
    category = max(output_mu,key=output_mu.get)

    return pred,fuzzy,pd.DataFrame(active),category,output_mu

# ============================================================
# UI
# ============================================================
st.title("📊 Sistem Prediksi Persentase Penduduk Miskin")
st.subheader("Provinsi Kepulauan Bangka Belitung — Metode Fuzzy Mamdani")
st.caption("TPT, RLS, dan PDRB per Kapita → Fuzzy Mamdani → Estimasi Persentase Penduduk Miskin")

uploaded = st.file_uploader(
    "Unggah file Excel penelitian (.xlsm / .xlsx)",
    type=["xlsm","xlsx"]
)

if uploaded is None:
    st.info("⬆️ Unggah file Excel yang sama dengan yang digunakan pada notebook. Sheet yang dibaca: **Data Lengkap**.")
    st.stop()

try:
    data = load_data(uploaded)
    params = build_membership_params(data)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

if len(data) != 63:
    st.warning(f"Data setelah seleksi berjumlah {len(data)} observasi. Notebook targetnya 63 observasi.")
else:
    st.success("Data berhasil dibaca: 63 observasi, 7 kabupaten/kota, periode 2017–2025.")

tab1, tab2, tab3 = st.tabs(["🔵 Data Historis","🟢 Data Baru","📋 27 Rule"])

with tab1:
    st.markdown("### Prediksi berdasarkan data historis")
    col1,col2 = st.columns(2)
    wilayah = col1.selectbox("Wilayah", sorted(data["Wilayah"].unique()))
    tahun = col2.selectbox("Tahun", sorted(data["Tahun"].unique()))

    if st.button("🔮 Hitung Prediksi Historis", type="primary"):
        row = data[(data["Wilayah"]==wilayah)&(data["Tahun"]==tahun)].iloc[0]
        pred,fuzzy,active,category,output_mu = predict_fuzzy(
            row["X1_TPT"],row["X2_RLS"],row["X3_PDRB_Per_Kapita"],params,data
        )

        a,b,c,d = st.columns(4)
        a.metric("TPT",f"{row['X1_TPT']:.4f}")
        b.metric("RLS",f"{row['X2_RLS']:.4f}")
        c.metric("PDRB/Kapita",f"{row['X3_PDRB_Per_Kapita']:.4f}")
        d.metric("Aktual",f"{row['Y_Kemiskinan']:.4f}%")

        st.success(f"### Estimasi Kemiskinan: **{pred:.4f}%**")
        st.write(f"Kategori output: **{category}**")
        st.write(f"Error: **{pred-row['Y_Kemiskinan']:.4f} poin persentase**")

        st.markdown("#### Fuzzifikasi")
        fuzz_rows = []
        for col in VARS_MODEL:
            fuzz_rows.append({
                "Variabel":LABELS[col],
                "Nilai":row[col],
                "Rendah":fuzzy[(col,"Rendah")],
                "Sedang":fuzzy[(col,"Sedang")],
                "Tinggi":fuzzy[(col,"Tinggi")]
            })
        st.dataframe(pd.DataFrame(fuzz_rows).style.format({
            "Nilai":"{:.4f}","Rendah":"{:.4f}","Sedang":"{:.4f}","Tinggi":"{:.4f}"
        }),use_container_width=True)

        st.markdown("#### α-Predikat / Rule Aktif")
        if len(active):
            st.dataframe(active.sort_values("α-predikat",ascending=False).style.format({"α-predikat":"{:.4f}"}),use_container_width=True)
        else:
            st.warning("Tidak ada rule aktif.")

        st.markdown("#### Derajat Keanggotaan Output")
        st.json({k:round(v,4) for k,v in output_mu.items()})

with tab2:
    st.markdown("### Prediksi data baru")
    st.caption("Y aktual tidak diperlukan. Masukkan TPT, RLS, dan PDRB; rule 27 tetap.")

    c1,c2 = st.columns(2)
    wilayah_baru = c1.text_input("Wilayah",placeholder="Contoh: Bangka Tengah")
    tahun_baru = c2.number_input("Tahun",min_value=1900,max_value=2100,value=int(data["Tahun"].max())+1)

    c1,c2,c3 = st.columns(3)
    tpt = c1.number_input("TPT",value=float(data["X1_TPT"].median()),format="%.4f")
    rls = c2.number_input("RLS",value=float(data["X2_RLS"].median()),format="%.4f")
    pdrb = c3.number_input("PDRB per Kapita",value=float(data["X3_PDRB_Per_Kapita"].median()),format="%.4f")

    if st.button("🚀 Prediksi Data Baru",type="primary"):
        pred,fuzzy,active,category,output_mu = predict_fuzzy(tpt,rls,pdrb,params,data)
        st.success(f"### Estimasi Kemiskinan: **{pred:.4f}%**")
        st.write(f"Wilayah: **{wilayah_baru or 'Tidak diisi'}** | Tahun: **{int(tahun_baru)}**")
        st.write(f"Kategori output: **{category}**")

        warnings_out=[]
        for col,val in {"X1_TPT":tpt,"X2_RLS":rls,"X3_PDRB_Per_Kapita":pdrb}.items():
            mn=float(data[col].min()); mx=float(data[col].max())
            if val<mn or val>mx:
                warnings_out.append(f"{LABELS[col]} = {val:.4f} di luar rentang model [{mn:.4f}, {mx:.4f}]")
        for msg in warnings_out:
            st.warning(msg)

        fuzz_rows=[]
        for col,val in {"X1_TPT":tpt,"X2_RLS":rls,"X3_PDRB_Per_Kapita":pdrb}.items():
            fuzz_rows.append({
                "Variabel":LABELS[col],"Nilai":val,
                "Rendah":fuzzy[(col,"Rendah")],
                "Sedang":fuzzy[(col,"Sedang")],
                "Tinggi":fuzzy[(col,"Tinggi")]
            })
        st.markdown("#### Fuzzifikasi")
        st.dataframe(pd.DataFrame(fuzz_rows).style.format({
            "Nilai":"{:.4f}","Rendah":"{:.4f}","Sedang":"{:.4f}","Tinggi":"{:.4f}"
        }),use_container_width=True)

        st.markdown("#### α-Predikat / Rule Aktif")
        st.dataframe(active.sort_values("α-predikat",ascending=False).style.format({"α-predikat":"{:.4f}"}),use_container_width=True)

        st.markdown("#### Derajat Keanggotaan Output")
        st.json({k:round(v,4) for k,v in output_mu.items()})

with tab3:
    st.markdown("### Basis Aturan Fuzzy Mamdani — 27 Rule")
    st.dataframe(pd.DataFrame(rules),use_container_width=True)
    st.info("Konsekuen rule ditetapkan eksplisit berdasarkan logika substantif; Y aktual tidak digunakan untuk membentuk rule.")
