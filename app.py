import streamlit as st
import pandas as pd
import re
import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="ABSA Review - Aplikasi",
    layout="wide"
)

# ======================================================
# DICTIONARY & KEYWORD ASPEK (sesuai notebook)
# ======================================================
CUSTOM_DICT = {
    "tbtb": "tiba-tiba",
    "cs": "customer service",
    "brg": "barang",
    "unt": "untuk",
    "gbsa": "tidak bisa",
    "enggak": "tidak",
    "app": "aplikasi",
    "apk": "aplikasi",
    "spt": "seperti",
    "tlng": "tolong",
    "kk": "kakak",
    "kks": "kakak",
    "skli": "sekali",
    "sempet": "sempat",
    "min": "admin"
}

EMOJI_DICT = {
    "😍": "senang",
    "😊": "senang",
    "😡": "marah",
    "😭": "sedih",
    "👍": "bagus"
}

ASPEK_KEYWORD = {
    "Pengiriman": ["kirim", "pengiriman", "paket", "ongkir",
                   "antar", "driver", "kurir", "pickup", "estimasi", "lacak", "resi",
                   "packing", "bungkus", "alamat",
                   "diundur", "jam", "dikirim", "waktu"],
    "Aplikasi": ["aplikasi", "fitur", "update", "bug", "akun", "login", "install",
                 "loading", "terbaru", "versi", "kode otp", "uninstall",
                 "berat", "navigasi", "menu", "tampilan", "design",
                 "notifikasi", "ngelag", "daftar", "verifikasi", "instal"],
    "Harga": ["harga", "biaya", "cashback", "voucher", "promo", "mahal", "murah", "gratis",
              "terjangkau", "overprice", "stok", "katalog", "diskon"],
    "Layanan": ["layanan", "customer", "service", "respon", "bantuan",
                "komplain", "informasi", "call", "center", "proses",
                "admin", "chat", "bayar", "pembayaran", "rekening", "kasir",
                "transfer", "spaylatteer", "checkout",
                "struk", "poin", "transaksi", "dana", "gopay",
                "ovo", "linkaja", "pending", "uang"]
}

ASPEK_MAPPING = ["Pengiriman", "Aplikasi", "Harga", "Layanan"]
LABEL_MAP = {0: "Negatif", 1: "Positif"}

MODEL_PATH = "fadhyas/absa-indobert"
KAMUS_PATH = os.path.join("dataset", "kamuskatabaku.csv")
# ======================================================
# FUNGSI PREPROCESSING
# ======================================================
@st.cache_data(show_spinner=False)
def build_singkatan_patterns():
    """Membangun pattern regex untuk normalisasi singkatan dari file kamus (opsional) + custom dict."""
    singkatan_dict = {}
    if os.path.exists(KAMUS_PATH):
        try:
            import io
            kamus_df = pd.read_csv(KAMUS_PATH, sep=';')
            singkatan_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))
        except Exception as e:
            st.warning(f"Gagal membaca file kamus: {e}")

    singkatan_dict.update(CUSTOM_DICT)

    patterns = {
        re.compile(r'\b' + re.escape(str(k)) + r'\b'): str(v)
        for k, v in singkatan_dict.items()
    }
    return patterns


def case_folding(text):
    return text.lower()


def caracter_repetition(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def tanda_baca_fixing(text):
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text


def space_fixing(text):
    return re.sub(r'([,.!?])(?=[^\s])', r'\1 ', text)


def double_space_fixing(text):
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def convert_emoji(text):
    for emo, arti in EMOJI_DICT.items():
        text = text.replace(emo, f" {arti} ")
    return text


def normalisasi_singkatan(text, singkatan_patterns):
    for singkatan, lengkap in singkatan_patterns.items():
        text = singkatan.sub(lengkap, text)
    return text


def preprocess_indobert(text, singkatan_patterns):
    if not isinstance(text, str):
        return ""
    text = case_folding(text)
    text = caracter_repetition(text)
    text = tanda_baca_fixing(text)
    text = space_fixing(text)
    text = double_space_fixing(text)
    text = remove_url(text)
    text = convert_emoji(text)
    text = normalisasi_singkatan(text, singkatan_patterns)
    return text


# ======================================================
# EKSTRAKSI ASPEK
# ======================================================
def extract_aspek(text):
    text = str(text).lower()
    hasil = []
    for aspek in ASPEK_MAPPING:
        for kw in ASPEK_KEYWORD[aspek]:
            if kw in text:
                hasil.append({'aspek': aspek, 'kata_kunci': kw})
                break

    seen = set()
    unik = []
    for item in hasil:
        if item['aspek'] not in seen:
            seen.add(item['aspek'])
            unik.append(item)
    return unik


# ======================================================
# LOAD MODEL
# ======================================================
def get_checkpoint_path(model_path):
    if not os.path.isdir(model_path):
        return model_path
    subfolders = [
        os.path.join(model_path, d)
        for d in os.listdir(model_path)
        if d.startswith('checkpoint')
    ]
    if subfolders:
        return sorted(subfolders)[-1]
    return model_path


@st.cache_resource(show_spinner=False)
def load_model():
    checkpoint_path = get_checkpoint_path(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device


def predict_sentimen(review, kata_kunci, tokenizer, model, device):
    enc = tokenizer(
        review,        # sentence A
        kata_kunci,    # sentence B
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return LABEL_MAP[torch.argmax(logits, dim=1).item()]


def predict_absa(review, singkatan_patterns, tokenizer, model, device):
    clean = preprocess_indobert(review, singkatan_patterns)
    aspek_list = extract_aspek(clean)

    if not aspek_list:
        return clean, pd.DataFrame({'Aspek': ['Tidak ditemukan'], 'Kata Kunci': ['-'], 'Sentimen': ['-']})

    hasil = []
    for item in aspek_list:
        aspek_kategori = item['aspek']
        kata_kunci = item['kata_kunci']
        sentimen = predict_sentimen(clean, kata_kunci, tokenizer, model, device)
        hasil.append({'Review': clean, 'Aspek': aspek_kategori, 'Kata Kunci': kata_kunci, 'Sentimen': sentimen})

    return clean, pd.DataFrame(hasil)


# ======================================================
# UI STREAMLIT
# ======================================================
st.title(" Aspect-Based Sentiment Analysis (ABSA)")
st.caption("Analisis sentimen review berdasarkan aspek menggunakan model IndoBERT")

singkatan_patterns = build_singkatan_patterns()
 
model_ready = False
try:
    with st.spinner("Memuat model, mohon tunggu..."):
        tokenizer, model, device = load_model()
    model_ready = True
except Exception as e:
    st.error(f"Gagal memuat model dari '{MODEL_PATH}': {e}")

st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 18px;
    }
    .stTextArea label {
        font-size: 100px;
    }
    </style>
""", unsafe_allow_html=True)
review_input = st.text_area("Masukkan teks review", 
                            height=100, 
                            )

if st.button("Analisis", type="primary"):
    if not model_ready:
        st.error("Model belum berhasil dimuat. Cek kembali folder model")
    elif review_input.strip() == "":
        st.warning("Mohon masukkan teks review.")
    else:
        with st.spinner("Menganalisis..."):
            clean, hasil_df = predict_absa(
                review_input,
                singkatan_patterns,
                tokenizer,
                model,
                device
            )

        st.markdown(f"**Teks setelah preprocessing:** _{clean}_")

        for _, row in hasil_df.iterrows():
            if row["Sentimen"] == "Positif":
                st.success(f"**{row['Aspek']}** → {row['Sentimen']} (kata kunci: {row['Kata Kunci']})")
            elif row["Sentimen"] == "Negatif":
                st.error(f"**{row['Aspek']}** → {row['Sentimen']} (kata kunci: {row['Kata Kunci']})")
            else:
                st.info(f"{row['Aspek']}: {row['Sentimen']}")

        st.dataframe(hasil_df, use_container_width=True, hide_index=True)

