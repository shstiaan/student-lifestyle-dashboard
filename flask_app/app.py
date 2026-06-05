from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stress_level_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

load_dotenv(BASE_DIR / ".env")


def read_positive_int_env(name, default):
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = os.environ.get(
    "GEMINI_FALLBACK_MODELS",
    "gemini-2.5-flash-lite,gemini-2.0-flash-lite,gemini-2.0-flash",
)
GEMINI_TIMEOUT_SECONDS = read_positive_int_env("GEMINI_TIMEOUT_SECONDS", 45)
GEMINI_MAX_OUTPUT_TOKENS = read_positive_int_env("GEMINI_MAX_OUTPUT_TOKENS", 900)
GEMINI_RATE_LIMIT_COOLDOWN_SECONDS = read_positive_int_env("GEMINI_RATE_LIMIT_COOLDOWN_SECONDS", 120)
GEMINI_COOLDOWN_UNTIL = 0.0
GEMINI_RECOMMENDATION_CACHE = {}

app = Flask(__name__)


def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifact belum tersedia. Jalankan `python train_model.py` terlebih dahulu."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


MODEL, METADATA = load_artifacts()
FEATURES = METADATA["features"]
FEATURE_SCHEMA = METADATA["feature_schema"]
STRESS_LABELS = {int(key): value for key, value in METADATA["stress_labels"].items()}

FEATURE_GROUP_DEFINITIONS = [
    {
        "id": "psychology",
        "title": "Psikologis",
        "badge": "01",
        "accent": "blue",
        "features": ["anxiety_level", "self_esteem", "mental_health_history", "depression"],
    },
    {
        "id": "body",
        "title": "Fisiologis",
        "badge": "02",
        "accent": "yellow",
        "features": ["headache", "blood_pressure", "sleep_quality", "breathing_problem"],
    },
    {
        "id": "campus",
        "title": "Lingkungan & Akademik",
        "badge": "03",
        "accent": "mint",
        "features": [
            "noise_level",
            "living_conditions",
            "safety",
            "basic_needs",
            "academic_performance",
            "study_load",
            "teacher_student_relationship",
            "future_career_concerns",
        ],
    },
    {
        "id": "social",
        "title": "Sosial",
        "badge": "04",
        "accent": "coral",
        "features": ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
    },
]

RESULT_ASSETS = {
    0: "assets/result-low.png",
    1: "assets/result-medium.png",
    2: "assets/result-high.png",
}

GEMINI_RECOMMENDATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "priority_factors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "value": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["label", "value", "reason"],
            },
        },
        "activities": {"type": "ARRAY", "items": {"type": "STRING"}},
        "prevention": {"type": "ARRAY", "items": {"type": "STRING"}},
        "coping": {"type": "ARRAY", "items": {"type": "STRING"}},
        "songs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "artist": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["title", "artist", "reason"],
            },
        },
        "seven_day_plan": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "day": {"type": "STRING"},
                    "focus": {"type": "STRING"},
                    "action": {"type": "STRING"},
                },
                "required": ["day", "focus", "action"],
            },
        },
        "professional_help": {"type": "STRING"},
        "disclaimer": {"type": "STRING"},
    },
    "required": [
        "summary",
        "activities",
        "prevention",
        "coping",
        "songs",
        "professional_help",
        "disclaimer",
    ],
}

FEATURE_HELP = {
    "anxiety_level": {
        "label": "Anxiety Level",
        "description": "Menggambarkan tingkat kecemasan yang dirasakan mahasiswa.",
        "direction": "Semakin besar nilainya, semakin tinggi kecemasan dan umumnya risiko stres meningkat.",
        "low": "0 = sangat rendah",
        "high": "21 = sangat tinggi",
    },
    "self_esteem": {
        "label": "Self Esteem",
        "description": "Menggambarkan rasa percaya diri dan penilaian positif terhadap diri sendiri.",
        "direction": "Semakin besar nilainya, semakin baik harga diri dan umumnya risiko stres menurun.",
        "low": "0 = sangat rendah",
        "high": "30 = sangat baik",
    },
    "mental_health_history": {
        "label": "Mental Health History",
        "description": "Menunjukkan apakah ada riwayat masalah kesehatan mental.",
        "direction": "Nilai 1 berarti ada riwayat; nilai 0 berarti tidak ada riwayat.",
        "low": "0 = tidak ada",
        "high": "1 = ada",
    },
    "depression": {
        "label": "Depression",
        "description": "Menggambarkan tingkat gejala depresi seperti sedih berkepanjangan atau kehilangan minat.",
        "direction": "Semakin besar nilainya, semakin tinggi gejala depresi dan umumnya risiko stres meningkat.",
        "low": "0 = sangat rendah",
        "high": "27 = sangat tinggi",
    },
    "headache": {
        "label": "Headache",
        "description": "Menggambarkan frekuensi atau intensitas sakit kepala.",
        "direction": "Semakin besar nilainya, semakin sering/berat sakit kepala dan dapat berkaitan dengan stres lebih tinggi.",
        "low": "0 = tidak pernah/sangat rendah",
        "high": "5 = sangat sering/tinggi",
    },
    "blood_pressure": {
        "label": "Blood Pressure",
        "description": "Kategori tekanan darah pada data.",
        "direction": "Semakin besar nilainya, kategori tekanan darah makin tinggi.",
        "low": "1 = rendah/normal rendah",
        "high": "3 = tinggi",
    },
    "sleep_quality": {
        "label": "Sleep Quality",
        "description": "Menggambarkan kualitas tidur mahasiswa.",
        "direction": "Semakin besar nilainya, semakin baik kualitas tidur dan umumnya risiko stres menurun.",
        "low": "0 = sangat buruk",
        "high": "5 = sangat baik",
    },
    "breathing_problem": {
        "label": "Breathing Problem",
        "description": "Menggambarkan tingkat gangguan pernapasan yang dirasakan.",
        "direction": "Semakin besar nilainya, semakin berat gangguan pernapasan dan bisa berkaitan dengan stres lebih tinggi.",
        "low": "0 = tidak ada/sangat rendah",
        "high": "5 = sangat tinggi",
    },
    "noise_level": {
        "label": "Noise Level",
        "description": "Menggambarkan tingkat kebisingan lingkungan tempat tinggal atau belajar.",
        "direction": "Semakin besar nilainya, semakin bising lingkungan dan umumnya risiko stres meningkat.",
        "low": "0 = sangat tenang",
        "high": "5 = sangat bising",
    },
    "living_conditions": {
        "label": "Living Conditions",
        "description": "Menggambarkan kualitas kondisi tempat tinggal.",
        "direction": "Semakin besar nilainya, semakin baik kondisi tempat tinggal dan umumnya risiko stres menurun.",
        "low": "0 = sangat buruk",
        "high": "5 = sangat baik",
    },
    "safety": {
        "label": "Safety",
        "description": "Menggambarkan rasa aman di lingkungan sehari-hari.",
        "direction": "Semakin besar nilainya, semakin tinggi rasa aman dan umumnya risiko stres menurun.",
        "low": "0 = sangat tidak aman",
        "high": "5 = sangat aman",
    },
    "basic_needs": {
        "label": "Basic Needs",
        "description": "Menggambarkan terpenuhinya kebutuhan dasar seperti makan, tempat tinggal, dan kebutuhan harian.",
        "direction": "Semakin besar nilainya, semakin terpenuhi kebutuhan dasar dan umumnya risiko stres menurun.",
        "low": "0 = tidak terpenuhi",
        "high": "5 = sangat terpenuhi",
    },
    "academic_performance": {
        "label": "Academic Performance",
        "description": "Menggambarkan performa akademik mahasiswa.",
        "direction": "Semakin besar nilainya, semakin baik performa akademik dan umumnya risiko stres menurun.",
        "low": "0 = sangat buruk",
        "high": "5 = sangat baik",
    },
    "study_load": {
        "label": "Study Load",
        "description": "Menggambarkan beratnya beban belajar, tugas, dan tuntutan akademik.",
        "direction": "Semakin besar nilainya, semakin berat beban belajar dan umumnya risiko stres meningkat.",
        "low": "0 = sangat ringan",
        "high": "5 = sangat berat",
    },
    "teacher_student_relationship": {
        "label": "Teacher Student Relationship",
        "description": "Menggambarkan kualitas hubungan mahasiswa dengan pengajar/dosen.",
        "direction": "Semakin besar nilainya, semakin baik hubungan tersebut dan umumnya risiko stres menurun.",
        "low": "0 = sangat buruk",
        "high": "5 = sangat baik",
    },
    "future_career_concerns": {
        "label": "Future Career Concerns",
        "description": "Menggambarkan tingkat kekhawatiran tentang karier masa depan.",
        "direction": "Semakin besar nilainya, semakin besar kekhawatiran karier dan umumnya risiko stres meningkat.",
        "low": "0 = tidak khawatir",
        "high": "5 = sangat khawatir",
    },
    "social_support": {
        "label": "Social Support",
        "description": "Menggambarkan dukungan dari teman, keluarga, atau lingkungan sosial.",
        "direction": "Semakin besar nilainya, semakin kuat dukungan sosial dan umumnya risiko stres menurun.",
        "low": "0 = sangat rendah",
        "high": "3 = tinggi",
    },
    "peer_pressure": {
        "label": "Peer Pressure",
        "description": "Menggambarkan tekanan dari teman sebaya atau lingkungan pergaulan.",
        "direction": "Semakin besar nilainya, semakin tinggi tekanan teman sebaya dan umumnya risiko stres meningkat.",
        "low": "0 = tidak ada/sangat rendah",
        "high": "5 = sangat tinggi",
    },
    "extracurricular_activities": {
        "label": "Extracurricular Activities",
        "description": "Menggambarkan tingkat keterlibatan dalam kegiatan ekstrakurikuler.",
        "direction": "Nilai lebih besar berarti keterlibatan lebih tinggi; dampaknya bisa positif atau menambah beban tergantung konteks.",
        "low": "0 = tidak aktif",
        "high": "5 = sangat aktif",
    },
    "bullying": {
        "label": "Bullying",
        "description": "Menggambarkan tingkat pengalaman bullying atau perundungan.",
        "direction": "Semakin besar nilainya, semakin tinggi pengalaman bullying dan umumnya risiko stres meningkat.",
        "low": "0 = tidak ada",
        "high": "5 = sangat tinggi",
    },
}


def enrich_feature_schema(schema):
    enriched = []
    for field in schema:
        help_text = FEATURE_HELP.get(field["name"], {})
        enriched_field = {**field}
        enriched_field.update(help_text)
        enriched.append(enriched_field)
    return enriched


FEATURE_SCHEMA = enrich_feature_schema(FEATURE_SCHEMA)


def build_feature_groups():
    field_by_name = {field["name"]: field for field in FEATURE_SCHEMA}
    groups = []

    for group in FEATURE_GROUP_DEFINITIONS:
        fields = [field_by_name[name] for name in group["features"] if name in field_by_name]
        groups.append({**group, "fields": fields})

    grouped_names = {name for group in FEATURE_GROUP_DEFINITIONS for name in group["features"]}
    remaining_fields = [field for field in FEATURE_SCHEMA if field["name"] not in grouped_names]
    if remaining_fields:
        groups.append(
            {
                "id": "other",
                "title": "Lainnya",
                "badge": f"{len(groups) + 1:02d}",
                "accent": "purple",
                "features": [field["name"] for field in remaining_fields],
                "fields": remaining_fields,
            }
        )

    return groups


FEATURE_GROUPS = build_feature_groups()


def get_field_label(feature_name):
    for field in FEATURE_SCHEMA:
        if field["name"] == feature_name:
            return field["label"]
    return feature_name.replace("_", " ").title()


def predict_from_values(values):
    input_df = pd.DataFrame([values], columns=FEATURES)
    predicted_class = int(MODEL.predict(input_df)[0])
    prediction = {
        "class_id": predicted_class,
        "label": STRESS_LABELS[predicted_class],
        "asset": RESULT_ASSETS.get(predicted_class, RESULT_ASSETS[1]),
    }

    probabilities = None
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(input_df)[0]
        probabilities = [
            {
                "class_id": int(class_id),
                "label": STRESS_LABELS[int(class_id)],
                "probability": round(float(prob) * 100, 2),
            }
            for class_id, prob in zip(MODEL.classes_, proba)
        ]

    return prediction, probabilities


def get_priority_factors(values):
    rules = [
        ("anxiety_level", values.get("anxiety_level", 0) >= 15, "Kecemasan relatif tinggi.", 5),
        ("depression", values.get("depression", 0) >= 18, "Gejala depresi relatif tinggi.", 5),
        ("bullying", values.get("bullying", 0) >= 3, "Pengalaman bullying perlu diperhatikan.", 5),
        ("sleep_quality", values.get("sleep_quality", 5) <= 2, "Kualitas tidur masih rendah.", 4),
        ("study_load", values.get("study_load", 0) >= 4, "Beban belajar terasa berat.", 4),
        ("future_career_concerns", values.get("future_career_concerns", 0) >= 4, "Kekhawatiran karier cukup tinggi.", 4),
        ("social_support", values.get("social_support", 3) <= 1, "Dukungan sosial masih rendah.", 4),
        ("self_esteem", values.get("self_esteem", 30) <= 10, "Harga diri perlu diperkuat.", 3),
        ("safety", values.get("safety", 5) <= 2, "Rasa aman di lingkungan masih rendah.", 3),
        ("basic_needs", values.get("basic_needs", 5) <= 2, "Pemenuhan kebutuhan dasar perlu diperhatikan.", 3),
        ("peer_pressure", values.get("peer_pressure", 0) >= 4, "Tekanan teman sebaya cukup tinggi.", 3),
        ("noise_level", values.get("noise_level", 0) >= 4, "Lingkungan cukup bising.", 2),
        ("academic_performance", values.get("academic_performance", 5) <= 2, "Performa akademik sedang perlu dukungan.", 2),
    ]
    factors = [
        {
            "feature": feature,
            "label": get_field_label(feature),
            "value": values.get(feature),
            "reason": reason,
            "severity": severity,
        }
        for feature, matched, reason, severity in rules
        if matched
    ]
    factors.sort(key=lambda item: item["severity"], reverse=True)
    return factors[:6]


def fallback_recommendation(prediction, values, priority_factors):
    class_id = prediction["class_id"]
    if class_id == 0:
        summary = "Tingkat stres terprediksi rendah. Fokus utama adalah menjaga ritme sehat agar kondisi tetap stabil."
        activities = [
            "Pertahankan jadwal tidur dan belajar yang sudah berjalan.",
            "Lakukan aktivitas fisik ringan 20-30 menit, 3 kali seminggu.",
            "Sisihkan waktu sosial singkat dengan teman atau keluarga.",
            "Gunakan jeda 5 menit setelah 45-60 menit belajar.",
        ]
        songs = [
            {"title": "Here Comes The Sun", "artist": "The Beatles", "reason": "Nuansa ringan untuk menjaga mood positif."},
            {"title": "Good Life", "artist": "OneRepublic", "reason": "Energi optimis untuk aktivitas harian."},
            {"title": "Rehat", "artist": "Kunto Aji", "reason": "Pengingat untuk tetap memberi ruang istirahat."},
        ]
    elif class_id == 1:
        summary = "Tingkat stres terprediksi sedang. Prioritasnya adalah mengurangi tekanan utama dan membuat rutinitas lebih terkontrol."
        activities = [
            "Pecah tugas besar menjadi target kecil 25-30 menit.",
            "Buat daftar tiga prioritas harian, bukan semua hal sekaligus.",
            "Lakukan jalan santai atau peregangan 10-15 menit saat mulai tegang.",
            "Hubungi satu orang tepercaya untuk bercerita atau meminta bantuan praktis.",
        ]
        songs = [
            {"title": "Weightless", "artist": "Marconi Union", "reason": "Instrumental tenang untuk membantu relaksasi."},
            {"title": "Bloom", "artist": "The Paper Kites", "reason": "Tempo lembut untuk menurunkan intensitas pikiran."},
            {"title": "Secukupnya", "artist": "Hindia", "reason": "Cocok untuk refleksi saat tekanan terasa menumpuk."},
        ]
    else:
        summary = "Tingkat stres terprediksi tinggi. Fokus awal adalah menurunkan intensitas stres, mencari dukungan, dan mengurangi beban yang tidak mendesak."
        activities = [
            "Ambil jeda aman 10 menit: duduk, minum air, dan atur napas perlahan.",
            "Tunda keputusan besar sampai kondisi lebih stabil.",
            "Hubungi teman, keluarga, dosen wali, konselor kampus, atau tenaga profesional.",
            "Kurangi paparan pemicu yang bisa dikendalikan, seperti kebisingan atau tugas nonprioritas.",
        ]
        songs = [
            {"title": "Weightless", "artist": "Marconi Union", "reason": "Instrumental minimal untuk suasana lebih tenang."},
            {"title": "Fix You", "artist": "Coldplay", "reason": "Lagu reflektif untuk menemani fase berat tanpa lirik agresif."},
            {"title": "Evaluasi", "artist": "Hindia", "reason": "Mendorong refleksi dan penerimaan saat tekanan tinggi."},
        ]

    prevention = [
        "Tidur dan bangun pada jam yang relatif konsisten.",
        "Batasi belajar maraton; gunakan blok belajar pendek dengan jeda.",
        "Catat pemicu stres utama dan satu langkah kecil yang bisa dilakukan hari ini.",
        "Kurangi kafein berlebihan, terutama sore atau malam.",
    ]
    coping = [
        "Coba napas 4-4-6: tarik 4 detik, tahan 4 detik, hembus 6 detik.",
        "Gunakan grounding 5-4-3-2-1: sebutkan 5 hal yang terlihat sampai 1 hal yang dirasakan.",
        "Tulis kekhawatiran dalam 3 kolom: masalah, hal yang bisa dikontrol, langkah kecil.",
    ]
    seven_day_plan = [
        {"day": "Hari 1", "focus": "Stabilisasi", "action": "Rapikan jadwal tidur dan pilih satu tugas paling penting."},
        {"day": "Hari 2", "focus": "Beban akademik", "action": "Pecah tugas terbesar menjadi tiga bagian kecil."},
        {"day": "Hari 3", "focus": "Tubuh", "action": "Lakukan jalan santai atau peregangan 15 menit."},
        {"day": "Hari 4", "focus": "Dukungan", "action": "Hubungi satu orang tepercaya untuk bercerita."},
        {"day": "Hari 5", "focus": "Lingkungan", "action": "Kurangi satu pemicu seperti kebisingan atau distraksi belajar."},
        {"day": "Hari 6", "focus": "Refleksi", "action": "Tulis tiga hal yang membaik dan satu hal yang masih berat."},
        {"day": "Hari 7", "focus": "Evaluasi", "action": "Tentukan rutinitas kecil yang akan dipertahankan minggu depan."},
    ]

    return {
        "source": "fallback",
        "summary": summary,
        "priority_factors": priority_factors,
        "activities": activities,
        "prevention": prevention,
        "coping": coping,
        "songs": songs,
        "seven_day_plan": seven_day_plan,
        "professional_help": (
            "Jika stres terasa sangat berat, berlangsung lama, mengganggu tidur/makan/kuliah, "
            "atau muncul pikiran menyakiti diri, segera hubungi konselor kampus, psikolog, dokter, "
            "keluarga/orang tepercaya, atau layanan darurat setempat."
        ),
        "disclaimer": "Rekomendasi ini bersifat edukatif dan bukan diagnosis medis.",
    }


def normalize_song_items(items, fallback_items):
    if not isinstance(items, list) or not items:
        return fallback_items

    normalized = []
    for item in items:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            artist = str(item.get("artist", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if title:
                normalized.append(
                    {
                        "title": title,
                        "artist": artist or "Rekomendasi personal",
                        "reason": reason or "Dipilih untuk menemani proses regulasi emosi.",
                    }
                )
        elif isinstance(item, str) and item.strip():
            normalized.append(
                {
                    "title": item.strip(),
                    "artist": "Rekomendasi personal",
                    "reason": "Dipilih untuk menemani proses regulasi emosi.",
                }
            )

    return normalized or fallback_items


def normalize_plan_items(items, fallback_items):
    if not isinstance(items, list) or not items:
        return fallback_items

    normalized = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            action = str(item.get("action", "")).strip()
            if action:
                normalized.append(
                    {
                        "day": str(item.get("day", f"Hari {index}")).strip() or f"Hari {index}",
                        "focus": str(item.get("focus", "Langkah kecil")).strip() or "Langkah kecil",
                        "action": action,
                    }
                )
        elif isinstance(item, str) and item.strip():
            normalized.append({"day": f"Hari {index}", "focus": "Langkah kecil", "action": item.strip()})

    return normalized or fallback_items


def normalize_recommendation(recommendation, fallback):
    normalized = {**fallback, **recommendation}
    for key in ["priority_factors", "activities", "prevention", "coping"]:
        if not isinstance(normalized.get(key), list) or not normalized[key]:
            normalized[key] = fallback[key]
    normalized["songs"] = normalize_song_items(normalized.get("songs"), fallback["songs"])
    normalized["seven_day_plan"] = normalize_plan_items(normalized.get("seven_day_plan"), fallback["seven_day_plan"])
    for key in ["summary", "professional_help", "disclaimer"]:
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            normalized[key] = fallback[key]
    normalized["source"] = recommendation.get("source", fallback["source"])
    return normalized


def repair_common_json_issues(text):
    repaired = text.strip()
    repaired = re.sub(r"^```(?:json)?\s*", "", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\s*```$", "", repaired)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([}\]\"0-9])\s*\n\s*(\"[A-Za-z0-9_ -]+\"\s*:)", r"\1,\n\2", repaired)
    repaired = re.sub(r"(\})\s*\n\s*(\{)", r"\1,\n\2", repaired)
    return repaired


def extract_json_object(text):
    cleaned = repair_common_json_issues(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response LLM tidak berisi JSON object.")

    extracted = cleaned[start : end + 1]
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return json.loads(repair_common_json_issues(extracted))


def build_llm_prompt(prediction, values, priority_factors):
    safe_values = {key: values[key] for key in FEATURES if key in values}
    return (
        "Buat rekomendasi personal berbahasa Indonesia untuk mahasiswa berdasarkan prediksi stres dan input numerik berikut.\n"
        "Jangan memberi diagnosis medis, jangan menyarankan obat, dan jangan menulis lirik lagu.\n"
        "Output wajib JSON valid saja. Jangan gunakan markdown, bullet di luar JSON, komentar, atau teks pembuka.\n"
        "Semua nilai string harus memakai escaping JSON yang benar jika ada tanda kutip.\n\n"
        "Buat jawaban sangat ringkas: 3 aktivitas, 3 pencegahan, 2 coping, dan 2 lagu. "
        "Tidak perlu membuat rencana 7 hari atau mengulang faktor prioritas karena sistem sudah menyiapkannya.\n\n"
        f"Hasil prediksi: {prediction['label']} (kelas {prediction['class_id']}).\n"
        f"Faktor prioritas terdeteksi: {json.dumps(priority_factors, ensure_ascii=False)}.\n"
        f"Input fitur: {json.dumps(safe_values, ensure_ascii=False)}."
    )


def get_gemini_model_candidates():
    raw_models = [os.environ.get("GEMINI_MODEL", GEMINI_MODEL)]
    raw_models.extend(os.environ.get("GEMINI_FALLBACK_MODELS", GEMINI_FALLBACK_MODELS).split(","))

    models = []
    for model in raw_models:
        model = model.strip()
        if model and model not in models:
            models.append(model)
    return models


def get_retry_after_seconds(response):
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return GEMINI_RATE_LIMIT_COOLDOWN_SECONDS

    try:
        return max(int(retry_after), 1)
    except ValueError:
        return GEMINI_RATE_LIMIT_COOLDOWN_SECONDS


def build_gemini_cache_key(prediction, values, priority_factors):
    payload = {
        "class_id": prediction["class_id"],
        "values": values,
        "priority_factors": priority_factors,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def generate_recommendation_with_gemini(prediction, values, priority_factors):
    global GEMINI_COOLDOWN_UNTIL

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY belum tersedia. Menggunakan rekomendasi fallback lokal."

    cache_key = build_gemini_cache_key(prediction, values, priority_factors)
    if cache_key in GEMINI_RECOMMENDATION_CACHE:
        return copy.deepcopy(GEMINI_RECOMMENDATION_CACHE[cache_key]), None

    now = time.time()
    if now < GEMINI_COOLDOWN_UNTIL:
        wait_seconds = int(GEMINI_COOLDOWN_UNTIL - now)
        return (
            None,
            f"Kuota Gemini sedang dibatasi. Coba lagi sekitar {wait_seconds} detik lagi. "
            "Menggunakan rekomendasi fallback lokal.",
        )

    system_instruction = (
        "Kamu adalah asisten edukasi kesehatan mental mahasiswa. "
        "Berikan rekomendasi praktis, aman, spesifik, dan non-diagnostik. "
        "Selalu sarankan bantuan profesional untuk indikasi berat."
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_llm_prompt(prediction, values, priority_factors)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_RECOMMENDATION_SCHEMA,
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    last_error = None
    model_errors = []
    for model in get_gemini_model_candidates():
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(5, GEMINI_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            recommendation = extract_json_object(content)
            recommendation["source"] = f"gemini:{model}"
            GEMINI_RECOMMENDATION_CACHE[cache_key] = copy.deepcopy(recommendation)
            return recommendation, None
        except requests.exceptions.Timeout:
            last_error = f"request ke model {model} melewati batas {GEMINI_TIMEOUT_SECONDS} detik"
            model_errors.append(last_error)
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            if status_code == 429:
                wait_seconds = get_retry_after_seconds(response)
                GEMINI_COOLDOWN_UNTIL = max(GEMINI_COOLDOWN_UNTIL, time.time() + wait_seconds)
                last_error = (
                    f"kuota/rate limit model {model} sedang penuh. "
                    f"Coba lagi sekitar {wait_seconds} detik lagi"
                )
            else:
                last_error = f"model {model}: HTTP {status_code or 'error'}"
            model_errors.append(last_error)
        except (json.JSONDecodeError, ValueError):
            last_error = f"model {model} merespons, tetapi format JSON belum valid"
            model_errors.append(last_error)
            break
        except Exception as exc:
            last_error = f"model {model}: {exc}"
            model_errors.append(last_error)

    detail = "; ".join(model_errors[-3:]) if model_errors else last_error
    return None, f"Gemini tidak dapat digunakan saat ini: {detail}. Menggunakan rekomendasi fallback lokal."


def parse_form(form_data):
    values = {}
    errors = []

    for field in FEATURE_SCHEMA:
        name = field["name"]
        raw_value = form_data.get(name, "").strip()

        if raw_value == "":
            errors.append(f"{field['label']} wajib diisi.")
            continue

        try:
            value = int(float(raw_value))
        except ValueError:
            errors.append(f"{field['label']} harus berupa angka.")
            continue

        min_value = field["min"]
        max_value = field["max"]
        if value < min_value or value > max_value:
            errors.append(f"{field['label']} harus berada di rentang {min_value} sampai {max_value}.")

        values[name] = value

    return values, errors


@app.route("/")
def index():
    return render_template("index.html", metadata=METADATA)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probabilities = None
    errors = []
    form_values = {}

    if request.method == "POST":
        form_values, errors = parse_form(request.form)

        if not errors:
            prediction, probabilities = predict_from_values(form_values)

    return render_template(
        "predict.html",
        fields=FEATURE_SCHEMA,
        field_groups=FEATURE_GROUPS,
        form_values=form_values,
        prediction=prediction,
        probabilities=probabilities,
        errors=errors,
        metadata=METADATA,
    )


@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == "GET":
        return render_template(
            "recommendation.html",
            missing_payload=True,
            prediction=None,
            probabilities=None,
            recommendation=None,
            llm_notice=None,
            metadata=METADATA,
        )

    form_values, errors = parse_form(request.form)
    if errors:
        return render_template(
            "recommendation.html",
            missing_payload=True,
            prediction=None,
            probabilities=None,
            recommendation=None,
            llm_notice="Data prediksi tidak lengkap. Silakan isi ulang form prediksi.",
            metadata=METADATA,
        )

    prediction, probabilities = predict_from_values(form_values)
    priority_factors = get_priority_factors(form_values)
    fallback = fallback_recommendation(prediction, form_values, priority_factors)
    llm_recommendation, llm_notice = generate_recommendation_with_gemini(
        prediction,
        form_values,
        priority_factors,
    )
    final_recommendation = normalize_recommendation(llm_recommendation, fallback) if llm_recommendation else fallback

    return render_template(
        "recommendation.html",
        missing_payload=False,
        prediction=prediction,
        probabilities=probabilities,
        recommendation=final_recommendation,
        llm_notice=llm_notice,
        metadata=METADATA,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
