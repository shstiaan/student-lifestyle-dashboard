from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "StressLevelDataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "stress_level_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

TARGET_COL = "stress_level"
RANDOM_STATE = 42

STRESS_LABELS = {
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress",
}


def build_feature_schema(df: pd.DataFrame, feature_columns: list[str]) -> list[dict]:
    descriptions = {
        "anxiety_level": "Tingkat kecemasan",
        "self_esteem": "Skor harga diri",
        "mental_health_history": "Riwayat kesehatan mental",
        "depression": "Skor depresi",
        "headache": "Frekuensi sakit kepala",
        "blood_pressure": "Kategori tekanan darah",
        "sleep_quality": "Kualitas tidur",
        "breathing_problem": "Gangguan pernapasan",
        "noise_level": "Paparan kebisingan",
        "living_conditions": "Kondisi tempat tinggal",
        "safety": "Rasa aman",
        "basic_needs": "Pemenuhan kebutuhan dasar",
        "academic_performance": "Performa akademik",
        "study_load": "Beban belajar",
        "teacher_student_relationship": "Relasi dosen/guru dan mahasiswa",
        "future_career_concerns": "Kekhawatiran karier masa depan",
        "social_support": "Dukungan sosial",
        "peer_pressure": "Tekanan teman sebaya",
        "extracurricular_activities": "Aktivitas ekstrakurikuler",
        "bullying": "Pengalaman bullying",
    }

    schema = []
    for column in feature_columns:
        min_value = int(df[column].min())
        max_value = int(df[column].max())
        median_value = float(df[column].median())
        schema.append(
            {
                "name": column,
                "label": column.replace("_", " ").title(),
                "description": descriptions.get(column, column.replace("_", " ")),
                "min": min_value,
                "max": max_value,
                "step": 1,
                "default": int(round(median_value)),
            }
        )
    return schema


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    feature_columns = [column for column in df.columns if column != TARGET_COL]
    X = df[feature_columns]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    metadata = {
        "model_name": "Hist Gradient Boosting",
        "model_type": "sklearn.pipeline.Pipeline",
        "target": TARGET_COL,
        "features": feature_columns,
        "stress_labels": {str(key): value for key, value in STRESS_LABELS.items()},
        "feature_schema": build_feature_schema(df, feature_columns),
        "dataset_path": str(DATA_PATH.relative_to(BASE_DIR)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
    }

    metrics = {
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "cv_f1_macro_mean_train_only": float(cv_scores.mean()),
        "cv_f1_macro_std_train_only": float(cv_scores.std()),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=[STRESS_LABELS[key] for key in sorted(STRESS_LABELS)],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to      : {MODEL_PATH}")
    print(f"Metadata saved to   : {METADATA_PATH}")
    print(f"Metrics saved to    : {METRICS_PATH}")
    print(f"Test accuracy       : {test_accuracy:.4f}")
    print(f"Test F1 macro       : {test_f1_macro:.4f}")
    print(f"CV F1 macro mean    : {cv_scores.mean():.4f}")


if __name__ == "__main__":
    main()
