from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "model_artifacts" / "xgb_best_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR.parent.parent / "model_artifacts" / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

def predict_career(input_data: dict):
    """
    Minimal inputs for SS1-SS3 students:
    - Required: Gender, Department, School_Type
    - Optional: Subject choices, aptitude scores, etc.
    """
    df_input = pd.DataFrame([input_data])
    df_input.columns = (
        df_input.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )

    categorical_columns = [
        "gender", "school_type", "department", "academic_strength",
        "best_subject_category", "confidence_level", "career_influence",
        "mathematics", "english", "civic_education", "physics", "chemistry",
        "biology", "further_mathematics", "agricultural_science", "geography",
        "technical_drawing", "computer_studies", "yoruba", "igbo_hausa",
        "data_processing", "literature_in_english", "history",
        "christian_religious_studies/islamic_studies", "french", "creative_arts",
        "economics", "financial_accounting", "commerce", "government", "marketing"
    ]

    numerical_columns = [
        "age", "waec_year", "waec_credits", "cgpa", "course_alignment",
        "aptitude_score_10", "cognitive_score_10", "psychometric_avg_5",
        "sentiment_avg_5"
    ]

    features = categorical_columns + numerical_columns

    defaults = {
        "gender": None,
        "school_type": None,
        "department": None,
        "academic_strength": None,
        "best_subject_category": None,
        "confidence_level": None,
        "career_influence": None,
        "mathematics": "UNKNOWN",
        "english": "UNKNOWN",
        "physics": "UNKNOWN",
        "chemistry": "UNKNOWN",
        "biology": "UNKNOWN",
        "further_mathematics": "UNKNOWN",
        "agricultural_science": "UNKNOWN",
        "geography": "UNKNOWN",
        "technical_drawing": "UNKNOWN",
        "computer_studies": "UNKNOWN",
        "yoruba": "UNKNOWN",
        "igbo_hausa": "UNKNOWN",
        "data_processing": "UNKNOWN",
        "literature_in_english": "UNKNOWN",
        "christian_religious_studies/islamic_studies": "UNKNOWN",
        "creative_arts": "UNKNOWN",
        "economics": "UNKNOWN",
        "financial_accounting": "UNKNOWN",
        "commerce": "UNKNOWN",
        "government": "UNKNOWN",
        "marketing": "UNKNOWN",
        "waec_credits": 5.0,
        "cgpa": None,
        "course_alignment": None,
        "aptitude_score_10": None,
        "cognitive_score_10": None,
        "psychometric_avg_5": None,
        "sentiment_avg_5": None,
    }

    for col, default_value in defaults.items():
        if col not in df_input.columns:
            df_input[col] = default_value

    grade_map = {
        "A": 8,
        "B": 6,
        "C": 5,
        "D": 3,
        "E": 2,
        "F": 1,
        "UNKNOWN": 5
    }

    grade_cols = [
        "mathematics", "english", "physics", "chemistry", "biology", "further_mathematics",
        "agricultural_science", "geography", "technical_drawing", "computer_studies",
        "yoruba", "igbo_hausa", "data_processing", "literature_in_english", "history",
        "christian_religious_studies/islamic_studies", "french", "creative_arts",
        "economics", "financial_accounting", "commerce", "government", "marketing"
    ]

    for col in grade_cols:
        if col in df_input.columns:
            df_input[col] = (
                df_input[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(grade_map)
                .fillna(5)
            )

    for col in features:
        if col not in df_input.columns:
            df_input[col] = 5.0 if col == "waec_credits" else "unknown" if col in categorical_columns else 0.0

    df_input = df_input[features]

    for col in numerical_columns:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)

    df_input = df_input.astype({col: "float64" if col in numerical_columns else "object" for col in features})
    math_grade = df_input.at[0, "mathematics"]
    eng_grade = df_input.at[0, "english"]
    if math_grade == 1 or eng_grade == 1:
        return {
            "predicted_career": "None, please improve your grades in Mathematics and English to get career recommendations.",
            "confidence_percent": 0.0,
            "top_3": []
        }
    pred_encoded = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    career = le.inverse_transform([pred_encoded])[0]
    confidence = float(np.max(proba))
    top_three = sorted(zip(le.classes_, proba), key=lambda x: x[1], reverse=True)[:3]

    return {
        "predicted_career": str(career),
        "confidence_percent": float(round(confidence * 100, 1)),
        "top_3": [
            {
                "career": str(career_name),
                "confidence_percent": float(round(float(prob) * 100, 1))
            }
            for career_name, prob in top_three
        ]
    }

def transform_for_model(input_data: dict):
    """Return the preprocessed DataFrame used by the model for a single input dict."""
    df_input = pd.DataFrame([input_data])
    df_input.columns = (
        df_input.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )

    categorical_columns = [
        "gender", "school_type", "department", "academic_strength", "best_subject_category",
        "confidence_level", "career_influence",
        "mathematics", "english", "civic_education", "physics", "chemistry",
        "biology", "further_mathematics", "agricultural_science", "geography", "technical_drawing",
        "computer_studies", "yoruba", "igbo_hausa", "data_processing", "literature_in_english",
        "history", "christian_religious_studies/islamic_studies", "french", "creative_arts", "economics",
        "financial_accounting", "commerce", "government", "marketing"
    ]

    numerical_columns = [
        "age", "waec_year", "waec_credits", "cgpa", "course_alignment",
        "aptitude_score_10", "cognitive_score_10", "psychometric_avg_5", "sentiment_avg_5"
    ]

    features = categorical_columns + numerical_columns

    defaults = {
        "gender": None,
        "school_type": None,
        "department": None,
        "academic_strength": None,
        "best_subject_category": None,
        "confidence_level": None,
        "career_influence": None,
        "mathematics": "UNKNOWN",
        "english": "UNKNOWN",
        "physics": "UNKNOWN",
        "chemistry": "UNKNOWN",
        "biology": "UNKNOWN",
        "further_mathematics": "UNKNOWN",
        "agricultural_science": "UNKNOWN",
        "geography": "UNKNOWN",
        "technical_drawing": "UNKNOWN",
        "computer_studies": "UNKNOWN",
        "yoruba": "UNKNOWN",
        "igbo_hausa": "UNKNOWN",
        "data_processing": "UNKNOWN",
        "literature_in_english": "UNKNOWN",
        "history": "UNKNOWN",
        "christian_religious_studies/islamic_studies": "UNKNOWN",
        "french": "UNKNOWN",
        "creative_arts": "UNKNOWN",
        "economics": "UNKNOWN",
        "financial_accounting": "UNKNOWN",
        "commerce": "UNKNOWN",
        "government": "UNKNOWN",
        "marketing": "UNKNOWN",
        "waec_credits": 5.0,
        "cgpa": 0.0,
        "course_alignment": 0,
        "aptitude_score_10": 5,
        "cognitive_score_10": 5,
        "psychometric_avg_5": 3.0,
        "sentiment_avg_5": 3.0,
    }

    for col, default_value in defaults.items():
        if col not in df_input.columns:
            df_input[col] = default_value

    grade_map = {"A": 8, "B": 6, "C": 5, "D": 3, "E": 2, "F": 1, "UNKNOWN": 5}
    grade_cols = [
        "mathematics", "english", "physics", "chemistry", "biology", "further_mathematics",
        "agricultural_science", "geography", "technical_drawing", "computer_studies",
        "yoruba", "igbo_hausa", "data_processing", "literature_in_english", "history",
        "christian_religious_studies/islamic_studies", "french", "creative_arts",
        "economics", "financial_accounting", "commerce", "government", "marketing", "civic_education"
    ]

    for col in grade_cols:
        if col in df_input.columns:
            df_input[col] = (
                df_input[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(grade_map)
                .fillna(5)
            )

    for col in features:
        if col not in df_input.columns:
            df_input[col] = 5.0 if col == "waec_credits" else "unknown" if col in categorical_columns else 0.0

    for col in numerical_columns:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)

    df_input = df_input[features]
    df_input = df_input.astype({col: "float64" if col in numerical_columns else "object" for col in features})
    return df_input

# Quick test
if __name__ == "__main__":
    sample = {
        'Gender': 'Male',
        'Age': '23',
        'School_Type': 'Mission / Faith School',
        'Department': 'Science',
        'Mathematics': 'C',
        'English': 'B',
        'Civic Education': 'D',
        'Physics': 'E',
        'Chemistry': 'A',
        'Biology': 'A',
        'agricultural Science': 'B',
        'geography': 'D',
        'Academic_Strength': 'Low',
        'Best_Subject_Category': 'Science',
        'Confidence_Level': 'Not very confident',
        'Career_Influence': 'Personal passion'
    }

    result = predict_career(sample)

    print("\n=== SS Student Career Prediction ===")
    print(f"**Predicted Career:** {result['predicted_career']}")
    print(f"Confidence: {result['confidence_percent']:.1f}%\n")
    print("Top 3:")
    for item in result['top_3']:
        print(f"  • {item['career']} ({item['confidence_percent']:.1f}%)")