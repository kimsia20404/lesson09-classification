import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/kimsia20404/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_notes.json"

COLUMNS_KOREAN = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

# 숫자로 저장되어 있지만 범주형으로 해석하는 열
BINARY_OR_CATEGORY_COLUMNS = {
    "hypertension",
    "heart_disease",
    "stroke",
}


# --------------------------------------------------
# 기록 파일 읽기 / 저장하기
# --------------------------------------------------
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {"meanings": {}, "source": ""}

        return {
            "meanings": data.get("meanings", {}),
            "source": data.get("source", ""),
        }
    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """입력한 내용을 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미를 직접 정리해 보는 공간입니다.")
st.divider()

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.exception(error)
