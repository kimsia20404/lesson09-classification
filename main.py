import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/kimsia20404/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_dictionary.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------------------
# 기록 파일 읽기 / 저장하기
# ------------------------------------------------------------
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 읽습니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        return {
            "meanings": saved.get("meanings", {}),
            "source": saved.get("source", ""),
        }
    except (json.JSONDecodeError, OSError, AttributeError):
        st.warning("기존 기록 파일을 읽지 못했습니다. 새 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """입력한 내용을 UTF-8 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 데이터 읽기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    # bmi의 빈 값은 결측치(NaN)로 읽습니다. 원본 데이터는 수정하지 않습니다.
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소와 인터넷 연결을 확인해 주세요.")
    st.exception(error)
    st.stop()

required_columns = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()

if "stroke" not in df.columns:
    st.error("stroke 열을 찾을 수 없습니다.")
    st.stop()

saved_records = load_records()
saved_meanings = saved_records["meanings"]
saved_source = saved_records["source"]


# ------------------------------------------------------------
# 화면
# ------------------------------------------------------------
st.title(f"🧠 {APP_TITLE}")
st.write(
