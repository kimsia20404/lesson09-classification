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
    st.stop()


# --------------------------------------------------
# 저장된 학습 기록 불러오기
# --------------------------------------------------
records = load_records()
saved_meanings = records.get("meanings", {})
saved_source = records.get("source", "")


# --------------------------------------------------
# 1. 데이터 소개 및 숫자 카드
# --------------------------------------------------
st.header("1. 데이터 한눈에 보기")
st.write(
    "이 데이터는 여러 사람의 건강 및 생활 습관 정보와 "
    "뇌졸중 발생 여부를 담고 있습니다. 각 열을 살펴보며 "
    "어떤 정보가 예측에 활용될 수 있을지 생각해 봅시다."
)

total_people = len(df)
column_count = len(df.columns)

if "stroke" in df.columns:
    stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum())
else:
    stroke_count = 0

stroke_rate = (stroke_count / total_people * 100) if total_people else 0

card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()


# --------------------------------------------------
# 2. 열 이름과 뜻 정리
# --------------------------------------------------
st.header("2. 열 이름과 뜻 알아보기")
st.write(
    "각 열이 어떤 정보를 나타내는지 교재를 참고해 "
    "**우리말 뜻** 칸에 직접 입력해 보세요. "
    "입력한 내용은 아래의 저장 버튼을 누르면 JSON 파일에 기록됩니다."
)


def describe_values(series, column_name):
    """열의 값 종류를 학습용 문장으로 정리합니다."""
    non_null = series.dropna()
    unique_count = non_null.nunique()

    if column_name == "id":
        return f"식별 번호 (고유값 {unique_count:,}개)"

    if column_name in BINARY_OR_CATEGORY_COLUMNS:
        values = sorted(non_null.unique().tolist())
        return "범주형: " + ", ".join(str(value) for value in values)

    if pd.api.types.is_numeric_dtype(series):
        return f"숫자형 (고유값 {unique_count:,}개)"

    values = sorted(non_null.astype(str).unique().tolist())
    if len(values) <= 12:
        return "범주형: " + ", ".join(values)
    return f"범주형 (고유값 {unique_count:,}개)"


table_rows = []

for column in df.columns:
    table_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": saved_meanings.get(column, ""),
            "값의 종류": describe_values(df[column], column),
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

dictionary_df = pd.DataFrame(table_rows)

edited_dictionary = st.data_editor(
    dictionary_df,
    use_container_width=True,
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 직접 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
    key="dictionary_editor",
)

st.divider()


# --------------------------------------------------
# 3. 원본 데이터 미리 보기
# --------------------------------------------------
st.header("3. 데이터 처음 다섯 줄")
st.write("데이터를 가공하지 않은 상태로 처음 다섯 행을 표시합니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()


# --------------------------------------------------
# 4. 데이터 출처와 기록 저장
# --------------------------------------------------
st.header("4. 데이터 출처")
st.write("교재에 소개된 데이터 출처를 아래 칸에 적어 보세요.")

source_text = st.text_area(
    "데이터 출처",
    value=saved_source,
    placeholder="교재를 참고해 데이터 출처를 입력하세요.",
    height=100,
    key="source_editor",
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    meanings_to_save = {}

    for _, row in edited_dictionary.iterrows():
        meanings_to_save[str(row["열 이름"])] = str(row["우리말 뜻"])

    try:
        save_records(meanings_to_save, source_text)
        st.success(f"기록을 저장했습니다: {RECORD_FILE}")
    except OSError as error:
        st.error("기록 파일을 저장하지 못했습니다.")
        st.exception(error)

st.caption(
    "저장 위치: 내_기록/stroke_notes.json"
)
st.caption(
    "참고: Streamlit Cloud에서는 앱 재배포나 서버 환경 초기화 시 "
    "로컬 파일이 유지되지 않을 수 있습니다. 영구 보관이 필요하면 "
    "외부 저장소를 연결해야 합니다."
)
