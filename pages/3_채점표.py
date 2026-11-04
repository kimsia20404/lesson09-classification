
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📝",
    layout="wide"
)

st.title("📝 3. 채점표")
st.caption("모델의 예측 결과를 실제 정답과 비교하여 평가합니다.")

st.divider()


# ==================================================
# 2. 데이터 불러오기
# ==================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "kimsia20404/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==================================================
# 3. 우리말 이름
# ==================================================

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

ALL_FEATURES = list(FEATURE_NAMES.keys())

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# ==================================================
# 4. 예측에 사용할 속성 선택
# ==================================================

st.header("📋 1. 예측에 사용할 속성")

st.write(
    """
    분류 모델 페이지와 동일하게 예측에 사용할 속성을 선택합니다.
    처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다.
    """
)

selected_features = st.multiselect(
    "입력 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 골라 주세요.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[x] for x in selected_features)
)

st.divider()


# ==================================================
# 5. 학습용과 테스트용 데이터 나누기
# ==================================================

st.header("✂️ 2. 학습용과 테스트용 데이터 나누기")

st.write(
    """
    사람의 번호(id)를 기준으로 오름차순 정렬합니다.

    열 명씩 묶은 다음 각 묶음의 앞 세 명은 테스트용,
    나머지 일곱 명은 학습용으로 사용합니다.
    """
)

if "id" not in df.columns or "stroke" not in df.columns:
    st.error("데이터에 id 또는 stroke 열이 없습니다.")
    st.stop()

df = df.copy()

df["_id_sort"] = pd.to_numeric(
    df["id"],
    errors="coerce"
)

df = df.sort_values(
    "_id_sort",
    kind="stable",
    na_position="last"
).reset_index(drop=True)

# 각 10명 묶음의 앞 3명
test_indices = []

for start in range(0, len(df), 10):
    test_indices.extend(
        range(start, min(start + 3, len(df)))
    )

test_indices = set(test_indices)

test_mask = df.index.isin(test_indices)

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

# 정답 데이터
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 입력 데이터
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# 숫자로 변환
for col in selected_features:
    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )


# ==================================================
# 6. BMI 빈 값 처리
# ==================================================

bmi_median = None

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습용 BMI 데이터에 유효한 값이 없습니다.")
        st.stop()

    # 학습용 중앙값으로 양쪽 데이터의 결측값 대체
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# 나머지 결측값 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 BMI 이외의 빈 값이 있습니다.")
    st.stop()

# 무한대 값 확인
if (
    X_train.isin([np.inf, -np.inf]).any().any()
    or X_test.isin([np.inf, -np.inf]).any().any()
):
    st.error("입력 데이터에 무한대 값이 있습니다.")
    st.stop()

if y_train.nunique() < 2:
    st.error("학습용 데이터에 두 종류의 정답이 모두 있어야 합니다.")
    st.stop()


# ==================================================
# 7. 데이터 개수 확인
# ==================================================

total_people = len(df)
train_people = len(train_df)
test_people = len(test_df)

test_stroke_people = int((y_test == 1).sum())

st.subheader("데이터 분할 결과")

c1, c2, c3 = st.columns(3)

c1.metric("전체 사람 수", f"{total_people:,}명")
c2.metric("학습용 사람 수", f"{train_people:,}명")
c3.metric("테스트용 사람 수", f"{test_people:,}명")

if total_people == 5110:
    if train_people != 3577 or test_people != 1533:
        st.error("예상한 데이터 분할 인원수와 다릅니다.")
        st.stop()

if bmi_median is not None:
    st.caption(
        f"BMI 빈 값 대체에 사용한 학습용 중앙값: {bmi_median:.2f}"
    )

st.divider()


# ==================================================
# 8. 크기 맞추기
# ==================================================

st.header("📏 3. 크기 맞추기")

st.write(
    """
    표준화 기준은 학습용 데이터에서만 계산합니다.
    테스트용 데이터에는 같은 기준을 적용합니다.
    """
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success("학습용 데이터만 이용하여 크기 맞추기를 완료했습니다.")

st.divider()


# ==================================================
# 9. 두 분류 모델 학습
# ==================================================

st.header("🧠 4. 분류 모델 학습")

st.write(
    """
    **로지스틱 회귀(확률로 답하는 모델)**

    뇌졸중일 확률을 계산하고, 확률이 0.5 이상이면
    뇌졸중(1)이라고 판단합니다.

    **의사결정트리(질문으로 답하는 모델)**

    질문을 최대 세 번까지만 던지며,
    마지막 마디에 사람이 다섯 명 미만이면 더 나누지 않습니다.
    """
)

# --------------------------------------------------
# 로지스틱 회귀
# --------------------------------------------------

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# --------------------------------------------------
# 의사결정트리
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# 세 모델의 테스트 예측
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)

# 학습용 데이터에서 더 많은 정답을 찾음
majority_class = int(y_train.value_counts().idxmax())

# 입력을 보지 않고 다수 클래스만 예측하는 모델
baseline_pred = np.full(
    len(y_test),
    majority_class,
    dtype=int
)

st.success("세 모델의 학습과 예측을 완료했습니다.")

st.divider()


# ==================================================
# 10. 혼동행렬과 평가 지표 계산 함수
# ==================================================

def calculate_confusion(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 실제 0, 예측 0
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    # 실제 0, 예측 1
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    # 실제 1, 예측 0
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # 실제 1, 예측 1
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def calculate_metrics(counts):

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    total = tn + fp + fn + tp

    # 정확도
    accuracy = (
        (tp + tn) / total
        if total > 0 else None
    )

    # 재현율: 실제 뇌졸중인 사람 중 찾아낸 비율
    recall_denominator = tp + fn

    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자 비율
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = None
        f1 = None

    else:
        precision = tp / precision_denominator

        # F1 = 2 * 정밀도 * 재현율 / (정밀도 + 재현율)
        if recall is None:
            f1 = None

        elif precision + recall == 0:
            f1 = 0.0

        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# ==================================================
# 11. 모델별 혼동행렬 계산
# ==================================================

logistic_counts = calculate_confusion(
    y_test,
    logistic_pred
)

tree_counts = calculate_confusion(
    y_test,
    tree_pred
)

baseline_counts = calculate_confusion(
    y_test,
    baseline_pred
)

logistic_metrics = calculate_metrics(logistic_counts)
tree_metrics = calculate_metrics(tree_counts)
baseline_metrics = calculate_metrics(baseline_counts)


# ==================================================
# 12. 테스트 데이터 안내
# ==================================================

st.header("🎯 5. 테스트 데이터 채점")

st.write(
    f"""
    테스트용 사람은 **{test_people:,}명**이며,
    그중 실제 뇌졸중 환자는 **{test_stroke_people:,}명**입니다.
    """
)

st.caption(
    f"실제 뇌졸중이 아닌 사람: "
    f"{test_people - test_stroke_people:,}명"
)

st.divider()


# ==================================================
# 13. 혼동행렬 표시 함수
# ==================================================

def make_confusion_table(counts):

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    # 행 합계
    actual_0_total = tn + fp
    actual_1_total = fn + tp

    # 열 합계
    predicted_0_total = tn + fn
    predicted_1_total = fp + tp

    total = tn + fp + fn + tp

    matrix = pd.DataFrame(
        [
            [
                f"{tn:,}명 (TN)",
                f"{fp:,}명 (FP)",
                f"{actual_0_total:,}명"
            ],
            [
                f"{fn:,}명 (FN)",
                f"{tp:,}명 (TP)",
                f"{actual_1_total:,}명"
            ],
            [
                f"{predicted_0_total:,}명",
                f"{predicted_1_total:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 없음 (0)",
            "실제 뇌졸중 있음 (1)",
            "열 합계"
        ],
        columns=[
            "예측 뇌졸중 없음 (0)",
            "예측 뇌졸중 있음 (1)",
            "행 합계"
        ]
    )

    return matrix


# ==================================================
# 14. 두 모델의 혼동행렬
# ==================================================

st.header("📊 6. 혼동행렬 비교")

st.write(
    """
    혼동행렬은 실제 정답과 모델의 예측을 비교한 표입니다.

    - TN: 실제로 뇌졸중이 아니고, 아니라고 예측
    - FP: 실제로 뇌졸중이 아닌데, 뇌졸중이라고 예측
    - FN: 실제로 뇌졸중인데, 아니라고 예측
    - TP: 실제로 뇌졸중이고, 뇌졸중이라고 예측
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("로지스틱 회귀(확률로 답하는 모델)")

    st.dataframe(
        make_confusion_table(logistic_counts),
        use_container_width=True
    )

with col2:
    st.subheader("의사결정트리(질문으로 답하는 모델)")

    st.dataframe(
        make_confusion_table(tree_counts),
        use_container_width=True
    )

st.divider()


# ==================================================
# 15. 정확도, 재현율, 정밀도, F1 비교
# ==================================================

st.header("📈 7. 평가 지표 비교")

st.write(
    """
    네 가지 지표를 이용해 세 모델의 성능을 비교합니다.

    - 정확도: 전체 예측 중 맞힌 비율
    - 재현율: 실제 뇌졸중 환자 중 찾아낸 비율
    - 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자의 비율
    - F1: 정밀도와 재현율을 함께 고려한 지표
    """
)

metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"])
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            "정확도": format_metric(tree_metrics["정확도"]),
            "재현율": format_metric(tree_metrics["재현율"]),
            "정밀도": format_metric(tree_metrics["정밀도"]),
            "F1": format_metric(tree_metrics["F1"])
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": format_metric(baseline_metrics["정확도"]),
            "재현율": format_metric(baseline_metrics["재현율"]),
            "정밀도": format_metric(baseline_metrics["정밀도"]),
            "F1": format_metric(baseline_metrics["F1"])
        }
    ]
)

st.dataframe(
    metrics_table,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "정밀도와 F1은 뇌졸중이라고 예측한 사람이 0명이면 "
    "'계산할 수 없음'으로 표시합니다."
)

st.divider()


# ==================================================
# 16. 놓친 환자와 헛짚은 사람 목록
# ==================================================

st.header("🔍 8. 놓친 환자와 헛짚은 사람 찾기")

st.write(
    """
    모델이 틀린 사람들을 직접 확인합니다.

    - 놓친 환자(FN): 실제 뇌졸중 환자인데 뇌졸중이 아니라고 예측한 사람
    - 헛짚은 사람(FP): 실제로 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람
    """
)

# --------------------------------------------------
# 모델 선택
# --------------------------------------------------

selected_model = st.selectbox(
    "확인할 모델 선택",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ]
)

prediction_dict = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_pred,
    "의사결정트리(질문으로 답하는 모델)": tree_pred,
    "훈련용에서 많은 쪽으로만 답하는 모델": baseline_pred
}

counts_dict = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_counts,
    "의사결정트리(질문으로 답하는 모델)": tree_counts,
    "훈련용에서 많은 쪽으로만 답하는 모델": baseline_counts
}

selected_pred = prediction_dict[selected_model]
selected_counts = counts_dict[selected_model]


# --------------------------------------------------
# 실제 정답과 예측 결과 합치기
# --------------------------------------------------

result_df = test_df.copy()

result_df["실제 정답"] = y_test.to_numpy()
result_df["예측 결과"] = selected_pred


# --------------------------------------------------
# 놓친 환자와 헛짚은 사람 추출
# --------------------------------------------------

# FN: 실제 1, 예측 0
missed_patients = result_df[
    (result_df["실제 정답"] == 1)
    & (result_df["예측 결과"] == 0)
].copy()

# FP: 실제 0, 예측 1
false_alarm_people = result_df[
    (result_df["실제 정답"] == 0)
    & (result_df["예측 결과"] == 1)
].copy()


# --------------------------------------------------
# 나이 많은 순서로 정렬
# --------------------------------------------------

missed_patients["age"] = pd.to_numeric(
    missed_patients["age"],
    errors="coerce"
)

false_alarm_people["age"] = pd.to_numeric(
    false_alarm_people["age"],
    errors="coerce"
)

missed_patients = missed_patients.sort_values(
    "age",
    ascending=False,
    na_position="last"
)

false_alarm_people = false_alarm_people.sort_values(
    "age",
    ascending=False,
    na_position="last"
)


# ==================================================
# 17. 목록 표시 함수
# ==================================================

def prepare_person_table(data):

    display_df = pd.DataFrame({
        "번호": data["id"],
        "나이": data["age"],
        "평균 혈당": data["avg_glucose_level"],
        "고혈압 여부": data["hypertension"].map({
            0: "없음",
            1: "있음"
        }),
        "심장병 여부": data["heart_disease"].map({
            0: "없음",
            1: "있음"
        })
    })

    display_df["나이"] = pd.to_numeric(
        display_df["나이"],
        errors="coerce"
    ).round(1)

    display_df["평균 혈당"] = pd.to_numeric(
        display_df["평균 혈당"],
        errors="coerce"
    ).round(2)

    return display_df


# --------------------------------------------------
# 왼쪽: 놓친 환자
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    missed_count = len(missed_patients)

    st.subheader(
        f"🚨 놓친 환자 (FN): {missed_count:,}명"
    )

    st.caption(
        "실제 뇌졸중 환자인데 아니라고 예측한 사람"
    )

    if missed_count > 0:

        missed_display = prepare_person_table(
            missed_patients.head(20)
        )

        st.dataframe(
            missed_display,
            hide_index=True,
            use_container_width=True
        )

        if missed_count > 20:
            st.caption(
                f"전체 {missed_count:,}명 중 "
                "나이가 많은 앞 20명만 표시했습니다."
            )

    else:
        st.success("놓친 환자가 없습니다.")


# --------------------------------------------------
# 오른쪽: 헛짚은 사람
# --------------------------------------------------

with col2:

    false_alarm_count = len(false_alarm_people)

    st.subheader(
        f"⚠️ 헛짚은 사람 (FP): {false_alarm_count:,}명"
    )

    st.caption(
        "실제로는 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람"
    )

    if false_alarm_count > 0:

        false_alarm_display = prepare_person_table(
            false_alarm_people.head(20)
        )

        st.dataframe(
            false_alarm_display,
            hide_index=True,
            use_container_width=True
        )

        if false_alarm_count > 20:
            st.caption(
                f"전체 {false_alarm_count:,}명 중 "
                "나이가 많은 앞 20명만 표시했습니다."
            )

    else:
        st.success("헛짚은 사람이 없습니다.")


# ==================================================
# 18. 목록과 혼동행렬 일치 확인
# ==================================================

st.divider()

st.subheader("📌 목록과 혼동행렬 확인")

fn_matrix = selected_counts["FN"]
fp_matrix = selected_counts["FP"]

st.write(
    f"""
    **놓친 환자 목록 {missed_count:,}명 = 혼동행렬 FN {fn_matrix:,}명**

    **헛짚은 사람 목록 {false_alarm_count:,}명 = 혼동행렬 FP {fp_matrix:,}명**
    """
)

if missed_count == fn_matrix and false_alarm_count == fp_matrix:
    st.success(
        "두 목록의 사람 수가 혼동행렬의 FN, FP와 정확히 일치합니다."
    )
else:
    st.error(
        "목록의 인원수와 혼동행렬의 인원수가 일치하지 않습니다."
    )


st.divider()


# ==================================================
# 19. 마무리
# ==================================================

st.header("📝 학습 정리")

st.write(
    """
    다음 질문에 답해 보세요.

    1. 정확도가 높은 모델이 반드시 뇌졸중 환자를 잘 찾아낼까요?
    2. 놓친 환자(FN)가 많으면 어떤 문제가 생길까요?
    3. 헛짚은 사람(FP)이 많으면 어떤 문제가 생길까요?
    4. 재현율과 정밀도 중 뇌졸중 예측에서는 무엇을 중요하게 생각해야 할까요?
    5. 세 모델 중 어떤 모델을 선택하고 싶은가요? 그 이유는 무엇인가요?
    """
)

st.caption("뇌졸중 예측 실습실 | 채점표")
