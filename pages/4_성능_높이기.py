
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람은 전체에서 차지하는 비율이 매우 낮습니다. "
    "이번 페이지에서는 두 집단의 가중치를 맞추기 전과 후에 "
    "분류 모델의 성능이 어떻게 달라지는지 비교합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

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
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

st.subheader("1. 모델에 입력할 속성 고르기")

selected_features = st.multiselect(
    "분류 모델이 사용할 속성을 선택하세요.",
    options=list(FEATURE_NAMES.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("산점도와 모델 학습을 위해 속성을 두 개 이상 골라 주세요.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[x] for x in selected_features)
)


# --------------------------------------------------
# 4. 데이터 나누기
#    채점표 페이지와 같은 방식
# --------------------------------------------------

df = df.copy()

# id를 숫자로 바꾸고 오름차순 정렬
df["_id_sort"] = pd.to_numeric(df["id"], errors="coerce")
df = df.sort_values("_id_sort").reset_index(drop=True)

# 10명씩 묶었을 때 앞의 3명은 테스트, 나머지 7명은 훈련
test_mask = (df.index % 10) < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

# 정답 데이터
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 선택한 속성만 사용
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()


# --------------------------------------------------
# 5. BMI 결측값 처리
#    훈련용 데이터의 중앙값으로 채우기
# --------------------------------------------------

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# 혹시 남아 있는 숫자형 결측값이 있다면 훈련용 중앙값으로 처리
for col in selected_features:
    if X_train[col].isna().any():
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)


# --------------------------------------------------
# 6. 표준화
#    훈련용 데이터로 기준을 만들고 테스트용에 적용
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. 가중치 조정 전후 모델 학습
# --------------------------------------------------

# 로지스틱 회귀
logistic_before = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight=None
)

logistic_after = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight="balanced"
)

# 의사결정트리
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight=None
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

# 학습
logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train_scaled, y_train)
tree_after.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 8. 예측
# --------------------------------------------------

def get_predictions(model):
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


logistic_before_pred, logistic_before_prob = get_predictions(
    logistic_before
)
logistic_after_pred, logistic_after_prob = get_predictions(
    logistic_after
)

tree_before_pred, tree_before_prob = get_predictions(tree_before)
tree_after_pred, tree_after_prob = get_predictions(tree_after)


# --------------------------------------------------
# 9. 혼동행렬과 성능 지표 계산
# --------------------------------------------------

def calculate_confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
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

    accuracy = (tp + tn) / total if total > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_number(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.3f}"


def make_comparison_table(before_counts, after_counts):
    before_metrics = calculate_metrics(before_counts)
    after_metrics = calculate_metrics(after_counts)

    rows = [
        ("TN · 정상으로 맞힌 사람", before_counts["TN"], after_counts["TN"]),
        ("FP · 헛짚은 사람", before_counts["FP"], after_counts["FP"]),
        ("FN · 놓친 환자", before_counts["FN"], after_counts["FN"]),
        ("TP · 찾아낸 환자", before_counts["TP"], after_counts["TP"]),
        ("정확도", format_number(before_metrics["정확도"]),
         format_number(after_metrics["정확도"])),
        ("재현율", format_number(before_metrics["재현율"]),
         format_number(after_metrics["재현율"])),
        ("정밀도", format_number(before_metrics["정밀도"]),
         format_number(after_metrics["정밀도"])),
        ("F1", format_number(before_metrics["F1"]),
         format_number(after_metrics["F1"]))
    ]

    return pd.DataFrame(
        rows,
        columns=["평가 항목", "가중치 조정 전", "가중치 조정 후"]
    )


# 혼동행렬 계산
logistic_before_counts = calculate_confusion(
    y_test, logistic_before_pred
)
logistic_after_counts = calculate_confusion(
    y_test, logistic_after_pred
)

tree_before_counts = calculate_confusion(
    y_test, tree_before_pred
)
tree_after_counts = calculate_confusion(
    y_test, tree_after_pred
)


# --------------------------------------------------
# 10. 테스트 데이터 정보
# --------------------------------------------------

st.divider()
st.subheader("2. 테스트 데이터")

actual_patients = int((y_test == 1).sum())
actual_non_patients = int((y_test == 0).sum())

col1, col2, col3 = st.columns(3)

col1.metric("테스트 인원", f"{len(y_test):,}명")
col2.metric("실제 뇌졸중 환자", f"{actual_patients:,}명")
col3.metric("실제 뇌졸중이 아닌 사람", f"{actual_non_patients:,}명")

st.caption(
    f"훈련 인원은 {len(y_train):,}명이며, "
    f"테스트 인원은 {len(y_test):,}명입니다."
)


# --------------------------------------------------
# 11. 모델별 가중치 조정 전후 비교
# --------------------------------------------------

st.divider()
st.subheader("3. 가중치 조정 전후 성능 비교")

st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")

logistic_table = make_comparison_table(
    logistic_before_counts,
    logistic_after_counts
)

st.dataframe(
    logistic_table,
    hide_index=True,
    use_container_width=True
)


def show_change_summary(before_counts, after_counts):
    before_found = before_counts["TP"]
    after_found = after_counts["TP"]

    before_missed = before_counts["FN"]
    after_missed = after_counts["FN"]

    before_false_alarm = before_counts["FP"]
    after_false_alarm = after_counts["FP"]

    st.write(
        f"**찾아낸 환자 수:** {before_found}명 → {after_found}명"
        f"　|　**놓친 환자 수:** {before_missed}명 → {after_missed}명"
        f"　|　**헛짚은 사람 수:** {before_false_alarm}명 → {after_false_alarm}명"
    )


show_change_summary(
    logistic_before_counts,
    logistic_after_counts
)

st.markdown("#### 의사결정트리(질문으로 답하는 모델)")

tree_table = make_comparison_table(
    tree_before_counts,
    tree_after_counts
)

st.dataframe(
    tree_table,
    hide_index=True,
    use_container_width=True
)

show_change_summary(
    tree_before_counts,
    tree_after_counts
)

st.caption(
    "TP는 실제 환자를 환자로 찾아낸 경우, FN은 환자를 놓친 경우입니다. "
    "FP는 환자가 아닌 사람을 환자로 잘못 판단한 경우입니다. "
    "재현율은 실제 환자 중 찾아낸 비율이고, "
    "정밀도는 환자라고 예측한 사람 중 실제 환자의 비율입니다."
)


# --------------------------------------------------
# 12. 산점도에 사용할 모델과 축 선택
# --------------------------------------------------

st.divider()
st.subheader("4. 가중치 조정 전후의 0.5 경계 비교")

model_options = {
    "로지스틱 회귀(확률로 답하는 모델)": (
        logistic_before,
        logistic_after
    ),
    "의사결정트리(질문으로 답하는 모델)": (
        tree_before,
        tree_after
    )
}

selected_model_name = st.selectbox(
    "경계선을 확인할 모델",
    list(model_options.keys())
)

before_model, after_model = model_options[selected_model_name]

st.write(
    "점의 색은 실제 뇌졸중 여부를 나타냅니다. "
    "두 경계는 각각 가중치 조정 전과 조정 후에 "
    "모델이 뇌졸중 확률 0.5라고 판단하는 위치입니다."
)


# --------------------------------------------------
# 13. 공통 함수: 원래 속성값을 모델 입력으로 변환
# --------------------------------------------------

# 훈련용 데이터의 중앙값을 각 속성의 대표값으로 사용
feature_medians = X_train.median().to_dict()


def make_scaled_grid(grid_df):
    """속성값 표를 모델에 넣을 수 있도록 표준화한다."""
    return scaler.transform(grid_df[selected_features])


def predict_grid_probability(model, grid_df):
    """격자점마다 뇌졸중일 확률을 계산한다."""
    scaled = make_scaled_grid(grid_df)
    return model.predict_proba(scaled)[:, 1]


# --------------------------------------------------
# 14. 속성이 두 개이면 2차원 산점도
# --------------------------------------------------

if len(selected_features) == 2:

    axis_col1, axis_col2 = st.columns(2)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    with axis_col2:
        y_feature = st.selectbox(
            "세로축",
            selected_features,
            index=1 if selected_features[1] != x_feature else 0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    if x_feature == y_feature:
        st.warning("가로축과 세로축은 서로 다른 속성을 선택해 주세요.")
        st.stop()

    # 테스트 데이터의 실제 값
    plot_df = test_df.copy()
    plot_df[x_feature] = X_test[x_feature].values
    plot_df[y_feature] = X_test[y_feature].values

    x_min = float(plot_df[x_feature].min())
    x_max = float(plot_df[x_feature].max())
    y_min = float(plot_df[y_feature].min())
    y_max = float(plot_df[y_feature].max())

    # 그림 범위를 조금 넓게 설정
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    if x_padding == 0:
        x_padding = 1
    if y_padding == 0:
        y_padding = 1

    x_range = np.linspace(
        x_min - x_padding,
        x_max + x_padding,
        80
    )

    y_range = np.linspace(
        y_min - y_padding,
        y_max + y_padding,
        80
    )

    xx, yy = np.meshgrid(x_range, y_range)

    grid_df = pd.DataFrame(
        [feature_medians] * xx.size
    )

    grid_df[x_feature] = xx.ravel()
    grid_df[y_feature] = yy.ravel()

    # 전후 확률 계산
    before_grid_prob = predict_grid_probability(
        before_model, grid_df
    ).reshape(xx.shape)

    after_grid_prob = predict_grid_probability(
        after_model, grid_df
    ).reshape(xx.shape)

    fig = go.Figure()

    # 실제 뇌졸중이 아닌 사람
    normal_mask = y_test.values == 0

    fig.add_trace(
        go.Scatter(
            x=plot_df.loc[normal_mask, x_feature],
            y=plot_df.loc[normal_mask, y_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3B82F6",
                size=7,
                opacity=0.65
            )
        )
    )

    # 실제 뇌졸중 환자
    patient_mask = y_test.values == 1

    fig.add_trace(
        go.Scatter(
            x=plot_df.loc[patient_mask, x_feature],
            y=plot_df.loc[patient_mask, y_feature],
            mode="markers",
            name="실제 뇌졸중 환자",
            marker=dict(
                color="#EF4444",
                size=8,
                opacity=0.8
            )
        )
    )

    # 경계선이 그림 범위 안에 있는지 확인
    before_has_boundary = (
        before_grid_prob.min() <= 0.5
        and before_grid_prob.max() >= 0.5
    )

    after_has_boundary = (
        after_grid_prob.min() <= 0.5
        and after_grid_prob.max() >= 0.5
    )

    # 가중치 조정 전 경계선
    if before_has_boundary:
        fig.add_trace(
            go.Contour(
                x=x_range,
                y=y_range,
                z=before_grid_prob,
                name="가중치 조정 전 0.5 경계",
                showscale=False,
                hoverinfo="skip",
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.01,
                    coloring="none"
                ),
                line=dict(
                    color="#F59E0B",
                    width=3,
                    dash="dash"
                )
            )
        )

    # 가중치 조정 후 경계선
    if after_has_boundary:
        fig.add_trace(
            go.Contour(
                x=x_range,
                y=y_range,
                z=after_grid_prob,
                name="가중치 조정 후 0.5 경계",
                showscale=False,
                hoverinfo="skip",
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.01,
                    coloring="none"
                ),
                line=dict(
                    color="#10B981",
                    width=3
                )
            )
        )

    fig.update_layout(
        title="실제 뇌졸중 여부와 0.5 경계선",
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if not before_has_boundary:
        st.info(
            "가중치 조정 전의 0.5 경계선은 "
            "현재 그림 범위 안에 없습니다."
        )

    if not after_has_boundary:
        st.info(
            "가중치 조정 후의 0.5 경계선은 "
            "현재 그림 범위 안에 없습니다."
        )

    st.caption(
        "경계선을 그릴 때 선택하지 않은 속성은 훈련용 데이터의 "
        "중앙값으로 고정했습니다."
    )


# --------------------------------------------------
# 15. 속성이 세 개 이상이면 3차원 산점도
# --------------------------------------------------

else:

    st.write("세 가지 속성을 골라 3차원으로 살펴봅니다.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축 (X)",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    with axis_col2:
        y_default = 1 if len(selected_features) > 1 else 0
        y_feature = st.selectbox(
            "세로축 (Y)",
            selected_features,
            index=y_default,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    with axis_col3:
        z_default = 2 if len(selected_features) > 2 else 0
        z_feature = st.selectbox(
            "높이축 (Z)",
            selected_features,
            index=z_default,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    if len({x_feature, y_feature, z_feature}) != 3:
        st.warning("3차원 산점도의 세 축은 서로 다른 속성으로 선택해 주세요.")
        st.stop()

    plot_df = test_df.copy()
    plot_df[x_feature] = X_test[x_feature].values
    plot_df[y_feature] = X_test[y_feature].values
    plot_df[z_feature] = X_test[z_feature].values

    # 축별 범위
    x_min, x_max = (
        float(plot_df[x_feature].min()),
        float(plot_df[x_feature].max())
    )
    y_min, y_max = (
        float(plot_df[y_feature].min()),
        float(plot_df[y_feature].max())
    )
    z_min, z_max = (
        float(plot_df[z_feature].min()),
        float(plot_df[z_feature].max())
    )

    # 경계면을 계산할 격자
    x_values = np.linspace(x_min, x_max, 24)
    y_values = np.linspace(y_min, y_max, 24)
    z_values = np.linspace(z_min, z_max, 40)

    # 각 x, y 위치에서 z 방향으로 확률을 계산
    grid_x, grid_y, grid_z = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij"
    )

    grid_df = pd.DataFrame(
        [feature_medians] * grid_x.size
    )

    grid_df[x_feature] = grid_x.ravel()
    grid_df[y_feature] = grid_y.ravel()
    grid_df[z_feature] = grid_z.ravel()

    before_prob_3d = predict_grid_probability(
        before_model, grid_df
    ).reshape(grid_x.shape)

    after_prob_3d = predict_grid_probability(
        after_model, grid_df
    ).reshape(grid_x.shape)

    # 각 x, y 위치에서 확률 0.5가 되는 z값을 보간
    def find_boundary_surface(probability_grid):
        boundary = np.full(
            (len(x_values), len(y_values)),
            np.nan
        )

        for i in range(len(x_values)):
            for j in range(len(y_values)):
                values = probability_grid[i, j, :] - 0.5

                crossings = np.where(
                    values[:-1] * values[1:] <= 0
                )[0]

                if len(crossings) == 0:
                    continue

                k = crossings[0]

                z1 = z_values[k]
                z2 = z_values[k + 1]

                p1 = values[k]
                p2 = values[k + 1]

                if p2 == p1:
                    boundary[i, j] = z1
                else:
                    boundary[i, j] = (
                        z1 + (0 - p1) * (z2 - z1) / (p2 - p1)
                    )

        return boundary

    before_surface = find_boundary_surface(before_prob_3d)
    after_surface = find_boundary_surface(after_prob_3d)

    before_has_surface = np.isfinite(before_surface).any()
    after_has_surface = np.isfinite(after_surface).any()

    fig = go.Figure()

    # 실제 뇌졸중이 아닌 사람
    normal_mask = y_test.values == 0

    fig.add_trace(
        go.Scatter3d(
            x=plot_df.loc[normal_mask, x_feature],
            y=plot_df.loc[normal_mask, y_feature],
            z=plot_df.loc[normal_mask, z_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3B82F6",
                size=3,
                opacity=0.65
            )
        )
    )

    # 실제 뇌졸중 환자
    patient_mask = y_test.values == 1

    fig.add_trace(
        go.Scatter3d(
            x=plot_df.loc[patient_mask, x_feature],
            y=plot_df.loc[patient_mask, y_feature],
            z=plot_df.loc[patient_mask, z_feature],
            mode="markers",
            name="실제 뇌졸중 환자",
            marker=dict(
                color="#EF4444",
                size=4,
                opacity=0.85
            )
        )
    )

    # 가중치 조정 전 0.5 경계면
    if before_has_surface:
        fig.add_trace(
            go.Surface(
                x=x_values,
                y=y_values,
                z=before_surface.T,
                name="가중치 조정 전 0.5 경계면",
                colorscale=[
                    [0, "#F59E0B"],
                    [1, "#F59E0B"]
                ],
                opacity=0.55,
                showscale=False,
                connectgaps=False,
                hoverinfo="skip"
            )
        )

    # 가중치 조정 후 0.5 경계면
    if after_has_surface:
        fig.add_trace(
            go.Surface(
                x=x_values,
                y=y_values,
                z=after_surface.T,
                name="가중치 조정 후 0.5 경계면",
                colorscale=[
                    [0, "#10B981"],
                    [1, "#10B981"]
                ],
                opacity=0.55,
                showscale=False,
                connectgaps=False,
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title="실제 뇌졸중 여부와 0.5 경계면",
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            zaxis_title=FEATURE_NAMES[z_feature]
        ),
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if not before_has_surface:
        st.info(
            "가중치 조정 전의 0.5 경계면은 "
            "현재 3차원 그림 범위에서 찾을 수 없습니다."
        )

    if not after_has_surface:
        st.info(
            "가중치 조정 후의 0.5 경계면은 "
            "현재 3차원 그림 범위에서 찾을 수 없습니다."
        )

    st.caption(
        "마우스로 그림을 드래그하면 3차원 공간을 돌려 볼 수 있습니다. "
        "선택하지 않은 속성은 훈련용 데이터의 중앙값으로 고정했습니다. "
        "경계면은 각 X·Y 위치에서 Z 방향으로 확률 0.5가 되는 지점을 "
        "계산해 표시합니다."
    )


# --------------------------------------------------
# 16. 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("생각해 보기")

st.write(
    """
    - 가중치를 맞춘 뒤 실제 환자를 더 많이 찾아냈나요?
    - 환자를 더 많이 찾아내는 대신 헛짚은 사람도 늘어났나요?
    - 정확도만 비교했을 때와 재현율·정밀도를 함께 비교했을 때
      어떤 차이가 있나요?
    - 뇌졸중처럼 한쪽 집단이 드문 문제에서는 어떤 지표를
      중요하게 살펴봐야 할까요?
    """
)

st.caption(
    "가중치 조정은 모델이 드문 집단의 사례도 학습 과정에서 "
    "중요하게 다루도록 돕는 방법입니다. "
    "가중치 조정 후의 성능이 모든 지표에서 반드시 좋아지는 것은 아닙니다."
)
