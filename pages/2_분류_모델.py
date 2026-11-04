
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🤖 2. 분류 모델")
st.caption("건강 정보를 이용하여 뇌졸중 발생 여부를 예측합니다.")

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

# 처음에는 BMI를 제외한 네 가지 선택
DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# ==================================================
# 4. 입력 속성 선택
# ==================================================

st.header("📋 1. 예측에 사용할 속성 고르기")

st.write(
    """
    뇌졸중 발생 여부를 예측할 때 사용할 속성을 선택하세요.
    처음에는 체질량지수를 제외한 네 가지가 선택되어 있습니다.
    """
)

selected_features = st.multiselect(
    "예측에 사용할 속성",
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
    사람의 번호(id) 순서로 정렬한 다음 열 명씩 묶습니다.

    - 각 묶음의 앞 세 명: 테스트용
    - 각 묶음의 나머지 일곱 명: 학습용

    테스트용 데이터는 모델을 학습하거나 크기 맞추기 기준을
    계산할 때 사용하지 않습니다.
    """
)

if "id" not in df.columns or "stroke" not in df.columns:
    st.error("데이터에 id 또는 stroke 열이 없습니다.")
    st.stop()

# 번호를 숫자로 변환하여 정렬
df = df.copy()
df["_id_sort"] = pd.to_numeric(df["id"], errors="coerce")
df = df.sort_values(
    "_id_sort",
    kind="stable",
    na_position="last"
).reset_index(drop=True)

# 10명씩 묶어서 앞의 3명만 테스트용으로 지정
test_indices = []

for start in range(0, len(df), 10):
    test_indices.extend(
        range(start, min(start + 3, len(df)))
    )

test_indices = set(test_indices)

test_mask = df.index.isin(test_indices)

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

# 학습용과 테스트용의 정답
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 선택한 속성만 사용
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# 숫자형으로 변환
for col in selected_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# BMI를 선택한 경우에만 학습용 중앙값으로 결측값 대체
bmi_median = None

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습용 BMI 데이터에 유효한 값이 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 다른 선택 속성에 결측값이 있다면 중단
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 BMI 이외의 빈 값이 있습니다.")
    st.stop()

if X_train.isin([np.inf, -np.inf]).any().any():
    st.error("학습용 데이터에 무한대 값이 있습니다.")
    st.stop()

if X_test.isin([np.inf, -np.inf]).any().any():
    st.error("테스트용 데이터에 무한대 값이 있습니다.")
    st.stop()

# 훈련용 데이터에 두 종류의 정답이 모두 있는지 확인
if y_train.nunique() < 2:
    st.error("학습용 데이터에 뇌졸중 있음과 없음이 모두 있어야 합니다.")
    st.stop()

if y_test.nunique() < 2:
    st.warning("테스트용 데이터에 한 종류의 정답만 있습니다.")

# 데이터 개수 카드
c1, c2, c3 = st.columns(3)

c1.metric("전체 사람 수", f"{len(df):,}명")
c2.metric("학습용 사람 수", f"{len(train_df):,}명")
c3.metric("테스트용 사람 수", f"{len(test_df):,}명")

if len(df) == 5110:
    if len(test_df) != 1533 or len(train_df) != 3577:
        st.error("데이터 분할 인원수가 예상과 다릅니다.")
        st.stop()

    st.success("테스트용 1,533명, 학습용 3,577명으로 분리했습니다.")

else:
    st.info(
        f"현재 데이터는 {len(df):,}명입니다. "
        f"규칙에 따라 테스트용 {len(test_df):,}명, "
        f"학습용 {len(train_df):,}명으로 분리했습니다."
    )

if bmi_median is not None:
    st.write(
        f"체질량지수 결측값 대체에 사용한 학습용 중앙값: "
        f"**{bmi_median:.2f}**"
    )

st.divider()


# ==================================================
# 6. 크기 맞추기
# ==================================================

st.header("📏 3. 크기 맞추기")

st.write(
    """
    로지스틱 회귀 모델은 속성의 크기에 영향을 받으므로
    표준화(평균 0, 표준편차 1)를 합니다.

    기준은 학습용 데이터에서만 계산합니다.
    테스트용 데이터에는 학습용에서 계산한 기준을 그대로 적용합니다.
    """
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success("학습용 데이터로 크기 맞추기를 완료했습니다.")


# ==================================================
# 7. 모델 만들기
# ==================================================

st.header("🧠 4. 분류 모델 만들기")

# 로지스틱 회귀
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=2000
)

logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

st.write(
    """
    **로지스틱 회귀(확률로 답하는 모델)**

    입력한 속성을 이용해 뇌졸중일 확률을 계산합니다.
    확률이 0.5 이상이면 뇌졸중 있음(1),
    0.5 미만이면 뇌졸중 없음(0)으로 판단합니다.
    """
)

st.write(
    """
    **의사결정트리(질문으로 답하는 모델)**

    속성에 관한 질문을 차례로 던져 답을 찾아갑니다.
    질문은 최대 세 번까지만 하고,
    마지막 마디에 사람이 다섯 명 미만이면 더 나누지 않습니다.
    """
)

st.divider()


# ==================================================
# 8. 정확도 계산
# ==================================================

st.header("🎯 5. 모델 정확도 비교")

# 훈련용과 테스트용 예측
logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

# 정확도
logistic_train_acc = (logistic_train_pred == y_train).mean()
logistic_test_acc = (logistic_test_pred == y_test).mean()

tree_train_acc = (tree_train_pred == y_train).mean()
tree_test_acc = (tree_test_pred == y_test).mean()

# 입력 속성을 하나도 보지 않는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = np.full(len(y_train), majority_class)
baseline_test_pred = np.full(len(y_test), majority_class)

baseline_train_acc = (baseline_train_pred == y_train).mean()
baseline_test_acc = (baseline_test_pred == y_test).mean()

st.write(
    """
    정확도는 모델이 예측한 결과와 실제 정답이 일치한 비율입니다.
    테스트 데이터의 정확도를 중심으로 세 모델을 비교합니다.
    """
)


def accuracy_card(title, train_acc, test_acc):
    with st.container(border=True):
        st.subheader(title)

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_acc * 100:.2f}%"
        )

        small1, small2 = st.columns(2)

        small1.caption(
            f"훈련 데이터 정확도\n\n{train_acc * 100:.2f}%"
        )

        small2.caption(
            f"테스트 데이터 정확도\n\n{test_acc * 100:.2f}%"
        )


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀",
        logistic_train_acc,
        logistic_test_acc
    )

with col2:
    accuracy_card(
        "의사결정트리",
        tree_train_acc,
        tree_test_acc
    )

with col3:
    accuracy_card(
        "아무것도 보지 않는 모델",
        baseline_train_acc,
        baseline_test_acc
    )

st.caption(
    f"아무것도 보지 않는 모델은 학습용 데이터에서 "
    f"더 많은 정답인 {majority_class}만 계속 예측합니다."
)

st.divider()


# ==================================================
# 9. 두 속성으로 예측 영역 살펴보기
# ==================================================

st.header("🗺️ 6. 두 속성으로 예측 영역 살펴보기")

st.write(
    """
    선택한 속성 가운데 두 개를 골라 가로축과 세로축으로 사용합니다.

    - 점: 테스트 데이터에 있는 실제 사람
    - 점의 색: 실제 뇌졸중 발생 여부
    - 선: 로지스틱 회귀 모델이 확률 0.5로 나누는 경계
    - 옅은 색 영역: 의사결정트리 모델이 나눈 예측 영역
    """
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:
    y_options = [x for x in selected_features if x != x_feature]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# --------------------------------------------------
# 9-1. 나머지 속성은 테스트 데이터 중앙값으로 고정
# --------------------------------------------------

st.subheader("나머지 속성을 고정하는 값")

fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[k]}: {v:.2f}"
        for k, v in fixed_values.items()
    )

    st.write(
        "가로축과 세로축이 아닌 속성은 "
        f"테스트 데이터의 중앙값에 고정합니다: **{fixed_text}**"
    )

else:
    st.write("선택한 속성이 두 개뿐이므로 고정할 나머지 속성이 없습니다.")


# --------------------------------------------------
# 9-2. 그래프용 격자 생성
# --------------------------------------------------

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_range = x_max - x_min
y_range = y_max - y_min

# 모든 값이 같을 때도 그래프를 그릴 수 있도록 처리
if x_range == 0:
    x_range = max(abs(x_min) * 0.1, 1)

if y_range == 0:
    y_range = max(abs(y_min) * 0.1, 1)

x_grid = np.linspace(
    x_min - x_range * 0.05,
    x_max + x_range * 0.05,
    150
)

y_grid = np.linspace(
    y_min - y_range * 0.05,
    y_max + y_range * 0.05,
    150
)

xx, yy = np.meshgrid(x_grid, y_grid)


# --------------------------------------------------
# 9-3. 격자에 나머지 속성의 중앙값 채우기
# --------------------------------------------------

grid_data = pd.DataFrame({
    feature: np.full(xx.size, float(X_test[feature].median()))
    for feature in selected_features
})

grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()

# 로지스틱 회귀 확률 계산
grid_scaled = scaler.transform(grid_data)

positive_index = list(logistic_model.classes_).index(1)

probability_grid = logistic_model.predict_proba(
    grid_scaled
)[:, positive_index].reshape(xx.shape)

# 의사결정트리 예측 영역 계산
tree_region = tree_model.predict(
    grid_data
).reshape(xx.shape)


# --------------------------------------------------
# 9-4. Plotly 그림 만들기
# --------------------------------------------------

fig = go.Figure()

# 의사결정트리 영역을 먼저 그리기
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=tree_region,
        name="의사결정트리 영역",
        showscale=False,
        opacity=0.22,
        contours=dict(
            start=-0.5,
            end=1.5,
            size=2,
            coloring="fill",
            showlines=False
        ),
        colorscale=[
            [0, "rgb(100, 160, 230)"],
            [1, "rgb(240, 120, 120)"]
        ],
        hoverinfo="skip"
    )
)

# 로지스틱 회귀의 확률 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=probability_grid,
        name="로지스틱 회귀 0.5 경계선",
        showscale=False,
        contours=dict(
            start=0.5,
            end=0.5,
            size=0.01,
            coloring="lines",
            showlabels=False
        ),
        line=dict(
            color="black",
            width=3
        ),
        hoverinfo="skip"
    )
)

# 실제 테스트 데이터의 점
test_plot = X_test.copy()
test_plot["실제 정답"] = y_test.to_numpy()

# 뇌졸중이 아닌 사람
negative = test_plot[test_plot["실제 정답"] == 0]

fig.add_trace(
    go.Scatter(
        x=negative[x_feature],
        y=negative[y_feature],
        mode="markers",
        name="실제 뇌졸중 없음 (0)",
        marker=dict(
            color="blue",
            size=7,
            opacity=0.7
        ),
        hovertemplate=(
            f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
            f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
            "실제 결과: 뇌졸중 없음<extra></extra>"
        )
    )
)

# 뇌졸중이 있는 사람
positive = test_plot[test_plot["실제 정답"] == 1]

fig.add_trace(
    go.Scatter(
        x=positive[x_feature],
        y=positive[y_feature],
        mode="markers",
        name="실제 뇌졸중 있음 (1)",
        marker=dict(
            color="red",
            size=8,
            opacity=0.85
        ),
        hovertemplate=(
            f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
            f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
            "실제 결과: 뇌졸중 있음<extra></extra>"
        )
    )
)

fig.update_layout(
    title="테스트 데이터와 두 모델의 예측 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 9-5. 경계선이 그림 안에 있는지 확인
# --------------------------------------------------

# 확률 0.5의 경계가 격자 영역을 통과하는지 확인
prob_min = float(probability_grid.min())
prob_max = float(probability_grid.max())

st.subheader("로지스틱 회귀의 경계선")

if prob_min <= 0.5 <= prob_max:
    st.success(
        "확률 0.5 경계선이 현재 그림 영역 안에 있습니다."
    )
else:
    st.warning(
        "확률 0.5 경계선이 현재 그림 영역 밖에 있습니다. "
        f"현재 그림 영역에서 계산한 확률 범위는 "
        f"{prob_min:.3f}~{prob_max:.3f}입니다."
    )

st.caption(
    "참고: 이 그림은 선택한 두 속성만 변화시키고 "
    "나머지 속성은 테스트 데이터의 중앙값으로 고정한 결과입니다."
)

st.divider()


# ==================================================
# 10. 의사결정트리 가지 그림
# ==================================================

st.header("🌳 7. 의사결정트리가 던진 질문")

st.write(
    """
    의사결정트리는 질문을 세 번까지만 던집니다.

    각 마디에는 그곳까지 도달한 훈련용 사람 수,
    실제 뇌졸중인 사람 수, 뇌졸중 비율을 표시합니다.
    """
)

tree_structure = tree_model.tree_

# 훈련용 데이터가 각 마디에 도달했는지 확인
train_decision_path = tree_model.decision_path(X_train)

node_samples = np.asarray(train_decision_path.sum(axis=0)).ravel()

# 실제 뇌졸중인 사람의 마디별 수 계산
positive_samples = np.zeros(tree_structure.node_count, dtype=int)

for node_id in range(tree_structure.node_count):
    reached = train_decision_path[:, node_id].toarray().ravel().astype(bool)
    positive_samples[node_id] = int(y_train.to_numpy()[reached].sum())


# --------------------------------------------------
# 10-1. DOT 문자열 만들기
# --------------------------------------------------

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11, margin="0.16,0.10"];',
    'edge [fontname="sans-serif", fontsize=10, arrowsize=0.7];'
]

leaf_nodes = []
negative_leaf_nodes = []
used_features = set()


def add_tree_node(node_id):
    left = tree_structure.children_left[node_id]
    right = tree_structure.children_right[node_id]

    total = int(node_samples[node_id])
    positive_count = int(positive_samples[node_id])
    ratio = positive_count / total * 100 if total > 0 else 0

    # 리프 노드인지 확인
    is_leaf = left == right

    if is_leaf:
        leaf_nodes.append(node_id)

        # 예측 결과
        prediction = int(tree_model.classes_[
            np.argmax(tree_structure.value[node_id][0])
        ])

        if prediction == 0:
            negative_leaf_nodes.append(node_id)
            answer = "최종 답: 뇌졸중 없음 (0)"
            fill_color = "#DCEBFA"
        else:
            answer = "최종 답: 뇌졸중 있음 (1)"
            fill_color = "#F9DADA"

        label = (
            f"마디 {node_id}\\n"
            f"훈련용 사람: {total}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {ratio:.1f}%\\n"
            f"{answer}"
        )

    else:
        feature_index = tree_structure.feature[node_id]
        feature = selected_features[feature_index]
        threshold = tree_structure.threshold[node_id]

        used_features.add(feature)

        label = (
            f"마디 {node_id}\\n"
            f"{FEATURE_NAMES[feature]} ≤ {threshold:.2f} ?\\n"
            f"훈련용 사람: {total}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {ratio:.1f}%"
        )

        fill_color = "#FFF1C9"

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
    )

    if not is_leaf:
        # 왼쪽 가지: 조건을 만족하는 예
        dot_lines.append(
            f'{node_id} -> {left} [label="예"];'
        )

        # 오른쪽 가지: 조건을 만족하지 않는 아니요
        dot_lines.append(
            f'{node_id} -> {right} [label="아니요"];'
        )

        add_tree_node(left)
        add_tree_node(right)


add_tree_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# --------------------------------------------------
# 10-2. 가지 그림 표시
# --------------------------------------------------

st.subheader("의사결정트리 구조")

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 10-3. 나무 요약
# --------------------------------------------------

st.subheader("나무 요약")

st.write(
    f"**답을 내는 마디(리프 노드)는 모두 {len(leaf_nodes)}칸입니다.**"
)

st.write(
    f"그중 **뇌졸중 없음(0)**이라고 답하는 마디는 "
    f"**{len(negative_leaf_nodes)}칸**입니다."
)

if used_features:
    used_feature_names = [
        FEATURE_NAMES[x]
        for x in selected_features
        if x in used_features
    ]

    st.write(
        "**이 나무가 실제로 질문한 속성:** "
        + ", ".join(used_feature_names)
    )
else:
    st.write("이 나무는 질문을 만들지 않고 바로 답을 내렸습니다.")

st.divider()


# ==================================================
# 11. 마무리
# ==================================================

st.header("📝 학습 정리")

st.write(
    """
    이번 페이지에서 살펴본 내용을 정리해 보세요.

    1. 입력 속성을 바꾸면 정확도는 어떻게 달라질까요?
    2. 로지스틱 회귀와 의사결정트리 중 테스트 정확도가 더 높은 모델은 무엇인가요?
    3. 입력을 보지 않는 기준 모델보다 두 분류 모델의 정확도가 높은가요?
    4. 의사결정트리는 어떤 속성을 이용해 질문을 만들었나요?
    5. 훈련 정확도와 테스트 정확도에 차이가 있다면 그 이유는 무엇일까요?
    """
)

st.caption("뇌졸중 예측 실습실 | 분류 모델")
