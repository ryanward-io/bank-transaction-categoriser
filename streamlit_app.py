"""Bank transaction categoriser.

Learns spending categories (Food, Myki, Supermarket, ...) from the text of a bank transaction
description. Runs entirely in the browser via stlite (Streamlit on Pyodide); see index.html.

Rebuilt in 2026 as a scikit-learn pipeline with an honest evaluation and an interactive front end.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Bank transaction categoriser", layout="centered")

VECTORISERS = {
    "Word counts": lambda: CountVectorizer(token_pattern=r"[a-z0-9]+"),
    "Word TF-IDF": lambda: TfidfVectorizer(token_pattern=r"[a-z0-9]+"),
    "Character TF-IDF": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
}
MODELS = {
    "Logistic regression": lambda: OneVsRestClassifier(LogisticRegression(C=10, solver="liblinear")),
    "Multinomial naive Bayes": lambda: MultinomialNB(),
    "Complement naive Bayes": lambda: ComplementNB(),
    "Random forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "Decision tree": lambda: DecisionTreeClassifier(random_state=0),
}
EXAMPLES = [
    "WOOLWORTHS 3201",
    "PAYPAL *UBER AU",
    "NETFLIX.COM",
    "BP CONNECT BENDIGO",
    "Interest Credit",
    "MYKI TOP UP SOUTHERN CROSS",
    "THE LITTLE CAFE FITZROY",
    "Powershop - Direct Debit",
]
SEED = 0


@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")


def build(vec_name, model_name):
    return make_pipeline(VECTORISERS[vec_name](), MODELS[model_name]())


@st.cache_resource
def fit_full(vec_name, model_name):
    df = load_data()
    return build(vec_name, model_name).fit(df["description"], df["category"])


@st.cache_data
def honest_cv(vec_name, model_name):
    """5-fold cross-validation over *distinct* descriptions, so no description is in both train and test."""
    df = load_data()
    cv = KFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(build(vec_name, model_name), df["description"], df["category"], cv=cv)
    return pred


@st.cache_data
def leaky_accuracy(vec_name, model_name):
    """Score a random split of the raw rows, where repeats of a description land on both sides (shown only for comparison)."""
    df = load_data()
    rows = df.loc[df.index.repeat(df["count"])].reset_index(drop=True)
    scores = []
    for seed in range(5):
        x_tr, x_te, y_tr, y_te = train_test_split(
            rows["description"], rows["category"], test_size=0.2, random_state=seed
        )
        scores.append(accuracy_score(y_te, build(vec_name, model_name).fit(x_tr, y_tr).predict(x_te)))
    return float(np.mean(scores)), len(rows)


@st.cache_data
def compare_all():
    df = load_data()
    cv = KFold(5, shuffle=True, random_state=SEED)
    out = []
    for vec_name in VECTORISERS:
        for model_name in MODELS:
            pred = cross_val_predict(build(vec_name, model_name), df["description"], df["category"], cv=cv)
            out.append(
                {
                    "Features": vec_name,
                    "Model": model_name,
                    "Accuracy": accuracy_score(df["category"], pred),
                    "Macro F1": f1_score(df["category"], pred, average="macro", zero_division=0),
                }
            )
    return pd.DataFrame(out)


data = load_data()
n_distinct = len(data)
n_rows = int(data["count"].sum())
majority = data["category"].value_counts().idxmax()
baseline = float((data["category"] == majority).mean())

st.title("Bank transaction categoriser")
st.write(
    "Can a model learn your spending categories from nothing but the text on a bank statement? "
    "Type a transaction below and see which category it picks, then look at how well it does."
)
st.caption(
    f"Trained on {n_rows} labelled transactions ({n_distinct} distinct descriptions) from my own account. "
    "Person-to-person payments, account numbers, names and dates were removed, and the employer is generic."
)

st.header("1. Try it")
col_a, col_b = st.columns(2)
vec_name = col_a.selectbox("Features", list(VECTORISERS), index=1)
model_name = col_b.selectbox("Model", list(MODELS), index=0)

if "txn" not in st.session_state:
    st.session_state["txn"] = EXAMPLES[0]


def pick(example):
    st.session_state["txn"] = example


st.text_input("Transaction description", key="txn")
st.caption("Or try an example:")
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    cols[i % 4].button(ex, key=f"ex{i}", on_click=pick, args=(ex,), use_container_width=True)

model = fit_full(vec_name, model_name)
text = st.session_state["txn"].strip()
if text:
    proba = model.predict_proba([text])[0]
    top = pd.DataFrame({"Category": model.classes_, "Probability": proba}).sort_values(
        "Probability", ascending=False
    )
    best = top.iloc[0]
    st.subheader(f"{best['Category']}  ({best['Probability']:.0%})")
    if best["Probability"] < 0.5:
        st.warning("Low confidence. The model has not seen anything much like this.")
    chart = (
        alt.Chart(top.head(5))
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Category:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=["Category", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=alt.Step(28))
    )
    st.altair_chart(chart, use_container_width=True)
    if model_name == "Complement naive Bayes":
        st.caption("Complement naive Bayes scores are not well calibrated probabilities; read them as a ranking.")

st.header("2. How well does it work?")
pred = honest_cv(vec_name, model_name)
acc = accuracy_score(data["category"], pred)
macro = f1_score(data["category"], pred, average="macro", zero_division=0)
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{acc:.0%}")
m2.metric("Macro F1", f"{macro:.2f}")
m3.metric(f"Always guess '{majority}'", f"{baseline:.0%}")
st.write(
    "Accuracy is measured with 5-fold cross-validation over the distinct descriptions: the model is always scored on "
    "descriptions it has never seen. The last number is what you get by never thinking at all and always picking "
    f"the most common category, so that is the bar to beat. Macro F1 averages every category equally, so it is "
    "much lower: the rare categories (Petrol, Spotify, Swimming) have only a handful of examples."
)

with st.expander("Confusion matrix and per-category scores"):
    labels = sorted(data["category"].unique())
    cm = confusion_matrix(data["category"], pred, labels=labels)
    cm_df = pd.DataFrame(
        [(a, b, int(cm[i, j])) for i, a in enumerate(labels) for j, b in enumerate(labels)],
        columns=["Actual", "Predicted", "Count"],
    )
    base = alt.Chart(cm_df).encode(
        x=alt.X("Predicted:N", sort=labels),
        y=alt.Y("Actual:N", sort=labels, axis=alt.Axis(labelOverlap=False)),
    )
    heat = base.mark_rect().encode(
        color=alt.condition(
            alt.datum.Count > 0, alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None), alt.value("white")
        )
    )
    nums = base.mark_text(fontSize=10).encode(
        text=alt.condition(alt.datum.Count > 0, "Count:Q", alt.value("")),
        color=alt.condition(alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")),
    )
    st.altair_chart((heat + nums).properties(height=alt.Step(24)), use_container_width=True)
    report = pd.DataFrame(
        classification_report(data["category"], pred, output_dict=True, zero_division=0)
    ).T.loc[labels]
    report = report.rename(columns={"support": "Distinct descriptions"})
    report.index.name = "Category"
    report["Distinct descriptions"] = report["Distinct descriptions"].astype(int)
    st.dataframe(report.round(2))

with st.expander("Why score on unseen descriptions?"):
    leaky, n_leaky = leaky_accuracy(vec_name, model_name)
    st.write(
        f"The {n_rows} transactions contain only {n_distinct} distinct descriptions ('Powershop - Direct Debit' "
        "alone appears 49 times). A random split of the raw rows would put copies of the same description in both "
        f"train and test and overstate the result: about {leaky:.0%} for this model, against {acc:.0%} on "
        "descriptions it has never seen."
    )

st.header("3. Compare every combination")
if st.button("Run all 15 combinations"):
    with st.spinner("Fitting 75 models..."):
        results = compare_all()
    pivot = results.pivot(index="Model", columns="Features", values="Accuracy")
    st.write("Accuracy on unseen descriptions")
    st.dataframe(pivot.style.format("{:.1%}").highlight_max(axis=None, color="#bfdbfe"))
    pivot_f1 = results.pivot(index="Model", columns="Features", values="Macro F1")
    st.write("Macro F1")
    st.dataframe(pivot_f1.style.format("{:.2f}").highlight_max(axis=None, color="#bfdbfe"))
    st.write(
        "On a dataset this small the differences between the better models are within noise. "
        "Multinomial naive Bayes does badly on macro F1 because it leans towards the biggest category."
    )

with st.expander("About the data"):
    counts = (
        data.groupby("category")
        .agg(Descriptions=("description", "count"), Transactions=("count", "sum"))
        .sort_values("Descriptions", ascending=False)
    )
    st.dataframe(counts)
    st.write(
        "'Food' is a catch-all in my own labels (shops, tickets and taxis are in there too), and some "
        "descriptions are labelled inconsistently, so a perfect score is not possible. "
        "Descriptions come from card purchases, direct debits and bank fees. See the README for how the data was anonymised."
    )
    st.dataframe(data.sort_values("count", ascending=False).head(30), hide_index=True)
