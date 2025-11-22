# app.py — All-in-one enhanced Streamlit app for Predictive Analytics Hackathon
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import shap
import time
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

st.set_page_config(page_title="Agricultural Yield Optimizer", layout="wide", initial_sidebar_state="expanded")

# ---------- Constants ----------
DEFAULT_CSV_PATH = "Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv"
# the uploaded docx path (provided). Use as link or resource.
HACKATHON_DOC_PATH = "/mnt/data/Predictive Analytics Hackathon.docx"

# ---------- Utility functions ----------
@st.cache_data
def load_default_df(path=DEFAULT_CSV_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # median impute the known columns
    for col in ['soil_N', 'rainfall_mm', 'fertilizer_kg_per_ha']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # season mapping
    if "month" in df.columns:
        df["season"] = df["month"].map({
            12:"Winter",1:"Winter",2:"Winter",
            3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",
            9:"Autumn",10:"Autumn",11:"Autumn"
        })
    # encode crop_type
    if "crop_type" in df.columns:
        le = LabelEncoder()
        df["crop_type_enc"] = le.fit_transform(df["crop_type"])
        return df, le
    return df, None

def cost_model(fert, irr, pest):
    # simple cost model, can be adjusted to use dataset's input_cost_total if available
    # unit costs (₹): fertilizer per kg = 12, irrigation per mm = 3, pesticide per ml = 8
    return fert*12 + irr*3 + pest*8

def env_model(fert, irr, pest):
    # simple environmental score
    return fert*2 + irr*1.5 + pest*3

def create_download_link_df(df, filename="optimized.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

def generate_pdf_report(summary_text, charts_bytes, filename="report.pdf"):
    # charts_bytes is a list of (name, png_bytes)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Agricultural Yield Optimization Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", '', 11)
    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 6, line)
    pdf.ln(6)
    # add charts
    for (name, png) in charts_bytes:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, name, ln=True)
        # write png bytes to temp and insert
        tmp = io.BytesIO(png)
        tmp.seek(0)
        pdf.image(tmp, x=10, y=30, w=180)
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ---------- Load data (default) ----------
st.sidebar.title("Project files & dataset")
st.sidebar.markdown(f"- Hackathon brief: [{HACKATHON_DOC_PATH}]({HACKATHON_DOC_PATH})")
st.sidebar.markdown("Upload CSV (optional) — otherwise the default dataset is used.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])
    for col in ['soil_N', 'rainfall_mm', 'fertilizer_kg_per_ha']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "month" in df.columns:
        df["season"] = df["month"].map({
            12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"
        })
    if "crop_type" in df.columns:
        le = LabelEncoder()
        df["crop_type_enc"] = le.fit_transform(df["crop_type"])
    st.sidebar.success("Custom dataset loaded.")
else:
    df, label_encoder = load_default_df()
    st.sidebar.info(f"Using default dataset: {DEFAULT_CSV_PATH}")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview & EDA",
    "Train & Compare Models",
    "Model Explainability (SHAP)",
    "Personalized Optimizer",
    "Multi-Scenario Optimization",
    "Download Reports & CSV"
])

# ---------- Overview & EDA ----------
if page == "Overview & EDA":
    st.title("Dataset Overview & EDA")
    st.write("Basic dataset preview and summary statistics.")
    st.dataframe(df.head(200))
    st.write("### Summary statistics")
    st.write(df.describe())

    # Missing values
    st.write("### Missing Values")
    na = df.isna().sum()
    st.table(na[na>0])

    # Correlation heatmap (numeric only)
    st.write("### Correlation Heatmap (numeric columns)")
    numeric = df.select_dtypes(include=['float64','int64'])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # yield distribution and boxplots
    if "yield_kg_per_ha" in df.columns:
        st.write("### Yield distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["yield_kg_per_ha"], kde=True, ax=ax)
        st.pyplot(fig)

    # yield by crop
    if "crop_type" in df.columns:
        st.write("### Yield by Crop Type")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=df, x="crop_type", y="yield_kg_per_ha", ax=ax, ci="sd")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # input vs yield scatter plots
    st.write("### Inputs vs Yield")
    cols = ["fertilizer_kg_per_ha","irrigation_mm","pesticide_ml","rainfall_mm"]
    for c in cols:
        if c in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=c, y="yield_kg_per_ha", ax=ax)
            st.pyplot(fig)

# ---------- Train & Compare Models ----------
elif page == "Train & Compare Models":
    st.title("Train & Compare Models")

    # features selection (safe default)
    features = [c for c in [
        'soil_pH','soil_N','soil_P','rainfall_mm','temp_avg',
        'fertilizer_kg_per_ha','irrigation_mm','pesticide_ml',
        'month','day_of_year','crop_type_enc'
    ] if c in df.columns]

    X = df[features]
    y = df["yield_kg_per_ha"]

    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write("Models to train:")
    use_gbr = st.checkbox("GradientBoostingRegressor (good baseline)", value=True)
    use_rf = st.checkbox("RandomForestRegressor", value=True)
    use_et = st.checkbox("ExtraTreesRegressor", value=True)

    if st.button("Train selected models"):
        models = {}
        if use_gbr:
            models["GBR"] = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=42)
        if use_rf:
            models["RF"] = RandomForestRegressor(n_estimators=300, random_state=42)
        if use_et:
            models["ET"] = ExtraTreesRegressor(n_estimators=300, random_state=42)

        results = {}
        for name, m in models.items():
            t0 = time.time()
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results[name] = {"model": m, "r2": r2, "rmse": rmse, "time_s": time.time()-t0}
            st.write(f"**{name}** — R²: {r2:.4f} — RMSE: {rmse:.2f} — train_time: {results[name]['time_s']:.1f}s")

        # pick best by r2
        best_name = max(results.items(), key=lambda x: x[1]["r2"])[0]
        st.success(f"Best model: {best_name}")
        st.session_state["models"] = {k:v["model"] for k,v in results.items()}
        st.session_state["best_model_name"] = best_name
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["features"] = features

# ---------- SHAP Explainability ----------
elif page == "Model Explainability (SHAP)":
    st.title("Model Explainability with SHAP")
    if "models" not in st.session_state:
        st.warning("Train models first on the 'Train & Compare Models' page.")
    else:
        models = st.session_state["models"]
        best = st.session_state["best_model_name"]
        st.write(f"Using best model: {best}")
        model = models[best]
        X_train = st.session_state["X_train"]

        # compute SHAP once
        with st.spinner("Computing SHAP values (TreeExplainer)..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)

        st.subheader("SHAP summary plot (global importance)")
        fig = shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(bbox_inches='tight')

        st.subheader("Top features by importance (built-in)")
        fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(fi)

        st.subheader("SHAP dependence plot (choose feature)")
        feat = st.selectbox("Feature", X_train.columns)
        fig2 = shap.dependence_plot(feat, shap_values, X_train, show=False)
        st.pyplot(bbox_inches='tight')

# ---------- Personalized Optimizer ----------
elif page == "Personalized Optimizer":
    st.title("Personalized Input Optimizer")

    st.markdown("Enter the plot-specific values (or use dataset medians):")
    # user inputs with defaults from dataset medians
    soil_pH = st.number_input("soil_pH", value=float(df["soil_pH"].median()))
    soil_N = st.number_input("soil_N (kg/ha)", value=float(df["soil_N"].median()))
    soil_P = st.number_input("soil_P", value=float(df["soil_P"].median()))
    rainfall = st.number_input("rainfall_mm", value=float(df["rainfall_mm"].median()))
    temp_avg = st.number_input("temp_avg", value=float(df["temp_avg"].median()))
    month = st.slider("month", 1, 12, 6)
    day_of_year = st.slider("day_of_year", 1, 365, 150)

    crop_options = list(df["crop_type"].unique()) if "crop_type" in df.columns else ["Crop"]
    crop_choice = st.selectbox("Crop type", crop_options)
    if "crop_type" in df.columns and "crop_type_enc" in df.columns:
        crop_enc = int(df.loc[df["crop_type"]==crop_choice, "crop_type_enc"].mode()[0])
    else:
        crop_enc = 0

    st.write("Optimization preferences:")
    budget_limit = st.number_input("Budget limit (₹ per hectare)", value=12000)
    env_limit = st.number_input("Environmental score limit", value=10000)

    # crop-specific ranges (can be tuned)
    crop_ranges = {
        "Wheat": {"fert": (50,300,10), "irr": (0,200,10), "pest":(0,200,10)},
        "Maize": {"fert": (50,350,10), "irr": (0,250,10), "pest":(0,200,10)},
        "Barley": {"fert": (30,250,10), "irr": (0,180,10), "pest":(0,150,10)},
        "Rice": {"fert": (80,350,10), "irr": (50,400,10), "pest":(0,200,10)},
    }
    # default range if crop not in key
    r = crop_ranges.get(crop_choice, {"fert":(0,300,10),"irr":(0,400,20),"pest":(0,200,10)})

    scenario = st.radio("Optimization scenario", ("Yield Priority","Cost Priority","Environment Priority"))

    if st.button("Find optimal inputs (personalized)"):
        if "models" not in st.session_state:
            st.warning("Train models first on 'Train & Compare Models'.")
        else:
            best_model = st.session_state["models"][st.session_state["best_model_name"]]
            best_val = -1
            best_combo = None
            fert_start, fert_end, fert_step = r["fert"]
            irr_start, irr_end, irr_step = r["irr"]
            pest_start, pest_end, pest_step = r["pest"]

            # grid search
            for fert in range(fert_start, fert_end+1, fert_step):
                for irr in range(irr_start, irr_end+1, irr_step):
                    for pest in range(pest_start, pest_end+1, pest_step):
                        cost = cost_model(fert, irr, pest)
                        if cost > budget_limit: continue
                        env = env_model(fert, irr, pest)
                        if env > env_limit: continue

                        row = pd.DataFrame([[
                            soil_pH, soil_N, soil_P, rainfall, temp_avg,
                            fert, irr, pest,
                            month, day_of_year, crop_enc
                        ]], columns=st.session_state["features"])

                        pred = best_model.predict(row)[0]

                        # scenario weighting
                        if scenario == "Yield Priority":
                            score = pred
                        elif scenario == "Cost Priority":
                            score = pred - 0.001*cost
                        else: # Environment Priority
                            score = pred - 0.01*env

                        if score > best_val:
                            best_val = score
                            best_combo = {"fert":fert, "irr":irr, "pest":pest, "cost":cost, "env":env, "predicted_yield":pred}

            if best_combo is None:
                st.error("No feasible combination found with current limits.")
            else:
                st.success("Found optimal inputs!")
                st.write(f"**Fertilizer:** {best_combo['fert']} kg/ha")
                st.write(f"**Irrigation:** {best_combo['irr']} mm")
                st.write(f"**Pesticide:** {best_combo['pest']} ml")
                st.write(f"**Total Cost:** ₹{best_combo['cost']}")
                st.write(f"**Environmental Score:** {best_combo['env']}")
                st.write(f"**Predicted Yield:** {best_combo['predicted_yield']:.2f} kg/ha")

                # small insight generator using SHAP if available
                try:
                    model = best_model
                    X_train = st.session_state["X_train"]
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)
                    # basic summary statement
                    top_feats = np.array(X_train.columns)[np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:3]]
                    insight = (f"Top features influencing yield (global): {', '.join(top_feats)}. "
                               f"Model suggests fertilizer={best_combo['fert']} and irrigation={best_combo['irr']} are best under your constraints.")
                    st.info(insight)
                except Exception as e:
                    st.info("Optimization completed. (SHAP insight not available.)")

                # store result for CSV/PDF downloads
                st.session_state["last_opt"] = best_combo

# ---------- Multi-Scenario Optimization ----------
elif page == "Multi-Scenario Optimization":
    st.title("Multi-Scenario Optimization Comparison")
    st.write("Run three scenarios and compare results: Yield, Cost, Environment priorities.")

    if st.button("Run multi-scenario (auto)"):
        if "models" not in st.session_state:
            st.warning("Train models first on 'Train & Compare Models'.")
        else:
            # use dataset medians
            soil_pH = df["soil_pH"].median()
            soil_N = df["soil_N"].median()
            soil_P = df["soil_P"].median()
            rainfall = df["rainfall_mm"].median()
            temp_avg = df["temp_avg"].median()
            month = int(df["month"].median())
            day_of_year = int(df["day_of_year"].median())
            crop_enc = int(df["crop_type_enc"].mode()[0]) if "crop_type_enc" in df.columns else 0

            best_model = st.session_state["models"][st.session_state["best_model_name"]]
            scenarios = ["Yield Priority","Cost Priority","Environment Priority"]
            all_results = []

            # default search ranges
            fert_range = range(0,301,10)
            irr_range = range(0,401,20)
            pest_range = range(0,201,10)

            for scenario in scenarios:
                best_val = -1; best_combo = None
                for fert in fert_range:
                    for irr in irr_range:
                        for pest in pest_range:
                            cost = cost_model(fert, irr, pest)
                            if cost > 12000: continue
                            env = env_model(fert, irr, pest)
                            if env > 10000: continue
                            row = pd.DataFrame([[
                                soil_pH, soil_N, soil_P, rainfall, temp_avg,
                                fert, irr, pest, month, day_of_year, crop_enc
                            ]], columns=st.session_state["features"])
                            pred = best_model.predict(row)[0]
                            if scenario == "Yield Priority":
                                score = pred
                            elif scenario == "Cost Priority":
                                score = pred - 0.001*cost
                            else:
                                score = pred - 0.01*env
                            if score > best_val:
                                best_val = score
                                best_combo = {"scenario":scenario, "fert":fert, "irr":irr, "pest":pest, "cost":cost, "env":env, "predicted_yield":pred}
                all_results.append(best_combo)

            df_results = pd.DataFrame(all_results)
            st.dataframe(df_results)
            st.session_state["multi_results"] = df_results
            st.success("Scenarios completed!")

# ---------- Download reports & CSV ----------
elif page == "Download Reports & CSV":
    st.title("Download Results & PDF Report")
    st.write("You can download the latest optimization result as CSV and a PDF report.")

    if "last_opt" in st.session_state:
        res = st.session_state["last_opt"]
        df_out = pd.DataFrame([res])
        csv_href = create_download_link_df(df_out, "optimal_inputs.csv")
        st.markdown(f"[Download optimized CSV](data:file/csv;base64,{base64.b64encode(df_out.to_csv(index=False).encode()).decode()})", unsafe_allow_html=True)

        st.write("Generate PDF report including a few charts and summary.")
        # create a small set of charts
        charts = []
        # correlation chart
        fig, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(df.select_dtypes(include=['float64','int64']).corr(), ax=ax)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        charts.append(("Correlation Heatmap", buf.getvalue()))
        plt.close(fig)

        # yield distribution
        fig, ax = plt.subplots(); sns.histplot(df["yield_kg_per_ha"], kde=True, ax=ax)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        charts.append(("Yield Distribution", buf.getvalue()))
        plt.close(fig)

        summary_lines = f"""Optimization Summary:
Optimal inputs found:
Fertilizer: {res['fert']} kg/ha
Irrigation: {res['irr']} mm
Pesticide: {res['pest']} ml
Total cost: ₹{res['cost']}
Environmental score: {res['env']}
Predicted yield: {res['predicted_yield']:.2f} kg/ha

Top features (from model): {', '.join(st.session_state['features'][:3])}
"""
        pdf_bytes = generate_pdf_report(summary_lines, charts)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="optimization_report.pdf", mime="application/pdf")

    else:
        st.info("Run an optimization first (Personalized Optimizer or Multi-Scenario) to generate downloadable outputs.")

# End of app
