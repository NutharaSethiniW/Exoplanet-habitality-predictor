
# app.py | Exoplanet Habitability Predictor
# Streamlit Community Cloud deployment

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import requests
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Exoplanet Habitability Predictor",
    layout="wide"
)


@st.cache_resource
def load_models():
    xgb        = joblib.load("models/xgboost.pkl")
    esi_reg    = joblib.load("models/esi_regressor.pkl")
    explainer  = joblib.load("models/shap_explainer.pkl")
    threshold  = pd.read_csv("data/processed/chosen_threshold.csv")
    threshold  = float(threshold["chosen_threshold"].iloc[0])
    return xgb, esi_reg, explainer, threshold

@st.cache_data
def load_data():
    df     = pd.read_csv("data/processed/dataset_features.csv")
    scores = pd.read_csv("data/processed/model_scores.csv")
    return df, scores

xgb, esi_reg, explainer, THRESHOLD = load_models()
df, model_scores = load_data()

FEATURE_COLS = [
    "pl_orbper", "pl_insol", "pl_eqt", "pl_rade", "pl_bmasse",
    "pl_orbeccen", "st_teff", "st_met",
    "hz_flux_ratio", "in_habitable_zone", "density_proxy",
    "temp_delta_earth", "star_G", "star_Hot", "star_K", "star_M",
]


def engineer_features(orbper, insol, eqt, rade, bmasse,
                       eccen, st_teff, st_met):
    hz_flux_ratio    = insol / 1.0
    in_hz            = int(0.25 <= insol <= 1.5)
    density_proxy    = bmasse / (rade ** 3) if rade > 0 else 0
    temp_delta_earth = abs(eqt - 255.0)

    teff = st_teff
    star_G   = int(5300 <= teff < 6000)
    star_Hot = int(teff >= 7300)
    star_K   = int(3900 <= teff < 5300)
    star_M   = int(teff < 3900)

    return pd.DataFrame([{
        "pl_orbper": orbper, "pl_insol": insol, "pl_eqt": eqt,
        "pl_rade": rade, "pl_bmasse": bmasse, "pl_orbeccen": eccen,
        "st_teff": st_teff, "st_met": st_met,
        "hz_flux_ratio": hz_flux_ratio,
        "in_habitable_zone": in_hz,
        "density_proxy": density_proxy,
        "temp_delta_earth": temp_delta_earth,
        "star_G": star_G, "star_Hot": star_Hot,
        "star_K": star_K, "star_M": star_M,
    }])


def predict(X_input):
    prob     = xgb.predict_proba(X_input)[0][1]
    habitable = int(prob >= THRESHOLD)
    esi      = float(esi_reg.predict(X_input)[0])
    esi      = max(0.0, min(1.0, esi))
    return prob, habitable, esi


def shap_chart(X_input):
    shap_vals = explainer.shap_values(X_input)[0]
    feat_names = FEATURE_COLS
    order = np.argsort(np.abs(shap_vals))[::-1][:8]
    names  = [feat_names[i] for i in order]
    values = [shap_vals[i] for i in order]
    colors = ["steelblue" if v > 0 else "salmon" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value (positive = pushes toward habitable)")
    ax.set_title("Feature Contributions to This Prediction")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def hz_chart(insol, eqt, planet_name="Input Planet"):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Known reference points [insol, temp, name]
    refs = [
        (1.0,   255, "Earth", "green"),
        (1.91,  310, "Venus", "orange"),
        (0.43,  210, "Mars",  "red"),
    ]
    for r_insol, r_temp, r_name, r_color in refs:
        ax.scatter(r_insol, r_temp, s=120,
                   color=r_color, zorder=5, label=r_name)
        ax.annotate(r_name, (r_insol, r_temp),
                    textcoords="offset points", xytext=(8, 4))

    # Input planet
    ax.scatter(insol, eqt, s=200, color="blue",
               marker="*", zorder=6, label=planet_name)
    ax.annotate(planet_name, (insol, eqt),
                textcoords="offset points", xytext=(8, 4),
                color="blue", fontweight="bold")

    # Habitable zone shading
    ax.axvspan(0.25, 1.5, alpha=0.1, color="green", label="Habitable Zone (flux)")
    ax.axhspan(200, 320, alpha=0.1, color="blue", label="Habitable Zone (temp)")

    ax.set_xlabel("Stellar Flux (Earth = 1.0)")
    ax.set_ylabel("Equilibrium Temperature (K)")
    ax.set_title("Habitable Zone Comparison")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# NASA TAP live lookup
def fetch_planet(name):
    cols = "pl_name,pl_orbper,pl_insol,pl_eqt,pl_rade,pl_bmasse,pl_orbeccen,st_teff,st_met"
    url  = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query=SELECT+{cols}+FROM+pscomppars"
        f"+WHERE+pl_name+like+'{name}'"
        "&format=csv"
    )
    try:
        r = requests.get(url, timeout=15)
        data = pd.read_csv(pd.io.common.StringIO(r.text))
        if len(data) > 0:
            return data.iloc[0], None
        return None, "Planet not found in NASA archive."
    except Exception as e:
        return None, str(e)


st.title(" Exoplanet Habitability Predictor")
st.markdown(
    "Predict whether a confirmed exoplanet is potentially habitable "
    "and estimate its **Earth Similarity Index (ESI)** using a trained "
    "XGBoost model on NASA + PHL data."
)

tab1, tab2, tab3 = st.tabs([
    " Manual Input",
    " Live NASA Lookup",
    " Model Performance"
])

# TAB 1: Manual Input 
with tab1:
    st.subheader("Enter Planet Parameters")
    st.markdown("Adjust the sliders to describe a planet and get an instant prediction.")

    col1, col2 = st.columns(2)
    with col1:
        orbper  = st.slider("Orbital Period (days)",       0.5,   1000.0, 365.0,  step=0.5)
        insol   = st.slider("Stellar Flux (Earth = 1.0)",  0.01,  10.0,   1.0,    step=0.01)
        eqt     = st.slider("Equilibrium Temperature (K)", 50.0,  2000.0, 255.0,  step=5.0)
        rade    = st.slider("Planet Radius (Earth radii)", 0.3,   20.0,   1.0,    step=0.1)
    with col2:
        bmasse  = st.slider("Planet Mass (Earth masses)",  0.1,   1000.0, 1.0,    step=0.1)
        eccen   = st.slider("Orbital Eccentricity",        0.0,   0.9,    0.02,   step=0.01)
        st_teff = st.slider("Stellar Temperature (K)",     2500.0,10000.0,5778.0, step=50.0)
        st_met  = st.slider("Stellar Metallicity",        -2.0,   1.0,    0.0,    step=0.05)

    if st.button("Predict Habitability", type="primary"):
        X_input = engineer_features(orbper, insol, eqt, rade,
                                     bmasse, eccen, st_teff, st_met)
        prob, habitable, esi = predict(X_input)

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if habitable:
                st.success("POTENTIALLY HABITABLE")
            else:
                st.error("NOT HABITABLE")
        with res_col2:
            st.metric("Habitability Probability", f"{prob:.3f}",
                      delta=f"threshold={THRESHOLD}")
        with res_col3:
            st.metric("Earth Similarity Index", f"{esi:.3f}",
                      delta="Earth=1.0")

        st.divider()
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(hz_chart(insol, eqt, "Your Planet"))
        with chart_col2:
            st.pyplot(shap_chart(X_input))

# TAB 2: Live NASA Lookup 
with tab2:
    st.subheader("Look Up a Real Planet")
    st.markdown(
        "Enter a planet name from the NASA Exoplanet Archive "
        "(e.g. `Kepler-186 f`, `TOI-700 d`)"
    )

    planet_input = st.text_input("Planet name", placeholder="e.g. Kepler-186 f")

    if st.button("Fetch & Predict", type="primary"):
        if planet_input.strip():
            with st.spinner(f"Querying NASA archive for {planet_input}..."):
                row, error = fetch_planet(planet_input.strip())

            if error:
                st.error(f" {error}")
            else:
                st.success(f"Found: {row['pl_name']}")

                # fill missing values with Earth defaults
                def safe(val, default):
                    return float(val) if pd.notna(val) else default

                orbper  = safe(row["pl_orbper"],  365.0)
                insol   = safe(row["pl_insol"],   1.0)
                eqt     = safe(row["pl_eqt"],     255.0)
                rade    = safe(row["pl_rade"],     1.0)
                bmasse  = safe(row["pl_bmasse"],  1.0)
                eccen   = safe(row["pl_orbeccen"],0.02)
                st_teff = safe(row["st_teff"],    5778.0)
                st_met  = safe(row["st_met"],     0.0)

                st.markdown("**Retrieved parameters:**")
                param_df = pd.DataFrame({
                    "Parameter": ["Orbital Period", "Stellar Flux",
                                  "Eq. Temperature", "Radius",
                                  "Mass", "Eccentricity",
                                  "Stellar Temp", "Metallicity"],
                    "Value": [f"{orbper:.2f} days", f"{insol:.3f} Earth",
                              f"{eqt:.1f} K", f"{rade:.2f} R⊕",
                              f"{bmasse:.2f} M⊕", f"{eccen:.3f}",
                              f"{st_teff:.0f} K", f"{st_met:.2f}"]
                })
                st.dataframe(param_df, hide_index=True)

                X_input = engineer_features(orbper, insol, eqt, rade,
                                             bmasse, eccen, st_teff, st_met)
                prob, habitable, esi = predict(X_input)

                st.divider()
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    if habitable:
                        st.success("POTENTIALLY HABITABLE")
                    else:
                        st.error("NOT HABITABLE")
                with res_col2:
                    st.metric("Habitability Probability", f"{prob:.3f}")
                with res_col3:
                    st.metric("Earth Similarity Index", f"{esi:.3f}")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.pyplot(hz_chart(insol, eqt, row["pl_name"]))
                with chart_col2:
                    st.pyplot(shap_chart(X_input))
        else:
            st.warning("Please enter a planet name.")

# TAB 3: Model Performance
with tab3:
    st.subheader("Model Performance Summary")

    st.markdown("**Classification metrics across all models:**")
    display_scores = model_scores[[
        "model", "f1", "recall", "precision", "roc_auc"
    ]].dropna(subset=["f1"])
    st.dataframe(display_scores, hide_index=True)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final F1 Score",  "0.714")
    col2.metric("Recall",          "0.833")
    col3.metric("ROC-AUC",         "0.997")
    col4.metric("Threshold",       "0.12")

    st.divider()
    st.markdown("**Top features by SHAP importance:**")
    shap_importance = pd.DataFrame({
        "Feature": ["temp_delta_earth", "pl_insol", "pl_rade",
                    "st_teff", "pl_orbper", "pl_eqt"],
        "Mean |SHAP|": [3.509, 2.037, 1.726, 1.023, 0.690, 0.540],
        "Description": [
            "Temperature difference from Earth",
            "Stellar flux (Earth units)",
            "Planet radius (Earth radii)",
            "Stellar temperature (K)",
            "Orbital period (days)",
            "Equilibrium temperature (K)",
        ]
    })
    st.dataframe(shap_importance, hide_index=True)

    st.caption(
        "Model: XGBoost - Data: NASA PSCompPars + PHL HWC - "
        "6,298 planets - 61 habitable - Decision threshold: 0.12"
    )