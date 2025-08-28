import os
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
HEALTH_URL = f"{API_BASE}/health"
AT_RISK_URL = f"{API_BASE}/at_risk"

st.set_page_config(
    page_title="Churn – Dashboard API",
    page_icon="📉",
    layout="wide",
    menu_items={"About": "Demo POC – lecture des prédictions via API"}
)

# -------------------------
# Header
# -------------------------
st.title("📉 Churn – Dashboard (API client)")
st.caption("Cette app interroge l'API pour récupérer les clients à risque (lecture du CSV de prédictions).")

# -------------------------
# Sidebar (paramètres)
# -------------------------
with st.sidebar:
    st.header("⚙️ Paramètres API")
    api_base = st.text_input("URL de l'API", value=API_BASE, help="Ex: http://localhost:8000")
    health_url = f"{api_base}/health"
    at_risk_url = f"{api_base}/at_risk"

    st.markdown("---")
    st.header("🔎 Filtres")
    min_proba = st.slider("Seuil probabilité minimum", 0.0, 1.0, 0.5, 0.01)
    gender_filter = st.multiselect("Filtrer par genre", options=["Female", "Male"], default=["Female", "Male"])
    top_n = st.number_input("Limiter à N premiers", min_value=1, value=50, step=1)
    search_id = st.text_input("Recherche (customerID contient)")

# -------------------------
# Utils
# -------------------------
@st.cache_data(show_spinner=False, ttl=30)
def get_health(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=3)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=30)
def get_at_risk(url: str, min_proba: float) -> List[Dict[str, Any]]:
    r = requests.get(url, params={"min_proba": min_proba}, timeout=5)
    r.raise_for_status()
    return r.json()

def to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["customerID", "gender", "churn_proba", "churn_pred"])
    df = pd.DataFrame(items)
    # types & arrondi
    if "churn_proba" in df.columns:
        df["churn_proba"] = pd.to_numeric(df["churn_proba"], errors="coerce")
    if "churn_pred" in df.columns:
        df["churn_pred"] = pd.to_numeric(df["churn_pred"], errors="coerce").astype("Int64")
    df = df.sort_values("churn_proba", ascending=False)
    return df

# -------------------------
# Actions (boutons)
# -------------------------
col1, col2, col3, col4 = st.columns([1.2,1,1,1])

with col1:
    st.markdown("#### Actions")
with col2:
    if st.button("🩺 Tester l'API (health)", use_container_width=True):
        try:
            resp = get_health(health_url)
            st.success(f"API OK – status: {resp.get('status')}")
        except Exception as e:
            st.error(f"Échec health: {e}")

with col3:
    load_clicked = st.button("⚡ Charger clients à risque", type="primary", use_container_width=True)
with col4:
    clear_cache = st.button("♻️ Vider le cache", use_container_width=True)
    if clear_cache:
        st.cache_data.clear()
        st.toast("Cache vidé.", icon="✅")

st.markdown("---")

# -------------------------
# Data loading
# -------------------------
df = pd.DataFrame()
if load_clicked:
    with st.spinner("Récupération depuis l'API…"):
        try:
            items = get_at_risk(at_risk_url, min_proba=min_proba)
            df = to_dataframe(items)
            st.toast(f"{len(df)} clients reçus.", icon="✅")
        except Exception as e:
            st.error(f"Impossible de récupérer les données : {e}")

# -------------------------
# UI Résultats
# -------------------------
if not df.empty:
    # Filtres UI
    df_ui = df.copy()

    if gender_filter:
        df_ui = df_ui[df_ui["gender"].isin(gender_filter)]

    if search_id:
        df_ui = df_ui[df_ui["customerID"].astype(str).str.contains(search_id, case=False, na=False)]

    df_ui = df_ui.head(top_n)

    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Clients à risque (affichés)", len(df_ui))
    with k2:
        st.metric("Probabilité moyenne", f"{df_ui['churn_proba'].mean():.2f}" if not df_ui.empty else "—")
    with k3:
        st.metric("Probabilité max", f"{df_ui['churn_proba'].max():.2f}" if not df_ui.empty else "—")

    # Tableau
    st.subheader("Détails")
    st.dataframe(
        df_ui[["customerID", "gender", "churn_proba", "churn_pred"]],
        use_container_width=True,
        hide_index=True
    )

    # Graphe
    st.subheader("Distribution des probabilités (top N)")
    st.bar_chart(df_ui.set_index("customerID")["churn_proba"])

    # Export
    csv_bytes = df_ui.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exporter CSV (filtré)",
        data=csv_bytes,
        file_name="at_risk_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("Clique sur **Charger clients à risque** pour interroger l'API.")
