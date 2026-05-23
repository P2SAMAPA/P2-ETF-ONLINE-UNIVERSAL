import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Universal Portfolio", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌐 Universal Portfolio Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cover (2010) log‑optimal universal portfolio | Exponential gradient | Parameter‑free | Top N holdings (100% allocated proportionally)</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🌐 Universal Portfolio")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Learning rate:** {config.LEARNING_RATE} | **Adaptive:** {config.USE_ADAPTIVE_ETA}")
st.sidebar.markdown(f"**Top holdings:** {config.TOP_N}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'universal_portfolio_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header(f"🏆 Top {config.TOP_N} Holdings (100% Allocated Proportionally)")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - **Universal Portfolio** (Cover, 1991, 2010) is a parameter‑free, online algorithm that asymptotically achieves the same growth rate as the best constant‑rebalanced portfolio (BCRP) in hindsight.
    - It uses an **exponential gradient (Soft‑Max)** update: \( w_{t+1,i} = w_{t,i} \exp(\eta r_{t,i}) / Z \), where \(\eta\) is a learning rate.
    - The algorithm does not assume any statistical model and is provably log‑optimal.
    - We run it on a rolling window of past returns (63–4032 days). The final portfolio weights are used to allocate capital **only to the top 5 ETFs**, but **the weights among them are proportional to the original universal portfolio weights** (summing to 100%).
    - For each ETF, the rolling window that gives the **highest weight** is selected.
    """)

for universe_name, uni_data in universes.items():
    top_assets = uni_data.get("top_assets", [])
    if not top_assets:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    # Display top holdings in a row of cards (up to 5)
    cols = st.columns(min(len(top_assets), 5))
    for idx, asset in enumerate(top_assets):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{asset['ticker']}</div>
                <div class="etf-score">weight = {asset['weight']:.2%}</div>
                <div class="etf-score">best window = {asset.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Show cumulative wealth for the best window
    win_res = uni_data.get("window_results", {})
    if win_res:
        best_win = top_assets[0]['best_window'] if top_assets else None
        if best_win is not None and str(best_win) in win_res:
            cum_wealth = win_res[str(best_win)].get("final_cum_wealth", None)
            if cum_wealth is not None:
                st.info(f"**Cumulative wealth (window {best_win}d):** {cum_wealth:.3f} (starting at 1.0)")
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    weight = info.get("weight", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    weight = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Portfolio Weight": weight, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Portfolio Weight"] = pd.to_numeric(df["Portfolio Weight"], errors='coerce')
            df = df.dropna(subset=["Portfolio Weight"]).sort_values("Portfolio Weight", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Universal portfolio (Cover, 2010) uses exponential gradient update. Only the top 5 ETFs receive allocation, but the weights among them are proportional to the original universal portfolio weights.")
