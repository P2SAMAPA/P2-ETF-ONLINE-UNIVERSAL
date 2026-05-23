# Universal Portfolio Engine

Implements Cover's Universal Portfolio (Cover, 1991, 2010) using the exponential gradient (Soft‑Max) update. The algorithm is parameter‑free, online, and asymptotically log‑optimal. It processes daily returns sequentially and maintains a portfolio that is a weighted average of all constant‑rebalanced portfolios. At the end of the training window, the final weights are used to allocate capital. The engine outputs the top \(N\) (default 5) holdings per universe. Multi‑window evaluation selects the best window per ETF (highest weight).

- **Algorithm:** Exponential gradient (Soft‑Max) with constant or adaptive learning rate
- **Properties:** No statistical assumptions, provable log‑optimal growth
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top \(N\) ETFs per universe by portfolio weight

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
