import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from universal_portfolio import UniversalPortfolio

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Universal Portfolio) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_assets": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:].dropna(axis=1, how='any')
            if ret_win.shape[1] != len(tickers):
                ret_win = ret_win[tickers].dropna()
            n_assets = ret_win.shape[1]
            if n_assets < 2:
                continue
            up = UniversalPortfolio(n_assets, learning_rate=config.LEARNING_RATE, adaptive=config.USE_ADAPTIVE_ETA)
            history, cum_wealth = up.run_online(ret_win)
            final_weights = up.current_weights()
            asset_names = ret_win.columns.tolist()
            # Sort by weight descending, take top N
            sorted_idx = np.argsort(final_weights)[::-1][:config.TOP_N]
            top_weights = {asset_names[i]: final_weights[i] for i in sorted_idx}
            # Normalise top weights to sum to 1
            total = sum(top_weights.values())
            if total > 0:
                top_weights = {etf: w / total for etf, w in top_weights.items()}
            else:
                top_weights = {etf: 1.0/len(top_weights) for etf in top_weights}
            # For ETFs not in top N, weight = 0
            all_weights = {etf: top_weights.get(etf, 0.0) for etf in asset_names}
            window_results[win] = {
                "weights": all_weights,
                "final_cum_wealth": float(cum_wealth[-1]) if len(cum_wealth) > 0 else 1.0,
                "top_etfs": list(top_weights.keys())
            }
            for etf, w in all_weights.items():
                if etf not in best_per_etf or w > best_per_etf[etf][0]:
                    best_per_etf[etf] = (w, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to equal weights")
            eq = 1.0/len(tickers)
            for etf in tickers:
                best_per_etf[etf] = (eq, 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_assets": []}
                continue

        full_scores = {ticker: {"weight": float(w), "best_window": win} for ticker, (w, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_assets = [{"ticker": ticker, "weight": float(w), "best_window": win} for ticker, (w, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top {config.TOP_N} assets by universal portfolio weight (normalised to 100%): {[e['ticker'] for e in top_assets]}")
        all_results[universe_name] = {
            "top_assets": top_assets,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/universal_portfolio_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Universal Portfolio Engine complete ===")

if __name__ == "__main__":
    main()
