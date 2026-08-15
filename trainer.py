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

        # Get the earliest date where ALL tickers have data
        ticker_data_availability = {}
        for ticker in tickers:
            if ticker in df.columns:
                # Get first non-null date for this ticker
                first_valid = df[ticker].first_valid_index()
                if first_valid is not None:
                    ticker_data_availability[ticker] = first_valid
                else:
                    ticker_data_availability[ticker] = df.index[0]  # fallback

        # Find the maximum start date among all tickers (latest start date)
        latest_start = max(ticker_data_availability.values()) if ticker_data_availability else df.index[0]
        
        # Filter returns to only include data from the latest start date
        returns = returns.loc[latest_start:]

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            
            # Get the last 'win' days of data
            ret_win = returns.iloc[-win:].dropna(axis=1, how='any')
            
            # Only use tickers that exist in the window and have complete data
            available_tickers = [t for t in tickers if t in ret_win.columns]
            
            if len(available_tickers) < 2:
                print(f"  Not enough available tickers for window {win}d (have {len(available_tickers)})")
                continue
                
            ret_win = ret_win[available_tickers]
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
            # Only use available tickers for equal weights
            available_tickers = [t for t in tickers if t in returns.columns]
            if not available_tickers:
                available_tickers = tickers
            eq = 1.0/len(available_tickers)
            for etf in available_tickers:
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
