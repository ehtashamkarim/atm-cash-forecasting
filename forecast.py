"""
ATM Cash Demand Forecasting & Replenishment Optimisation
--------------------------------------------------------
1. Simulates daily withdrawals for 60 ATMs (urban / semi-urban / rural)
   over 2 years with weekday, salary-week, month-end and festival effects
   (simulated data - patterns modelled on typical Indian ATM usage).
2. Forecasts next-7-day demand per ATM: seasonal-naive baseline vs a
   Gradient Boosting model on calendar + lag features.
3. Back-tests replenishment over the last 90 days:
   current policy  = fill to fixed capacity every 3 days
   proposed policy = load forecast demand for next refill cycle + safety buffer
   Compares idle-cash cost, CIT (cash-in-transit) trip cost and cash-outs.

Run:  python forecast.py
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(11)
os.makedirs("outputs", exist_ok=True)

N_ATM = 60
dates = pd.date_range("2024-07-01", "2026-06-30", freq="D")
festivals = pd.to_datetime(["2024-08-15", "2024-10-12", "2024-11-01", "2024-12-25",
                            "2025-03-14", "2025-03-31", "2025-08-15", "2025-08-27",
                            "2025-10-02", "2025-10-20", "2025-12-25",
                            "2026-03-04", "2026-03-20"])
atms = pd.DataFrame({
    "atm_id": [f"ATM{i:03d}" for i in range(1, N_ATM + 1)],
    "location": rng.choice(["Urban", "Semi-urban", "Rural"], N_ATM, p=[.5, .3, .2]),
})
base_map = {"Urban": 9.0, "Semi-urban": 6.0, "Rural": 3.8}   # Rs lakh / day
atms["base"] = atms.location.map(base_map) * rng.uniform(0.75, 1.25, N_ATM)

rows = []
for _, a in atms.iterrows():
    d = pd.DataFrame({"date": dates})
    dow = d.date.dt.dayofweek
    dom = d.date.dt.day
    mult = (1 + np.where(dow >= 5, 0.18, 0) - np.where(dow == 1, 0.05, 0)
            + np.where(dom <= 7, 0.35 if a.location != "Rural" else 0.20, 0)
            + np.where(dom >= 28, 0.10, 0))
    fest = np.zeros(len(d))
    for f in festivals:
        gap = (f - d.date).dt.days
        fest += np.where((gap >= 0) & (gap <= 3), 0.45, 0)
    trend = 1 + 0.04 * np.arange(len(d)) / 365          # slow growth
    noise = rng.lognormal(0, 0.15, len(d))
    d["withdrawal_lakh"] = (a.base * mult * (1 + fest) * trend * noise).round(2)
    d["atm_id"], d["location"] = a.atm_id, a.location
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
df.to_csv("outputs/atm_daily_withdrawals.csv", index=False)

# ---------------- features ----------------
df = df.sort_values(["atm_id", "date"])
fest_set = set(festivals)
df["dow"] = df.date.dt.dayofweek
df["dom"] = df.date.dt.day
df["month"] = df.date.dt.month
df["days_to_fest"] = df.date.apply(lambda x: min([(f - x).days for f in festivals if (f - x).days >= 0] or [99]))
df["loc_code"] = df.location.map({"Urban": 0, "Semi-urban": 1, "Rural": 2})
g = df.groupby("atm_id").withdrawal_lakh
for L in [7, 14, 28]:
    df[f"lag{L}"] = g.shift(L)
df["roll7_lag7"] = g.transform(lambda s: s.shift(7).rolling(7).mean())
df["roll28_lag7"] = g.transform(lambda s: s.shift(7).rolling(28).mean())
df = df.dropna()

cut = pd.Timestamp("2026-04-01")
tr, te = df[df.date < cut], df[df.date >= cut].copy()
feats = ["dow", "dom", "month", "days_to_fest", "loc_code", "lag7", "lag14", "lag28", "roll7_lag7", "roll28_lag7"]
m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=3)
m.fit(tr[feats], tr.withdrawal_lakh)
te["pred_gbm"] = m.predict(te[feats])
te["pred_naive"] = te.lag7

def mape(y, p): return float(np.mean(np.abs(y - p) / y) * 100)
mape_naive = mape(te.withdrawal_lakh, te.pred_naive)
mape_gbm = mape(te.withdrawal_lakh, te.pred_gbm)

# ---------------- replenishment back-test (last 90 days) ----------------
CAPACITY = 40.0           # Rs lakh cassette capacity
CYCLE = 3                 # refill every 3 days
TRIP_COST = 0.035         # Rs lakh per CIT visit (Rs 3,500)
COST_OF_FUNDS = 0.065 / 365   # daily opportunity cost of idle cash
STOCKOUT_PEN = 0.25       # Rs lakh penalty per cash-out day (lost fees, complaints, RBI cash-out norms)

def simulate(policy, data, buffer=1.15):
    idle = trips = stockouts = 0.0
    for atm, s in data.groupby("atm_id"):
        s = s.sort_values("date").reset_index(drop=True)
        cash = 0.0
        for i in range(len(s)):
            if i % CYCLE == 0:
                if policy == "fixed":
                    load = CAPACITY
                else:
                    load = min(CAPACITY, s.pred_gbm.iloc[i:i + CYCLE].sum() * buffer)
                if load > cash:
                    cash = load; trips += 1
            need = s.withdrawal_lakh.iloc[i]
            if need > cash:
                stockouts += 1; cash = 0
            else:
                cash -= need
            idle += cash
    return {"idle_cash_lakh_days": idle, "trips": trips, "stockout_days": stockouts,
            "total_cost_lakh": idle * COST_OF_FUNDS + trips * TRIP_COST + stockouts * STOCKOUT_PEN}

# choose safety buffer on a separate validation window (Jan-Mar 2026), not on the test set
va_cut = pd.Timestamp("2026-01-01")
m_va = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=3)
m_va.fit(df[df.date < va_cut][feats], df[df.date < va_cut].withdrawal_lakh)
va = df[(df.date >= va_cut) & (df.date < cut)].copy()
va["pred_gbm"] = m_va.predict(va[feats])
fixed_va = simulate("fixed", va)
grid = {}
for b in [1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6]:
    r = simulate("forecast", va, b)
    grid[b] = r
ok = [b for b, r in grid.items() if r["stockout_days"] <= fixed_va["stockout_days"]]
BUFFER = min(ok, key=lambda b: grid[b]["total_cost_lakh"]) if ok else max(grid)
print("buffer chosen on validation:", BUFFER)
fixed, smart = simulate("fixed", te), simulate("forecast", te, BUFFER)
days = te.date.nunique()
summary = {
    "atms": N_ATM, "safety_buffer": BUFFER, "history_days": len(dates), "test_days": days,
    "mape_seasonal_naive_pct": round(mape_naive, 1),
    "mape_gbm_pct": round(mape_gbm, 1),
    "error_reduction_pct": round(100 * (mape_naive - mape_gbm) / mape_naive, 1),
    "avg_idle_cash_fixed_lakh_per_atm": round(fixed["idle_cash_lakh_days"] / days / N_ATM, 2),
    "avg_idle_cash_forecast_lakh_per_atm": round(smart["idle_cash_lakh_days"] / days / N_ATM, 2),
    "idle_cash_reduction_pct": round(100 * (1 - smart["idle_cash_lakh_days"] / fixed["idle_cash_lakh_days"]), 1),
    "stockout_days_fixed": fixed["stockout_days"], "stockout_days_forecast": smart["stockout_days"],
    "stockout_rate_fixed_pct": round(100 * fixed["stockout_days"] / (days * N_ATM), 2),
    "stockout_rate_forecast_pct": round(100 * smart["stockout_days"] / (days * N_ATM), 2),
    "cost_fixed_lakh_90d": round(fixed["total_cost_lakh"], 2),
    "cost_forecast_lakh_90d": round(smart["total_cost_lakh"], 2),
    "cash_released_cr": round((fixed["idle_cash_lakh_days"] - smart["idle_cash_lakh_days"]) / days / 100, 2),
}
imp = pd.Series(np.abs(te[feats].corrwith(te.withdrawal_lakh)), index=feats).sort_values(ascending=False)
summary["cost_saving_pct"] = round(100*(1-smart["total_cost_lakh"]/fixed["total_cost_lakh"]),1)
summary["top_correlated_features"] = list(imp.index[:4])
te[["atm_id", "date", "withdrawal_lakh", "pred_naive", "pred_gbm"]].to_csv("outputs/test_forecasts.csv", index=False)
json.dump(summary, open("outputs/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
