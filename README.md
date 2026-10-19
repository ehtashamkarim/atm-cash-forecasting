# ATM Cash Demand Forecasting & Replenishment Optimisation

Forecasts daily cash withdrawals for **60 ATMs** (urban, semi-urban, rural) and tests whether forecast-driven cash loading beats the usual "fill the cassette every 3 days" rule. Idle cash in ATMs earns nothing, so every rupee sitting in a machine costs the bank its cost of funds, while running out of cash costs fees, complaints and regulatory penalties.

> Data is simulated: 2 years (Jul 2024 – Jun 2026) of daily withdrawals with weekday, salary-week (1st–7th), month-end, festival and growth effects typical of Indian ATM usage.

## Results (from `outputs/summary.json`, 91-day out-of-sample test)
| Metric | Value |
|---|---|
| Forecast error, seasonal-naive (last week) | MAPE 22.5% |
| Forecast error, Gradient Boosting | **MAPE 12.5% (44% lower)** |
| Avg idle cash per ATM | Rs 22.4 lakh → **Rs 14.6 lakh (−35%)** |
| Cash released across 60 ATMs | **~Rs 4.7 Cr** |
| Cash-out rate | 3.52% → 3.53% (unchanged) |
| Total cost (idle cash + CIT trips + cash-outs) | **−5.7%** |

## Method
1. **Features**: day of week, day of month, month, days to next festival, location type, lags (7/14/28 days), rolling 7- and 28-day means (all lagged ≥ 7 days, so a 7-day-ahead forecast is possible).
2. **Model**: Gradient Boosting regressor vs seasonal-naive baseline; train to Mar 2026, test Apr–Jun 2026.
3. **Replenishment back-test**: every 3 days, load forecast demand × safety buffer (capped at Rs 40 lakh). The **safety buffer (1.4x) was chosen on a separate Jan–Mar 2026 validation window**, not on the test data, with the rule "no more cash-outs than today's policy".

## Talking points
- Why MAPE and not R²: operations teams think in "% off per day".
- Trade-off: smaller loads mean less idle cash but more cash-out risk; the buffer is the lever.
- Festival and salary-week spikes are why a fixed schedule over-stocks most days and still runs out on peak days.

## Run
```
pip install pandas numpy scikit-learn
python forecast.py
```
