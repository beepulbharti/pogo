# primary imports
import os
import logging
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('prophet').disabled = True
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.handlers = []
cmdstanpy_logger.setLevel(logging.CRITICAL)

# third party imports
import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm


def forecast_stock(
    data: pd.DataFrame,
    train_size: int = 100,
    window_size: int = 252,
    vol_window: int = 21,
    vol_median_window: int = 63,
) -> pd.DataFrame:
    """
    Next-day opening price prediction (via log(open) with Prophet, refit each day).
    Also creates regime/calendar groups and conformity score on PRICE scale.
    """

    df = data[['date', 'open']].copy()
    df.columns = ['ds', 'price']
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # Target for Prophet: log price
    df['y'] = np.log(df['price'])

    # Regime features computed from past RETURNS (no leakage)
    df['r'] = df['price'].pct_change()
    r_lag = df['r'].shift(1)
    df['volatility'] = r_lag.rolling(vol_window, min_periods=vol_window).std()
    df['trend'] = r_lag.rolling(vol_window, min_periods=vol_window).mean()
    df['vol_median'] = df['volatility'].rolling(
        vol_median_window, min_periods=vol_median_window
    ).median()

    start_forecast_idx = max(train_size, vol_window + vol_median_window)

    true_open, pred_open, dates = [], [], []
    is_high_vol, is_low_vol, is_uptrend, is_downtrend = [], [], [], []

    start_idx = 0
    for idx in tqdm(range(start_forecast_idx, len(df)), desc="Forecasting"):
        # start_idx = max(0, idx - window_size)
        train_data = df[['ds', 'y']].iloc[start_idx:idx]

        model = Prophet()
        model.fit(train_data)

        target_date = df['ds'].iloc[idx]
        future = pd.DataFrame({'ds': [target_date]})
        forecast = model.predict(future)

        pred_logp = float(forecast['yhat'].iloc[0])
        pred_p = float(np.exp(pred_logp))          # map back to price
        true_p = float(df['price'].iloc[idx])

        dates.append(target_date)
        true_open.append(true_p)
        pred_open.append(pred_p)

        current_vol = df['volatility'].iloc[idx]
        current_vol_median = df['vol_median'].iloc[idx]
        current_trend = df['trend'].iloc[idx]

        is_high_vol.append(int(current_vol > current_vol_median))
        is_low_vol.append(int(current_vol <= current_vol_median))
        is_uptrend.append(int(current_trend > 0))
        is_downtrend.append(int(current_trend <= 0))

    results = pd.DataFrame({
        'date': dates,
        'open': true_open,
        'open_pred': pred_open,
        'is_high_vol': is_high_vol,
        'is_low_vol': is_low_vol,
        'is_uptrend': is_uptrend,
        'is_downtrend': is_downtrend,
    })

    # Conformity score on PRICE scale (absolute dollar error)
    results['conformity_score'] = (results['open'] - results['open_pred']).abs()

    # Calendar groups
    results['is_monday'] = (results['date'].dt.dayofweek == 0).astype(int)
    results['is_tuesday'] = (results['date'].dt.dayofweek == 1).astype(int)
    results['is_wednesday'] = (results['date'].dt.dayofweek == 2).astype(int)
    results['is_thursday'] = (results['date'].dt.dayofweek == 3).astype(int)
    results['is_friday'] = (results['date'].dt.dayofweek == 4).astype(int)

    results['is_q1'] = (results['date'].dt.quarter == 1).astype(int)
    results['is_q2'] = (results['date'].dt.quarter == 2).astype(int)
    results['is_q3'] = (results['date'].dt.quarter == 3).astype(int)
    results['is_q4'] = (results['date'].dt.quarter == 4).astype(int)

    for m in range(1, 13):
        results[f'is_{pd.Timestamp(2000, m, 1).strftime("%B").lower()}'] = (
            results['date'].dt.month == m
        ).astype(int)

    return results


def process_stock(stock: str, data_folder: str, output_folder: str, **kwargs) -> bool:
    try:
        column_names = ['date', 'close', 'high', 'low', 'open', 'volume']
        data = pd.read_csv(
            os.path.join(data_folder, stock + '_data.csv'),
            skiprows=3,
            names=column_names,
            usecols=range(len(column_names))
        )

        results = forecast_stock(data, **kwargs)

        output_path = os.path.join(output_folder, f"{stock}_forecast.csv")
        results.to_csv(output_path, index=False)

        print(f"✓ {stock} completed: {len(results)} predictions")
        return True

    except Exception as e:
        print(f"✗ {stock} failed: {e}")
        return False


if __name__ == '__main__':
    data_folder = 'data/raw_data'
    output_folder = 'data/processed_data'
    os.makedirs(output_folder, exist_ok=True)

    stocks = ['AAPL', 'BA', 'DAL', 'META', 'MRNA']

    # define your params dict (you referenced **params before)
    params = dict(train_size=100, window_size=252, vol_window=21, vol_median_window=63)

    successful, failed = [], []
    for stock in tqdm(stocks):
        print(f"\nProcessing {stock}...")
        if process_stock(stock, data_folder, output_folder, **params):
            successful.append(stock)
        else:
            failed.append(stock)

    print("\n" + "=" * 50)
    print(f"Completed: {len(successful)}/{len(stocks)}")
    print(f"Successful: {successful}")
    if failed:
        print(f"Failed: {failed}")
        with open(os.path.join(output_folder, 'failed_stocks.txt'), 'w') as f:
            for stock in failed:
                f.write(stock + '\n')