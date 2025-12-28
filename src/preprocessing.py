import pandas as pd
from . import config

def load_and_preprocess():
    # Load Data
    data_df = pd.read_csv(config.DATA_PATH)
    stock_df = pd.read_csv(config.STOCK_PATH)

    # Convert dates from string to datetime
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Sort the dates
    data_df = data_df.sort_values('Date')
    stock_df = stock_df.sort_values('Date')

    # Handle missing values in the 'Data' series: forward-fill then back-fill
    data_df['Data'] = data_df['Data'].ffill().bfill()

    # Basic feature engineering: lags, percent change, rolling stats
    data_df['Data_Lag1'] = data_df['Data'].shift(1)
    data_df['Data_Lag2'] = data_df['Data'].shift(2)
    data_df['Data_Lag3'] = data_df['Data'].shift(3)
    data_df['Data_Lag4'] = data_df['Data'].shift(4)
    data_df['Data_Lag5'] = data_df['Data'].shift(5)

    # Day-over-day change and percent change
    data_df['Data_Change_PrevDay'] = data_df['Data_Lag1'] - data_df['Data_Lag2']
    data_df['Data_Pct_Change'] = data_df['Data'].pct_change()

    # Rolling statistics: 5-day mean and 5-day std (volatility proxy)
    data_df['Data_Rolling_Mean'] = data_df['Data_Lag1'].rolling(window=5).mean()
    data_df['Data_Rolling_STD'] = data_df['Data_Lag1'].rolling(window=5).std()

    # Merge datasets on Date
    df = pd.merge(data_df, stock_df, on='Date', how='inner')

    # Target: next-day price change
    df['Price_Change'] = df['Price'] - df['Price'].shift(1)

    # Drop any remaining NaNs produced by shifting/rolling
    df_clean = df.dropna().copy()

    return df_clean