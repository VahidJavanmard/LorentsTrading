def calculate_linear_regression_channel(df, window=14):
    if isinstance(df, pd.DataFrame):
        prices = df['close'].values
        mid_channel = np.full(len(prices), np.nan)
        upper_channel = np.full(len(prices), np.nan)
        lower_channel = np.full(len(prices), np.nan)

        for i in range(len(prices) - window + 1):
            y = prices[i:i + window]
            x = np.arange(window)
            slope, intercept, _, _, _ = linregress(x, y)
            regression_values = slope * x + intercept

            std_dev = np.std(y - regression_values)
            mid_channel[i + window - 1] = regression_values[-1]
            upper_channel[i + window - 1] = regression_values[-1] + std_dev
            lower_channel[i + window - 1] = regression_values[-1] - std_dev

        df['mid_channel'] = mid_channel
        df['upper_channel'] = upper_channel
        df['lower_channel'] = lower_channel

        return df
    else:
        raise ValueError("Input must be a DataFrame")

def calculate_bollinger_bands(df, window=20, window_dev=2):
    if isinstance(df, pd.DataFrame):
        # محاسبه بولینگر بند
        indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=window, window_dev=window_dev)

        # اضافه کردن ستون های بولینگر بند به DataFrame
        df['bb_mavg'] = indicator_bb.bollinger_mavg()
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_lower'] = indicator_bb.bollinger_lband()

        return df
    else:
        raise ValueError("Input must be a DataFrame")
     #توابع مربوط به محاسبات
def calculate_rsi(df, period=14):
    if isinstance(df, pd.DataFrame):
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        # df['rsi'] = df['rsi'].fillna(0)  # اصلاح این خط
        return df
    else:
        raise ValueError("Input must be a DataFrame")

def calculate_adx(df, period=14):
    if isinstance(df, pd.DataFrame):
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        # df['adx'] = df['adx'].fillna(0)  # اصلاح این خط
        return df
    else:
        raise ValueError("Input must be a DataFrame")



def calculate_di(df, period=14):
    # Calculate True Range (TR)
    df['tr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=1)

    # Calculate Positive and Negative Directional Movement (+DM, -DM)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = (df['up_move'] > df['down_move']) & (df['up_move'] > 0)
    df['minus_dm'] = (df['down_move'] > df['up_move']) & (df['down_move'] > 0)

    # Set +DM and -DM values
    df.loc[df['plus_dm'], 'plus_dm'] = df['up_move']
    df['plus_dm'] = df['plus_dm'].astype(bool)
    df['minus_dm'] = df['minus_dm'].astype(bool)
    df.loc[df['minus_dm'], 'minus_dm'] = df['down_move']

    # Calculate ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)

    # Drop NaN values
    plus_dm = df['plus_dm'].dropna()
    atr = df['atr'].dropna()

    # Calculate +DI and -DI
    df['plus_di'] = 100 * ta.trend.ema_indicator(plus_dm, window=period) / atr
    df['minus_di'] = 100 * ta.trend.ema_indicator(df['minus_dm'], window=period) / atr

    return data


def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    # Calculate Short and Long EMA
    short_ema = ta.trend.ema_indicator(df['close'], window=short_window)
    long_ema = ta.trend.ema_indicator(df['close'], window=long_window)

    # Calculate MACD line
    macd_line = short_ema - long_ema

    # Calculate Signal line
    signal_line = ta.trend.ema_indicator(macd_line, window=signal_window)

    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line

    # Add MACD indicators to DataFrame
    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['macd_histogram'] = macd_histogram

    return df

def calculate_mfi(df, period=14):
    if isinstance(df, pd.DataFrame):
        df['positive_money_flow'] = df['high'].diff(1) * df['volume']
        df['negative_money_flow'] = df['low'].diff(1) * df['volume'].abs()
        df['mfi'] = 100 * (df['positive_money_flow'].rolling(period).sum() /
                           df['negative_money_flow'].rolling(period).sum())

        # جایگزینی مقادیر NaN به جای inf و سپس پر کردن مقادیر NaN با مقدار ایندکس قبلی
        # df['mfi'] = df['mfi'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # df['mfi'] = df['mfi'].replace([np.inf, -np.inf], np.nan).dropna()
        return df
    else:
        raise ValueError("Input must be a DataFrame")


def calculate_supertrend(df, period=7, multiplier=3):
    if isinstance(df, pd.DataFrame):
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
        df['upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
        df['lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])

        supertrend = [0.0] * len(df)
        in_uptrend = [True] * len(df)

        for i in range(1, len(df)):
            if df['close'][i] > df['upper_band'][i-1]:
                in_uptrend[i] = True
            elif df['close'][i] < df['lower_band'][i-1]:
                in_uptrend[i] = False
            else:
                in_uptrend[i] = in_uptrend[i-1]

                if in_uptrend[i] and df['lower_band'][i] < df['lower_band'][i-1]:
                    df.loc[i, 'lower_band'] = df['lower_band'][i-1]
                if not in_uptrend[i] and df['upper_band'][i] > df['upper_band'][i-1]:
                    df.loc[i, 'upper_band'] = df['upper_band'][i-1]

            if in_uptrend[i]:
                supertrend[i] = df['lower_band'][i]
            else:
                supertrend[i] = df['upper_band'][i]

        df['supertrend'] = supertrend
        df['in_uptrend'] = in_uptrend
        return df
    else:
        raise ValueError("Input must be a DataFrame")

def calculate_bollinger_bands(df, period=20, std_dev_multiplier=2):
    if isinstance(df, pd.DataFrame):
        # محاسبه میانگین متحرک ساده (SMA)
        df['SMA'] = df['close'].rolling(window=period).mean()

        # محاسبه انحراف معیار (Standard Deviation)
        df['std_dev'] = df['close'].rolling(window=period).std()

        # محاسبه باندهای بالا و پایین بولینگر
        df['upper_band'] = df['SMA'] + (std_dev_multiplier * df['std_dev'])
        df['lower_band'] = df['SMA'] - (std_dev_multiplier * df['std_dev'])

        # پر کردن مقادیر NaN با 0
        df[['SMA', 'upper_band', 'lower_band']].fillna(0, inplace=True)

        return df
    else:
        raise ValueError("Input must be a DataFrame")
