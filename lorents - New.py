from imports import *
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     BeautyChart.main()
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#تنظیمات api , تلگرام
APIURL="https://open-api.bingx.com"
# APIKEY="yhMMfr0fbRsinaHaPnhesLhhaE5nur7D3W034jCpyURP1CFnI1T0Yjxmt77TpdGcCD3wnTDI5LXoVL0yYF9w"
# SECRETKEY="HMaxI039IQpRkKPIaxI6PUSI0I11pDDId6gEzWXO2yDARgp1ZgDvSB0GQVvB92OZ9sn9jNbp4I1SoRIrw"
APIKEY="oM0qKrqgmibx0a2VHVBwn4oBuWLLdqQBdlQ73kvLnHIqlbzgvewRrt2kk1YbUURcP7K0e1izVtaqjyehrg"
SECRETKEY="usj3lCWX2NKN17PqyE7HB8G3Jb0D4NpHBzGyKy3FUphKjJF6bx2wTNiB8I1jWFyun6pB3JGoAFhIzIQDHhwg"
telegram_bot_token='142615795:AAGdGxG0GpxEi-iu23sKAuVctdLqQ5hgKB4'
telegram_channel_id='@tradinghistorytest'
# telegram_channel_id=70074075
# bot=Bot(token=telegram_bot_token)
bot=telebot.TeleBot(telegram_bot_token)
BotId="My PC"
cycle=[]
cycle_status=False
csv=False
signalPrice=0
lastSignalAction=3
lastSignalExcell=3
actionStatus=""
profit=0.00
sendMessage=False
closedMethod=""
#تعداد کندل برای محاسبه درصد تغییر
shiftedNumber=1
#درصد تغییر برای قرار داد ن  در کلاس no Action
expectedChange=0
#درصد دقت برای سیگنال
expectedAccuracy=0.30

shiftedNumber_range = range(1, 10)
expectedChange_range = np.arange(0.1, 0.5, 0.1)

expectedCombineModel=0.40
expectedLstmModel=0.4
expectedSvmModel=0.40

trainTimeStamp=datetime.now

#بارگذاری داده ها
# symbol="BTC-USDT"
symbol="BTC-USDT"
# symbol="FTM-USDT"

indicatorSource='close'
timeframe="4h"
limit=5000
percentTest=0.05
n_Candle_Train=10

lstm_model=None
svm_model=None
combine_model=None
accuracy=-1




#indicator Sell Buy Zone

rsi_overbought = 70
rsi_oversold = 30
macd_threshold = 0
adx_threshold = 25
cci_overbought = 100
cci_oversold = -100
stochastic_high = 80
stochastic_low = 20


scaler= StandardScaler()

# accuracy=None
# تنظیم زمان به قبرس
cyprus_timezone=pytz.timezone('Asia/Nicosia')


def DrawChartNew():
    if __name__ == '__main__':
        chart = Chart(toolbox=True)
            
            # Columns: time | open | high | low | close | volume 
            # df = pd.read_csv('ohlcv.csv')
        # df=data
        df = pd.read_csv('BTC-USDT4h2024-05-24 08-00-00.csv')
        df['date'] = pd.to_datetime(df['timestamp'])  # تبدیل ستون timestamp به فرمت datetime
            # df['date'] = df['timestamp'].map(mdates.date2num)
            # df.to_csv("test.csv")
        chart.set(df, render_drawings=True)
        for index in  df.itertuples(index=True, name='Pandas'):
            i = index
            if(i.marker=='B'):
                chart.marker(i.date,'inside','arrow_up','green',i.close)
            elif (i.marker=='C'):
                chart.marker( i.date,'inside','circle','white',i.close)
            elif (i.marker=='CB'):
                chart.marker(i.date,'inside','arrow_up','green',i.close)
            elif (i.marker=='CB'):
                chart.marker(i.date,'inside','arrow_down','red',i.close)
            elif (i.marker=='S'):
                chart.marker(i.date,'inside','arrow_down','red',i.close)

            
        chart.watermark('BTC-USDT 1D', color='rgba(180, 180, 240, 0.7)')
        chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                            horz_color='#FFFFFF', horz_style='dotted')

        chart.show(block=True)
    

# DrawChartNew()
#import n*timefrime
def add_time(current_time, time_string):
    unit = time_string[-1]
    value = int(time_string[:-1])
    
    if unit == 'm':  # minutes
        delta = timedelta(minutes=value)
    elif unit == 'h':  # hours
        delta = timedelta(hours=value)
    elif unit == 's':  # seconds
        delta = timedelta(seconds=value)
    elif unit == 'd':  # days
        delta = timedelta(days=value)
    else:
        raise ValueError("Unsupported time unit. Use 's' for seconds, 'm' for minutes, 'h' for hours, or 'd' for days.")
    
    return current_time + delta

def add_time_multiple(current_time, time_string, count):
    for _ in range(count):
        current_time = add_time(current_time, time_string)
    return current_time



# def add_time(current_time, time_string):
#     unit = time_string[-1]
#     value = int(time_string[:-1])
    
#     if unit == 'm':  # minutes
#         delta = timedelta(minutes=value)
#     elif unit == 'h':  # hours
#         delta = timedelta(hours=value)
#     else:
#         raise ValueError("Unsupported time unit. Use 'm' for minutes or 'h' for hours.")
    
#     return current_time + delta

def get_cyprus_time():
    utc_time=datetime.now(timezone.utc)
    # # utc_time=datetime.now(utc)
    utc_time=utc_time.replace(tzinfo=pytz.utc)
    local_time=utc_time.astimezone(cyprus_timezone)
    return local_time

def get_sign(api_secret,payload):
    signature=hmac.new(api_secret.encode("utf-8"),payload.encode("utf-8"),hashlib.sha256).hexdigest()
    return signature

def parseParam(paramsMap):
    sortedKeys=sorted(paramsMap)
    paramsStr="&".join([f"{key}={paramsMap[key]}" for key in sortedKeys])
    return paramsStr + "&timestamp"+ str(int(time.time()*1000))

def send_request(method, path, url_params, payload=""):
    url =f"{APIURL}{path}?{url_params}&signature={get_sign(SECRETKEY, url_params)}"
    headers = {'X-BX-APIKEY': APIKEY}
    response = requests.request(method, url, headers-headers, data-payload)
    return response.json()

def curentdata():
    exchange=ccxt.bingx()
    ticker = exchange.fetch_ticker(symbol)
    lastprice= ticker['last']
    print("lastprice",lastprice)
    return lastprice

def getLastPrice(data):
    return curentdata()


def add_indicators(df):
    global data
    # محاسبه 20 اندیکاتور فنی
    df['rsi'] = ta.momentum.RSIIndicator(df[indicatorSource]).rsi()
    df['macd'] = ta.trend.MACD(df[indicatorSource]).macd()
    df['macd_diff'] = ta.trend.MACD(df[indicatorSource]).macd_diff()
    df['ema'] = ta.trend.EMAIndicator(df[indicatorSource]).ema_indicator()
    df['bollinger_mavg'] = ta.volatility.BollingerBands(df[indicatorSource]).bollinger_mavg()
    df['bollinger_hband'] = ta.volatility.BollingerBands(df[indicatorSource]).bollinger_hband()
    df['bollinger_lband'] = ta.volatility.BollingerBands(df[indicatorSource]).bollinger_lband()
    df['stochastic_oscillator'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['stochastic'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    # df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['force_index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
    df['tsi'] = ta.momentum.TSIIndicator(df[indicatorSource]).tsi()
    df['ultimate_oscillator'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
    df['kama'] = ta.momentum.KAMAIndicator(df[indicatorSource]).kama()
    df['dpo'] = ta.trend.DPOIndicator(df[indicatorSource]).dpo()
    df=calculate_linear_regression_channel(df)
    
    df = dropNaFix(df)
    

    return df

def dropNaFix(df):
    last_10_rows = df.tail(shiftedNumber)
    # اعمال dropna بر روی سایر سطرها
    df_cleaned = df.iloc[:-shiftedNumber].dropna()
    
    # بازگرداندن ده سطر آخر به DataFrame
    df_cleaned = pd.concat([df_cleaned, last_10_rows])
    # تنظیم ایندکس‌ها
    df_cleaned = df_cleaned.reset_index(drop=True)
    # df = df.dropna()
    return df_cleaned
     #توابع مربوط به محاسبات
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


# def calculate_supertrend(df, period=7, multiplier=3):
#     if isinstance(df, pd.DataFrame):
#         # محاسبه میانگین متحرک نمایی
#         df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()

#         # محاسبه خط های بالا و پایین Supertrend
#         df['upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
#         df['lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])

#         # تعیین جهت روند
#         supertrend = [0.0] * len(df)
#         for i in range(1, len(df)):
#             if df['close'][i] > df['upper_band'][i-1]:
#                 supertrend[i] = df['lower_band'][i]
#             elif df['close'][i] < df['lower_band'][i-1]:
#                 supertrend[i] = df['upper_band'][i]
#             else:
#                 supertrend[i] = supertrend[i-1]

#         df['supertrend'] = supertrend
#         df['supertrend'].fillna(0, inplace=True)
#         return df
#     else:
#         raise ValueError("Input must be a DataFrame")

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

def drawChart():

#رسم نمودار
    plt.figure(figsize=(15,10))
    #نمودار RSI
    plt.subplot(4,1,1)
    plt.plot(data['rsi'],label='RSI')
    plt.axhline(50,color='grey',linestyle='--')
    plt.title('Relative Stregth Index (Rsi)')
    plt.legend()

    #نمودار ADX
    plt.subplot(4,1,2)
    plt.plot(data['adx'],label='ADX')
    plt.axhline(25,color='grey',linestyle='--')
    plt.title('Average Directional Index (ADX)')
    plt.legend()

    #نمودار DI+ و DI-
    # plt.subplot(4,1,3)
    # plt.plot(data['plus_di'],label='DI+')
    # plt.plot(data['minus_di'],label='DI-')
    # plt.title('Directional Movement Indicators')
    # plt.legend()

    #نمودار سیگنال های خرید و فروش
    plt.subplot(4,1,4)
    plt.plot(data['close'],label='Close Price')
    buy_signals=data[data['trade_signal']=='buy']
    sell_signals=data[data['trade_signal']=='sell']
    plt.scatter(buy_signals.index,buy_signals['close'],marker='^',color='g',label='Buy Signal',alpha=1)
    plt.scatter(sell_signals.index,sell_signals['close'],marker='v',color='r',label='Sell Signal',alpha=1)
    plt.title('Buy/Sell Signals')
    plt.legend()

    #نمایش نمودار ها
    plt.tight_layout()
    # plt.draw()
    # plt.show(block=False)
    # plt.ion()
    # plt.ion()
    # plt.show()



def cosine_distance(x1, x2):
    return cosine(x1, x2)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))


def lorentzian_distance(x, y):
    return np.sum(np.log1p(np.abs(x - y) ** 2))

def calculate_buy_sell_signals(data,delay_period=1):

    #محاسبه شاخص های adx ,rsi
    data['adx']=ta.trend.adx(data['high'],data['low'],data['close'],window=14,fillna=True)
    data['rsi']=ta.momentum.rsi(data['close'],window=14,fillna=True)

    #مقدار دهی اولیه متغییر ها
    cycle_open=False
    cycle_type=None
    last_cycle_type=None
    data['trade_signal']=None
    data['cycle_status']=None
    potential_signal_time=None
    potential_signal_type=None

    #(crossover) برای بررسی عبور DI- , DI+ ذخیره مقادیر قبلی
    previous_plus_di=data['plus_di'].shift(1)
    previous_plus_di=previous_plus_di.dropna()
    previous_minus_di=data['minus_di'].shift(1)
    previous_minus_di=previous_minus_di.dropna()

    # if cycle[-1]="closed"
    for i in range(1,len(data)):
        current_time=data.loc[i,'timestamp']

        #بررسی استمرار شرایط سیگنال پس ار گذشت تاخیر
        if potential_signal_time and (current_time- potential_signal_time).total_seconds()>=delay_period*60:
            if potential_signal_type =='buy' and (data.loc[i,'plus_di']<data.loc[i,'minus_di']):
                cycle_open=True
                cycle_type='buy'
                data.loc[i,'trade_signal']='buy'
                data.loc[i,'cycle_status']='open'
            elif potential_signal_type=='sell' and (data.loc[i,'minus_di']>data.loc[i,'plus_di']):
                cycle_open=True
                cycle_type='sell'
                data.loc[i,'trade_signal']='sell'
                data.loc[i,'cycle_status']='open'
            potential_signal_time=None

            #بررسی شرایط سیگنال بدون تاخیر
            if not cycle_open:
                if(data.loc[i,'plus_di']>data.loc[i,'minus_di']) and   (data.loc[i,'ADX'] >25) and (data.loc[i,'RSI']<50):
                    potential_signal_time=current_time
                    potential_signal_type='buy'
                elif (data.loc[i,'minus_di']<data.loc[i,'plus_di']) and   (data.loc[i,'ADX'] <25) and (data.loc[i,'RSI']<50):
                    potential_signal_time=current_time
                    potential_signal_type='sell'

            #برای برستن سیکل cross over بررسی عبور
            if cycle_open:
                if cycle_type=='buy' and (data.loc[i,'minus_di']<data.loc[i,'plus_di']) and (previous_plus_di[i]<previous_minus_di[i]):
                    cycle_open=False
                    last_cycle_type='buy'
                    data.loc[i,'trade_signal']='sell'
                    data.loc[i,'cycle_status']='closed'
                elif cycle_type=='sell' and (data.loc[i,'plus_du']<data.loc[i,'minus_di']) and (previous_minus_di[i] < previous_plus_di[i]):
                    cycle_open=False
                    last_cycle_type='sell'
                    data.loc[i,'trade_signal']='buy'
                    data.loc[i,'cycle_status']='closed'
    return data
def CloseSignalWithIndicators(data, delay_period=1,CheckCycle=True,signalAction=lastSignalAction):
    # محاسبه شاخص های adx, rsi, mfi
    data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14, fillna=True)
    data['rsi'] = ta.momentum.rsi(data['close'], window=14, fillna=True)
    data['mfi'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14, fillna=True)

    # مقداردهی اولیه متغیرها
    # cycle_open = False
    # cycle_type = None
    # last_cycle_type = None
    # potential_signal_time = None
    # potential_signal_type = None

    # # اضافه کردن ستون های trade_signal و cycle_status
    # data['trade_signal'] = None
    # data['cycle_status'] = None

    # # ذخیره مقادیر قبلی MFI
    # previous_mfi = data['mfi'].shift(1)
    # if cycle_status==False:
    # # حلقه اصلی برای بررسی سیگنال‌ها
    #     for i in range(1, len(data)):
    #         currentMfi = data.loc[i, 'mfi']
    #         current_time = data.loc[i, 'timestamp']
    #         # بررسی استمرار شرایط سیگنال پس از گذشت تأخیر
    #         if potential_signal_time and (current_time - potential_signal_time).total_seconds() >= delay_period * 60:
    #             if potential_signal_type == 'buy' and currentMfi > 20:
    #                 cycle_open = True
    #                 cycle_type = 'buy'
    #                 data.loc[i, 'trade_signal'] = 'buy'
    #                 data.loc[i, 'cycle_status'] = 'open'
    #                 # cycle.append('open')
    #                 print(i)
    #             elif potential_signal_type == 'sell' and currentMfi < 80:
    #                 cycle_open = True
    #                 cycle_type = 'sell'
    #                 data.loc[i, 'trade_signal'] = 'sell'
    #                 data.loc[i, 'cycle_status'] = 'open'
    #                 cycle.append('open')
    #             potential_signal_time = None

    #         # بررسی شرایط سیگنال بدون تأخیر
    #         if not cycle_status:
    #             if currentMfi < 20 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] < 50:
    #                 potential_signal_time = current_time
    #                 potential_signal_type = 'buy'
    #             elif currentMfi > 80 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] > 50:
    #                 potential_signal_time = current_time
    #                 potential_signal_type = 'sell'
    # currentMfi = data.loc[-1, 'mfi']
    currentMfi = data['mfi'].iloc[-1]

    # return False
    #previous_mfi = data.loc[-2, 'mfi']
    #         # بررسی عبور برای بستن سیکل

    if CheckCycle==False or cycle_status:
        if signalAction == 1 and currentMfi > 80 :
            print(f"closed buy :{currentMfi}")
            return True
        elif signalAction == 0 and currentMfi < 20 :
            print(f"closed sell: {currentMfi}")
            return True

    return False
    # return True

def OpenSignalWithIndicators(data, delay_period=1,CheckCycle=True,signalAction=lastSignalAction):
    # محاسبه شاخص های adx, rsi, mfi
    data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14, fillna=True)
    data['rsi'] = ta.momentum.rsi(data['close'], window=14, fillna=True)
    data['mfi'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14, fillna=True)

    # مقداردهی اولیه متغیرها
    # cycle_open = False
    # cycle_type = None
    # last_cycle_type = None
    # potential_signal_time = None
    # potential_signal_type = None

    # # اضافه کردن ستون های trade_signal و cycle_status
    # data['trade_signal'] = None
    # data['cycle_status'] = None

    # # ذخیره مقادیر قبلی MFI
    # previous_mfi = data['mfi'].shift(1)
    # if cycle_status==False:
    # # حلقه اصلی برای بررسی سیگنال‌ها
    #     for i in range(1, len(data)):
    #         currentMfi = data.loc[i, 'mfi']
    #         current_time = data.loc[i, 'timestamp']
    #         # بررسی استمرار شرایط سیگنال پس از گذشت تأخیر
    #         if potential_signal_time and (current_time - potential_signal_time).total_seconds() >= delay_period * 60:
    #             if potential_signal_type == 'buy' and currentMfi > 20:
    #                 cycle_open = True
    #                 cycle_type = 'buy'
    #                 data.loc[i, 'trade_signal'] = 'buy'
    #                 data.loc[i, 'cycle_status'] = 'open'
    #                 # cycle.append('open')
    #                 print(i)
    #             elif potential_signal_type == 'sell' and currentMfi < 80:
    #                 cycle_open = True
    #                 cycle_type = 'sell'
    #                 data.loc[i, 'trade_signal'] = 'sell'
    #                 data.loc[i, 'cycle_status'] = 'open'
    #                 cycle.append('open')
    #             potential_signal_time = None

    #         # بررسی شرایط سیگنال بدون تأخیر
    #         if not cycle_status:
    #             if currentMfi < 20 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] < 50:
    #                 potential_signal_time = current_time
    #                 potential_signal_type = 'buy'
    #             elif currentMfi > 80 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] > 50:
    #                 potential_signal_time = current_time
    #                 potential_signal_type = 'sell'
    # currentMfi = data.loc[-1, 'mfi']
    currentMfi = data['mfi'].iloc[-1]
    print(f"Current MFi {currentMfi}")

    # return False
    #previous_mfi = data.loc[-2, 'mfi']
    #         # بررسی عبور برای بستن سیکل

    if CheckCycle==False or cycle_status:
        if signalAction == 0 and currentMfi > 80 :
            print(f"Open Sell :{currentMfi}")
            return True
        elif signalAction == 1 and currentMfi < 20 :
            print(f"Open buy: {currentMfi}")
            return True

    return False
    # return True

def calculate_buy_sell_signalsSuperTrend(data, delay_period=1):
    # محاسبه شاخص های adx, rsi, mfi و supertrend
    data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14, fillna=True)
    data['rsi'] = ta.momentum.rsi(data['close'], window=14, fillna=True)
    data['mfi'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14, fillna=True)

    data = calculate_supertrend(data)

    # مقداردهی اولیه متغیرها
    cycle_open = False
    cycle_type = None
    potential_signal_time = None
    potential_signal_type = None

    # اضافه کردن ستون های trade_signal و cycle_status
    data['trade_signal'] = None
    data['cycle_status'] = None

    # ذخیره مقادیر قبلی MFI
    previous_mfi = data['mfi'].shift(1)

    # حلقه اصلی برای بررسی سیگنال‌ها
    for i in range(1, len(data)):
        currentMfi = data.loc[i, 'mfi']
        current_time = data.loc[i, 'timestamp']
        in_uptrend = data.loc[i, 'in_uptrend']

        # بررسی استمرار شرایط سیگنال پس از گذشت تأخیر
        if potential_signal_time and (current_time - potential_signal_time).total_seconds() >= delay_period * 60:
            if potential_signal_type == 'buy' and currentMfi > 20 and in_uptrend:
                cycle_open = True
                cycle_type = 'buy'
                data.loc[i, 'trade_signal'] = 'buy'
                data.loc[i, 'cycle_status'] = 'open'
                cycle.append('open')
            elif potential_signal_type == 'sell' and currentMfi < 80 and not in_uptrend:
                cycle_open = True
                cycle_type = 'sell'
                data.loc[i, 'trade_signal'] = 'sell'
                data.loc[i, 'cycle_status'] = 'open'
                cycle.append('open')
            potential_signal_time = None

        # بررسی شرایط سیگنال بدون تأخیر
        if not cycle_open:
            if currentMfi < 20 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] < 50 and in_uptrend:
                potential_signal_time = current_time
                potential_signal_type = 'buy'
            elif currentMfi > 80 and data.loc[i, 'adx'] > 25 and data.loc[i, 'rsi'] > 50 and not in_uptrend:
                potential_signal_time = current_time
                potential_signal_type = 'sell'

        # بررسی عبور برای بستن سیکل
        if cycle_open:
            if cycle_type == 'buy' and currentMfi > 80 and previous_mfi[i] < 80:
                cycle_open = False
                data.loc[i, 'trade_signal'] = 'sell'
                data.loc[i, 'cycle_status'] = 'closed'
                cycle.append('closed')
            elif cycle_type == 'sell' and currentMfi < 20 and previous_mfi[i] > 20:
                cycle_open = False
                data.loc[i, 'trade_signal'] = 'buy'
                data.loc[i, 'cycle_status'] = 'closed'
                cycle.append('closed')

    return data



# تابغ برای دریافت داده های تاریخی
def fetch_historical_data(symbol, timeframe,limit):
    if csv==False:
        exchange=ccxt.bingx()
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe,limit=limit)
        df=pd.DataFrame(ohlcv,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        # df['current'] = df['close']
        df['current'] = 0
        # df['current'].iloc[-1]=getLastPrice(df)
        print(df["current"].iloc[-1])
        print(df["close"].iloc[-1])
        return df
    else:
        cv=pd.read_csv("data.csv")

        # cv['timestamp']=pd.to_datetime(cv['timestamp'],unit='ms')
        cv2=cv.filter(['timestamp','open','high','low','close','volume'])
        cv2['timestamp'] = pd.to_datetime(cv2['timestamp'], format="%Y-%m-%d %H:%M:%S")
        # time_format = "%Y-%m-%d %H:%M:%S"
        # timestamp = datetime.strptime(cv2['timestamp'], time_format)
        # cv2['timestamp']=timestamp
        return cv2


# تابغ برای دریافت آحرین قیمت بازار
def get_latest_market_price(symbol,timeframe='1m',limit=1):
            if csv==False:
                data=fetch_historical_data(symbol,timeframe,limit)

                if not data.empty:
                    return data['close'].iloc[-1]
                else:
                    return None
            else:
                 cv=pd.read_csv("data.csv")
                 return cv['close'].iloc[-1]

def MlpClassifier():
    # Initialize empty dictionaries to store best hyperparameters and models
    best_params = {}
    best_models = {}
    # Initialize an empty list to store the best models
    all_best_models = []  # This list will store the best model for each column

    # Assuming `data` is already defined and contains your dataset
    timestamp = data['timestamp'].iloc[-1]
    current_time = timestamp.strftime("%Y-%m-%d %H-%M-%S")
    data.to_csv(f"Data{current_time}.csv")

    # Extract the target variable (label) for the current column
    y = data["target"]
    X = data[['rsi', 'mid_channel', 'upper_channel', 'lower_channel', 'mfi']]
    X = X.fillna(X.mean())

    # Calculate the length of the data
    data_length = len(X)

    # Calculate the number of test data points (20% of the data)
    test_size = int(data_length * 0.02)

    # Separate the last 20% of the data for testing
    X_train = X[:-test_size]
    X_test = X[-test_size:-shiftedNumber]
    y_train = y[:-(test_size)]
    y_test = y[-test_size:-shiftedNumber]

    # Define hyperparameters to test for MLP
    hidden_layer_sizes_options = [(50, 50), (100,), (50, 100, 50)]
    learning_rate_options = ['constant', 'invscaling', 'adaptive']
    best_acc = 0  # Initialize the best accuracy to 0

    # Iterate over the hyperparameters to find the best model
    for hidden_layer_sizes in hidden_layer_sizes_options:
        for learning_rate in learning_rate_options:
            # Create and train an MLP model
            mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate=learning_rate, max_iter=1000)
            mlp_model.fit(X_train, y_train)

            # Evaluate the model's performance on the testing set
            accuracy = accuracy_score(y_test, mlp_model.predict(X_test))

            # Update best values if current accuracy is higher
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = {'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate': learning_rate}
                best_model = mlp_model

    # Store the best hyperparameters and model
    best_models[0] = best_model

    # Append the best model to the all_best_models list
    all_best_models.append(best_models[0])

    predictions = all_best_models[0].predict(X_test)

    accuracy=recall_score(y_test,all_best_models[0].predict(X_test),average='macro')
    conf = confusion_matrix(y_test,predictions)
    print(f"confusion Matrix:\n{conf}")
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,predictions))
    print('recall_score = ',recall_score(y_test,predictions,average='macro') )

    accuracy = accuracy_score(y_test, all_best_models[0].predict(X_test))
    if accuracy > expectedAccuracy:
        predictions = all_best_models[0].predict(X[-1:])
    array = np.array(predictions)
    predictionNumber = int(array[-1])
    return all_best_models, accuracy, predictionNumber

def CalculatePerAcc(y_test, predictions, conf):
    precision_per_class = precision_score(y_test, predictions, average=None)
    # print("Precision per class:")
    # for idx, precision in enumerate(precision_per_class):
    #     print(f"Class {idx}: {precision:.2f}")

    # محاسبه تعداد نمونه‌های هر کلاس
    class_counts = conf.sum(axis=1)

    # محاسبه میانگین وزنی Precision دو کلاس اول
    weight_0 = class_counts[0]
    weight_1 = class_counts[1]
    total_weight = weight_0 + weight_1

    weighted_precision = (precision_per_class[0] * weight_0 + precision_per_class[1] * weight_1) / total_weight

    # print(f"Weighted Precision for classes 0 and 1: {weighted_precision:.2f}")
    return weighted_precision


# def CalculatePerAcc(y_true, y_pred, conf_matrix):
#     per_acc = (conf_matrix[1,1] + conf_matrix[0,0]) / np.sum(conf_matrix)
#     return per_acc

# def CalculatePerAcc(y_true, y_pred, conf_matrix):
#     if conf_matrix.shape == (3, 3):
#         per_acc = (conf_matrix[1, 1] + conf_matrix[0, 0]) / np.sum(conf_matrix)
#         return per_acc
#     else:
#         return 0


def Knn(X_train,X_test,y_train,y_test):
    
    # pca = PCA(n_components=0.95)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
   

    # از الگوریتم knn  برای پیش بینی استفاده کردیم
    # Initialize empty dictionaries to store best n_neighbors and models
    best_n_neighbors = {}
    best_models = {}
    #برای هر ستون یک مدل جداگانه اموزش دادیم و مقادیر بهترین  تعداد همسایه برای هر ستون پیدا کردیم
    # مدلی که بیشترین دقت را داشت ذخیره کردیم
    # Initialize an empty list to store the best models
    all_best_models = []  # This list will store the best model for each column

    # Iterate through each column (feature)

    # y = data["target"]
    # X = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','close']]


    # pca = PCA(n_components=0.95)  # Keep 95% of the variance
    # X = pca.fit_transform(X)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #X=X.fillna(X.mean())
    # timestamp=data['timestamp'].iloc[-1]
    # current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")
    # data.to_csv(f"Data{current_time}.csv")

        # فرض می‌کنیم X و y دیتاهای شما هستند
    # data_length = len(X)

    # محاسبه تعداد داده‌های تست (20% انتهایی)
    # test_size = int(data_length * percentTest)

    # جدا کردن 20% انتهایی داده‌ها برای تست
    # X_train = X[:-test_size]
    # X_test = X[-test_size:-shiftedNumber]
    # y_train = y[:-(test_size)]
    # y_test = y[-test_size:-shiftedNumber]


    # X=data[['rsi','adx','mfi','supertrend']]

    # scaler = MinMaxScaler()


            # نرمال سازی X
    # X = scaler.fit_transform(X)
    knn_model_best=[]
    # Find the optimal n_neighbors for this column
    best_n, best_acc = 0, 0  # Initialize best values
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    for n_neighbors in range(3, 100):  # Test different n_neighbors values
        # Split data into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        # Create and train a KNN model
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)

        # Evaluate the model's performance on the testing set
        # accuracy = accuracy_score(y_test, knn_model.predict(X_test))
        # accuracy=recall_score(y_test,knn_model.predict(X_test),average='macro')
        perdictTemp=knn_model.predict(X_train)
        conf = confusion_matrix(y_train,perdictTemp)
        accuracy=CalculatePerAcc(y_train, perdictTemp, conf)
    #  print(accuracy)

        # Update best values if current accuracy is higher
        if accuracy > best_acc:
            best_n, best_acc = n_neighbors, accuracy
            knn_model_best = knn_model

    # Store the best n_neighbors and model for this column
    best_n_neighbors[0] = best_n
    best_models[0] = knn_model_best

    # Append the best model for this column to the all_best_models list
    all_best_models.append(best_models[0])

    predictions=all_best_models[0].predict(X_test)


    # data["predictions"]=predictions
    # accuracy = accuracy_score(y_test, all_best_models[0].predict(X_test))
    # accuracy=recall_score(y_test,all_best_models[0].predict(X_test),average='macro')
    conf = confusion_matrix(y_test,predictions)
    accuracy = CalculatePerAcc(y_test, predictions, conf)

    print(f"confusion Matrix:\n{conf}")
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,predictions))
    print('recall_score = ',recall_score(y_test,predictions,average='macro') )
    print(f'accuracy:{accuracy}')
    # print(f"Weighted Precision for classes 0 and 1: {CalculatePerAcc(y_test, predictions, conf):.2f}")
    # predictions=all_best_models[0].predict(X[-1:])

    # array=np.array(predictions)
    # predictionNumber= int( array [-1] )
    # print(f'predictions:{predictions}')

    # if predictionNumber !=  lastSignalExcell :
    
    #     ExtractToExcell(predictionNumber,'Signal.csv')

    if accuracy>expectedAccuracy :
        return predictions,all_best_models[0],accuracy
    
    return predictions,all_best_models[0],accuracy
    # else:
    #     return None,None,None

def calculate_auc_pr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def CalculatePerAccGrid(y_true, y_pred):
    precision_per_class = precision_score(y_true, y_pred, average='weighted')
    conf = confusion_matrix(y_true, y_pred)

    class_counts = conf.sum(axis=1)
    weight_0 = class_counts[0]
    weight_1 = class_counts[1]
    total_weight = weight_0 + weight_1

    weighted_precision = (precision_per_class[0] * weight_0 + precision_per_class[1] * weight_1) / total_weight

    return weighted_precision





def SvmOld(X_train, X_test, y_train, y_test):
    # Perform PCA on training and test data
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Define values to iterate over for C and gamma
    c_values = [0.01, 0.1, 1, 10, 100]
    gamma_values = [0.01, 0.001, 0.005, 0.5, 0.1, 1, 10, 100]

    # Define parameter grid
    param_grid = {
        'C': c_values,
        'gamma': gamma_values,
        'kernel': ['rbf']
    }

    # Initialize SVM classifier
    svm_model = SVC()

    # Initialize GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

    # Fit GridSearchCV on PCA-transformed training data
    grid_search.fit(X_train_pca, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Best accuracy
    best_accuracy = grid_search.best_score_
    print(f"Best cross-validation accuracy: {best_accuracy * 100:.2f}%")

    # Train final SVM model with best parameters
    best_svm_model = SVC(**best_params)
    best_svm_model.fit(X_train_pca, y_train)

    # Predict using the best model on test set
    final_predictions = best_svm_model.predict(X_test_pca)
    final_predictions_train = best_svm_model.predict(X_train_pca)

    # Calculate accuracy and confusion matrix on test set
    final_acc = accuracy_score(y_test, final_predictions)
    final_conf = confusion_matrix(y_test, final_predictions)

    # Calculate accuracy and confusion matrix on training set
    final_acc_train = accuracy_score(y_train, final_predictions_train)
    final_conf_train = confusion_matrix(y_train, final_predictions_train)

    # Print final metrics
    print(f"Final accuracy on test set: {final_acc * 100:.2f}%")
    print(f"Confusion matrix on test set:\n{final_conf}")
    print(f"Final accuracy on training set: {final_acc_train * 100:.2f}%")
    print(f"Confusion matrix on training set:\n{final_conf_train}")

    # Print classification report on test set
    print('Classification Report on test set:\n', classification_report(y_test, final_predictions))

    return final_predictions, best_svm_model

# Example usage:
# Assuming X_train, X_test, y_train, y_test are already defined
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# final_predictions, best_svm_model = Svm(X_train, X_test, y_train, y_test)


def Svm(X_train, X_test, y_train, y_test):
    pca = PCA(n_components=0.95)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    X_train_pca = X_train
    X_test_pca = X_test
    # Define values to iterate over for C and gamma
    c_values = [0.01, 0.1, 1, 10, 100, 1000] 
    gamma_values = [0.01,0.001,0.005,0.5, 0.1, 1, 10, 100,1000,0.0001,0.9,0.8,0.7,0.6,0.4]
    # gamma_values = [0.01,0.001,0.005,0.5, 0.1, 1, 10, 100,1000,0.0001,0.9,0.8,0.7,0.6,0.4]

    best_acc = -1
    best_params = {'C': None, 'gamma': None}

    # Iterate over all combinations of C and gamma
    for C in c_values:
        for gamma in gamma_values:
            # Initialize SVM classifier with current parameters
            svm_model = SVC(C=C, gamma=gamma, kernel='rbf')

            # Fit SVM model on PCA transformed training data
            svm_model.fit(X_train_pca, y_train)

            # Predict on test data
            # svm_predictions = svm_model.predict(X_test_pca)
            svm_predictions = svm_model.predict(X_test_pca)

            # Calculate accuracy
            # acc = CalculatePerAccGrid(y_test, svm_predictions)
            # acc = CalculatePerAccGrid(y_test, svm_predictions)
            acc = precision_score(y_test, svm_predictions, average='weighted')
            # acc = precision_score(y_test, svm_predictions, average='weighted')

            # Print current parameters and accuracy
            print(f"C={C}, gamma={gamma}, Accuracy={acc * 100:.2f}%")

            # Check if current model is better than previous best
            if acc > best_acc:
                best_acc = acc
                best_params['C'] = C
                best_params['gamma'] = gamma

    print(f"Best parameters found: {best_params}")
    print(f"Best accuracy found: {best_acc * 100:.2f}%")

    # Train final SVM model with best parameters
    best_svm_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
    best_svm_model.fit(X_train_pca, y_train)

    # Predict using the best model
    final_predictions = best_svm_model.predict(X_test_pca)
    final_predictions2 = best_svm_model.predict(X_train_pca)

    final_conf2 = confusion_matrix(y_train, final_predictions2)

    # Calculate accuracy and confusion matrix
    final_acc = accuracy_score(y_test, final_predictions)
    final_conf = confusion_matrix(y_test, final_predictions)
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,final_predictions))
    print('metrics.classification_report2:=\n',metrics.classification_report(y_train,final_predictions2))
    print(f"Final accuracy: {final_acc * 100:.2f}%")
    print(f"Confusion matrix:\n{final_conf}")
    print(f"Confusion matrix2:\n{final_conf2}")

    return final_predictions, best_svm_model
def Lstmold(X_train, X_test, y_train, y_test, expectedLstmModel=expectedLstmModel):
    # pca = PCA(n_components=0.95)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)


    X_train_pca = X_train
    X_test_pca = X_test

    # Reshape for LSTM input
    X_train_lstm = X_train_pca.values.reshape((-1, X_train_pca.shape[1], 1))
    X_test_lstm = X_test_pca.values.reshape((-1, X_test_pca.shape[1], 1))

    # Define the LSTM model
    model_lstm = Sequential()

    # Define the model architecture
    model_lstm.add(Input(shape=(X_train_pca.shape[1], 1)))
    model_lstm.add(LSTM(units=100, return_sequences=True, activation='relu'))
    model_lstm.add(LSTM(units=50, activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=25, activation='relu'))
    model_lstm.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    acc = -1
    while acc < expectedLstmModel:
        # Train the LSTM model
        model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

        # Predict with the LSTM model
        lstm_predictions = model_lstm.predict(X_test_lstm)
        lstm_predictions = (lstm_predictions > 0.5).astype(int)

        # Calculate confusion matrix and prediction accuracy
        conf = confusion_matrix(y_test, lstm_predictions)
        accuracy_lstm = accuracy_score(y_test, lstm_predictions)
        acc = accuracy_lstm
        print(f"LSTM Accuracy: {accuracy_lstm * 100:.2f}%")

    return lstm_predictions, model_lstm, accuracy_lstm

def Lstm(X_train, X_test, y_train, y_test):



    model_lstm = Sequential()
    
    # Define the model architecture
    model_lstm.add(Input(shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(units=100, return_sequences=True, activation='relu'))
    # model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=50, activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=25, activation='relu'))
    model_lstm.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification
    
    # Compile the model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    
    # Reshape the features for LSTM
    X_train_lstm = X_train.values.reshape((-1, X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((-1, X_test.shape[1], 1))

    # Setup EarlyStopping
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the LSTM model
    # model_lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    acc=-1
    while acc<expectedLstmModel:
        model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2)

        # Predict with the LSTM model
        lstm_predictions = model_lstm.predict(X_test_lstm)
        lstm_predictions = (lstm_predictions > 0.5).astype(int)  # Convert to 0 and 1

        # Calculate confusion matrix and prediction accuracy
        conf = confusion_matrix(y_test, lstm_predictions)
        accuracy_lstm = accuracy_score(y_test, lstm_predictions)
        acc=accuracy_lstm
        print(f"LSTM Accuracy: {accuracy_lstm * 100:.2f}%")
    # accuracy_lstm=round(((accuracy_lstm)*100),2)
    return lstm_predictions, model_lstm,accuracy_lstm





# تابع تصمیم‌گیری برای برچسب‌گذاری با رای‌گیری
def determine_label(row):
    votes = []
    
    # اضافه کردن شرایط بر اساس اندیکاتورهای مختلف
    if row['rsi'] > rsi_overbought:
        votes.append(0)  # سل
    elif row['rsi'] < rsi_oversold:
        votes.append(1)  # بای
    
    if row['macd'] > macd_threshold:
        votes.append(0)  # سل
    elif row['macd'] < macd_threshold:
        votes.append(1)  # بای
    
    # if row['adx'] > adx_threshold:
    #     votes.append(0)  # سل
    # elif row['adx'] < adx_threshold:
    #     votes.append(1)  # بای
    
    if row['cci'] > cci_overbought:
        votes.append(0)  # سل
    elif row['cci'] < cci_oversold:
        votes.append(1)  # بای
    
    if row['stochastic_oscillator'] > stochastic_high:
        votes.append(0)  # سل
    elif row['stochastic_oscillator'] < stochastic_low:
        votes.append(1)  # بای
    
    # if row['ema_short'] > row['ema_long']:
    #     votes.append(0)  # سل
    # elif row['ema_short'] < row['ema_long']:
    #     votes.append(1)  # بای
    
    # تعیین برچسب نهایی بر اساس اکثریت رای‌ها
    if len(votes) == 0:
        return 2  # No Action (هیچکدام)
    else:
        majority_vote = np.bincount(votes).argmax()
        return majority_vote

def TrainLstmSVm():
    global data
    global accuracy
    # global LstmAccuracy
    global scaler
    accuracy=0
    LstmAccuracy=0
    CombineAccuracy=0
    combined_predictions=-3
    # data=data.dropna()
    data = dropNaFix(data)
    y = data["target"]
    # X = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','current']]
    X = data[['close','volume','rsi','macd','macd_diff','williams_r','dpo']]

    # pca = PCA(n_components=0.95)  # Keep 95% of the variance
    # X = pca.fit_transform(X)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #X=X.fillna(X.mean())
    timestamp=data['timestamp'].iloc[-1]
    current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")
    data.to_csv(f"Data{current_time}.csv")

        # فرض می‌کنیم X و y دیتاهای شما هستند
    data_length = len(X)

    # محاسبه تعداد داده‌های تست (20% انتهایی)
    test_size = int(data_length * percentTest)

    # جدا کردن 20% انتهایی داده‌ها برای تست
    # X_train = X[:-test_size]
    # X_test = X[-test_size:-shiftedNumber]
    # y_train = y[:-(test_size)]
    # y_test = y[-test_size:-shiftedNumber]
    
    X_train = X[:-test_size]
    X_test = X[-test_size:-shiftedNumber]
    y_train = y[:-test_size]
    y_test = y[-test_size:-shiftedNumber]
    finalScore=-3
    # نرمال‌سازی داده‌ها
   # X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
   # # ویژگی‌ها و برچسب‌ها را تعیین کنید
    # while expectedCombineModel>accuracy or  CombineAccuracy<0.6 :

        
 


    # آموزش مدل LSTM
    lstm_predictions,lstm_Model,LstmAccuracy = Lstm(X_train, X_test, y_train, y_test)
    knn_predictions,knn_Model,KnnAccuracy = Knn(X_train, X_test, y_train, y_test)
    
    # آموزش مدل SVM
    svm_predictions,svm_Model = Svm(X_train, X_test, y_train, y_test)
    
    # finalScore=lstm_predictions
    results=[]
    for i in range (X_test.shape[0]):
        if lstm_predictions[i]==knn_predictions[i]:
            results.append(lstm_predictions[i])
        elif lstm_predictions[i]==svm_predictions[i]:
            results.append(lstm_predictions[i])
        else:
            results.append(svm_predictions[i])


        
        


    # svm_Model=None
    # combined_model=None
    # ترکیب پیش‌بینی‌های LSTM و اسکورهای SVM
    # combined_X = np.column_stack((lstm_predictions, scores))

        # ساخت مدل ترکیبی (مثلاً Random Forest)
    # combined_model = RandomForestClassifier()
    # combined_model.fit(combined_X, y_test)
    # # پیش‌بینی خرید و فروش با مدل ترکیبی
    # combined_predictions = combined_model.predict(combined_X)

    # combined_predictions=lstm_predictions
    print(f"shape YTest:{y_test.shape}")
    # results=np.array(results)
    # print(f"Shape Result{results.shape}")
    # results = np.array(results.values, dtype=object)
    # results = results.values
    results = [x.item() if isinstance(x, np.ndarray) else x for x in results]
    results = np.array(results)
    print(results)
    # results = np.array(results)
    conf=confusion_matrix(y_test,results)
    # conf=confusion_matrix(y_test,combined_predictions)
    # محاسبه درصد درستی پیش‌بینی‌ها
    accuracy = CalculatePerAcc(y_test, results,conf)
    # accuracy = CalculatePerAcc(y_test, combined_predictions,conf)
    print(f"accuracy {accuracy * 100:.2f}%")
    CombineAccuracy = accuracy_score(y_test, results)
    # CombineAccuracy = accuracy_score(y_test, combined_predictions)
    print(f"accuracy Combine {CombineAccuracy * 100:.2f}%")


    # X_test_lstm = X[-1:].values.reshape((-1, 8, 1))
    # lstm_predictions=lstm_Model.predict(X_test_lstm)
    # lstm_predictions = (lstm_predictions > 0.5).astype(int)
    # svm_predictions = svm_Model.predict(X[-1:])


    # ایجاد اسکور بر اساس پیش‌بینی‌های SVM
    # scores = [0 if prediction == 0 else 1 for prediction in svm_predictions]
    # محاسبه درصد درستی پیش‌بینی‌ها
    # combined_X = np.column_stack((lstm_predictions, scores))

    # ساخت مدل ترکیبی (مثلاً Random Forest)
    # combined_model = RandomForestClassifier()
    # combined_model.fit(combined_X, y_test)

    # پیش‌بینی خرید و فروش با مدل ترکیبی
    # combined_predictions = combined_model.predict(combined_X)
    # conf = confusion_matrix(y_test,combined_predictions)
    # accuracy=CalculatePerAcc(y_test, combined_predictions, conf)
    print(f"confusion Matrix:\n{conf}")
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,results))
    # print('metrics.classification_report:=\n',metrics.classification_report(y_test,combined_predictions))
    # return lstm_Model,svm_Model,combined_model,accuracy
    return lstm_Model,svm_Model,knn_Model,accuracy

def Election(lstm_Model,svm_Model,knn_Model):
    global data
    global accuracy
    data = dropNaFix(data)
    y = data["target"]
    # X = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','close']]
    # X2 = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','current']]
    X2 = data[['close','volume','rsi','macd','macd_diff','williams_r','dpo']]

    pca = PCA(n_components=0.95)
    


       # فرض می‌کنیم X و y دیتاهای شما هستند
    data_length = len(X2)

    # محاسبه تعداد داده‌های تست (20% انتهایی)
    test_size = int(data_length * percentTest)
  # x2=scaler.transform(X2)
   # x=scaler.transform(X2)
    x2=X2
    x=X2

    X_Predict = x2[-1:]       
    X_test = x[-test_size:-shiftedNumber]
    y_test = y[-test_size:-shiftedNumber]
    # X_test_pca = pca.fit_transform(x[:-test_size])
    # X_predict_pca = pca.transform(X_Predict)
    X_test_pca = X_test
    X_predict_pca = X_Predict


    # pca = PCA(n_components=0.95)  # Keep 95% of the variance
    # X = pca.fit_transform(X)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #X=X.fillna(X.mean())
    timestamp=data['timestamp'].iloc[-1]
    current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")
    data.to_csv(f"Data{current_time}.csv")

 
    
    # محاسبه درصد درستی پیش‌بینی‌ها
    print(f" x2[-1] :{X2[-1:]}")
    print(f" lastTest :{X_Predict}")
   
    
    X_Predict_lstm = X_Predict.values.reshape((-1,  X_Predict.shape[1], 1))
    X_test_lstm = X_test.values.reshape((-1, X_test.shape[1], 1))

    lstm_predictions=lstm_model.predict(X_Predict_lstm)
    lstm_predictionsTest=lstm_model.predict(X_test_lstm)

    lstm_predictions = (lstm_predictions > 0.5).astype(int)
    lstm_predictionsTest = (lstm_predictionsTest > 0.5).astype(int)

    
    svm_predictions = svm_model.predict(X_predict_pca)
    svm_predictionsTest = svm_model.predict(X_test_pca)
    # ایجاد اسکور بر اساس پیش‌بینی‌های SVM
    svm_predictions = [0 if prediction == 0 else 1 for prediction in svm_predictions]
    svm_predictionsTest = [0 if prediction2 == 0 else 1 for prediction2 in svm_predictionsTest]
    # محاسبه درصد درستی پیش‌بینی‌ها
    knn_predictions = knn_Model.predict(X_predict_pca)
    knn_predictionsTest = knn_Model.predict(X_test_pca)
   # print(f" combined_X :{combined_X}")
    #print(f" combined_XTest :{combined_XTest}")

    resultTest=[]
    for i in range (X_test.shape[0]):
        if lstm_predictionsTest[i]==knn_predictionsTest[i]:
            resultTest.append(lstm_predictionsTest[i])
        elif lstm_predictionsTest[i]==svm_predictionsTest[i]:
            resultTest.append(lstm_predictionsTest[i])
        else:
            resultTest.append(svm_predictionsTest[i])
    resultTest = [x.item() if isinstance(x, np.ndarray) else x for x in resultTest]
    resultTest = np.array(resultTest)
    print(resultTest)
    conf=confusion_matrix(y_test,resultTest)
    accuracy = CalculatePerAcc(y_test, resultTest,conf)
    print(f"accuracy {accuracy * 100:.2f}%")
    CombineAccuracy = accuracy_score(y_test, resultTest)
    print(f"accuracy Combine {CombineAccuracy * 100:.2f}%")
    print(f"confusion Matrix:\n{conf}")
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,resultTest))


    print(f" Predict Lstm {lstm_predictions[0]}")
    print(f" Predict Svm {svm_predictions[0]}")
    print(f" Predict Knn {knn_predictions[0]}")
    results=-3
    # if lstm_predictions[0]==knn_predictions[0]:
    #     results=lstm_predictions[0]
    # elif lstm_predictions[0]==svm_predictions[0]:
    #     results=lstm_predictions[0]
    # else:
    #     results=svm_predictions[0]
   # combined_predictions=results
    # combined_predictions=results
    combined_predictions=svm_predictions[0]


    #combined_predictionsTest=lstm_predictionsTest
    #combined_predictions=lstm_predictions
    # print(f"درصد درستی پیش‌بینی ترکیبی: {accuracy * 100:.2f}%")
    # print(f"combine prediction: {combined_predictions}")

    # conf = confusion_matrix(y_test,combined_predictionsTest)
    # accuracy = CalculatePerAcc(y_test, combined_predictionsTest, conf)

    # print(f"confusion Matrix:\n{conf}")
    # print('metrics.classification_report:=\n',metrics.classification_report(y_test,combined_predictionsTest))
    # return True,accuracy,combined_predictions

    
    # array=np.array(combined_predictions)
    # predictionNumber= int( array [-1] )
    predictionNumber= combined_predictions
    print(f'combined_predictions:{combined_predictions}')
    return True,accuracy,predictionNumber



def PredictCombine(lstm_model,svm_model,combine_model):

    global data
    global accuracy

    data = dropNaFix(data)
    y = data["target"]
    # X = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','close']]
    # X2 = data[['rsi', 'macd', 'bollinger_hband', 'bollinger_lband',  'mfi', 'stochastic','mid_channel','current']]
    X2 = data[['close','volume','rsi','macd','macd_diff','williams_r','dpo']]


       # فرض می‌کنیم X و y دیتاهای شما هستند
    data_length = len(X2)

    # محاسبه تعداد داده‌های تست (20% انتهایی)
    test_size = int(data_length * percentTest)
  # x2=scaler.transform(X2)
   # x=scaler.transform(X2)
    x2=X2
    x=X2

    X_Predict = x2[-1:]
    X_test = x[-test_size:-shiftedNumber]
    y_test = y[-test_size:-shiftedNumber]
    


    # pca = PCA(n_components=0.95)  # Keep 95% of the variance
    # X = pca.fit_transform(X)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #X=X.fillna(X.mean())
    timestamp=data['timestamp'].iloc[-1]
    current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")
    data.to_csv(f"Data{current_time}.csv")

 
    
    # محاسبه درصد درستی پیش‌بینی‌ها
    print(f" x2[-1] :{X2[-1:]}")
    print(f" lastTest :{X_Predict}")
   
    
    X_Predict_lstm = X_Predict.values.reshape((-1, 7, 1))
    X_test_lstm = X_test.values.reshape((-1, 7, 1))
    lstm_predictions=lstm_model.predict(X_Predict_lstm)
    lstm_predictionsTest=lstm_model.predict(X_test_lstm)
    lstm_predictions = (lstm_predictions > 0.5).astype(int)
    lstm_predictionsTest = (lstm_predictionsTest > 0.5).astype(int)
    svm_predictions = svm_model.predict(X_Predict)
    svm_predictionsTest = svm_model.predict(X_test)
    # ایجاد اسکور بر اساس پیش‌بینی‌های SVM
    scores = [0 if prediction == 0 else 1 for prediction in svm_predictions]
    scoresTest = [0 if prediction2 == 0 else 1 for prediction2 in svm_predictionsTest]
    # محاسبه درصد درستی پیش‌بینی‌ها
    combined_X = np.column_stack((lstm_predictions, scores))
    combined_XTest = np.column_stack((lstm_predictionsTest, scoresTest))
   # print(f" combined_X :{combined_X}")
    #print(f" combined_XTest :{combined_XTest}")
    combined_predictions = combine_model.predict(combined_X)
    combined_predictionsTest = combine_model.predict(combined_XTest)
    # accuracy = accuracy_score(y_test, combine_model)

    #combined_predictionsTest=lstm_predictionsTest
    #combined_predictions=lstm_predictions
    print(f"درصد درستی پیش‌بینی ترکیبی: {accuracy * 100:.2f}%")
    print(f"combine prediction: {combined_predictions}")

    conf = confusion_matrix(y_test,combined_predictionsTest)
    accuracy = CalculatePerAcc(y_test, combined_predictionsTest, conf)

    print(f"confusion Matrix:\n{conf}")
    print('metrics.classification_report:=\n',metrics.classification_report(y_test,combined_predictionsTest))
    return True,accuracy,combined_predictions


def ExtractToExcell(predictionNumber,fileName):
    global lastSignalExcell
    timestamp=data['timestamp'].iloc[-1]
    current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")

    lastSignalExcell=predictionNumber
    # get_latest_market_price(symbol,'1m',1)
            # قیمت فعلی و پیش‌بینی را در یک مجموعه داده ذخیره کنید
    dataExcell = pd.DataFrame({
        "current_price": data['close'].iloc[-1],
        "prediction": predictionNumber,
        "timeStamp":current_time
        }, index=[0])


        # data.to_csv(f"Signal.csv")
        # dataExcell.to_csv(f"Signal.csv", mode='a', header=False, index=False)
    file_path=fileName
    if os.path.exists(file_path):
        dataExcell.to_csv(file_path, mode='a', header=False, index=False)
    else:
            dataExcell.to_csv(file_path, mode='w', header=True, index=False)


def send_prediction_to_telegram(prediction,timestamp,symbol,accuracy,price,actionStatus,timeframe,best_k,profit,signalPrice,closedMethod):
    if accuracy<expectedAccuracy:
        # print('دقت مدل کمتر از 57 درصد است و سیگنال منتشر نمیشود')
        percent=expectedAccuracy*100
        print(f'accuracy is :{accuracy} and less than {percent}%')
        return

    current_time=timestamp.strftime("%Y-%m-%d %H:%M:%S")
    # action="Buy" if prediction==1 elif pre "Sell"
    if prediction==0:
        action="Sell"
    elif prediction==1:
        action="Buy"
    else:
        action="No Action"

    if actionStatus=='Open':
         actionStatus_message="Cycle Open"
    elif actionStatus=='Close':
        actionStatus_message= "Cycle Closed By"+closedMethod
    else:
        actionStatus_message="none"

    message= (
        f"Date and Time:{current_time} \n"
        f"Symbol: {symbol} \n"
        f"Action: {action} \n"
        f"Signal Price: {signalPrice} \n"
        f"TimeFrame: {timeframe} \n"
        f"Cycle Status: {actionStatus_message} \n"
        f"Current Price: {price} \n"
        f"Profit Percent: {profit} \n"
        f"Model Accuracy: {accuracy:.2f}\n"
        f"best_k: {best_k} \n"
        f"Robot ID: {BotId}"
    )

    print(message)
    # bot.sendMessage(chat_id=telegram_channel_id,text=message)
    if csv==False:
        try:
            bot.send_message(telegram_channel_id, message)
        except:
            print("could not connect to telegram")

    try:
        ExtractToExcell(prediction,'SignalControl.csv')
    except:
            print("could write to SignalControl.csv")







#تعریف اولیه داده
data=fetch_historical_data(symbol,timeframe,limit)
# data=calculate_rsi(data)
# data=calculate_adx(data)
# data=calculate_mfi(data)
# data=calculate_linear_regression_channel(data)
# data=calculate_supertrend(data)
# data=calculate_macd(data)
data=add_indicators(data)
data['cycle_status']=None
data['trade_signal']=None
last_data_fetch_time=get_cyprus_time()
# data=CloseSignalWithIndicators(data)
# data=calculate_buy_sell_signalsSuperTrend(data)
# data=calculate_di(data)
# data=calculate_di(data)
# test=calculate_di(data)
# data['plus_di']=test[1][0]
# data['plus_di'],data['minus_di']=
#محاسبه سیگنال های خرید و فروش با استفاده از داده های بارگذاری شده
# signals=calculate_buy_sell_signals(data)
# signals=calculate_buy_sell_signals(data)

#تعریف زمان برای آخرین بارگذاری داده ها

#محاسبه سیگنال های خرید و فروش و بروزرسانی وضعیت سیکل
# data=calculate_buy_sell_signals(data)
#رسم نمودار
# drawChart()
# chart_thread = threading.Thread(target=drawChart)
# chart_thread.start()


def backtest(data,lot_size,intial_balance):
    balance=initial=intial_balance
    postion=0
    trade_log=[]
    cycle_log=[]

    for i in range(len(data)):
        signal=data.loc[i,'trade_signal']
        close_price=data.loc[i,'close']
        cycle_status=data.loc[i,'cycle_status']

        if signal=='buy' and postion==0:
            #خرید و باز کردن سیکل
            postion=lot_size/close_price
            balance-=lot_size
            trade_log.append(('buy',close_price,balance))
            if cycle_status=='open':
                cycle_log.append(('Cycle Opened',i))

        elif signal=='sell' and postion >0:
            #فروش و بستن سیکل
            balance+=postion*close_price
            postion=0
            trade_log.append(('sell',close_price,balance))
            if cycle_status=='closed':
                cycle_log.append(('Cycle Closed',i))

    #بستن پوزیشن باقی مانده در انتهای داده ها
    if postion >0:
        balance+=postion * data.iloc[-1]['close']
        trade_log.append(('sell',data.iloc[-1]['close'],balance))
        postion=0

    return balance,trade_log,cycle_log


    #مثال برای نمایش نتایج  بک تست
# final_balance,trades,cycles=backtest(data,lot_size=1000,initial_balance=10000)

# print(f"Final Balance:{final_balance}")
# for trade in trades:
#     print(f"Trade: {trade}")
# for cycle in cycles:
#     print(f"Cycle: {cycle}")


# final_balance,trades,cycles=backtest(data,lot_size=1000,intial_balance=10000)

# print(f"Final Balance: {final_balance}")
# for trade in trades:
#     print(f"Trade: {trade}")
# for cycle in cycles:
#     print(f"Cycle: {cycle}")
# تابع برای تعیین مقدار ستون target براساس درصد تغییر
def determine_target(percent_change,expectedChange):
    if percent_change > expectedChange:
        return 1
    elif percent_change < -expectedChange:
        return 0
    else:
        return 3

# حلقه اصلی


def FirstTrain():
    global data
    global trainTimeStamp
    print("while On")
    current_time=get_cyprus_time()

    #1. بارگذاری داده ها
    new_data=fetch_historical_data(symbol,timeframe,limit)
    #اطمینان از اینکه داده های جدید حاوی NaN نباشند
    new_data=new_data.ffill()

    #2. بررسی تغییرات در داده ها
    if not new_data.equals(data):
        data=new_data
        data['next_close'] = data['close'].shift(-shiftedNumber)
        data['target']= (data['next_close']>data['close']).astype(int)
        data=add_indicators(data)
        CalculateProfit()     
                    
    
    # lstm_model,svm_model,combine_model,accuracy=TrainLstmSVm()
    # printAcc= round(((accuracy)*100),3)
    # if(lstm_model!= None):
    #     try:
    #         joblib.dump(lstm_model, f'model/lstm_model-{symbol}-{timeframe}-{printAcc}.joblib')
    #     except e:
    #         print("could not Save Lstm Model :{e}")
    # if(svm_model!= None):
    #     try:
    #         joblib.dump(svm_model, f'model/svm_model{symbol}-{timeframe}-{printAcc}.joblib')
    #     except e:
    #         print("could not Save Svm Model :{e}")
    # if(combine_model!= None):
    #     try:
    #         joblib.dump(combine_model, f'model/combine_model{symbol}-{timeframe}-{printAcc}.joblib')
    #     except e:
    #         print("could not Save combine Model :{e}")
    # trainTimeStamp=datetime.now()
    return lstm_model,svm_model,combine_model,accuracy

def CalculateProfit():
    data['label'] = data.apply(determine_label, axis=1)
    data['signal']=3

    data['signal'] = data.apply(lambda row: row['target'] if row['label'] == row['target'] else None, axis=1)
    data['signal'] = data.apply(lambda row: 2 if row['label'] != row['target'] else row['target'], axis=1)

    transactions = []
    buy_price = None
    open_signal = None

    df = data
    transactions = []
    buy_price = None

    open_signal = None
    data['transaction']=0
    data['marker']=0

    for i in range(len(df)):
        signal = df.at[i, 'signal']
        if signal == 0:
            if buy_price is None:  # اگر معامله باز نیست، یک معامله باز کنید
                buy_price = df.at[i, 'close']
                open_signal = 0
                data['marker'][i]='B'
            elif open_signal == 1:  # اگر معامله باز هست و از 1 به 0 تغییر کرده است
                sell_price = df.at[i, 'close']
                profit_percent = ((sell_price - buy_price) / buy_price) * 100
                transactions.append(profit_percent)
                data['transaction'][i]=profit_percent
                data['marker'][i]='CB'
                buy_price = None  # Reset buy price after the transaction
                open_signal = None
        elif signal == 1:
            if buy_price is None:  # اگر معامله باز نیست، یک معامله باز کنید
                buy_price = df.at[i, 'close']
                open_signal = 1
                data['marker'][i]='B'
            elif open_signal == 0:  # اگر معامله باز هست و از 0 به 1 تغییر کرده است
                data['marker'][i]='CS'
                sell_price = df.at[i, 'close']
                profit_percent = ((sell_price - buy_price) / buy_price) * 100
                transactions.append(profit_percent)
                data['transaction'][i]=profit_percent
                buy_price = None  # Reset buy price after the transaction
                open_signal = None
        elif signal == 2:
            if buy_price is not None:  # اگر معامله باز هست و به 2 رسید
                sell_price = df.at[i, 'close']
                data['marker'][i]='C'
                if (open_signal==1):
                    profit_percent = ((sell_price - buy_price) / buy_price) * 100
                elif (open_signal==0):
                    profit_percent = ((sell_price - buy_price) / buy_price) * 100*-1
                data['transaction'][i]=profit_percent
                data['marker'][i]='C'
                transactions.append(profit_percent)
                buy_price = None  # Reset buy price after the transaction
                open_signal = None


    timestamp=data['timestamp'].iloc[-1]
    current_time=timestamp.strftime("%Y-%m-%d %H-%M-%S")

    data.to_csv(f"{symbol}{timeframe}{current_time}.csv")


            

    # if __name__ == '__main__':

        # process = multiprocessing.Process(target=DrawChartNew)
        # process.start()
        # process.join()
        # DrawChartNew()


    #رسم نمودار
    # plt.figure(figsize=(15,10))
    #نمودار RSI
    # فرض می‌کنیم data یک DataFrame پانداس است و ستون‌های 'timestamp', 'open', 'high', 'low', 'close' را دارد
    # data['timestamp'] = pd.to_datetime(data['timestamp'])  # تبدیل ستون timestamp به فرمت datetime
    # data['timestamp'] = data['timestamp'].map(mdates.date2num)  # تبدیل به اعداد ترتیبی


    # # ohlc = data.loc['timestamp' 'open', 'high', 'low', 'close']
    # ohlc = data.loc[:, ['timestamp', 'open', 'high', 'low', 'close']]
    # #نمودار سیگنال های خرید و فروش
    # fig, ax = plt.subplots() 
    # candlestick_ohlc(ax, ohlc.values, width=0.6, 
    #              colorup='green', colordown='red', alpha=0.8) 
    
    # ax.xaxis_date()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # ax.set_ylabel('Price') 
    # ax.set_xlabel('Date') 
    # fig.suptitle(f'Daily Candlestick Chart of {symbol}') 
    # date_format = mpl_dates.DateFormatter('%d-%m-%Y') 
    # ax.xaxis.set_major_formatter(date_format) 
    # fig.autofmt_xdate() 
    
    # fig.tight_layout() 
    
    # # fig.set_dpi(200)
    # fig.set_figwidth(640)
    # fig.set_figheight(320)
    # fig.set_size_inches(12.0, 6.0)
    # plt.axis('auto')
    # plt.show() 


    # plt.subplot(1,1,1)
    # plt.plot(data['close'],label='Close Price')
    # buy_signals=data[data['marker']=='B']
    # sell_signals=data[data['marker']=='S']
    # close_signals=data[data['marker']=='C']
    # close_signalWithSell=data[data['marker']=='CS']
    # close_signalWithBuy=data[data['marker']=='CB']
    # plt.scatter(buy_signals.index,buy_signals['close'],marker="^",color='g',label='Buy Signal',alpha=1)
    # plt.scatter(sell_signals.index,sell_signals['close'],marker="v",color='r',label='Sell Signal',alpha=1)
    # plt.scatter(close_signals.index,close_signals['close'],marker="_",color='black',label='Close Signal',alpha=1)
    # plt.scatter(close_signalWithSell.index,close_signalWithSell['close'],marker='v',color='r',label='Close With Sell',alpha=1)
    # plt.scatter(close_signalWithBuy.index,close_signalWithBuy['close'],marker='^',color='g',label='Close With Buy',alpha=1)
    # plt.title('Buy/Sell Signals')
    # plt.legend()

    # #نمایش نمودار ها
    # plt.tight_layout()
    # plt.show()

    # print(transactions)




lstm_model,svm_model,combine_model,accuracy=FirstTrain()
while True:
    try:


        print("while On")
        current_time=get_cyprus_time()

        #1. بارگذاری داده ها
        new_data=fetch_historical_data(symbol,timeframe,limit)
        #اطمینان از اینکه داده های جدید حاوی NaN نباشند
        new_data=new_data.ffill()
       

        #2. بررسی تغییرات در داده ها
        if not new_data.equals(data):
            data=new_data
            print("dataCurrent:")
            print(data["current"].iloc[-1])
            print("\n dataClose:")
            print(data["close"].iloc[-1])

            nextTrainTime = add_time_multiple(trainTimeStamp, timeframe,n_Candle_Train)
            if(nextTrainTime<datetime.now()):
                lstm_model,svm_model,combile_model,accuracy=FirstTrain()
            # فرض کنیم data یک DataFrame از pandas باشد و شامل ستون 'close' باشد

            # قیمت بسته شدن دو روز بعد
            # data['next_close'] = data['close'].shift(-shiftedNumber)

            # محاسبه درصد تغییر قیمت دو روز بعد نسبت به قیمت فعلی
            # data['percent_change'] = ((data['next_close'] - data['close']) / data['close']) * 100



            # اعمال تابع بر روی ستون percent_change برای ایجاد ستون target
            # data['target'] = data['percent_change'].apply(determine_target(expectedChange,data['percent_change']))

            # حذف ستون‌های موقتی
            # data = data.drop(columns=['next_close', 'percent_change'])

            # print(data)

            # data['next_close'] = data['close'].shift(-1)
            data['next_close'] = data['close'].shift(-shiftedNumber)

            data['target']= (data['next_close']>data['close']).astype(int)
            # data.dropna(inplace=True)

            #3.محاسبه شاخص ها
            # data['RSI']=calculate_rsi(data['close'])
            # data=calculate_rsi(data)
            # data['ADX']=calculate_adx(data['high'],data['low'],data['close'])
            # data=calculate_adx(data)
            # data['plus_di'],data['minus_di']=calculate_di(data)
            # data=calculate_di(data)

            # data=calculate_mfi(data)
            # data=calculate_linear_regression_channel(data)
            # data=calculate_macd(data)
            data=add_indicators(data)
            # data=calculate_supertrend(data)

            #4. محاسبه سیگنال های خرید و فروش و بروزرسانی  وضعیت سیکل
            # data=calculate_buy_sell_signals(data)

            # data=calculate_buy_sell_signalsSuperTrend(data)
            # drawChart()

            #5. آماده سازی داده ها برای مدل
            # X=data[['RSI','ADX','plus_di','minus_di']]
            # X=data[['rsi','adx','mfi']]
            # y=data['target']


            #پاکسازی داده ها از NAN
            # X=X.fillna(X.mean())
             #جاگزین کردن NaN با میانگین

            # ایجاد یک نمونه از MinMaxScaler
            # scaler = MinMaxScaler()

            # نرمال سازی X
            # X = scaler.fit_transform(X)

            #K برای یافتن بهترین Cross-validation
            # knn=KNeighborsClassifier(metric=lorentzian_distance)
            # k_range=range(1,20)
            # # cross_val_score=[]
            # cv_scores=[]
            # for k in k_range:
            #     if k%2==0:
            #         continue
            #     # print(f" k is:{k}")
            #     knn.n_neighbors=k
            #     # scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
            #     # print(f"k:{k} and score: {scores}")
            #     # cv_scores.append(scores.mean())
            # best_k=k_range[np.argmax(cross_val_score)]

            # #آموزش مدل با بهترین KNN
            # knn.n_neighbors=best_k
            # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            # print(X_train)
            # knn.fit(X_train,y_train)

            #قیمت فعلی بازار است که از داده های اخیر بدست میآید ' latest_price ' فرض کنید
            # latest_price=data['current'].iloc[-1]
            latest_price=data['close'].iloc[-1]

            #7. ارسال آخرین پیش بینی به تلگرام با اضافه کردن قیمت
            latest_timestamp=data['timestamp'].iloc[-1]
            # latest_prediction=predictions[-1]
            #اضافه کردن وضعیت آخرین سیکل به پارامتر های تابع
            # latest_cycle_status=data['cycle_status'].iloc[-1]
            # all_best_models,accuracy,predictions=Knn()
            # all_best_models,accuracy,predictions=LstmSVm()
            # all_best_models,accuracy,predictions=PredictCombine(lstm_model,svm_model,combine_model)
            all_best_models,accuracy,predictions=Election(lstm_model,svm_model,combine_model)
            # predictions, accuracy, predictionNumber = Knn(data, shiftedNumber_range, expectedChange_range, expectedAccuracy)
            # all_best_models,accuracy,predictions=MlpClassifier()
            #باز کردن سیگنال جدید
            if(all_best_models==None or accuracy==None or predictions==None):
                continue
            elif cycle_status==False and accuracy> expectedAccuracy :

                cycle_status=True
                lastSignalAction=predictions
                # latest_prediction=predictions
                signalPrice=latest_price
                actionStatus="Open"
                profit=0
                sendMessage=True
                closedMethod=""
            #بستن سیگنال جاری با knn
            # elif (cycle_status==True and lastSignalAction!=predictions and accuracy> expectedAccuracy)or(CloseSignalWithIndicators(data,1,True,lastSignalAction)):
            elif (cycle_status==True and lastSignalAction!=predictions and accuracy> expectedAccuracy):

                # if CloseSignalWithIndicators(data,1,True,lastSignalAction):
                #     closedMethod="Mfi"
                # else:
                #     closedMethod="KNN"
                closedMethod="LStm"
                actionStatus="Close"
                cycle_status=False
                profit= round(((latest_price-signalPrice)*100)/signalPrice,3)
                if lastSignalAction==0:
                    profit=profit*-1

                lastSignalAction=3
                sendMessage=True
                # predictions=lastSignalAction

            # predictions=all_best_models[0].predict(X_train)
            # print(predictions[-1])

            # predictions=all_best_models[0].predict(X_train)
            # accuracy = accuracy_score(y_test, all_best_models[0].predict(X_test))
            # best_k=all_best_models[0].n_neighbors

            # best_k=all_best_models.n_neighbors
            best_k=1
            #6. پیش بینی
            # predictions=knn.predict(X_test)
            # accuracy=accuracy_score(y_test,predictions)
            # confusion_matrix = confusion_matrix(y_test, predictions)
            # print(f"accuracy is:{accuracy}")
            # print(confusion_matrix)


            # send_prediction_to_telegram(latest_prediction,latest_timestamp,symbol,accuracy,latest_price,latest_cycle_status,timeframe,best_k)
            if sendMessage:
                send_prediction_to_telegram(lastSignalAction,latest_timestamp,symbol,accuracy,latest_price,actionStatus,timeframe,best_k,profit,signalPrice,closedMethod)
                sendMessage=False
            # یک تاخیر کوتاه برای جلوگیری از اتلاف منابغ
            time.sleep(0.1)
    except Exception as e:
        print("Error:", e)
















