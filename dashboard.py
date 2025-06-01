import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def ma_crossover_strategy(data, short_window=20, long_window=50):
    """Стратегия пересечения скользящих средних"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0
    )
    signals['positions'] = signals['signal'].diff()
    return signals

def rsi_strategy(data, period=14, low=30, high=70):
    """Стратегия RSI (индекс относительной силы)"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['rsi'] = rsi
    signals['signal'] = 0
    signals['signal'] = np.where(rsi < low, 1, np.where(rsi > high, -1, 0))
    signals['positions'] = signals['signal'].diff()
    return signals

def macd_strategy(data, fast=12, slow=26, signal=9):
    """Стратегия MACD"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['macd'] = macd
    signals['signal_line'] = signal_line
    signals['signal'] = np.where(macd > signal_line, 1, -1)
    signals['positions'] = signals['signal'].diff()
    return signals

def bollinger_strategy(data, window=20, std_dev=2):
    """Стратегия Bollinger Bands"""
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['upper_band'] = upper_band
    signals['lower_band'] = lower_band
    signals['signal'] = np.where(
        data['Close'] < lower_band, 1, np.where(data['Close'] > upper_band, -1, 0))
    signals['positions'] = signals['signal'].diff()
    return signals

def volume_spike_strategy(data, multiplier=2, window=20):
    """Стратегия Volume Spike"""
    avg_volume = data['Volume'].rolling(window=window).mean()
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['volume'] = data['Volume']
    signals['avg_volume'] = avg_volume
    signals['signal'] = np.where(
        data['Volume'] > (multiplier * avg_volume), 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Функция оптимизации параметров для MA Crossover
def optimize_ma(data, short_range, long_range):
    best_return = -np.inf
    best_params = None
    
    for short in short_range:
        for long in long_range:
            if long <= short:
                continue
            try:
                signals = ma_crossover_strategy(data, short, long)
                portfolio = calculate_returns(signals)
                total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                
                if total_return > best_return:
                    best_return = total_return
                    best_params = (short, long)
            except:
                continue
                
    return best_return, best_params

# Функция оптимизации параметров для RSI
def optimize_rsi(data, period_range, low_range, high_range):
    best_return = -np.inf
    best_params = None
    
    for period in period_range:
        for low in low_range:
            for high in high_range:
                if low >= high:
                    continue
                try:
                    signals = rsi_strategy(data, period, low, high)
                    portfolio = calculate_returns(signals)
                    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                    
                    if total_return > best_return:
                        best_return = total_return
                        best_params = (period, low, high)
                except:
                    continue
                    
    return best_return, best_params

# Функция оптимизации параметров для MACD
def optimize_macd(data, fast_range, slow_range, signal_range):
    best_return = -np.inf
    best_params = None
    
    for fast in fast_range:
        for slow in slow_range:
            for signal in signal_range:
                if slow <= fast:
                    continue
                try:
                    signals = macd_strategy(data, fast, slow, signal)
                    portfolio = calculate_returns(signals)
                    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                    
                    if total_return > best_return:
                        best_return = total_return
                        best_params = (fast, slow, signal)
                except:
                    continue
                    
    return best_return, best_params

# Функция оптимизации параметров для Bollinger Bands
def optimize_bollinger(data, window_range, std_dev_range):
    best_return = -np.inf
    best_params = None
    
    for window in window_range:
        for std_dev in std_dev_range:
            try:
                signals = bollinger_strategy(data, window, std_dev)
                portfolio = calculate_returns(signals)
                total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                
                if total_return > best_return:
                    best_return = total_return
                    best_params = (window, std_dev)
            except:
                continue
                    
    return best_return, best_params

# Функция оптимизации параметров для Volume Spike
def optimize_volume(data, multiplier_range, window_range):
    best_return = -np.inf
    best_params = None
    
    for multiplier in multiplier_range:
        for window in window_range:
            try:
                signals = volume_spike_strategy(data, multiplier, window)
                portfolio = calculate_returns(signals)
                total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                
                if total_return > best_return:
                    best_return = total_return
                    best_params = (multiplier, window)
            except:
                continue
                    
    return best_return, best_params

def calculate_returns(signals, initial_capital=10000):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['price'] = signals['price']
    portfolio['signal'] = signals['signal']
    
    # Инициализация
    portfolio['position'] = 0
    portfolio['shares'] = 0.0
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    
    in_position = False
    current_cash = initial_capital
    current_shares = 0
    
    for i in range(len(portfolio)):
        price = portfolio['price'].iloc[i]
        signal = portfolio['signal'].iloc[i]
        
        if signal == 1 and not in_position:
            shares_bought = current_cash / price
            current_shares = shares_bought
            current_cash = 0
            in_position = True
        
        elif signal == -1 and in_position:
            cash_after_sale = current_shares * price
            current_cash = cash_after_sale
            current_shares = 0
            in_position = False
        
        portfolio_value = current_cash + current_shares * price
        
        portfolio.iloc[i, portfolio.columns.get_loc('position')] = int(in_position)
        portfolio.iloc[i, portfolio.columns.get_loc('shares')] = current_shares
        portfolio.iloc[i, portfolio.columns.get_loc('cash')] = current_cash
        portfolio.iloc[i, portfolio.columns.get_loc('total')] = portfolio_value
    
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio

def calculate_metrics(returns):
    """Расчет ключевых метрик"""
    total_return = (returns['total'][-1] / returns['total'][0] - 1) * 100
    
    peak = returns['total'].cummax()
    drawdown = (returns['total'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    risk_free_rate = 0.0
    sharpe_ratio = (returns['returns'].mean() - risk_free_rate) / returns['returns'].std()
    
    return {
        'Доходность (%)': total_return,
        'Макс. просадка (%)': max_drawdown,
        'Коэф. Шарпа': sharpe_ratio
    }

# Настройка страницы
st.set_page_config(page_title="Анализ торговых стратегий", layout="wide")
st.title("📊 Сравнение торговых стратегий")

# Сайдбар для параметров
with st.sidebar:
    st.header("Параметры анализа")
    ticker = st.text_input("Тикер акции", "AAPL")
    start_date = st.date_input("Начальная дата", datetime(2020, 1, 1))
    end_date = st.date_input("Конечная дата", datetime(2023, 12, 31))
    
    st.subheader("Параметры стратегий")
    st.caption("MA Crossover")
    short_window = st.slider("Короткое окно (MA)", 10, 50, 20)
    long_window = st.slider("Длинное окно (MA)", 50, 200, 50)
    
    st.caption("RSI")
    rsi_period = st.slider("Период RSI", 10, 30, 14)
    rsi_low = st.slider("Нижний уровень RSI", 25, 40, 30)
    rsi_high = st.slider("Верхний уровень RSI", 60, 75, 70)
    
    st.caption("MACD")
    fast_period = st.slider("Быстрый период", 8, 15, 12)
    slow_period = st.slider("Медленный период", 20, 30, 26)
    signal_period = st.slider("Сигнальный период", 5, 12, 9)
    
    st.caption("Bollinger Bands")
    bb_window = st.slider("Окно Bollinger", 15, 30, 20)
    bb_std = st.slider("Стандартные отклонения", 1.0, 3.0, 2.0)
    
    st.caption("Volume Spike")
    volume_multiplier = st.slider("Множитель объема", 1.5, 3.0, 2.0)
    volume_window = st.slider("Окно объема", 10, 30, 20)


# Загрузка данных
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data = load_data(ticker, start_date, end_date)

# Расчет стратегий
ma_signals = ma_crossover_strategy(data, short_window, long_window)
rsi_signals = rsi_strategy(data, rsi_period, rsi_low, rsi_high)
macd_signals = macd_strategy(data, fast_period, slow_period, signal_period)
bollinger_signals = bollinger_strategy(data, bb_window, bb_std)
volume_signals = volume_spike_strategy(data, volume_multiplier, volume_window)

# Расчет доходности
ma_portfolio = calculate_returns(ma_signals)
rsi_portfolio = calculate_returns(rsi_signals)
macd_portfolio = calculate_returns(macd_signals)
bollinger_portfolio = calculate_returns(bollinger_signals)
volume_portfolio = calculate_returns(volume_signals)

# Визуализация результатов
tab1, tab2, tab3, tab4 = st.tabs(["Графики", "Метрики", "Сигналы", "Оптимизация"])  # Добавлена новая вкладка

with tab1:
    st.subheader("Рост капитала")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma_portfolio['total'], label='MA Crossover')
    ax.plot(rsi_portfolio['total'], label='RSI')
    ax.plot(macd_portfolio['total'], label='MACD')
    ax.plot(bollinger_portfolio['total'], label='Bollinger Bands')
    ax.plot(volume_portfolio['total'], label='Volume Spike')
    ax.set_title(f"Сравнение стратегий для {ticker}")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Ключевые метрики")
    metrics_df = pd.DataFrame({
        'MA Crossover': calculate_metrics(ma_portfolio),
        'RSI': calculate_metrics(rsi_portfolio),
        'MACD': calculate_metrics(macd_portfolio),
        'Bollinger Bands': calculate_metrics(bollinger_portfolio),
        'Volume Spike': calculate_metrics(volume_portfolio)
    }).T
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)
    
    # Экспорт данных
    csv = metrics_df.to_csv().encode('utf-8')
    st.download_button(
        label="Экспорт метрик в CSV",
        data=csv,
        file_name=f"{ticker}_strategy_metrics.csv",
        mime='text/csv'
    )

with tab3:
    st.subheader("Визуализация сигналов")
    strategy = st.selectbox("Выберите стратегию", [
        "MA Crossover", "RSI", "MACD", "Bollinger Bands", "Volume Spike"
    ])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if strategy == "MA Crossover":
        ax.plot(ma_signals['price'], label='Цена')
        ax.plot(ma_signals['short_ma'], label=f'{short_window}-дневная MA')
        ax.plot(ma_signals['long_ma'], label=f'{long_window}-дневная MA')
        ax.plot(ma_signals.loc[ma_signals.positions == 1.0].index, 
                ma_signals.short_ma[ma_signals.positions == 1.0],
                '^', markersize=10, color='g', label='Покупка')
        ax.plot(ma_signals.loc[ma_signals.positions == -1.0].index, 
                ma_signals.short_ma[ma_signals.positions == -1.0],
                'v', markersize=10, color='r', label='Продажа')
        ax.set_title('MA Crossover Strategy')
        
    elif strategy == "RSI":
        ax.plot(rsi_signals['rsi'], label='RSI')
        ax.axhline(70, linestyle='--', color='r', alpha=0.3)
        ax.axhline(30, linestyle='--', color='g', alpha=0.3)
        ax.set_title('RSI Strategy')
        
    elif strategy == "MACD":
        ax.plot(macd_signals['macd'], label='MACD')
        ax.plot(macd_signals['signal_line'], label='Signal Line')
        ax.set_title('MACD Strategy')
        
    elif strategy == "Bollinger Bands":
        ax.plot(bollinger_signals['price'], label='Цена')
        ax.plot(bollinger_signals['upper_band'], label='Верхняя полоса')
        ax.plot(bollinger_signals['lower_band'], label='Нижняя полоса')
        ax.fill_between(bollinger_signals.index, 
                       bollinger_signals['lower_band'], 
                       bollinger_signals['upper_band'], 
                       alpha=0.1)
        ax.set_title('Bollinger Bands Strategy')
        
    else:  # Volume Spike
        ax.bar(volume_signals.index, volume_signals['volume'], label='Объем')
        ax.plot(volume_signals['avg_volume'], label='Средний объем', color='orange')
        ax.set_title('Volume Spike Strategy')
        
    ax.legend()
    st.pyplot(fig)

with tab4:
    st.subheader("Оптимизация параметров стратегий")
    st.info("Оптимизация параметров для достижения максимальной доходности")
    
    if st.button("Запустить оптимизацию"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Оптимизация MA Crossover
        status_text.text("Оптимизация MA Crossover...")
        ma_return, ma_params = optimize_ma(
            data, 
            short_range=range(10, 51, 10), 
            long_range=range(50, 201, 50)
        )
        progress_bar.progress(20)
        
        # Оптимизация RSI
        status_text.text("Оптимизация RSI...")
        rsi_return, rsi_params = optimize_rsi(
            data, 
            period_range=range(10, 21, 4), 
            low_range=range(25, 36, 5), 
            high_range=range(65, 76, 5)
        )
        progress_bar.progress(40)
        
        # Оптимизация MACD
        status_text.text("Оптимизация MACD...")
        macd_return, macd_params = optimize_macd(
            data, 
            fast_range=range(10, 15, 2), 
            slow_range=range(20, 31, 5), 
            signal_range=range(7, 13, 2)
        )
        progress_bar.progress(60)
        
        # Оптимизация Bollinger Bands
        status_text.text("Оптимизация Bollinger Bands...")
        bb_return, bb_params = optimize_bollinger(
            data, 
            window_range=range(15, 31, 5), 
            std_dev_range=[1.5, 2.0, 2.5]
        )
        progress_bar.progress(80)
        
        # Оптимизация Volume Spike
        status_text.text("Оптимизация Volume Spike...")
        vol_return, vol_params = optimize_volume(
            data, 
            multiplier_range=[1.5, 2.0, 2.5, 3.0], 
            window_range=range(15, 31, 5)
        )
        progress_bar.progress(100)
        
        # Форматирование результатов
        results = {
            "Стратегия": ["MA Crossover", "RSI", "MACD", "Bollinger Bands", "Volume Spike"],
            "Доходность (%)": [
                f"+{ma_return:.1f}%" if ma_return > 0 else f"{ma_return:.1f}%",
                f"+{rsi_return:.1f}%" if rsi_return > 0 else f"{rsi_return:.1f}%",
                f"+{macd_return:.1f}%" if macd_return > 0 else f"{macd_return:.1f}%",
                f"+{bb_return:.1f}%" if bb_return > 0 else f"{bb_return:.1f}%",
                f"+{vol_return:.1f}%" if vol_return > 0 else f"{vol_return:.1f}%"
            ],
            "Лучшие параметры": [
                f"short={ma_params[0]}, long={ma_params[1]}",
                f"period={rsi_params[0]}, уровни={rsi_params[1]}/{rsi_params[2]}",
                f"fast={macd_params[0]}, slow={macd_params[1]}, signal={macd_params[2]}",
                f"window={bb_params[0]}, std_dev={bb_params[1]}",
                f"multiplier={vol_params[0]}, window={vol_params[1]}"
            ]
        }
        
        # Отображение результатов
        st.success("Оптимизация завершена!")
        st.table(pd.DataFrame(results))
        
        # Кнопка для применения лучших параметров
        if st.button("Применить лучшие параметры"):
            st.session_state.short_window = ma_params[0]
            st.session_state.long_window = ma_params[1]
            st.session_state.rsi_period = rsi_params[0]
            st.session_state.rsi_low = rsi_params[1]
            st.session_state.rsi_high = rsi_params[2]
            st.session_state.fast_period = macd_params[0]
            st.session_state.slow_period = macd_params[1]
            st.session_state.signal_period = macd_params[2]
            st.session_state.bb_window = bb_params[0]
            st.session_state.bb_std = bb_params[1]
            st.session_state.volume_multiplier = vol_params[0]
            st.session_state.volume_window = vol_params[1]
            
            st.experimental_rerun()