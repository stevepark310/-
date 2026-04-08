import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import datetime

# --- 1. 기본 설정 및 포트폴리오 유니버스 정의 ---
st.set_page_config(page_title="고변동성 레버리지 무한매수법 개선 프레임워크 백테스트", layout="wide")

UNIVERSE = {
    "TQQQ": "ProShares UltraPro QQQ (3x 기술주, ER 0.82%)",
    "SOXL": "Direxion Daily Semiconductor Bull 3X (3x 반도체, ER 0.75%)",
    "FNGU": "MicroSectors FANG+ 3X ETN (3x 빅테크, ER 2.60%)",
    "BULZ": "MicroSectors Solactive FANG & Innovation 3X ETN",
    "SPXL": "Direxion Daily S&P 500 Bull 3X (3x 대형주)",
    "UPRO": "ProShares UltraPro S&P500 (3x 대형주)",
    "TNA": "Direxion Daily Small Cap Bull 3X (3x 소형주)",
    "TECL": "Direxion Daily Technology Bull 3X (3x 기술섹터)",
    "FAS": "Direxion Daily Financial Bull 3X (3x 금융섹터)",
    "LABU": "Direxion Daily S&P Biotech Bull 3x (3x 바이오)"
}

# --- 세션 상태 관리 (버튼 초기화 방지 및 화면 전환용) ---
if 'show_manual' not in st.session_state:
    st.session_state.show_manual = False

if 'sim_executed' not in st.session_state:
    st.session_state.sim_executed = False

def toggle_manual():
    st.session_state.show_manual = not st.session_state.show_manual

# --- 2. 퀀트 보조지표 계산 로직 ---
def calculate_indicators(df):
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) 
    
    return df.dropna()

# --- [핵심 개선] 데이터 다운로드 캐싱 함수 ---
# 한 번 불러온 주가 데이터는 저장하여 체크박스 조작 시 API 재호출로 인한 딜레이 방지
@st.cache_data(show_spinner=False)
def get_market_data(ticker, start_d, end_d):
    fetch_start_date = start_d - datetime.timedelta(days=300)
    raw_data = yf.download(ticker, start=fetch_start_date, end=end_d)
    fx_data = yf.download("KRW=X", start=fetch_start_date, end=end_d)
    
    if raw_data.empty or fx_data.empty:
        return pd.DataFrame()
        
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.droplevel(1)
    if isinstance(fx_data.columns, pd.MultiIndex):
        fx_data.columns = fx_data.columns.droplevel(1)
        
    fx_close = fx_data[['Close']].rename(columns={'Close': 'FX_Rate'})
    merged_data = raw_data.join(fx_close, how='left')
    merged_data['FX_Rate'] = merged_data['FX_Rate'].ffill().bfill()
    
    merged_data['Close_KRW'] = merged_data['Close'] * merged_data['FX_Rate']
    
    processed = calculate_indicators(merged_data)
    processed = processed[processed.index >= pd.to_datetime(start_d)]
    return processed

# --- 숫자를 한글 단위(만, 억)로 변환해주는 유틸리티 함수 ---
def number_to_korean(num):
    if num == 0: return "0원"
    units = ['', '만', '억', '조']
    sub_units = ['천', '백', '십', '']
    
    def read_4_digits(n):
        res = ""
        n_str = str(n).zfill(4)
        for i in range(4):
            digit = int(n_str[i])
            if digit > 0:
                res += str(digit) + sub_units[i] + " "
        return res.strip()
    
    res = ""
    chunk_idx = 0
    temp = num
    while temp > 0:
        chunk = temp % 10000
        if chunk > 0:
            chunk_str = read_4_digits(chunk)
            res = chunk_str + units[chunk_idx] + " " + res
        temp //= 10000
        chunk_idx += 1
        
    return " ".join(res.split()) + "원"

# --- 3. 동적 리스크 제어 기반 개선된 무한매수법 코어 엔진 ---
def run_enhanced_infinite_buying(df, initial_capital_krw, splits, target_profit, stop_loss):
    cash = initial_capital_krw
    portfolio_value = initial_capital_krw
    
    shares_held = 0
    average_cost = 0.0
    invested_capital = 0.0
    buy_count = 0 
    
    history = [] 
    base_buy_amount = initial_capital_krw / splits 
    
    for date, row in df.iterrows():
        current_price_usd = row['Close']
        current_price_krw = row['Close_KRW']
        sma_200_usd = row['SMA_200']
        rsi = row['RSI']
        fx_rate = row['FX_Rate']
        
        daily_action = "-"
        daily_profit = 0.0 
        
        current_return = 0.0
        if invested_capital > 0:
            current_return = (current_price_krw - average_cost) / average_cost
            
        # [청산 모듈]
        if shares_held > 0:
            if current_return >= target_profit:
                sell_amount = shares_held * current_price_krw
                daily_profit = sell_amount - invested_capital
                cash += sell_amount
                shares_held = 0
                invested_capital = 0
                average_cost = 0
                buy_count = 0
                daily_action = "매도 (이익 실현)"
                
            elif current_return <= -stop_loss:
                sell_amount = shares_held * current_price_krw
                daily_profit = sell_amount - invested_capital
                cash += sell_amount
                shares_held = 0
                invested_capital = 0
                average_cost = 0
                buy_count = 0
                daily_action = "매도 (손절)"
                
            elif buy_count >= splits:
                sell_amount = shares_held * current_price_krw
                daily_profit = sell_amount - invested_capital
                cash += sell_amount
                shares_held = 0
                invested_capital = 0
                average_cost = 0
                buy_count = 0
                daily_action = "매도 (자금 고갈)"
                
        # [매수 모듈]
        if cash >= base_buy_amount and current_price_usd >= sma_200_usd:
            amount_to_invest = 0
            
            if rsi < 30: 
                amount_to_invest = base_buy_amount * 2 
            elif rsi > 70:
                amount_to_invest = 0 
            else:
                amount_to_invest = base_buy_amount
                
            if amount_to_invest > cash:
                amount_to_invest = cash
                
            if amount_to_invest > 0:
                shares_bought = amount_to_invest / current_price_krw
                
                total_cost = invested_capital + amount_to_invest
                shares_held += shares_bought
                average_cost = total_cost / shares_held
                invested_capital = total_cost
                
                cash -= amount_to_invest
                buy_count += (amount_to_invest / base_buy_amount)
                
                if daily_action == "-":
                    daily_action = "매수"
                else:
                    daily_action += " / 매수"
                    
        portfolio_value = cash + (shares_held * current_price_krw)
        
        history.append({
            '날짜': date,
            '달러 주가': current_price_usd,
            '당시 환율': fx_rate,
            '원화 주가': current_price_krw,
            '매매 구분': daily_action,
            '수익금': daily_profit,
            '수익률 (%)': current_return * 100 if shares_held > 0 else 0,
            '보유 현금': cash,
            '순자산': portfolio_value
        })
        
    res_df = pd.DataFrame(history).set_index('날짜')
    return res_df

# --- 4. Streamlit 대시보드 사이드바 (파라미터 입력부) ---
st.sidebar.header("투자 알고리즘 파라미터 설정")

st.sidebar.button(
    "📖 매뉴얼 닫기 (백테스트로 돌아가기)" if st.session_state.show_manual else "📖 라오어 무한매수법 매뉴얼 읽기", 
    on_click=toggle_manual, 
    type="primary",
    use_container_width=True
)
st.sidebar.markdown("---")

st.sidebar.subheader("📅 테스트 기간 설정")
default_end_date = datetime.date.today()
default_start_date = default_end_date - datetime.timedelta(days=365 * 5) 

start_date = st.sidebar.date_input("시작 날짜", value=default_start_date, help="백테스트를 시작할 과거의 특정 날짜를 지정합니다.")
end_date = st.sidebar.date_input("종료 날짜", value=default_end_date, help="백테스트를 종료할 날짜(기본값: 오늘)를 지정합니다.")
st.sidebar.markdown("---")

selected_ticker = st.sidebar.selectbox(
    "테스트 대상 종목 선택", 
    list(UNIVERSE.keys()), 
    format_func=lambda x: f"{x} - {UNIVERSE[x]}",
    help="무한매수법은 주로 주가 변동폭이 극심한 미국 3배 레버리지 ETF를 대상으로 합니다."
)

if 'capital_input' not in st.session_state:
    st.session_state.capital_input = "10,000,000"

def format_capital_input():
    raw_val = st.session_state.capital_widget.replace(",", "")
    if raw_val.isdigit():
        st.session_state.capital_input = f"{int(raw_val):,}"

st.sidebar.text_input(
    "초기 투자 자본 (KRW 원화 기준)", 
    key="capital_widget",
    value=st.session_state.capital_input,
    on_change=format_capital_input,
    help="무한매수법 시스템에 전적으로 할당할 '총 투자 시드머니'입니다."
)

raw_capital = st.session_state.capital_widget.replace(",", "") if 'capital_widget' in st.session_state else st.session_state.capital_input.replace(",", "")
initial_capital = int(raw_capital) if raw_capital.isdigit() else 10000000
st.sidebar.markdown(f"<span style='color: gray; font-size: 0.95em;'>입력 금액: <b>{number_to_korean(initial_capital)}</b></span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
splits = st.sidebar.slider(
    "분할 매수 횟수 (시드 분할 수준)", 20, 80, 40,
    help="총 시드머니를 며칠(몇 회)에 걸쳐 매수할지 결정합니다. 오리지널 방법론의 핵심은 '40분할'입니다."
)
target_profit = st.sidebar.slider(
    "1회전 목표 수익률 (%)", 5.0, 30.0, 10.0,
    help="보유한 주식의 평균 단가 대비 도달 시 전량 매도할 목표치입니다."
) / 100.0
stop_loss = st.sidebar.slider(
    "하드 스탑로스 강제 청산 한도 (%)", 10.0, 50.0, 20.0,
    help="손실률이 이 한계치에 도달하면 감정 없이 기계적으로 전량 손절하여 파산을 면합니다."
) / 100.0

# --- 시뮬레이션 실행 버튼 클릭 시 세션 상태 업데이트 ---
if st.sidebar.button("🚀 시뮬레이션 백테스트 실행", type="primary", use_container_width=True):
    st.session_state.sim_executed = True

# --- 메인 렌더링 영역 ---
if st.session_state.show_manual:
    st.title("📖 라오어의 미국주식 무한매수법 A to Z")
    st.markdown("""
    ### 1. 무한매수법이란?
    국내 서학개미들 사이에서 베스트셀러 서적과 커뮤니티, 유튜브를 통해 선풍적인 인기를 끈 투자 방법론입니다. **'싸게 사서 비싸게 판다'**는 주식투자의 정석을 '타이밍 예측'이라는 불가능한 영역에 맡기지 않고, **기계적인 분할매수를 통해 수리적으로 달성하려는 전략**입니다.

    ### 2. 무한매수법의 오리지널 핵심 원칙 (How-to)
    * **레버리지 종목 선정**: TQQQ, SOXL 등 주가 변동폭이 극심한 미국 3배 레버리지 ETF를 타겟으로 합니다.
    * **시드 40분할**: 투자 원금을 철저하게 **40분할**로 나눕니다.
    * **매일매일 기계적 매수**: 1일차는 장중 매수, 2일차~40일차는 평단가 및 평단가보다 10~15% 높은 가격으로 LOC 분할 매수.
    * **기계적 10% 익절**: 매일 내 평단가 대비 **+10%** 수익 지점에 매도를 걸어둡니다.

    ### 3. 무한매수법의 강력한 장점
    * **시장 타이밍 예측 불필요**: 박스권이나 약상승장에서 기계적으로 수익을 누적합니다.
    * **명확한 매도 기준**: 익절 라인 덕분에 수익을 실현합니다.

    ### 4. 수많은 후기가 증명하는 치명적인 단점 (리스크)
    * **거대한 대세 하락장에서의 붕괴**: 40 거래일 내내 하락하면 원금이 소진되고, 3배 레버리지 특유의 변동성 끌림으로 막대한 손실이 발생합니다.
    """)
    st.button("🔙 백테스트 시뮬레이터 화면으로 돌아가기", on_click=toggle_manual, use_container_width=True)

else:
    st.title("📈 개선된 무한매수법 정량적 백테스트 (원화 환율 동적 적용)")
    st.markdown("사용자가 지정한 시계열 데이터 및 **매일의 실제 환율(USD/KRW)을 동적으로 반영**하여 환리스크가 포함된 실질적인 원화 자산 곡선을 검증합니다.")

    # 버튼을 누른 적이 있어 sim_executed 상태가 True라면 항상 시뮬레이션 및 결과 화면 표시
    if st.session_state.sim_executed:
        
        if start_date >= end_date:
            st.error("시작 날짜는 종료 날짜보다 과거여야 합니다.")
            st.stop()
            
        with st.spinner(f"데이터 연동 및 백테스트 진행 중... (초기 로딩 외에는 즉시 렌더링됩니다)"):
            try:
                # 함수 상단에 추가한 캐싱 함수를 통해 빠르고 안전하게 데이터 호출
                processed_data = get_market_data(selected_ticker, start_date, end_date)
                
                if processed_data.empty:
                    st.error("데이터 서버에서 해당 기간의 정보를 불러오지 못했습니다. 종목 상장일 등을 확인해 주세요.")
                    st.stop()
                    
                results_df = run_enhanced_infinite_buying(
                    processed_data, 
                    initial_capital, 
                    splits, 
                    target_profit, 
                    stop_loss
                )
                
                final_value = results_df['순자산'].iloc[-1]
                total_return_pct = ((final_value / initial_capital) - 1) * 100
                
                roll_max = results_df['순자산'].cummax()
                drawdown = results_df['순자산'] / roll_max - 1.0
                max_drawdown = drawdown.min() * 100
                
                years = (end_date - start_date).days / 365.25
                cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
                
                bnh_shares = initial_capital / processed_data['Close_KRW'].iloc[0]
                bnh_portfolio_series = bnh_shares * processed_data['Close_KRW']
                
                bnh_final_value = bnh_portfolio_series.iloc[-1]
                bnh_return_pct = ((bnh_final_value / initial_capital) - 1) * 100
                bnh_cagr = ((bnh_final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
                
                bnh_roll_max = bnh_portfolio_series.cummax()
                bnh_drawdown = bnh_portfolio_series / bnh_roll_max - 1.0
                bnh_max_drawdown = bnh_drawdown.min() * 100
                
                current_fx_rate = processed_data['FX_Rate'].iloc[-1]
                st.info(f"💵 **최신 적용 환율 가이드:** 현재 시뮬레이션 종료일({end_date}) 기준 **1달러 = {current_fx_rate:,.2f}원**입니다.")

                st.subheader(f"📊 {selected_ticker} 사용자 지정 기간 종합 성과 요약 (KRW 기준)")
                
                st.markdown("##### 🟢 알고리즘 전략 (개선된 무한매수법) 성과")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("전략 최종 순자산", f"₩ {final_value:,.0f}", delta=f"단순보유 대비 ₩ {final_value - bnh_final_value:,.0f}")
                col2.metric("전략 누적 수익률", f"{total_return_pct:.2f}%", delta=f"단순보유 대비 {total_return_pct - bnh_return_pct:.2f}%p")
                col3.metric("전략 연평균 복리 수익률(CAGR)", f"{cagr:.2f}%", delta=f"단순보유 대비 {cagr - bnh_cagr:.2f}%p")
                col4.metric("전략 최대 낙폭(MDD)", f"{max_drawdown:.2f}%", delta=f"단순보유 대비 {max_drawdown - bnh_max_drawdown:.2f}%p 방어")
                
                st.markdown("##### 🟡 비교군: 단순 보유 (Buy & Hold) 성과")
                st.caption(f"선택하신 시작일({start_date})에 초기 자본을 전액 매수하여 단 한 번도 매도하지 않고 버텼을 때의 결과입니다.")
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("단순 보유 최종 순자산", f"₩ {bnh_final_value:,.0f}")
                col6.metric("단순 보유 누적 수익률", f"{bnh_return_pct:.2f}%")
                col7.metric("단순 보유 CAGR", f"{bnh_cagr:.2f}%")
                col8.metric("단순 보유 최대 낙폭(MDD)", f"{bnh_max_drawdown:.2f}%")
                
                st.markdown("---")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.08, 
                                    subplot_titles=(f"{selected_ticker} 원화 환산 자산 곡선 (Portfolio Equity Curve)", 
                                                    "실시간 미실현 수익률 진행 상황 (%)",
                                                    f"선택 기간 {selected_ticker} 주가 캔들 차트 (USD 달러 기준)"),
                                    row_heights=[0.45, 0.25, 0.3])
                
                fig.add_trace(plotly_go.Scatter(x=results_df.index, y=results_df['순자산'], 
                                                line=dict(color='#1f77b4', width=2), name="전략 운영 가치 (원화)"), row=1, col=1)
                
                fig.add_trace(plotly_go.Scatter(x=processed_data.index, y=processed_data['Close_KRW'], 
                                                line=dict(color='gray', width=1, dash='dot'), name="종목 원화 환산 종가"), row=1, col=1)
                
                fig.add_trace(plotly_go.Scatter(x=results_df.index, y=results_df['수익률 (%)'], 
                                                line=dict(color='#d62728', width=1), name="진행 수익률 (%)",
                                                fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.15)'), row=2, col=1)
                
                fig.add_trace(plotly_go.Candlestick(x=processed_data.index,
                                                    open=processed_data['Open'],
                                                    high=processed_data['High'],
                                                    low=processed_data['Low'],
                                                    close=processed_data['Close'],
                                                    name="주가 캔들 (USD)"), row=3, col=1)
                
                fig.update_yaxes(tickformat=",.0f", ticksuffix=" 원", row=1, col=1)
                fig.update_yaxes(tickformat=",.2f", ticksuffix=" %", row=2, col=1)
                fig.update_layout(height=1000, hovermode='x unified', template='plotly_white',
                                  title_text="알고리즘 트레이딩 원화 기준 진행 현황 심층 분석", showlegend=True,
                                  xaxis_rangeslider_visible=False) 
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("일자별 세부 백데이터 로그 확인 (매수/매도 내역 한눈에 보기)"):
                    
                    show_all_logs = st.checkbox("☑️ 전체보기 (매매 없는 날 포함하여 모두 표시)", value=False)
                    
                    if show_all_logs:
                        display_df = results_df
                    else:
                        display_df = results_df[results_df['매매 구분'] != "-"]
                    
                    def highlight_action(val):
                        if isinstance(val, str):
                            if '매수' in val:
                                return 'color: red; font-weight: bold;'
                            elif '매도' in val:
                                return 'color: blue; font-weight: bold;'
                        return ''
                        
                    def highlight_profit(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                return 'color: red; font-weight: bold;'
                            elif val < 0:
                                return 'color: blue; font-weight: bold;'
                        return ''

                    styled_df = display_df.style.format({
                        "달러 주가": "${:.2f}",
                        "당시 환율": "₩{:,.2f}",
                        "원화 주가": "₩{:,.0f}",
                        "수익금": lambda x: f"₩ {x:,.0f}" if x != 0 else "-",
                        "수익률 (%)": "{:.2f}%",
                        "보유 현금": "₩{:,.0f}",
                        "순자산": "₩{:,.0f}"
                    })
                    
                    if hasattr(styled_df, 'map'):
                        styled_df = styled_df.map(highlight_action, subset=['매매 구분'])
                        styled_df = styled_df.map(highlight_profit, subset=['수익금'])
                    else:
                        styled_df = styled_df.applymap(highlight_action, subset=['매매 구분'])
                        styled_df = styled_df.applymap(highlight_profit, subset=['수익금'])

                    if display_df.empty:
                        st.info("선택하신 기간 동안 매수/매도 동작이 한 번도 발생하지 않았습니다. (전체보기를 체크하면 주가 기록을 볼 수 있습니다)")
                    else:
                        st.dataframe(styled_df, use_container_width=True)

            except Exception as e:
                st.error(f"오류가 발생하여 시뮬레이션을 중단합니다. 상세 로그: {e}")