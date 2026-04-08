import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import datetime

# --- 1. 기본 설정 및 포트폴리오 유니버스 정의 ---
st.set_page_config(page_title="고변동성 레버리지 무한매수법 개선 프레임워크 백테스트", layout="wide")

# 거래량 및 유동성이 검증된 10개 ETF/ETN 리스트 (보고서 유니버스 기준)
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

# --- 2. 퀀트 보조지표 계산 로직 ---
def calculate_indicators(df):
    """거시 추세 판단용 이동평균선(SMA) 및 진입 강도 조절용 상대강도지수(RSI) 생성"""
    # 200일 이평선: 거시 하락장(Bear Market) 회피용 필터
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (14일): 단기 과열 및 침체 국면 판별
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # 데이터 초기값 50으로 중립 보정
    
    return df.dropna()

# --- 3. 동적 리스크 제어 기반 개선된 무한매수법 코어 엔진 ---
def run_enhanced_infinite_buying(df, initial_capital, splits, target_profit, stop_loss):
    """
    개선 논리 적용: 
    1. SMA 200 필터 작동 시 신규 진입 보류
    2. RSI < 30 (과매도 국면): 기본 진입액의 2배수 투입으로 단가 공격적 인하
    3. RSI > 70 (과열 국면): 추격 매수 억제 및 생략
    4. 하드 스탑로스 도달 시 강제 청산 (심리적/재무적 파산 방지)
    """
    cash = initial_capital
    portfolio_value = initial_capital
    
    shares_held = 0
    average_cost = 0.0
    invested_capital = 0.0
    buy_count = 0 # 시드 투입 횟수 (분할 도달 확인용)
    
    history = [] # 빈 리스트로 초기화 (수정됨)
    base_buy_amount = initial_capital / splits # 1회 배정된 투입 기준 금액
    
    for date, row in df.iterrows():
        current_price = row['Close']
        sma_200 = row['SMA_200'] # 컬럼 명시 (수정됨)
        rsi = row['RSI']         # 컬럼 명시 (수정됨)
        
        # 보유 포트폴리오의 실시간 미실현 수익률 산출
        current_return = 0.0
        if invested_capital > 0:
            current_return = (current_price - average_cost) / average_cost
            
        # [청산 모듈 1: 이익 실현] 지정된 목표 수익률 도달 시 즉시 전량 매도
        if shares_held > 0 and current_return >= target_profit:
            cash += shares_held * current_price
            shares_held = 0
            invested_capital = 0
            average_cost = 0
            buy_count = 0
            
        # [청산 모듈 2: 리스크 컷] 하드 스탑로스 도달 시 계좌 붕괴 방지를 위한 강제 손절
        elif shares_held > 0 and current_return <= -stop_loss:
            cash += shares_held * current_price
            shares_held = 0
            invested_capital = 0
            average_cost = 0
            buy_count = 0
            
        # [청산 모듈 3: 시간/자금 고갈] 분할 자본금 전액 소진 시 비자발적 장기투자 방지를 위한 청산
        elif buy_count >= splits:
            cash += shares_held * current_price
            shares_held = 0
            invested_capital = 0
            average_cost = 0
            buy_count = 0
            
        # [매수 모듈] 현금 보유고가 남아있고, 거시 추세가 200일선 위에 위치할 때만 가동
        if cash >= base_buy_amount and current_price >= sma_200:
            amount_to_invest = 0
            
            # 단기 모멘텀에 따른 동적 비중 스케일링
            if rsi < 30: 
                amount_to_invest = base_buy_amount * 2 # 패닉 셀링 구간에서의 공격적 방어
            elif rsi > 70:
                amount_to_invest = 0 # 환희의 구간에서의 매수 억제
            else:
                amount_to_invest = base_buy_amount
                
            # 가용 현금이 투입 예정액보다 적을 경우 남은 현금 전체 긁어모으기
            if amount_to_invest > cash:
                amount_to_invest = cash
                
            # 주식 실수량 매수 체결 및 평단가(ATP) 갱신
            if amount_to_invest > 0:
                shares_bought = amount_to_invest / current_price
                
                total_cost = invested_capital + amount_to_invest
                shares_held += shares_bought
                average_cost = total_cost / shares_held
                invested_capital = total_cost
                
                cash -= amount_to_invest
                buy_count += (amount_to_invest / base_buy_amount)
                
        # 장 마감 기준 일일 평가 금액 누적 기록
        portfolio_value = cash + (shares_held * current_price)
        history.append({
            'Date': date,
            'Price': current_price,
            'Portfolio Value': portfolio_value,
            'Cash': cash,
            'Return (%)': current_return * 100 if shares_held > 0 else 0
        })
        
    res_df = pd.DataFrame(history).set_index('Date')
    return res_df

# --- 4. Streamlit 대시보드 UI 및 프론트엔드 구성 ---
st.title("📈 개선된 무한매수법 알고리즘 정량적 백테스트 시스템")
st.markdown("과거 5년치(약 1250 거래일) 시계열 데이터를 활용해 고변동성 3배 레버리지 10종목에 대해 수리적 구조 리스크를 보완한 알고리즘 트레이딩 성과를 검증합니다.")

# 사이드바 사용자 정의 파라미터 컨트롤 패널
st.sidebar.header("투자 알고리즘 파라미터 설정")
selected_ticker = st.sidebar.selectbox("테스트 대상 종목 선택", list(UNIVERSE.keys()), format_func=lambda x: f"{x} - {UNIVERSE[x]}")
initial_capital = st.sidebar.number_input("초기 투자 자본 (USD 기준)", value=100000, step=10000)
splits = st.sidebar.slider("분할 매수 횟수 (시드 분할 수준)", 20, 80, 40)
target_profit = st.sidebar.slider("1회전 목표 수익률 (%)", 5.0, 30.0, 10.0) / 100.0
stop_loss = st.sidebar.slider("하드 스탑로스 강제 청산 한도 (%)", 10.0, 50.0, 20.0) / 100.0

st.sidebar.markdown("""
### 💡 알고리즘 개선 핵심 메커니즘
- **단기 과열 추격 매수 금지**: RSI > 70 초과 시 당일 진입 보류
- **과매도 패닉장 공격적 단가 인하**: RSI < 30 하회 시 2배수 진입
- **거시 하락장 회피 (MDD 최소화)**: SMA 200 하회 시 시스템 가동 중지
- **재무적/심리적 파산 방어선**: 설정된 최대 손실 임계점 도달 시 기계적 손절 로직 발동
""")

# 백테스트 엔진 구동 및 그래픽 렌더링
if st.sidebar.button("시뮬레이션 백테스트 실행"):
    with st.spinner(f"Yahoo Finance API 연동 중... {selected_ticker} 데이터 다운로드 및 시뮬레이션 계산 중..."):
        # 최근 5년치 기간 동적 설정
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * 5)
        
        try:
            # 주가 데이터 병렬 크롤링
            raw_data = yf.download(selected_ticker, start=start_date, end=end_date)
            if raw_data.empty:
                st.error("데이터 서버에서 정보를 불러오지 못했습니다. 종목 심볼 오류 또는 네트워크 차단 여부를 확인하세요.")
                st.stop()
                
            # yfinance 라이브러리 버전에 따른 다중 인덱스 반환 구조 방어 코드
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.droplevel(1)
                
            # 기술적 지표 생성 연산
            processed_data = calculate_indicators(raw_data)
            
            # 코어 백테스트 엔진 구동
            results_df = run_enhanced_infinite_buying(
                processed_data, 
                initial_capital, 
                splits, 
                target_profit, 
                stop_loss
            )
            
            # --- 성과 평가 지표(Metrics) 추출 ---
            final_value = results_df['Portfolio Value'].iloc[-1]
            total_return_pct = ((final_value / initial_capital) - 1) * 100
            
            # MDD (최대 낙폭) 산출
            roll_max = results_df['Portfolio Value'].cummax()
            drawdown = results_df['Portfolio Value'] / roll_max - 1.0
            max_drawdown = drawdown.min() * 100
            
            # CAGR (연환산 복리 수익률) 산출
            years = len(results_df) / 252 # 미국 증시 1년 평균 거래일 약 252일 적용
            cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
            
            # 벤치마크(단순 Buy & Hold) 비교 산출 (누락 인덱스 수정됨)
            bnh_shares = initial_capital / processed_data['Close'].iloc[0]
            bnh_final_value = bnh_shares * processed_data['Close'].iloc[-1]
            bnh_return_pct = ((bnh_final_value / initial_capital) - 1) * 100

            # --- 5. 백테스트 결과 리포팅 레이아웃 ---
            st.subheader(f"📊 {selected_ticker} 최근 5년 백테스트 종합 성과 요약")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("최종 포트폴리오 순자산", f"${final_value:,.2f}")
            col2.metric("전략 누적 수익률", f"{total_return_pct:.2f}%", delta=f"B&H 대비 {total_return_pct - bnh_return_pct:.2f}%p")
            col3.metric("연평균 복리 수익률 (CAGR)", f"{cagr:.2f}%")
            col4.metric("계좌 최대 낙폭 (MDD)", f"{max_drawdown:.2f}%")
            
            # Plotly 기반 인터랙티브 차트 구성
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, 
                                subplot_titles=(f"{selected_ticker} 모의투자 자산 곡선 (Portfolio Equity Curve)", "실시간 미실현 수익률 진행 상황"),
                                row_heights=[0.7, 0.3])
            
            # [상단 차트] 전략 구동에 따른 포트폴리오 가치 우상향 곡선
            fig.add_trace(plotly_go.Scatter(x=results_df.index, y=results_df['Portfolio Value'], 
                                            line=dict(color='#1f77b4', width=2), name="전략 운영 가치 (Equity)"), row=1, col=1)
            
            # 기초 자산 자체의 극단적 변동성 시각화를 위한 원본 주가(점선) 병기
            fig.add_trace(plotly_go.Scatter(x=processed_data.index, y=processed_data['Close'], 
                                            line=dict(color='gray', width=1, dash='dot'), name="종목 단순 종가 (Price)"), row=1, col=1)
            
            # [하단 차트] 매 사이클(회전) 단위의 미실현 수익/손실 물결 차트 (y축 기준 컬럼 지정 수정됨)
            fig.add_trace(plotly_go.Scatter(x=results_df.index, y=results_df['Return (%)'], 
                                            line=dict(color='#d62728', width=1), name="진행 수익률 (%)",
                                            fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.15)'), row=2, col=1)
            
            fig.update_layout(height=800, hovermode='x unified', template='plotly_white',
                              title_text="알고리즘 트레이딩 시계열 진행 현황 심층 분석", showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("일자별 세부 백데이터 로그 확인 (Raw Data Logs)"):
                st.dataframe(results_df.style.format({
                    "Price": "{:.2f}", 
                    "Portfolio Value": "{:.2f}", 
                    "Cash": "{:.2f}", 
                    "Return (%)": "{:.2f}"
                }))

        except Exception as e:
            st.error(f"오류가 발생하여 시뮬레이션을 중단합니다. 종목 심볼 지원 여부나 시계열 데이터 결측치를 확인하세요. 상세 로그: {e}")