import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Matplotlib는 Plotly가 아닌 다른 부분에서 사용될 수 있으므로 유지
import matplotlib.font_manager as fm # Matplotlib 폰트 관리를 위해 유지 (필요 없을 수도 있지만 안전하게)
from statsmodels.tsa.arima.model import ARIMA
import warnings
import datetime
import random 
import platform 
import os 

# Plotly 관련 임포트는 try-except로 감싸서, 설치 안 되었을 때도 앱이 실행되도록 합니다.
# 이전에 중복으로 임포트되던 plotly.express와 plotly.graph_objects는 이 try-except 블록 안으로 통합되었습니다.
try:
    import plotly.express as px 
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    plotly_available = False
    st.error("Plotly 라이브러리를 찾을 수 없습니다. requirements.txt 파일을 확인해주세요.") 

warnings.filterwarnings('ignore') 

st.set_page_config(layout="wide")

st.title("💰 경제수학 기반 예측 및 투자 시뮬레이션 앱")
st.markdown("---")

# --- 디버깅 정보 출력 (Streamlit Cloud에서 확인용) --- 
# 이 정보는 앱이 Streamlit Cloud에서 실행될 때 사이드바에 표시되어 디버깅에 도움을 줍니다.
st.sidebar.subheader("⚙️ 앱 디버깅 정보") 
st.sidebar.write(f"현재 작업 디렉토리: `{os.getcwd()}`")
st.sidebar.write("현재 디렉토리 파일 목록:")
try:
    for f in os.listdir('.'):
        st.sidebar.write(f"- `{f}`")
except Exception as e:
    st.sidebar.write(f"파일 목록을 가져오는 중 오류 발생: {e}")
st.sidebar.write(f"Plotly 사용 가능 여부: `{plotly_available}`")


# --- 0. 한국은행 기준금리 데이터 파싱 및 전처리 ---
# KOR_BASE_RATE_DATA_RAW 변수: 여러 줄의 문자열은 반드시 """ (큰따옴표 세 개)로 시작하고 끝나야 합니다.
# 이 부분이 이전 오류의 원인이었을 가능성이 높습니다.
KOR_BASE_RATE_DATA_RAW = """
202411월 28일3.00
202410월 11일3.25
202301월 13일3.50
202211월 24일3.25
202210월 12일3.00
202208월 25일2.50
202207월 13일2.25
202205월 26일1.75
202204월 14일1.50
202201월 14일1.25
202111월 25일1.00
202108월 26일0.75
202005월 28일0.50
202003월 17일0.75
201910월 16일1.25
201907월 18일1.50
201811월 30일1.75
201711월 30일1.50
201606월 09일1.25
201506월 11일1.50
201503월 12일1.75
201410월 15일2.00
201408월 14일2.25
201305월 09일2.50
201210월 11일2.75
201207월 12일3.00
201106월 10일3.25
201103월 10일3.00
201101월 13일2.75
201011월 16일2.50
201007월 09일2.25
""" # <-- 이 부분이 정확히 있는지, 오타는 없는지 다시 한번 확인해주세요.

@st.cache_data
def parse_korea_base_rate_data(raw_data):
    lines = raw_data.strip().split('\n')
    data = []
    for line in lines:
        try:
            # 202411월 28일3.00 -> 2024 11 28 3.00
            year = int(line[0:4])
            month_str = line[4:line.find('월')]
            day_str = line[line.find('월')+1:line.find('일')]
            rate_str = line[line.find('일')+1:]

            month = int(month_str)
            day = int(day_str)
            rate = float(rate_str)

            date = datetime.date(year, month, day)
            data.append({'Date': date, 'Rate': rate})
        except ValueError as e:
            st.error(f"데이터 파싱 오류 발생: {line} - {e}")
            continue
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').set_index('Date')
    # 월별 마지막 금리만 남기기 (해당 월의 최종 금리)
    df = df.resample('M').last().ffill() # 월말 기준으로 리샘플링하고, NaN 값은 이전 값으로 채움
    return df

korea_base_rate_df = parse_korea_base_rate_data(KOR_BASE_RATE_DATA_RAW)
# 실제 시뮬레이션에 사용될 금리 리스트 (퍼센트)
historical_base_rates_percent_app = korea_base_rate_df['Rate'].tolist()

# --- 1. 인플레이션율 예측 모델 (기존 코드와 동일) ---
@st.cache_data
def predict_inflation(cpi_data_series, p, d, q, forecast_years):
    # 인플레이션율 계산: 전년 동월 대비 CPI 변화율
    inflation_rate_monthly = cpi_data_series.pct_change(periods=12).dropna() * 100
    
    # 충분한 데이터가 있어야 모델 훈련 가능
    if len(inflation_rate_monthly) < 24: # 최소 2년치 데이터 필요 (ARIMA order 고려)
        st.warning("인플레이션 예측을 위한 CPI 데이터가 부족합니다. 기본 인플레이션율을 사용합니다.")
        return 0.025, inflation_rate_monthly, pd.Series()

    train_size = int(len(inflation_rate_monthly) * 0.8)
    train_data = inflation_rate_monthly[0:train_size]

    try:
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        forecast_steps = 12 * forecast_years
        
        # future_periods를 사용하여 예측 (새로운 predict 문법)
        forecast = model_fit.forecast(steps=forecast_steps)
        # forecast 인덱스 생성
        last_train_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_train_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        forecast.index = forecast_index

        predicted_future_inflation_rate = forecast.mean() / 100 # 예측 기간의 평균 인플레이션율
        return predicted_future_inflation_rate, inflation_rate_monthly, forecast
    except Exception as e:
        st.error(f"인플레이션 예측 오류 발생: {e}. 기본 인플레이션율 2.5%를 사용합니다.")
        return 0.025, inflation_rate_monthly, pd.Series()

@st.cache_data
def generate_cpi_data_for_app():
    start_date = '2000-01-01'
    end_date = datetime.date.today().strftime('%Y-%m-%d') # 현재 날짜까지
    date_rng = pd.date_range(start=start_date, end=end_date, freq='MS')
    initial_cpi = 80
    np.random.seed(42)
    # 실제 CPI 데이터와 유사한 패턴을 위해 더 작은 월별 변동 + 연간 추세
    monthly_cpi_increase = np.random.normal(0.002, 0.001, len(date_rng)).cumsum() # 월 0.2% 성장 기준
    cpi_data = pd.Series(initial_cpi * (1 + monthly_cpi_increase), index=date_rng)
    # 2020년 1월을 100으로 기준점 조정
    cpi_data = (cpi_data / cpi_data.loc['2020-01-01']) * 100
    return cpi_data

cpi_data_app = generate_cpi_data_for_app()

# --- 적립식 투자 시뮬레이션 함수 (다중 이자율 변동 적용) ---
# 매개변수 이름을 'change_months_option'에서 'change_months'로 변경했습니다.
def simulate_monthly_investment_multi_annual_rate(monthly_amount, historical_base_rates_percent, years, inflation_rate, tax_rate, change_months, bank_margin_rate_percent):
    """
    월별 적립식 투자를 시뮬레이션하며, 한국은행 기준금리 데이터를 기반으로 이자율이 변경됩니다.
    monthly_amount: 월별 투자 금액
    historical_base_rates_percent: 한국은행 기준금리 데이터 풀 (퍼센트)
    years: 총 투자 기간 (년)
    inflation_rate: 연간 인플레이션율 (비율)
    tax_rate: 이자소득세율 (비율)
    change_months: 이자율이 변경되는 월의 리스트 (예: [1, 7] for 1월, 7월)
    bank_margin_rate_percent: 은행 마진율 (예금금리 = 기준금리 + 마진율)
    """
    # 과거 기준금리 데이터에 은행 마진을 더하여 '예상 은행 금리' 풀 생성
    historical_bank_rates = [(rate + bank_margin_rate_percent) / 100 for rate in historical_base_rates_percent]
    
    months = years * 12
    monthly_inflation_factor = (1 + inflation_rate)**(1/12)

    total_invested_capital = 0
    total_portfolio_value = 0
    monthly_data = []
    
    # 첫 이자율 설정
    current_annual_return_rate = random.choice(historical_bank_rates)
    monthly_return_rate = current_annual_return_rate / 12

    for month_idx in range(1, months + 1):
        current_month_in_year = (month_idx - 1) % 12 + 1 

        # 이자율 변경 시점 확인
        # 'change_months_option' 대신 'change_months'를 사용합니다.
        if current_month_in_year in change_months: 
            current_annual_return_rate = random.choice(historical_bank_rates) # 과거 은행 금리 풀에서 무작위 선택
            monthly_return_rate = current_annual_return_rate / 12

        total_portfolio_value = total_portfolio_value * (1 + monthly_return_rate)
        total_portfolio_value += monthly_amount
        total_invested_capital += monthly_amount
        real_value = total_portfolio_value / (monthly_inflation_factor**month_idx)
        
        monthly_data.append({
            'Month': month_idx,
            'Year': month_idx / 12,
            'Invested Capital': total_invested_capital,
            'Portfolio Value': total_portfolio_value,
            'Real Value (Inflation Adjusted)': real_value,
            'Current Annual Return Rate': current_annual_return_rate # 현재 적용된 이자율 기록 (비율)
        })
    return pd.DataFrame(monthly_data)

# 현재 가치 계산 함수
def calculate_present_value(future_value, annual_discount_rate, years):
    if annual_discount_rate <= -1:
        return float('inf') if annual_discount_rate < -1 else float('nan')
    return future_value / ((1 + annual_discount_rate)**years)

# 연금 시뮬레이션 함수
def simulate_annuity_payout(initial_portfolio_value, retirement_age, life_expectancy, annual_inflation_rate, portfolio_return_rate_during_retirement):
    payout_years = life_expectancy - retirement_age
    if payout_years <= 0:
        return pd.DataFrame()

    remaining_value = initial_portfolio_value
    annual_payout_data = []
    
    # 4% 규칙을 초기 연금액으로 사용 (실제 은퇴 시 자산의 4%를 첫 해 인출)
    initial_annual_payout = initial_portfolio_value * 0.04 

    for year_idx in range(1, payout_years + 1):
        # 남은 자산에 은퇴 후 운용 수익률 적용
        remaining_value *= (1 + portfolio_return_rate_during_retirement)
        
        # 연금액을 인플레이션에 맞춰 증가 (명목 가치 증가)
        current_year_payout = initial_annual_payout * ((1 + annual_inflation_rate)**(year_idx - 1)) 

        # 잔액보다 연금액이 크면 잔액만큼만 지급
        if remaining_value < current_year_payout:
            current_year_payout = remaining_value 
            remaining_value = 0 
        else:
            remaining_value -= current_year_payout

        # 현재 연금액의 실질 구매력 계산 (인플레이션 반영)
        real_payout_value = current_year_payout / ((1 + annual_inflation_rate)**(year_idx - 1)) 

        annual_payout_data.append({
            'Age': retirement_age + year_idx -1,
            'Year': year_idx,
            'Annual Payout (Nominal)': current_year_payout,
            'Annual Payout (Real Value)': real_payout_value,
            'Remaining Portfolio Value': remaining_value
        })
        if remaining_value <= 0:
            break
    return pd.DataFrame(annual_payout_data)


# --- 1. 인플레이션율 예측 모델 섹션 ---
st.header("1. 인플레이션율 예측 모델")
st.write("과거 소비자물가지수(CPI) 데이터를 기반으로 미래 연간 인플레이션율을 예측합니다. (가상 데이터 사용)")

col1_inflation, col2_inflation = st.columns([1, 1])

with col1_inflation:
    st.subheader("인플레이션 예측 설정")
    p_param = st.slider("ARIMA p 값", 0, 10, 5, help="AR 모델의 차수: 과거 값에 대한 의존성")
    d_param = st.slider("ARIMA d 값", 0, 3, 1, help="차분 차수: 시계열을 정상 상태로 만들기 위한 차분 횟수")
    q_param = st.slider("ARIMA q 값", 0, 10, 0, help="MA 모델의 차수: 과거 예측 오차에 대한 의존성")
    forecast_years_inflation = st.slider("인플레이션 예측 기간 (년)", 1, 10, 5)

    if st.button("인플레이션 예측 실행"):
        predicted_inflation, inflation_rate_monthly, forecast_series = predict_inflation(
            cpi_data_app, p_param, d_param, q_param, forecast_years_inflation
        )
        st.session_state['predicted_inflation_rate'] = predicted_inflation
        st.success(f"예측된 미래 연간 평균 인플레이션율: **{predicted_inflation * 100:.2f}%**")

        with col2_inflation:
            st.subheader("인플레이션 예측 결과")
            if plotly_available: # Plotly 사용 가능할 때만 그래프 그림
                fig_inflation = px.line(
                    x=inflation_rate_monthly.index, 
                    y=inflation_rate_monthly, 
                    labels={'x':'날짜', 'y':'인플레이션율 (%)'},
                    title='연간 인플레이션율 예측',
                    line_shape="linear"
                )
                if not forecast_series.empty:
                    forecast_dates = pd.date_range(start=inflation_rate_monthly.index[-1] + pd.DateOffset(months=1), 
                                                   periods=len(forecast_series), freq='MS')
                    fig_inflation.add_trace(go.Scatter(x=forecast_dates, y=forecast_series, mode='lines', name='ARIMA 예측', line=dict(dash='dash')))
                fig_inflation.update_layout(
                    hovermode="x unified",
                    legend_title_text='데이터 종류',
                    font=dict(family="Arial, sans-serif") 
                )
                st.plotly_chart(fig_inflation, use_container_width=True)
            else:
                st.warning("Plotly 라이브러리가 없어 그래프를 표시할 수 없습니다.")


# --- 2. 적립식 투자 시뮬레이션 섹션 (변동 금리 적용) ---
st.markdown("---")
st.header("2. 적립식 투자 시뮬레이션 (기준금리 기반 변동 금리)")
st.write("월별 투자 금액과 투자 기간에 따른 자산 변화를 시뮬레이션합니다. 연간 이자율은 한국은행 기준금리 데이터를 바탕으로 무작위로 선택되어 특정 월에 변경됩니다.")

st.subheader("기준금리 데이터 및 은행 마진 설정")
st.info("시뮬레이션에 사용될 한국은행 기준금리 데이터입니다. 은행 예금 금리는 '기준금리 + 은행 마진율'로 계산됩니다.")

col_rate_data, col_margin = st.columns([1, 1])

with col_rate_data:
    st.dataframe(korea_base_rate_df.style.format({'Rate': '{:.2f}%'}))
    st.write(f"**활용되는 기준금리 범위:** {min(historical_base_rates_percent_app):.2f}% ~ {max(historical_base_rates_percent_app):.2f}%")

with col_margin:
    bank_margin_rate_percent = st.slider("은행 마진율 (%)", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                         help="은행 예금금리는 한국은행 기준금리에 은행별 마진이 더해져 결정됩니다. 이 마진율을 조절하여 실제 예금금리 수준을 시뮬레이션할 수 있습니다.")
    st.write(f"**시뮬레이션에 적용될 예상 은행 예금금리 범위:** "
             f"{(min(historical_base_rates_percent_app) + bank_margin_rate_percent):.2f}% ~ "
             f"{(max(historical_base_rates_percent_app) + bank_margin_rate_percent):.2f}%")


default_inflation_app = st.session_state.get('predicted_inflation_rate', 0.025)
st.write(f"현재 예측/적용 인플레이션율: **{default_inflation_app*100:.2f}%**")

col1_invest, col2_invest = st.columns([1, 1])

with col1_invest:
    st.subheader("투자 시뮬레이션 설정")
    monthly_invest_app = st.number_input("월별 투자 금액 (원)", min_value=100000, max_value=10000000, value=1000000, step=100000)
    tax_rate_app = st.slider("이자/배당 소득세율 (%)", min_value=0.0, max_value=30.0, value=15.4, step=0.1) / 100
    
    st.markdown("---")
    st.subheader("이자율 변경 월 설정")
    change_months_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    selected_change_months = st.multiselect(
        "매년 이자율이 변경될 월 선택",
        options=change_months_options,
        default=[1, 7], # 기본값: 1월, 7월
        help="매년 선택된 월에 이자율이 과거 기준금리 데이터 풀에서 무작위로 선택되어 새로 적용됩니다. 실제 금리 변동 시점에 가깝게 설정할 수 있습니다."
    )
    if not selected_change_months:
        st.warning("최소 한 개의 이자율 변경 월을 선택해야 합니다. 기본값인 1월이 적용됩니다.")
        selected_change_months = [1]


    investment_periods_app = st.multiselect(
        "시뮬레이션할 투자 기간 (년)",
        options=[5, 10, 15, 20, 25, 30, 35, 40],
        default=[10, 20, 30]
    )
    investment_periods_app.sort()

    if st.button("변동 금리 시뮬레이션 실행"):
        if not historical_base_rates_percent_app:
            st.error("사용할 수 있는 기준금리 데이터가 없습니다. 코드를 확인해주세요.")
        else:
            results_by_period_app = {}
            for years in investment_periods_app:
                df_result = simulate_monthly_investment_multi_annual_rate(
                    monthly_invest_app, historical_base_rates_percent_app, years,
                    inflation_rate=default_inflation_app, tax_rate=tax_rate_app,
                    change_months=selected_change_months, 
                    bank_margin_rate_percent=bank_margin_rate_percent
                )
                results_by_period_app[years] = df_result
            st.session_state['results_by_period_multi_annual'] = results_by_period_app

            with col2_invest:
                st.subheader("기간별 최종 자산 비교")
                if 'results_by_period_multi_annual' in st.session_state:
                    final_values_data_app = []
                    for years, df in st.session_state['results_by_period_multi_annual'].items():
                        total_profit = df['Portfolio Value'].iloc[-1] - df['Invested Capital'].iloc[-1]
                        tax_amount = total_profit * tax_rate_app if total_profit > 0 else 0
                        final_value_after_tax = df['Portfolio Value'].iloc[-1] - tax_amount

                        final_values_data_app.append({
                            'Investment Years': years,
                            '총 투자 원금': df['Invested Capital'].iloc[-1],
                            '세후 포트폴리오 가치': final_value_after_tax,
                            '실질 구매력 (인플레이션 조정)': df['Real Value (Inflation Adjusted)'].iloc[-1]
                        })
                    
                    final_df_app = pd.DataFrame(final_values_data_app)
                    st.dataframe(final_df_app.set_index('Investment Years').style.format('{:,.0f}'))

                    if plotly_available: # Plotly 사용 가능할 때만 그래프 그림
                        fig_bar = px.bar(
                            final_df_app, 
                            x='Investment Years', 
                            y=['총 투자 원금', '세후 포트폴리오 가치', '실질 구매력 (인플레이션 조정)'],
                            title=f'기간별 최종 자산 비교 ({len(selected_change_months)}회 변동 금리)',
                            barmode='group',
                            labels={'value':'금액 (원)', 'variable':'자산 종류', 'Investment Years':'투자 기간 (년)'},
                            color_discrete_map={ 
                                '총 투자 원금': 'blue', 
                                '세후 포트폴리오 가치': 'green', 
                                '실질 구매력 (인플레이션 조정)': 'red'
                            }
                        )
                        fig_bar.update_layout(
                            font=dict(family="Arial, sans-serif"), 
                            xaxis_title="투자 기간 (년)",
                            yaxis_title="금액 (원)"
                        )
                        fig_bar.update_yaxes(tickformat=".0f") 
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.warning("Plotly 라이브러리가 없어 그래프를 표시할 수 없습니다.")

                    st.subheader("기간별 자산 성장 곡선")
                    if plotly_available: # Plotly 사용 가능할 때만 그래프 그림
                        fig_line = px.line(
                            pd.concat([df.assign(Years=years) for years, df in st.session_state['results_by_period_multi_annual'].items()]), 
                            x='Year', 
                            y=['Portfolio Value', 'Real Value (Inflation Adjusted)'],
                            color='Years', 
                            title=f'기간별 자산 성장 곡선 ({len(selected_change_months)}회 변동 금리)',
                            labels={'value':'금액 (원)', 'variable':'자산 종류', 'Year':'기간 (년)'},
                            line_dash_map={ 
                                'Portfolio Value': 'solid', 
                                'Real Value (Inflation Adjusted)': 'dash'
                            }
                        )
                        fig_line.update_layout(
                            hovermode="x unified",
                            legend_title_text='데이터 종류',
                            font=dict(family="Arial, sans-serif") 
                        )
                        fig_line.update_yaxes(tickformat=".0f") 
                        st.plotly_chart(fig_line, use_container_width=True)
                    else:
                        st.warning("Plotly 라이브러리가 없어 그래프를 표시할 수 없습니다.")

# --- 3. 현재 가치와 연금 시뮬레이션 섹션 (기존과 동일) ---
st.markdown("---")
st.header("3. 현재 가치 및 연금 시뮬레이션")

col1_annuity, col2_annuity = st.columns([1,1])

with col1_annuity:
    st.subheader("미래 목표 금액의 현재 가치 계산")
    future_val = st.number_input("미래 목표 금액 (원)", min_value=10000000, value=500000000, step=10000000)
    # 할인율은 기준금리 데이터의 평균값 + 은행 마진을 기본으로 사용
    default_discount_rate = (np.mean(historical_base_rates_percent_app) + bank_margin_rate_percent) / 100 if historical_base_rates_percent_app else 0.03
    discount_rate_pv = st.number_input("연간 할인율 (%)", min_value=0.1, max_value=20.0, value=default_discount_rate * 100, step=0.1) / 100
    years_pv = st.number_input("기간 (년)", min_value=1, value=20, step=1)

    if st.button("현재 가치 계산"):
        present_value = calculate_present_value(future_val, discount_rate_pv, years_pv)
        st.info(f"**{years_pv}년 뒤 {future_val:,.0f}원**의 현재 가치는 약 **{present_value:,.0f}원**입니다.")

with col2_annuity:
    st.subheader("은퇴 후 연금 수령액 시뮬레이션")
    initial_portfolio_for_annuity_default = 1000000000 # 기본값
    # 투자 시뮬레이션 결과가 있다면 가장 긴 기간의 세후 자산 사용
    if 'results_by_period_multi_annual' in st.session_state and st.session_state['results_by_period_multi_annual']:
        investment_periods_app_keys = list(st.session_state['results_by_period_multi_annual'].keys())
        if investment_periods_app_keys: # 리스트가 비어있지 않은지 확인
            max_years = max(investment_periods_app_keys)
            df_max_yr_result = st.session_state['results_by_period_multi_annual'][max_years]
            total_profit_max_yr = df_max_yr_result['Portfolio Value'].iloc[-1] - df_max_yr_result['Invested Capital'].iloc[-1]
            tax_amount_max_yr = total_profit_max_yr * tax_rate_app if total_profit_max_yr > 0 else 0
            initial_portfolio_for_annuity_default = df_max_yr_result['Portfolio Value'].iloc[-1] - tax_amount_max_yr
            if initial_portfolio_for_annuity_default < 0: # 손실 났으면 0으로 표시
                initial_portfolio_for_annuity_default = 0


    initial_portfolio_for_annuity = st.number_input("은퇴 시점 총 자산 (원)", min_value=0, value=int(initial_portfolio_for_annuity_default), step=10000000)
    retirement_age_annuity = st.slider("은퇴 나이", 50, 80, 65)
    life_expectancy_annuity = st.slider("예상 수명", 70, 100, 90)
    portfolio_return_during_retirement = st.number_input("은퇴 후 자산 운용 수익률 (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

    if st.button("연금 시뮬레이션 실행", key="annuity_sim_button"):
        df_annuity = simulate_annuity_payout(
            initial_portfolio_for_annuity,
            retirement_age_annuity,
            life_expectancy_annuity,
            default_inflation_app,
            portfolio_return_during_retirement
        )
        if not df_annuity.empty:
            st.dataframe(df_annuity.set_index('Age').style.format('{:,.0f}'))

            if plotly_available: # Plotly 사용 가능할 때만 그래프 그림
                fig_annuity = px.line(
                    df_annuity, 
                    x='Age', 
                    y=['Annual Payout (Nominal)', 'Annual Payout (Real Value)', 'Remaining Portfolio Value'],
                    title='은퇴 후 연금 수령액 시뮬레이션',
                    labels={'value':'금액 (원)', 'variable':'자산 종류', 'Age':'나이'},
                    color_discrete_map={ 
                        'Annual Payout (Nominal)': 'blue', 
                        'Annual Payout (Real Value)': 'red', 
                        'Remaining Portfolio Value': 'green'
                    },
                    line_dash_map={ 
                        'Annual Payout (Nominal)': 'solid', 
                        'Annual Payout (Real Value)': 'dash', 
                        'Remaining Portfolio Value': 'dot'
                    }
                )
                fig_annuity.update_layout(
                    hovermode="x unified",
                    legend_title_text='데이터 종류',
                    font=dict(family="Arial, sans-serif") 
                )
                fig_annuity.update_yaxes(tickformat=".0f") 
                st.plotly_chart(fig_annuity, use_container_width=True)
            else:
                st.warning("Plotly 라이브러리가 없어 그래프를 표시할 수 없습니다.")
