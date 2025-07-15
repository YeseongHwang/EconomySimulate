import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.tsa.arima.model import ARIMA
import warnings
import datetime
import random
import platform
import plotly.express as px # <-- Plotly Express 임포트
import plotly.graph_objects as go
import os

# Plotly 관련 임포트는 try-except로 감싸서, 설치 안 되었을 때도 앱이 실행되도록 합니다.
try:
    import plotly.express as px 
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    plotly_available = False
    st.error("Plotly 라이브러리를 찾을 수 없습니다. requirements.txt 파일을 확인해주세요.") # <-- 디버깅 메시지

warnings.filterwarnings('ignore') 

st.set_page_config(layout="wide")

st.title("경제수학 기반 예측 및 투자 시뮬레이션")
st.markdown("---")

# --- 디버깅 정보 출력 (Streamlit Cloud에서 확인용) ---
st.sidebar.subheader("⚙️ 앱 디버깅 정보") # <-- 디버깅 정보 추가
st.sidebar.write(f"현재 작업 디렉토리: `{os.getcwd()}`")
st.sidebar.write("현재 디렉토리 파일 목록:")
for f in os.listdir('.'):
    st.sidebar.write(f"- `{f}`")
st.sidebar.write(f"Plotly 사용 가능 여부: `{plotly_available}`")




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
"""

@st.cache_data
def parse_korea_base_rate_data(raw_data):
    lines = raw_data.strip().split('\n')
    data = []
    for line in lines:
        try:
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
    df = df.resample('M').last().ffill()
    return df

korea_base_rate_df = parse_korea_base_rate_data(KOR_BASE_RATE_DATA_RAW)
historical_base_rates_percent_app = korea_base_rate_df['Rate'].tolist()

@st.cache_data
def predict_inflation(cpi_data_series, p, d, q, forecast_years):
    inflation_rate_monthly = cpi_data_series.pct_change(periods=12).dropna() * 100
    
    if len(inflation_rate_monthly) < 24:
        st.warning("인플레이션 예측을 위한 CPI 데이터가 부족합니다. 기본 인플레이션율을 사용합니다.")
        return 0.025, inflation_rate_monthly, pd.Series()

    train_size = int(len(inflation_rate_monthly) * 0.8)
    train_data = inflation_rate_monthly[0:train_size]

    try:
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        forecast_steps = 12 * forecast_years
    
        forecast = model_fit.forecast(steps=forecast_steps)
        last_train_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_train_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        forecast.index = forecast_index

        predicted_future_inflation_rate = forecast.mean() / 100 
        return predicted_future_inflation_rate, inflation_rate_monthly, forecast
    except Exception as e:
        st.error(f"인플레이션 예측 오류 발생: {e}. 기본 인플레이션율 2.5%를 사용합니다.")
        return 0.025, inflation_rate_monthly, pd.Series()

@st.cache_data
def generate_cpi_data_for_app():
    start_date = '2000-01-01'
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    date_rng = pd.date_range(start=start_date, end=end_date, freq='MS')
    initial_cpi = 80
    np.random.seed(42)
    monthly_cpi_increase = np.random.normal(0.002, 0.001, len(date_rng)).cumsum()
    cpi_data = pd.Series(initial_cpi * (1 + monthly_cpi_increase), index=date_rng)
    cpi_data = (cpi_data / cpi_data.loc['2020-01-01']) * 100
    return cpi_data

cpi_data_app = generate_cpi_data_for_app()

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
    
    historical_bank_rates = [(rate + bank_margin_rate_percent) / 100 for rate in historical_base_rates_percent]
    
    months = years * 12
    monthly_inflation_factor = (1 + inflation_rate)**(1/12)

    total_invested_capital = 0
    total_portfolio_value = 0
    monthly_data = []
    
    
    current_annual_return_rate = random.choice(historical_bank_rates)
    monthly_return_rate = current_annual_return_rate / 12

    for month_idx in range(1, months + 1):
        current_month_in_year = (month_idx - 1) % 12 + 1 

        
        if current_month_in_year in change_months: 
            current_annual_return_rate = random.choice(historical_bank_rates)
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
            'Current Annual Return Rate': current_annual_return_rate 
        })
    return pd.DataFrame(monthly_data)


def calculate_present_value(future_value, annual_discount_rate, years):
    if annual_discount_rate <= -1:
        return float('inf') if annual_discount_rate < -1 else float('nan')
    return future_value / ((1 + annual_discount_rate)**years)


def simulate_annuity_payout(initial_portfolio_value, retirement_age, life_expectancy, annual_inflation_rate, portfolio_return_rate_during_retirement):
    payout_years = life_expectancy - retirement_age
    if payout_years <= 0:
        return pd.DataFrame()

    remaining_value = initial_portfolio_value
    annual_payout_data = []
    
    initial_annual_payout = initial_portfolio_value * 0.04 

    for year_idx in range(1, payout_years + 1):
        remaining_value *= (1 + portfolio_return_rate_during_retirement)
        
        current_year_payout = initial_annual_payout * ((1 + annual_inflation_rate)**(year_idx - 1)) 

        if remaining_value < current_year_payout:
            current_year_payout = remaining_value 
            remaining_value = 0 
        else:
            remaining_value -= current_year_payout

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


st.markdown("---")
st.header("1. 적립식 투자 시뮬레이션 (기준금리 기반 변동 금리)")
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
        default=[1, 7],
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

                    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                    final_df_app.set_index('Investment Years').plot(kind='bar', ax=ax_bar, rot=0,
                                                                     color=['blue', 'green', 'red'])
                    ax_bar.set_title(f'기간별 최종 자산 비교 ({len(selected_change_months)}회 변동 금리)')
                    ax_bar.set_xlabel('투자 기간 (년)')
                    ax_bar.set_ylabel('금액 (원)')
                    ax_bar.ticklabel_format(style='plain', axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_bar)

                    st.subheader("기간별 자산 성장 곡선")
                    fig_line, ax_line = plt.subplots(figsize=(12, 6))
                    for years, df in st.session_state['results_by_period_multi_annual'].items():
                        ax_line.plot(df['Year'], df['Portfolio Value'], label=f'{years}년 (세전)', linestyle='-')
                        ax_line.plot(df['Year'], df['Real Value (Inflation Adjusted)'], label=f'{years}년 (실질)', linestyle='--')
                    ax_line.set_title(f'기간별 자산 성장 곡선 ({len(selected_change_months)}회 변동 금리)')
                    ax_line.set_xlabel('기간 (년)')
                    ax_line.set_ylabel('금액 (원)')
                    ax_line.legend()
                    ax_line.grid(True)
                    ax_line.ticklabel_format(style='plain', axis='y')
                    st.pyplot(fig_line)

st.markdown("---")
st.header("2. 현재 가치 및 연금 시뮬레이션")

col1_annuity, col2_annuity = st.columns([1,1])

with col1_annuity:
    st.subheader("미래 목표 금액의 현재 가치 계산")
    future_val = st.number_input("미래 목표 금액 (원)", min_value=10000000, value=500000000, step=10000000)
    default_discount_rate = (np.mean(historical_base_rates_percent_app) + bank_margin_rate_percent) / 100 if historical_base_rates_percent_app else 0.03
    discount_rate_pv = st.number_input("연간 할인율 (%)", min_value=0.1, max_value=20.0, value=default_discount_rate * 100, step=0.1) / 100
    years_pv = st.number_input("기간 (년)", min_value=1, value=20, step=1)

    if st.button("현재 가치 계산"):
        present_value = calculate_present_value(future_val, discount_rate_pv, years_pv)
        st.info(f"**{years_pv}년 뒤 {future_val:,.0f}원**의 현재 가치는 약 **{present_value:,.0f}원**입니다.")

with col2_annuity:
    st.subheader("은퇴 후 연금 수령액 시뮬레이션")
    initial_portfolio_for_annuity_default = 1000000000 
    if 'results_by_period_multi_annual' in st.session_state and st.session_state['results_by_period_multi_annual']:
        investment_periods_app_keys = list(st.session_state['results_by_period_multi_annual'].keys())
        if investment_periods_app_keys: 
            max_years = max(investment_periods_app_keys)
            df_max_yr_result = st.session_state['results_by_period_multi_annual'][max_years]
            total_profit_max_yr = df_max_yr_result['Portfolio Value'].iloc[-1] - df_max_yr_result['Invested Capital'].iloc[-1]
            tax_amount_max_yr = total_profit_max_yr * tax_rate_app if total_profit_max_yr > 0 else 0
            initial_portfolio_for_annuity_default = df_max_yr_result['Portfolio Value'].iloc[-1] - tax_amount_max_yr
            if initial_portfolio_for_annuity_default < 0: 
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

            fig_annuity, ax_annuity = plt.subplots(figsize=(12, 6))
            ax_annuity.plot(df_annuity['Age'], df_annuity['Annual Payout (Nominal)'], label='연간 수령액 (명목)', color='blue')
            ax_annuity.plot(df_annuity['Age'], df_annuity['Annual Payout (Real Value)'], label='연간 수령액 (실질 구매력)', color='red', linestyle='--')
            ax_annuity.plot(df_annuity['Age'], df_annuity['Remaining Portfolio Value'], label='남은 포트폴리오 가치', color='green', linestyle=':')
            ax_annuity.set_title('은퇴 후 연금 수령액 시뮬레이션')
            ax_annuity.set_xlabel('나이')
            ax_annuity.set_ylabel('금액 (원)')
            ax_annuity.legend()
            ax_annuity.grid(True)
            ax_annuity.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig_annuity)
