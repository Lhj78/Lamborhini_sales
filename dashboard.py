import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("lamborghini_stock_data.csv")

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Create Sales column (feature engineering)
    df['Sales'] = ((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']

    return df

df = load_data()

# -------------------- TITLE --------------------
st.title("📊 Lamborghini Sales Dashboard")
st.markdown("Sales engineered using **price × volume approximation**")

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔎 Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)

filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# -------------------- KPI METRICS --------------------
total_sales = filtered_df['Sales'].sum()
avg_sales = filtered_df['Sales'].mean()
max_sales = filtered_df['Sales'].max()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"{total_sales:,.0f}")
col2.metric("📊 Average Sales", f"{avg_sales:,.0f}")
col3.metric("🚀 Max Sales", f"{max_sales:,.0f}")

st.divider()

# -------------------- SALES TREND --------------------
st.subheader("📈 Sales Trend")

fig, ax = plt.subplots()
ax.plot(filtered_df['Date'], filtered_df['Sales'])
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.grid(True)

st.pyplot(fig)

# -------------------- MONTHLY SALES --------------------
st.subheader("📊 Monthly Sales")

monthly_sales = (
    filtered_df
    .set_index('Date')['Sales']
    .resample('ME')
    .sum()
)

fig, ax = plt.subplots()
monthly_sales.plot(kind='bar', ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Sales")

st.pyplot(fig)

# -------------------- SALES DISTRIBUTION --------------------
st.subheader("📦 Sales Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df['Sales'], bins=30)
ax.set_xlabel("Sales")
ax.set_ylabel("Frequency")

st.pyplot(fig)

# -------------------- SALES VS VOLUME --------------------
st.subheader("🔁 Sales vs Volume")

fig, ax = plt.subplots()
ax.scatter(filtered_df['Volume'], filtered_df['Sales'])
ax.set_xlabel("Volume")
ax.set_ylabel("Sales")

st.pyplot(fig)

# -------------------- CORRELATION --------------------
st.subheader("📊 Correlation with Sales")

numeric_df = filtered_df.select_dtypes(include='number')
corr = numeric_df.corr()['Sales']

fig, ax = plt.subplots()
corr.plot(kind='bar', ax=ax)
ax.set_ylabel("Correlation")

st.pyplot(fig)

# -------------------- RAW DATA --------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df)
