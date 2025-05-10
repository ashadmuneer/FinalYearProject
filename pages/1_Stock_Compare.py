import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
from datetime import datetime, timedelta
import google.generativeai as genai

# Title and description
st.title("Stock Comparison Dashboard")
st.write("Compare the performance of two stocks over a specified time period using Polygon.io data.")

# Predefined list of popular stock tickers
popular_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NFLX", "NVDA", "INTC", "ADBE",
    "JPM", "V", "MA", "DIS", "BABA",
    "WMT", "T", "PEP", "KO", "PFE"
]

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
stock1 = st.sidebar.selectbox("Select first stock:", popular_stocks, index=0)
stock2 = st.sidebar.selectbox("Select second stock:", popular_stocks, index=1)

# Date range selection
st.sidebar.header("Date Range")
default_end = datetime.today()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Polygon.io API key
POLYGON_API_KEY = "Uj_gHtqtdz9iBABRJh9t4Sc3zczYEYdI"  # Replace with your actual API key

# Function to fetch and normalize stock data
def get_stock_data(ticker, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "apiKey": POLYGON_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "results" not in data:
            st.error(f"No data found for {ticker}.")
            return None
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df["Close"] = df["c"]
        df["Normalized"] = 100 * df["Close"] / df["Close"].iloc[0]
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df[['Close', 'Normalized']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Plotting function
def plot_stock_performance(df1, df2, ticker1, ticker2):
    fig = go.Figure()
    if df1 is not None:
        fig.add_trace(go.Scatter(x=df1.index, y=df1['Normalized'], name=f"{ticker1} Normalized", line=dict(color='blue')))
    if df2 is not None:
        fig.add_trace(go.Scatter(x=df2.index, y=df2['Normalized'], name=f"{ticker2} Normalized", line=dict(color='orange')))
    fig.update_layout(title="Normalized Stock Price Performance", xaxis_title="Date", yaxis_title="Normalized Price (Starting at 100)", hovermode="x unified", template="plotly_white")
    return fig

# Difference calculation
def calculate_performance_difference(df1, df2, ticker1, ticker2):
    if df1 is None or df2 is None:
        return None
    diff = df1['Normalized'] - df2['Normalized']
    return pd.DataFrame({f"{ticker1} - {ticker2} (%)": diff}, index=df1.index)

# Session state init
if 'df1' not in st.session_state: st.session_state.df1 = None
if 'df2' not in st.session_state: st.session_state.df2 = None
if 'tickers' not in st.session_state: st.session_state.tickers = ("", "")

# Compare Stocks
if st.sidebar.button("Compare Stocks"):
    df1 = get_stock_data(stock1, start_date, end_date)
    df2 = get_stock_data(stock2, start_date, end_date)
    if df1 is not None and df2 is not None:
        st.session_state.df1 = df1
        st.session_state.df2 = df2
        st.session_state.tickers = (stock1, stock2)

# If data exists in session, show charts
if st.session_state.df1 is not None and st.session_state.df2 is not None:
    df1 = st.session_state.df1
    df2 = st.session_state.df2
    stock1, stock2 = st.session_state.tickers

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{stock1} Summary")
        st.write(f"Latest Price: ${df1['Close'].iloc[-1]:.2f}")
        st.write(f"Price Change: {(df1['Close'].iloc[-1] / df1['Close'].iloc[0] - 1) * 100:.2f}%")
    with col2:
        st.subheader(f"{stock2} Summary")
        st.write(f"Latest Price: ${df2['Close'].iloc[-1]:.2f}")
        st.write(f"Price Change: {(df2['Close'].iloc[-1] / df2['Close'].iloc[0] - 1) * 100:.2f}%")
    
    st.plotly_chart(plot_stock_performance(df1, df2, stock1, stock2), use_container_width=True)

    diff_df = calculate_performance_difference(df1, df2, stock1, stock2)
    if diff_df is not None:
        st.subheader("Performance Difference")
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=diff_df.index,
            y=diff_df[f"{stock1} - {stock2} (%)"],
            name="Performance Difference",
            line=dict(color='green')
        ))
        fig_diff.update_layout(
            title=f"Performance Difference ({stock1} - {stock2})",
            xaxis_title="Date",
            yaxis_title="Difference (%)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_diff, use_container_width=True)

# --- Chatbot Section ---
st.sidebar.header("Stock Chatbot")
st.sidebar.write("Ask questions about stocks or the dashboard!")

# Gemini API config
GEMINI_API_KEY = "AIzaSyBX_5S1b7N6NZTn9Pa3Lb9HAnXIyAa07pQ"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input form
with st.sidebar.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    try:
        response = model.generate_content(user_input)
        response_text = response.text if response.text else "Sorry, I couldn't generate a response."
        st.session_state.chat_history.append({"user": user_input, "bot": response_text})
    except Exception as e:
        st.sidebar.error(f"Error with Gemini API: {e}")
elif submit_button and not user_input:
    st.sidebar.warning("Please enter a question.")

# Display chat history
if st.session_state.chat_history:
    st.sidebar.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.write(f"**You**: {chat['user']}")
        st.sidebar.write(f"**Bot**: {chat['bot']}")