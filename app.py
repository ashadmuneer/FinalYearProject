import streamlit as st
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pandas import to_datetime

# Set page configuration
st.set_page_config(page_title="Stock Forecast with Sentiment Analysis", layout="wide")

# Custom CSS for news card styling
st.markdown("""
<style>
.news-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.news-card img {
    max-width: 100%;
    width: 100%;
    border-radius: 8px;
    object-fit: cover;
}
.news-link {
    color: #1e90ff;
    text-decoration: none;
}
.news-link:hover {
    text-decoration: underline;
}
.news-source {
    color: #6f6f6f;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Stock Prediction", "Stock Compare", "Demo Trading"],
    index=0
)

# Map page selection to file names (assuming multi-page app structure)
page_mapping = {
    "Stock Prediction": "app.py",
    "Stock Compare": "pages/1_Stock_Compare.py",
    "Demo Trading": "pages/2_Demo_Trading.py"
}

# Display warning if navigating to another page
if page != "Stock Prediction":
    st.warning(f"Please navigate to the '{page}' page using the sidebar.")
    st.stop()

# Constants
START = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
FMP_BASE_URL = "https://financialmodelingprep.com/api"
FMP_API_KEY = "odlMepfUe6PiMHYZPv0hsIBL5B38b79P"  # Replace with your actual FMP API key
PLACEHOLDER_IMAGE = "https://via.placeholder.com/150?text=No+Image"

st.title("Stock Forecast with Sentiment Analysis")

# Stock ticker symbols
stocks = (
    "TCS", "LTIM", "SBIN", "INFY", "ICICIBANK", "HDFCBANK",
    "AXISBANK", "BAJAJFINANCE", "WIPRO", "ITC", "POWERGRID",
    "MARUTI", "HCLTECH", "RELIANCE", "BHARTIARTL", "IDFCFIRSTB",
    "FEDERALBNK", "LUPIN", "SUZLON"
)

# User inputs
selected_stock = st.selectbox("Select dataset for prediction", stocks)
n_days = st.slider("Select number of days to predict", min_value=1, max_value=90, value=30)

@st.cache_data
def load_data(ticker, api_key):
    """Load historical stock data from Financial Modeling Prep."""
    try:
        # Try base ticker first
        url = f"{FMP_BASE_URL}/v3/historical-price-full/{ticker}?from={START}&to={TODAY}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or "historical" not in data or not data["historical"]:
            # Fallback: try with .NS suffix
            url = f"{FMP_BASE_URL}/v3/historical-price-full/{ticker}.NS?from={START}&to={TODAY}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "historical" not in data or not data["historical"]:
                st.error(f"No historical data found for {ticker} or {ticker}.NS.")
                return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["historical"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from FMP: {e}")
        return None
    except ValueError as e:
        st.error(f"Error processing FMP data: {e}")
        return None

@st.cache_data
def fetch_news(ticker):
    """Fetch news articles with title, description, pub_date, link, and image."""
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml-xml")  # Use lxml-xml parser
        articles = soup.find_all("item")[:10]
        news_data = []
        
        for article in articles:
            title = article.title.text
            description_html = article.description.text
            description_soup = BeautifulSoup(description_html, "html.parser")
            description = description_soup.get_text()
            source = "Unknown"
            font_tag = description_soup.find("font")
            if font_tag:
                source = font_tag.get_text()
            pub_date = article.pubDate.text
            link = article.link.text
            try:
                pub_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                pub_date = datetime.now()
            image_url = PLACEHOLDER_IMAGE
            try:
                article_response = requests.get(link, timeout=5)
                article_soup = BeautifulSoup(article_response.content, "html.parser")
                og_image = article_soup.find("meta", property="og:image")
                if og_image and og_image["content"]:
                    image_url = og_image["content"]
            except Exception:
                pass
            # Validate image_url
            image_url = image_url if image_url and image_url.startswith('http') else PLACEHOLDER_IMAGE
            news_data.append((title, description, pub_date, link, image_url, source))
        return news_data
    except Exception as e:
        st.warning(f"Error fetching news: {e}")
        return []

def analyze_sentiment(news_data, data_dates):
    """Perform sentiment analysis aligned with trading dates."""
    sentiment_dict = {date: [] for date in data_dates}
    for title, description, pub_date, _, _, _ in news_data:
        text = title + " " + description
        sentiment = TextBlob(text).sentiment.polarity
        pub_date = to_datetime(pub_date).date()
        closest_date = min(data_dates, key=lambda x: abs(x.date() - pub_date))
        sentiment_dict[closest_date].append(sentiment)
    sentiment_scores = [np.mean(sentiment_dict[date]) if sentiment_dict[date] else 0 for date in data_dates]
    return sentiment_scores

# Load data
with st.spinner("Loading data..."):
    data = load_data(selected_stock, FMP_API_KEY)
if data is None:
    st.stop()

st.subheader("Historical Stock Data")
st.write(data.tail())

# Plot historical stock data
def plot_stock_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock Close"))
    fig.layout.update(
        title_text="Historical Stock Prices with Rangeslider",
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title="Price (INR)"
    )
    st.plotly_chart(fig)

plot_stock_data()

# Fetch and analyze news
with st.spinner("Fetching news and analyzing sentiment..."):
    news_data = fetch_news(selected_stock)
    sentiment_scores = analyze_sentiment(news_data, data["Date"].tolist())

# Display news articles as cards
st.subheader("Recent News Articles and Sentiment Analysis")
if news_data:
    for i, (title, description, pub_date, link, image_url, source) in enumerate(news_data):
        with st.container():
            st.markdown('<div class="news-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image_url, caption="Article Image")  # Removed use_container_width
            with col2:
                st.markdown(f"**Article {i+1}: {title}**")
                st.markdown(f'<span class="news-source">{source}</span>', unsafe_allow_html=True)
                st.write(f"**Description:** {description}")
                st.write(f"**Published Date:** {pub_date}")
                sentiment = TextBlob(title + ' ' + description).sentiment.polarity
                st.write(f"**Sentiment Score:** {sentiment:.2f}")
                st.markdown(f'<a href="{link}" class="news-link" target="_blank">Read Full Article</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("---")
else:
    st.warning("No news articles found. Using neutral sentiment (0).")

# Prepare data for modeling
data["Sentiment"] = sentiment_scores
data["Day"] = range(len(data))
data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["Volatility"] = data["Close"].rolling(window=5).std()
data.interpolate(method="linear", inplace=True)
data.dropna(inplace=True)

# Prophet model
prophet_data = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=n_days)
forecast = prophet_model.predict(future)

# Random Forest model
X = data[["Day", "Sentiment", "SMA_5", "SMA_20", "Volatility"]]
y = data["Close"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Random Forest Model MAE: {mae:.2f} INR")

# Predict future prices
future_dates = pd.date_range(start=TODAY, periods=n_days).tolist()
future_prices = []
last_sma_5 = data["SMA_5"].iloc[-5:].tolist()
last_sma_20 = data["SMA_20"].iloc[-20:].tolist()
last_volatility = data["Volatility"].iloc[-1]

for i in range(n_days):
    future_day = pd.DataFrame({
        "Day": [len(data) + i],
        "Sentiment": [np.mean(sentiment_scores) if sentiment_scores else 0],
        "SMA_5": [np.mean(last_sma_5[-5:])],
        "SMA_20": [np.mean(last_sma_20[-20:])],
        "Volatility": [last_volatility]
    })
    future_day_scaled = scaler.transform(future_day)
    pred_price = model.predict(future_day_scaled)[0]
    future_prices.append(pred_price)
    last_sma_5.append(pred_price)
    last_sma_5 = last_sma_5[-5:]
    last_sma_20.append(pred_price)
    last_sma_20 = last_sma_20[-20:]
    last_volatility = np.std(last_sma_5) if len(last_sma_5) >= 5 else last_volatility

# Average predictions
average_prices = (np.array(future_prices) + forecast["yhat"][-n_days:].values) / 2

# Plot combined forecast
st.subheader(f"Stock Price Prediction for {n_days} Days")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Actual Price"))
fig2.add_trace(go.Scatter(x=future_dates, y=future_prices, name="Sentiment-based Prediction"))
fig2.add_trace(go.Scatter(x=forecast["ds"][-n_days:], y=forecast["yhat"][-n_days:], name="Prophet Forecast"))
fig2.add_trace(go.Scatter(
    x=forecast["ds"][-n_days:],
    y=forecast["yhat_upper"][-n_days:],
    mode="lines",
    line=dict(width=0),
    showlegend=False
))
fig2.add_trace(go.Scatter(
    x=forecast["ds"][-n_days:],
    y=forecast["yhat_lower"][-n_days:],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(0,100,80,0.2)",
    name="Prophet Confidence Interval"
))
fig2.layout.update(
    title_text="Predicted Stock Prices with Sentiment Analysis and Prophet",
    xaxis_rangeslider_visible=True,
    xaxis_title="Date",
    yaxis_title="Price (INR)"
)
st.plotly_chart(fig2)

# Plot average prediction
st.subheader("Average of Sentiment-based and Prophet Predictions")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Actual Price"))
fig3.add_trace(go.Scatter(x=future_dates, y=average_prices, name="Average Prediction"))
fig3.layout.update(
    title_text="Average of Sentiment-based and Prophet Predictions",
    xaxis_rangeslider_visible=True,
    xaxis_title="Date",
    yaxis_title="Price (INR)"
)
st.plotly_chart(fig3)

# Download predictions
download_data = pd.DataFrame({
    "Date": future_dates,
    "Sentiment-based Prediction": future_prices,
    "Prophet Prediction": forecast["yhat"][-n_days:].values,
    "Average Prediction": average_prices
})
st.download_button(
    label="Download Predictions as CSV",
    data=download_data.to_csv(index=False),
    file_name=f"{selected_stock}_predictions.csv",
    mime="text/csv"
)
