import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import bcrypt
import uuid
import json
import os
import streamlit.components.v1 as components

# File to persist user database
USERS_DB_FILE = "users_db.json"

# Load user database from JSON file
def load_users_db():
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, "r") as f:
                data = json.load(f)
                # Convert stored hashed passwords (bytes) back to bytes
                return {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {"testuser": bcrypt.hashpw("password123".encode(), bcrypt.gensalt())}
    return {"testuser": bcrypt.hashpw("password123".encode(), bcrypt.gensalt())}

# Save user database to JSON file
def save_users_db(users_db):
    # Convert hashed passwords (bytes) to strings for JSON serialization
    serializable_db = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in users_db.items()}
    with open(USERS_DB_FILE, "w") as f:
        json.dump(serializable_db, f, indent=4)

# Initialize users_db
users_db = load_users_db()

# FMP API key (replace with your actual API key)
API_KEY = "odlMepfUe6PiMHYZPv0hsIBL5B38b79P"  # Replace this with your actual FMP API key

USER_DATA_FILE = "user_data.json"

# Load user data from JSON
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Save user data to JSON
def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(user_data, f, indent=4)

# Initialize session state
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "balance" not in st.session_state:
        st.session_state.balance = 100000.0
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}
    if "transactions" not in st.session_state:
        st.session_state.transactions = []
    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}

# Load user data into session state
def load_user_session(username):
    user_data = load_user_data()
    if username in user_data:
        st.session_state.balance = user_data[username].get("balance", 100000.0)
        st.session_state.portfolio = user_data[username].get("portfolio", {})
        st.session_state.transactions = user_data[username].get("transactions", [])
    else:
        st.session_state.balance = 100000.0
        st.session_state.portfolio = {}
        st.session_state.transactions = []

# Save session state to user data
def save_user_session(username):
    user_data = load_user_data()
    user_data[username] = {
        "balance": st.session_state.balance,
        "portfolio": st.session_state.portfolio,
        "transactions": st.session_state.transactions
    }
    save_user_data(user_data)

# Registration page
def registration():
    st.title("📋 Registration Page")
    
    username = st.text_input("Enter a username", key="reg_username")
    password = st.text_input("Enter a password", type="password", key="reg_password")
    confirm_password = st.text_input("Confirm your password", type="password", key="reg_confirm_password")

    if password != confirm_password:
        st.error("Passwords do not match!")
        return

    if st.button("Register"):
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        if username in users_db:
            st.error("Username already exists!")
            return
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users_db[username] = hashed_password
        save_users_db(users_db)  # Persist the updated users_db
        
        # Store credentials in localStorage using JavaScript
        components.html(
            f"""
            <script>
                localStorage.setItem('username', '{username}');
                localStorage.setItem('password', '{password}');
                console.log('Credentials saved:', localStorage.getItem('username'), localStorage.getItem('password'));
            </script>
            """,
            height=0,
        )
        st.success("Registration successful! Please login.")

# Login page
def login():
    st.title("🔑 Login Page")
    
    username = st.text_input("Enter your username", key="login_username")
    password = st.text_input("Enter your password", type="password", key="login_password")

    if st.button("Login"):
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        # Debug: Check if the user exists in users_db
        if username not in users_db:
            st.error(f"Debug: Username '{username}' not found in users_db. Please register first.")
            return
        # Debug: Check the password match
        if bcrypt.checkpw(password.encode(), users_db[username]):
            st.session_state.logged_in = True
            st.session_state.username = username
            load_user_session(username)
            st.session_state.price_cache = {}
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials, please try again.")
            st.write("Debug: Password does not match the stored hashed password.")

# Cached function to fetch the latest stock price
@st.cache_data(ttl=60)
def get_latest_price(symbol):
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and "price" in data[0]:
            st.session_state.price_cache[symbol] = data[0]["price"]
            return data[0]["price"]
        return None
    except (requests.RequestException, ValueError, KeyError) as e:
        st.error(f"Error fetching price for {symbol}: {str(e)}")
        return None

# Cached function to fetch historical stock data
@st.cache_data(ttl=3600)
def get_historical_data(symbol):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=30&apikey={API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "historical" in data:
            return data["historical"]
        return None
    except (requests.RequestException, ValueError, KeyError) as e:
        st.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return None

# Stock Trading Application
def stock_trading_app():
    st.title("📈 Demat Stock Buy & Sell Simulator")

    # Sidebar for account info
    st.sidebar.header("💼 Account Info")
    st.sidebar.text(f"👤 Username: {st.session_state.username}")
    st.sidebar.text(f"🪙 Balance: ${st.session_state.balance:,.2f}")

    # Calculate portfolio value using cached prices
    portfolio_value = 0
    for symbol, qty in st.session_state.portfolio.items():
        if qty > 0:
            price = st.session_state.price_cache.get(symbol) or get_latest_price(symbol)
            if price:
                portfolio_value += qty * price
    st.sidebar.text(f"📈 Portfolio Value: ${portfolio_value:,.2f}")

    if st.sidebar.button("Logout"):
        save_user_session(st.session_state.username)
        st.session_state.clear()
        st.rerun()

    # Dropdown for selecting stock symbol
    stock_symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META"]
    symbol = st.selectbox("Select Stock Symbol", stock_symbols)

    if symbol:
        with st.spinner("Fetching stock data..."):
            price = get_latest_price(symbol)
            data = get_historical_data(symbol)

        if price is not None and data:
            st.subheader(f"{symbol} - Last 30 Days Candlestick Chart")

            # Prepare the data for the candlestick chart
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Candlestick Chart"
            )])

            fig.update_layout(
                title=f"{symbol} Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )

            st.plotly_chart(fig)

            # Live Price Display with manual refresh
            price_placeholder = st.empty()
            price_placeholder.markdown(f"**Live Price:** ${price:.2f} (USD)")
            if st.button("Refresh Price"):
                price = get_latest_price(symbol, _cache=False)
                if price:
                    price_placeholder.markdown(f"**Live Price:** ${price:.2f} (USD)")

            # Buy/Sell Interface
            col1, col2 = st.columns(2)
            with col1:
                qty = st.number_input("Buy Quantity", min_value=1, step=1, value=1)
                if st.button("Buy"):
                    total = qty * price
                    if total <= st.session_state.balance:
                        st.session_state.balance -= total
                        st.session_state.portfolio[symbol] = st.session_state.portfolio.get(symbol, 0) + qty
                        
                        # Add transaction to history
                        transaction = {
                            "ID": str(uuid.uuid4()),
                            "Symbol": symbol,
                            "Quantity": qty,
                            "Type": "Buy",
                            "Price": price,
                            "Total": total,
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.transactions.append(transaction)
                        
                        save_user_session(st.session_state.username)
                        st.success(f"Bought {qty} shares of {symbol} at ${price:.2f}")
                    else:
                        st.error("Insufficient balance.")

            with col2:
                qty_sell = st.number_input("Sell Quantity", min_value=1, step=1, value=1, key="sell_qty")
                if st.button("Sell"):
                    owned = st.session_state.portfolio.get(symbol, 0)
                    if owned >= qty_sell:
                        total = qty_sell * price
                        st.session_state.balance += total
                        st.session_state.portfolio[symbol] -= qty_sell
                        if st.session_state.portfolio[symbol] == 0:
                            del st.session_state.portfolio[symbol]
                        
                        # Add transaction to history
                        transaction = {
                            "ID": str(uuid.uuid4()),
                            "Symbol": symbol,
                            "Quantity": qty_sell,
                            "Type": "Sell",
                            "Price": price,
                            "Total": total,
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.transactions.append(transaction)
                        
                        save_user_session(st.session_state.username)
                        st.success(f"Sold {qty_sell} shares of {symbol} at ${price:.2f}")
                    else:
                        st.error("Not enough shares.")

            # Portfolio
            st.subheader("📊 Your Portfolio")
            if st.session_state.portfolio:
                portfolio_data = [
                    {
                        "Symbol": k, 
                        "Quantity": v, 
                        "Current Price": st.session_state.price_cache.get(k) or get_latest_price(k) or 0, 
                        "Total Value": v * (st.session_state.price_cache.get(k) or get_latest_price(k) or 0)
                    } 
                    for k, v in st.session_state.portfolio.items() if v > 0
                ]
                st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
            else:
                st.info("No stocks in portfolio yet.")

            # Transactions (show last 50 for performance)
            st.subheader("🧾 Transaction History")
            if st.session_state.transactions:
                st.dataframe(
                    pd.DataFrame(st.session_state.transactions[-50:]),
                    use_container_width=True
                )
            else:
                st.info("No transactions yet.")
        else:
            st.error("Invalid symbol or data unavailable.")

# Main Function to Control Flow
def main():
    init_session_state()
    if st.session_state.logged_in and st.session_state.username in users_db:
        load_user_session(st.session_state.username)
        stock_trading_app()
    else:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.sidebar.title("Login/Register")
        menu = ["Login", "Register"]
        choice = st.sidebar.radio("Menu", menu)
        
        if choice == "Login":
            login()
        elif choice == "Register":
            registration()

if __name__ == "__main__":
    main()