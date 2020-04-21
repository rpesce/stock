import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Define a list of stocks
stock_list = [
    "aapl",
    'msft',
    'goog',
    'fb',
    'intc',
    'ibm'
]

# Create empty data frame
df = pd.DataFrame()


def get_stocks(stocks):
    frame = pd.DataFrame()

    # Loop stocks and create stock object
    for stock in stocks:
        stock_instance = yf.Ticker(stock)
        fr = stock_instance.history(period='max')

        # Delete columns
        del fr['Dividends']
        del fr['Stock Splits']

        # Insert Symbol
        fr.insert(5, "Symbol", stock, allow_duplicates=True)

        # Compare price close tomorrow vs. price close today
        # Insert column 'Increased': True or False
        fr['Increased'] = (fr["Close"].shift(-1) > fr["Close"]).astype(int)

        # Append fr to frame
        frame = frame.append(fr)
    return frame


# Load all stocks into a DataFrame
df = get_stocks(stock_list)
df.sort_index(inplace=True)


# Split train vs. test data (25%)
splitter = int(df.shape[0]*0.25)

train = df[splitter:]
test = df[:splitter]

#Select variables
variables = ['Open', 'High', 'Low', 'Close']

# Get variables
X = train[variables]
y = train['Increased']

# Error analysis
results = []
kf = RepeatedKFold(n_splits=4, n_repeats=10, random_state=10)

for l_train, l_valid in kf.split(X):
    print("Train:", l_train.shape[0])
    print("Valid:", l_valid.shape[0])

    X_train, X_valid = X.iloc[l_train], X.iloc[l_valid]
    y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)

    p = model.predict(X_valid)

    acc = np.mean(y_valid == p)
    results.append(acc)
    print("Acc:", acc)
    print()

# Evaluate results
X_valid_check = train.iloc[l_valid].copy()
X_valid_check['p'] = p
print(X_valid_check.head(20))
# X_valid_check.shape

# Improve

