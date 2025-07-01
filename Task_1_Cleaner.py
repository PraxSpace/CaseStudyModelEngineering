import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Global PSP fee map
psp_fees = {
    'Moneycard': {'success_fee': 5.00, 'fail_fee': 2.00},
    'Goldcard': {'success_fee': 10.00, 'fail_fee': 5.00},
    'UK_Card': {'success_fee': 3.00, 'fail_fee': 1.00},
    'Simplecard': {'success_fee': 1.00, 'fail_fee': 0.50},
}

# 1. Data Preperation and cleanup
def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df['tmsp'] = pd.to_datetime(df['tmsp'])
    df = df.dropna()
    print("\nFile read sucessfully..!")
    return df

# 2. Feature Engineering
def engineer_features(df):
    df['hour'] = df['tmsp'].dt.hour
    df['day'] = df['tmsp'].dt.day
    df['weekday'] = df['tmsp'].dt.weekday
    df['month'] = df['tmsp'].dt.month
    df = df.sort_values('tmsp')
    df['prev_tmsp'] = df.groupby(['country', 'amount'])['tmsp'].shift(1)
    df['seconds_diff'] = (df['tmsp'] - df['prev_tmsp']).dt.total_seconds()
    df['is_retry'] = df['seconds_diff'].lt(60).fillna(False).astype(int)
    print("\nfeature related to retries added..!")
    return df



if __name__ == "__main__":
    filepath = "PSP_Jan_Feb_2019.xlsx"
    
    # Load + Process
    df = load_and_clean_data(filepath)
    df = engineer_features(df)


