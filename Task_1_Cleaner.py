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


# 3. Basic Exploratory Data Analysis
def run_eda(df):
    print("\nSuccess rate per PSP:")
    print(df.groupby("PSP")['success'].mean())

    print("\nSuccess rate by 3D Secure flag:")
    print(df.groupby("3D_secured")['success'].mean())

    pivot = df.pivot_table(index='hour', columns='PSP', values='success', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title("Success Rate by Hour and PSP")
    plt.show()

# -------------------------
# 4. Prepare Data for Model
# -------------------------
def prepare_model_data(df):
    df_model = pd.get_dummies(df, columns=['PSP', 'country', 'card'], drop_first=True)
    features = ['amount', '3D_secured', 'hour', 'weekday', 'is_retry'] + \
               [col for col in df_model.columns if col.startswith(('PSP_', 'country_', 'card_'))]
    X = df_model[features]
    y = df_model['success']
    print("\nModel prepared..!")
    return train_test_split(X, y, test_size=0.2, random_state=42), features

# -------------------------
# 5. Train Classifier
# -------------------------
def train_predictive_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    return model

# -------------------------
# 6. PSP Routing Simulation
# -------------------------
def train_routing_models(df, features):
    psp_models = {}
    psps = df['PSP'].unique()
    df_encoded = pd.get_dummies(df, columns=['PSP', 'country', 'card'], drop_first=True)

    for psp in psps:
        df_encoded['target'] = (df['PSP'] == psp) & (df['success'] == 1)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df_encoded[features], df_encoded['target'])
        psp_models[psp] = model
    return psp_models

def simulate_routing(psp_models, sample_tx):
    scores = {psp: model.predict_proba(sample_tx)[0][1] for psp, model in psp_models.items()}
    best_psp = max(scores, key=scores.get)
    print("\nPredicted Best PSP for Sample Transaction:")
    for psp, score in scores.items():
        print(f"{psp}: {score:.2f}")
    print(f"\nRute to: {best_psp}")
    return best_psp

def simulate_routing_with_cost(psp_models, sample_tx, psp_fees):
    scores = {}
    for psp, model in psp_models.items():
        prob_success = model.predict_proba(sample_tx)[0][1]
        fees = psp_fees[psp]
        expected_cost = prob_success * fees['success_fee'] + (1 - prob_success) * fees['fail_fee']
        scores[psp] = {'prob': prob_success, 'expected_cost': expected_cost}

    # Find PSP with lowest expected cost
    best_psp = min(scores.items(), key=lambda x: x[1]['expected_cost'])

    print("\nPSP Scores (Success Probability and Expected Cost):")
    for psp, data in scores.items():
        print(f"{psp}: P(success)={data['prob']:.2f}, Expected Cost={data['expected_cost']:.4f} €")

    print(f"\nOptimal PSP (lowest expected cost): {best_psp[0]} with cost {best_psp[1]['expected_cost']:.4f} €")
    return best_psp[0]

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    filepath = "PSP_Jan_Feb_2019.xlsx"
    
    # Load + Process
    df = load_and_clean_data(filepath)
    df = engineer_features(df)

    # EDA
    run_eda(df)

    # Modeling
    (X_train, X_test, y_train, y_test), features = prepare_model_data(df)
    model = train_predictive_model(X_train, y_train, X_test, y_test)

    # Routing Simulation
    psp_models = train_routing_models(df, features)

    # Pick a sample and simulate routing
    sample_tx = X_test.iloc[[0]]  # First row of test set
    simulate_routing_with_cost(psp_models, sample_tx, psp_fees)
    simulate_routing(psp_models, sample_tx)
