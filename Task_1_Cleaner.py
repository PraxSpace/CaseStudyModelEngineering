import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings; import sys
warnings.filterwarnings("ignore")
# Global PSP fee map as provided
psp_fees = {
    'Moneycard': {'success_fee': 5.00, 'fail_fee': 2.00},
    'Goldcard': {'success_fee': 10.00, 'fail_fee': 5.00},
    'UK_Card': {'success_fee': 3.00, 'fail_fee': 1.00},
    'Simplecard': {'success_fee': 1.00, 'fail_fee': 0.50},}
# 1. Data Preperation and cleanup of the CSV file provided
def load_and_clean_dataset(file):
    try:
        fileData = pd.read_excel(file)
        fileData['tmsp'] = pd.to_datetime(fileData['tmsp'])
        fileData = fileData.dropna()
        print("\nFile read and cleaned successfully.")
        return fileData
    except FileNotFoundError:
        print(f"\nFile not found: {file}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        sys.exit(1)
# 2. Feature Engineering data preparation as per the requirement
def engineer_features_data_prep(fileData):
    try:
        fileData['hour'] = fileData['tmsp'].dt.hour
        fileData['day'] = fileData['tmsp'].dt.day
        fileData['weekday'] = fileData['tmsp'].dt.weekday
        fileData['month'] = fileData['tmsp'].dt.month
        fileData = fileData.sort_values('tmsp')
        fileData['prev_tmsp'] = fileData.groupby(['country', 'amount'])['tmsp'].shift(1)
        fileData['seconds_diff'] = (fileData['tmsp'] - fileData['prev_tmsp']).dt.total_seconds()
        fileData['is_retry'] = fileData['seconds_diff'].lt(60).fillna(False).astype(int)
        print("\nFeature engineering completed.")
        return fileData
    except Exception as e:
        print(f"\nFeature engineering error: {str(e)}")
        sys.exit(1)
# 3. Basic Exploratory Data Analysis
def run_Exploratory_Data_Analysis(data):
    try:
        print("\nSuccess rate per PSP:")
        print(data.groupby("PSP")['success'].mean())
        print("\nSuccess rate by 3D Secure flag:")
        print(data.groupby("3D_secured")['success'].mean())
        pivot = data.pivot_table(index='hour', columns='PSP', values='success', aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='YlGnBu')
        plt.title("Success Rate by Hour and PSP:")
        plt.show()
    except Exception as e:
        print(f"\nError during EDA: {str(e)}")
        sys.exit(1)
# 4. Prepare Data for Model to train. 80/20 split applied here
def prepare_model_data(df):
    try:
        df_model = pd.get_dummies(df, columns=['PSP', 'country', 'card'], drop_first=True)
        features = ['amount', '3D_secured', 'hour', 'weekday', 'is_retry'] + \
                   [col for col in df_model.columns if col.startswith(('PSP_', 'country_', 'card_'))]
        X = df_model[features]
        y = df_model['success']
        print("\nModel prepared!")
        return train_test_split(X, y, test_size=0.2, random_state=42), features
    except Exception as e:
        print(f"\nError during model preparation: {str(e)}")
        sys.exit(1)
# 5. The prepared model trained to extract classifier report
def train_predictive_model(X_train, y_train, X_test, y_test):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        return model
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        sys.exit(1)
# 6. PSP Routing Simulation
def train_routing_models(df, features):
    try:
        psp_models = {}
        psps = df['PSP'].unique()
        df_encoded = pd.get_dummies(df, columns=['PSP', 'country', 'card'], drop_first=True)
        for psp in psps:
            df_encoded['target'] = (df['PSP'] == psp) & (df['success'] == 1)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df_encoded[features], df_encoded['target'])
            psp_models[psp] = model
        return psp_models
    except Exception as e:
        print(f"\nError during train routing model: {str(e)}")
        sys.exit(1)
def simulate_routing(psp_models, sample_tx):
    try:
        scores = {psp: model.predict_proba(sample_tx)[0][1] for psp, model in psp_models.items()}
        best_psp = max(scores, key=scores.get)
        print("\nPredicted Best PSP for Sample Transaction:")
        for psp, score in scores.items():
            print(f"{psp}: {score:.2f}")
        print(f"\nRute to: {best_psp}")
        return best_psp
    except Exception as e:
        print(f"\nError during routing simulation: {str(e)}")
        sys.exit(1)
def simulate_routing_with_cost(psp_models, sample_tx, psp_fees):
    try:
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
    except Exception as e:
        print(f"\nError during cost related simulation: {str(e)}")
        sys.exit(1)
# Main Execution
if __name__ == "__main__":
    filepath = "PSP_Jan_Feb_2019.xlsx"  #Reading the CSV file
    try:
        # Load + Process + data cleanup
        df = load_and_clean_dataset(filepath)
        df = engineer_features_data_prep(df)
        # Exploratory Data Analysis
        run_Exploratory_Data_Analysis(df)
        # Modeling
        (X_train, X_test, y_train, y_test), features = prepare_model_data(df)
        model = train_predictive_model(X_train, y_train, X_test, y_test)
        # Routing Simulation
        psp_models = train_routing_models(df, features)
        # Pick a sample and simulate routing
        if not X_test.empty:
            sample_tx = X_test.iloc[[0]]  # First row of test set
            simulate_routing_with_cost(psp_models, sample_tx, psp_fees)
            simulate_routing(psp_models, sample_tx)
        else:
            print(" No test data available to simulation routing.")
    except Exception as e:
        print(f"\n Unexpected error in main: {str(e)}")
        sys.exit(1)
