Project Overview
This repository contains the solution for automating Payment Service Provider (PSP) routing in online transactions using predictive modelling. The objective is to route each credit card transaction to the most suitable PSP to minimize payment failures and reduce processing fees. The project follows the CRISP-DM methodology and provides a transparent, reproducible workflow from data analysis to deployment.

Folders:

Raw and processed data CSV file (do not track sensitive data).
Production-ready Python scripts for data, modelling, and routing routines along with Exploratory data analysis and modelling (Jupyter).

How to Run the Code
Clone the repository and navigate to the project directory.

Install Dependencies

text
pip install -r requirements.txt
Open notebooks/Task_1_Cleaner_Converted.ipynb and execute each cell sequentially:

View results directly in the notebook, including model metrics, feature importances, confusion matrix, and business cost simulation.


Main Results
Model Performance:
The best model achieves an F1-score greater than 80%, indicating strong predictive ability in distinguishing successful from failed payment routes.

Cost-Aware Routing:
Expected processing costs per PSP are calculated using predicted probabilities, supporting business decisions that balance success probability with fee minimization.

Interpretability:
Key drivers of success (e.g., 3D secure, retry status, hour, country) are visualized and explained in both plots and tables in the notebook and report.