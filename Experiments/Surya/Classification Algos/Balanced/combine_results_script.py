import os
import pandas as pd

results_path = "./classification_results_2023"
algorithms = {
    "Decision Tree": "1DecisionTree",
    "K-NN": "2K-NN",
    "Logistic Regression": "LogisticRegression",
    "SVM": "SVM"
}

data = []
for patient_folder in os.listdir(results_path):
    patient_id = patient_folder
    patient_folder_path = os.path.join(results_path, patient_folder)
    
    for algorithm, algorithm_folder in algorithms.items():
        report_file = os.path.join(patient_folder_path, algorithm_folder, f"sklearn_models_balanced_for_train_metrics_{algorithm_folder}.csv")
        
        if os.path.isfile(report_file):
            df = pd.read_csv(report_file, index_col=0, header=None, skiprows=1)
            df = df.transpose()
            df["Patient ID"] = patient_id
            df["Algorithm"] = algorithm
            
            
            data.append(df)

result_df = pd.concat(data, ignore_index=True)
columns = ["Patient ID", "Algorithm"] + list(df.columns)
result_df = result_df[columns]
result_file = os.path.join(results_path, "Final_Report.xlsx")
result_df.to_excel(result_file, index=False)

print(f"Final report saved to {result_file}")