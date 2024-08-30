import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde, entropy  
import subprocess 
        

class Analyze:
    def __init__(self, original_df, synthetic_df):
        self.df1 = original_df
        self.df2 = synthetic_df

  

        if self.df1.shape[1] != self.df2.shape[1]:
            raise Exception("The number of columns in the original and synthetic datasets must match.")
        
        if self.df1.shape[0] != self.df2.shape[0]:
            raise Exception("The number of rows in the original and synthetic datasets are different ",
                             self.df1.shape[0], self.df2.shape[0], "may cause issues in the analysis. \
                                Comment this line if you want to proceed.")
            pass

    def meanAndStd(self) -> pd.DataFrame:
        output = {}
        for column in self.df1.columns:
            mean_df1 = self.df1[column].mean()
            std_df1  = self.df1[column].std()
            mean_df2 = self.df2[column].mean()
            std_df2  = self.df2[column].std()
            meandiff = abs(mean_df1 - mean_df2)

            output[column] = {'Mean_diff': meandiff,
                            'Mean_original_df': mean_df1, 'Mean_synthetic_df': mean_df2,
                            'Std_original_df' : std_df1,  'Std_synthetic_df' : std_df2}
        return pd.DataFrame(output).transpose().sort_values(by='Mean_diff', ascending=False)

    def synthetic_data_vault_project_metric(self) -> None:
        #%pip install sdv
        from sdv.metadata import SingleTableMetadata
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.evaluation.single_table import run_diagnostic

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.df1)
        
        quality_report = evaluate_quality(
            real_data=self.df1,
            synthetic_data=self.df2,
            metadata=metadata)
        print(quality_report)

        diagnostic_report = run_diagnostic(
            real_data=self.df1,
            synthetic_data=self.df2,
            metadata=metadata)
        print(diagnostic_report)

    def gretel_metric(self) -> None:
        #%pip install -U gretel-client
        from gretel_client import configure_session
        from gretel_client.evaluation.quality_report import QualityReport

        pd.set_option("max_colwidth", None)
        configure_session(api_key="prompt", cache="yes", validate=True)
        report = QualityReport(data_source=self.df2, ref_data=self.df1)
        report.run()
        # print(report.peek())
        # print((report.as_dict))
        return report.peek()

        # #gretel models create --config evaluate/default --in-data gretel_generated_minority_data.csv 
        # #       --ref-data gretel_original_minority_data.csv --output report-dir 
        # def run_gretel_command(args) -> None:
        #     try:
        #         result = subprocess.run(args, capture_output=True, text=True, check=True)
        #         return result.stdout.strip()
        #     except subprocess.CalledProcessError as e:
        #         return e.stderr.strip()

        # #self.df1.to_csv('gretel_original_minority_data.csv', index=False)
        # #self.df2.to_csv('gretel_generated_minority_data.csv', index=False)
 
        # cmd_args = [
        #     "gretel", "models", "create",
        #     "--config", "evaluate/default",
        #     "--in-data", "gretel_generated_minority_data.csv",
        #     "--ref-data", "gretel_original_minority_data.csv",
        #     "--output", "report-dir"
        # ]

        # log = run_gretel_command(cmd_args)
        # print(f"Command Executed ",{log}) 


    def plotHeatMaps(self,annot=False) -> None:
        _ , axes = plt.subplots(1, 2, figsize=(18, 10))
        sns.heatmap(self.df1.corr(), annot=annot, cmap='viridis', ax=axes[0])
        axes[0].set_title('Original')

        sns.heatmap(self.df2.corr(), annot=annot, cmap='viridis', ax=axes[1])
        axes[1].set_title('Synthetic')
        plt.tight_layout()
        plt.show()

    def plotScatterPlots(self, start_index, end_index) -> None:
        num_columns = len(self.df1.columns)
        num_rows = (num_columns + 4) // 5  

        fig, axs = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
        for i, column in enumerate(self.df1.columns):
            row_idx = i // 5
            col_idx = i % 5

            axs[row_idx, col_idx].scatter(self.df1.index[start_index:end_index], 
                                          self.df1[column][start_index:end_index], color='red', label='original_df')
            axs[row_idx, col_idx].scatter(self.df2.index[start_index:end_index], 
                                          self.df2[column][start_index:end_index], color='blue', label='synthetic_df')

            axs[row_idx, col_idx].set_title(f"Scatter Plot: {column}")
            axs[row_idx, col_idx].set_xlabel("Index")
            axs[row_idx, col_idx].set_ylabel(column)

            axs[row_idx, col_idx].legend()

        for i in range(num_columns, num_rows * 5):
            row_idx = i // 5
            col_idx = i % 5
            fig.delaxes(axs[row_idx, col_idx])

        plt.tight_layout()
        plt.show()

        for i in range(num_columns, num_rows * 5):
            row_idx = i // 5
            col_idx = i % 5
            fig.delaxes(axs[row_idx, col_idx])

        plt.tight_layout()
        plt.show()
    

    def plot_density(self) -> None:
        num_columns = len(self.df2.columns)
        num_rows = int(np.ceil(num_columns / 5))
        fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))

        highlighted_areas = {} 
        kl_divergences = {}  

        for i, column in enumerate(self.df1.columns):
            row = i // 5
            col = i % 5
            ax = axes[row, col] if num_rows > 1 else axes[col]

            sns.kdeplot(data=self.df1[column], color='blue', label='original_df', ax=ax)
            sns.kdeplot(data=self.df2[column], color='green', label='synthetic_df', ax=ax)

            # Fill between the curves
            x = np.linspace(0, 1, 1000)  
            kde_original = gaussian_kde(self.df1[column])
            kde_synthetic = gaussian_kde(self.df2[column])
            y1 = kde_original(x)
            y2 = kde_synthetic(x)
            ax.fill_between(x, y1, y2, where=(y1 > y2), interpolate=True, color='lightcoral', alpha=0.3)
            ax.fill_between(x, y1, y2, where=(y1 <= y2), interpolate=True, color='lightgreen', alpha=0.3)

            # Calculate and store the highlighted area for the column
            highlighted_area = np.sum(np.maximum(y1 - y2, 0) * np.diff(x)[0])
            highlighted_areas[column] = highlighted_area

            # Calculate and store the KL divergence for the column
            kl_divergence = entropy(y1, y2) # REF https://www.kaggle.com/code/nhan1212/some-statistical-distances
            kl_divergences[column] = kl_divergence

            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_xlim(0, 1)
            ax.legend()

        plt.tight_layout()
        #plt.show()

        total_highlighted_area = np.sum(list(highlighted_areas.values()))
        total_kl_divergence = np.sum(list(kl_divergences.values()))

        # for column, area in highlighted_areas.items():
        #     divergence = kl_divergences[column]
        #     print(f"{column}: {area:.2f}, {divergence:.6f}")

        # print(f"Total highlighted area: {total_highlighted_area:.2f}")
        # print(f"Average KL divergence: {total_kl_divergence / num_columns:.6f}")    

        return plt, total_highlighted_area, total_kl_divergence / num_columns    