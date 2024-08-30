import pandas as pd

class Duplicates:
    def __init__(self, original_df, synthetic_df):
        self.original_df = original_df
        self.synthetic_df = synthetic_df
        self.combined_df = pd.concat([self.original_df, self.synthetic_df], axis=0)

        if self.original_df.shape[1] != self.synthetic_df.shape[1]:
            raise Exception("The number of columns in the original and synthetic datasets must match, but they are different ", self.original_df.shape[1], self.synthetic_df.shape[1])
        
        if self.original_df.shape[0] != self.synthetic_df.shape[0]:
            raise Exception("The number of rows in the original and synthetic datasets are different ", self.original_df.shape[0], self.synthetic_df.shape[0])
        
    def identify_duplicates(self):
        num_duplicates_original_df = self.original_df.duplicated().sum()
        num_duplicates_synthetic_df = self.synthetic_df.duplicated().sum()
        rows_present_in_original = self.synthetic_df.isin(self.original_df.to_dict(orient='list')).all(axis=1)
        num_rows_present = rows_present_in_original.sum()

        return num_duplicates_original_df, num_duplicates_synthetic_df, num_rows_present
        # print("Number of duplicates in original_df:", num_duplicates_original_df)
        # print("Number of duplicates in synthetic_df:", num_duplicates_synthetic_df)
        # print("Number of rows in synthetic_df present in original_df:", num_rows_present)    

    def calculate_similarity(self, threshold):
        #A row is said to be similar, if all the values of the different columns 
        # in that row are within threshold% deviation of the values in same columns for a different row. 
        def get_similar_df(df1, threshold, **kwargs):

            df2 = kwargs.get('df2', None)
            if df2 is None:
                df2 = df1.copy()

            similar_rows_count = 0
            #similar_rows = []
            for i in range(len(df1)):
                for j in range(i+1, len(df1)):
                    row1, row2 = df1.iloc[i], df2.iloc[j]

                    if (row1 == 0).any() or (row2 == 0).any() or (row1.isna().any()) or (row2.isna().any()):
                        continue

                    deviation = abs((row1 - row2) / row1)
                    if (deviation <= threshold).all():
                        #similar_rows.append((row1.values, row2.values))
                        similar_rows_count += 1             
            return similar_rows_count    

        original_df_similarity = get_similar_df(self.original_df, threshold)
        # print(f"Total number of similar rows in original_df: {original_df_similarity}")
        # print(f"Duplicate ratio in original_df: {original_df_similarity / self.original_df.shape[0]:.3f}")

        synthetic_df_similarity = get_similar_df(self.synthetic_df, threshold)
        # print(f"Total number of similar rows in synthetic_df: {synthetic_df_similarity}")
        # print(f"Duplicate ratio in synthetic_df: {synthetic_df_similarity / self.synthetic_df.shape[0]:.3f}")

        synthetic_original_similarity = get_similar_df(self.original_df, threshold, df2=self.synthetic_df)
        # print(f"Total number of similar synthetic_df rows in original_df: {synthetic_original_similarity}")
        # print(f": {synthetic_original_similarity / self.combined_df.shape[0]:.3f}")

        return [original_df_similarity, original_df_similarity / self.original_df.shape[0], 
                synthetic_original_similarity, synthetic_df_similarity / self.synthetic_df.shape[0],
                synthetic_original_similarity / self.combined_df.shape[0]]