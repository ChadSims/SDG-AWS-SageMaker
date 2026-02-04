import miceforest as mf
import pandas as pd

from lib.data_transformer import DataTransformer


def impute(df: pd.DataFrame, metadata: dict, num_iter=10, random_state=42) -> pd.DataFrame:

    cat_features = metadata['cat_features']
    label = metadata['target_column']
    task = metadata['task']

    if cat_features or task == "classification":
        
        dt = DataTransformer()

        dt.fit(df, cat_features, label, task)

        transformed_df = dt.transform(df)

        imputation_kernel = mf.ImputationKernel(
            data=transformed_df,
            save_all_iterations_data=False,
            copy_data=True,
            random_state=random_state
        )
        imputation_kernel.mice(num_iter)

        imputed_df = imputation_kernel.complete_data()

        inverse_imputed_df = dt.inverse_transform(imputed_df)

        return inverse_imputed_df
    
    else: # regression
        imputation_kernel = mf.ImputationKernel(
            data=df,
            save_all_iterations_data=False,
            copy_data=True,
            random_state=random_state
        )
        imputation_kernel.mice(num_iter)

        imputed_df = imputation_kernel.complete_data()

        return imputed_df