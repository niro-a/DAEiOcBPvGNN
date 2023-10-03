def preprocess_dataframe(df, first_cols, split):
    """
    A function to reorder the columns of the dataframe, fill NaN values, perform one-hot encoding, and split columns.

    Args:
    df (pd.DataFrame): The dataframe to preprocess.
    first_cols (list): The columns that should be moved to the start of the dataframe.
    split (list): The columns that should be split.

    Returns:
    df (pd.DataFrame): The preprocessed dataframe.
    """

    import pandas as pd
    import numpy as np

    # Process the 'split' columns
    for col in split:
        temp = df[col]
        # Check if the data is not null and is of string type before attempting to split
        temp = temp[temp.notna() & temp.apply(lambda x: isinstance(x, str))].apply(lambda x: x[1:-1].split(', '))
        temp = pd.get_dummies(temp.apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)
        temp.columns = temp.columns.str.replace("'", "")
        temp.columns = temp.columns.str.replace(" ", "_")
        # Prepend original column name to new columns
        temp.columns = [f"{col}_{c}" for c in temp.columns]
        df = df.join(temp)
        df = df.drop(columns=[col])

    # Fill NaNs in the newly created columns with 0
    new_cols = [col for col in df.columns if any(s in col for s in split)]  # New columns contain original column name
    df[new_cols] = df[new_cols].fillna(0)

    # Reorder the columns such the first ones are the 'first_cols'
    df = df[first_cols + [col for col in df.columns if col not in first_cols]]

    # Find numerical columns with NaNs to fill with zeros, excluding 'first_cols'
    cols_na = df.select_dtypes(include=[np.number]).columns.difference(first_cols)
    df[cols_na] = df[cols_na].fillna(0)

    # Find categorical columns to perform one-hot encoding, excluding 'first_cols'
    categorical_cols = df.select_dtypes(include=[object]).columns.difference(first_cols).tolist()

    # Separate columns with NaNs and without NaNs
    one_hot_encode_na_cols = [col for col in categorical_cols if df[col].isna().any()]
    one_hot_encode_cols = [col for col in categorical_cols if col not in one_hot_encode_na_cols]

    # One-hot encoding categorical variables columns with NaNs
    df = pd.get_dummies(df, columns=one_hot_encode_na_cols, dummy_na=True, drop_first=True)

    # Impute NaNs with 0s categorical variables, columns without NaNs
    df = pd.get_dummies(df, columns=one_hot_encode_cols)

    # Adjust columns name
    df.columns = df.columns.str.replace(":", "_")

    return df