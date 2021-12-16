import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
from sklearn.impute import SimpleImputer



import pandas as pd
import numpy as np

from Preprocessing.b_data_profile import *
from Preprocessing.a_load_file import read_dataset


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    """

    # Method 1 - Using quantiles (25-75) we can remove the outlier
    try:
        if column in get_numeric_columns(df):
            quartile_1 = df[column].quantile(0.25)
            quartile_3 = df[column].quantile(0.75)
            IQR = quartile_3 - quartile_1
            lower_cutoff = quartile_1 - 1.5 * IQR
            upper_cutoff = quartile_3 + 1.5 * IQR
            df2 = df[(df[column] > lower_cutoff) & (df[column] < upper_cutoff)]

        # Method 2 - using z-scores we can remove outlier
            outliers = []
            max_dev = 3
            mean = np.mean(df[column])
            std = np.std(df[column])
            for i in df[column]:
                Z_score = (i - mean) / std
                if np.abs(Z_score) > max_dev:
                    outliers.append(i)

            df3 = df[~df[column].isin(set(outliers))]

        # # we don;t want much data to be removed so, we run both the methods and use hte one removes the leasyt no of outliers
        if(df2.shape[0] > df3.shape[0]):
            return  df2
        else:
            return df3

        print("Shape after fixing outliers in the column using inter quartile range  {}: {}", {column, df2.shape})
        print("Shape after fixing outliers in the column using z scores {}: {}", {column, df3.shape})

    except:
        print("Cannot process non-numeric column {}", {column})
        pass




def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    """

    # Checking for % of Null values in each column

    if column in df.columns:
        imp_col =[column]
        num_cols = get_numeric_columns(df)
        cat_cols = get_text_categorical_columns(df)
        bin_cols = get_binary_columns(df)

        # some columns can be both binary and categorical in that case we consider them to be binary only
        if(column in get_text_categorical_columns(df) and column in (get_binary_columns(df))):
            cat_cols.remove(column)

        # some columns can be both binary and numerical in that case we consider them to be binary only
        if (column in get_numeric_columns(df) and column in (get_binary_columns(df))):
            num_cols.remove(column)

        # print("cat_cols", cat_cols)
        # print("bin_cols", bin_cols)
        # print("num_cols", num_cols)
        # using SimpleImputer we impute the missing values as mean in numeric, mode in categorical and binary,
        # we can also use knn but it is computaionaly heavy
        if column in num_cols:

            imp = SimpleImputer(strategy='mean')
            imp.fit(df[imp_col])
            df[imp_col] = imp.transform(df[imp_col])

        elif column in cat_cols:
            # df[column] = df[column].astype("category")
            imp = SimpleImputer(strategy='most_frequent')
            imp.fit(df[imp_col])
            df[imp_col] = imp.transform(df[imp_col])
            # print(df[imp_col])

        elif column in bin_cols:
            df[imp_col] = (df[imp_col] * 1).astype('Int64')
            # imp_col = [column]
            imp = SimpleImputer(strategy='most_frequent')
            imp.fit(df[imp_col])
            df[imp_col] = imp.transform(df[imp_col])
            # print(df[imp_col])

        else:
        # if (df[column].dtype == 'datetime'):
            pass

        return df

def normalize_column(df_column: pd.Series) -> pd.Series:
    min_ = df_column.min()
    max_ = df_column.max()
    return (df_column - min_) / (max_ - min_)


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    """

    # Normalizeing using standization method which converts -1 to 1 with mean 0

    stand_df_column = 2*(df_column - min(df_column)) / (max(df_column) - min(df_column)) - 1
    return stand_df_column

