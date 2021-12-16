from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from Preprocessing.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    # Creates visualization report with all the basic functions and generates a html file
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    # Get the maximum value of a numeric column
    col = column_name
    try:
        if (df[col].dtypes == "float" or df[col].dtypes == "int"):
            print("The maximum value of the column is : ", round(float(max(df[col]))),2)
        return float(max(df[col]))

    except:
        print("Non numeric column, hence cannot calculate max")
        pass


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    # Get the minimum value of a numeric column
    col = column_name
    try:
        if (df[col].dtypes == "float" or df[col].dtypes == "int"):
            print("The minimum value of the column is : ", round(float(min(df[col])),2))

    except:
        print("Non numeric column, hence cannot calculate max")
        pass

    return float(min(df[col]))



def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    # Get the mean value of a numeric column
    col = column_name
    try:
        if (df[col].dtypes == "float" or df[col].dtypes == "int"):
            print("The mean value of the column is : ", round(float(np.mean(df[col])),2))

    except:
        print("Non numeric column, hence cannot calculate max")
        pass

    return float(np.mean(df[col]))


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    # Get the null values present in the column
    """
    This is also known as the number of 'missing values'
    """
    col = column_name
    try:
        # if (col in df.columns):
        print("The count of null values in the column is : ", float(df[col].isnull().sum().sum()))

    except:
        print("Cannot calculate the null values")
        pass

    return float(df[col].isnull().sum().sum())




def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    # Get the count of duplicate values present in the dataframe column
    col = column_name
    try:
        if (df[col].dtypes == "float" or df[col].dtypes == "int"):
            print("The count of duplicate values in the column is : ", float(df[col].duplicated().sum()))

    except:
        print("Non numeric column, hence cannot calculate the no of duplicate values")
        pass

    return float(df[col].duplicated().sum())


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
# Gets all the numeric columns in the dataset (be it int or float) and stores it into list
    num_cols = []
    for i in df.columns:
        if (df[i].dtypes == "float" or  df[i].dtypes == "int"):
            num_cols.append(i)

    # print("Numeric columns : ", num_cols)

    return num_cols


def get_binary_columns(df: pd.DataFrame) -> List[str]:
# Get the binary columns in the dataset (usaullay binary column contains 2 values(levels)
# so the col is object (non numeric) with level 2 we can consider that to binary - value present or not

    binary_cols = []
    for i in df.columns:
        if (df[i].dtype == "object"):
            if (df[i].nunique() == 2):
                binary_cols.append(i)

    # print("Binary columns : ", binary_cols)

    return binary_cols


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
# Gets all the categorical data from the dataset
    cat_cols = []
    for i in df.columns:
        if (df[i].dtype == "object" ):
            cat_cols.append(i)

    # print("Categorical columns : ", cat_cols)

    return cat_cols


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    # Calcualtes pearson correlation between two columns using corr function (by default - corr uses pearson correlation)
    """
    Calculate and return the pearson correlation between two columns
    """
    pcorr = df[col1].corr(df[col2])
    print("The Pearson correlation between the two columns is : ", pcorr)
    return pcorr


if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))
    a = pandas_profile(df)
    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
