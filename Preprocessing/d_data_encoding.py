
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Preprocessing.b_data_profile import *
from Preprocessing.c_data_cleaning import *
from Preprocessing.a_load_file import read_dataset


def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    """
    This method should generate a (sklearn version of a) label encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """

    #generate and fit the encoder
    label_encoder = LabelEncoder()
    return label_encoder.fit(df_column)

def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    """
    This method should generate a (sklearn version of a) one hot encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    # Fitting one hot  encoder
    df_column = df_column.astype("category")
    # print(df_column.dtype)
    ohe = OneHotEncoder()
    ohe.fit(pd.DataFrame(df_column))
    # oh_encoder.transform(df_column)

    return ohe

def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should replace the column of df with the label encoder's version of the column
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to replace the column
    :return: The df with the column replaced with the one from label encoder
    """

    #transform the encoder in order to replace with the previous column
    df1 = df.copy() #making a copy of df and returning the new updated one in order not to change df because it will be called in main function
    df1[column] = le.transform(df1[column])
    return df1


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    """
    This method should replace the column of df with all the columns generated from the one hot's version of the encoder
    Feel free to do it manually or through a sklearn ColumnTransformer
    :param df: Dataset
    :param column: column to be replaced
    :param ohe: the one hot encoder to be used to replace the column
    :param ohe_column_names: the names to be used as the one hot encoded's column names
    :return: The df with the column replaced with the one from label encoder
    """

    # transforming one hot encoder
    ohe_df  = ohe.transform(pd.DataFrame(df[column])).toarray()
    new_df = pd.concat([df,pd.DataFrame(ohe_df,columns=[ohe_column_names])],axis=1)
    new_df.drop(column,axis=1,inplace = True)
    return new_df


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_label_encoder
    The column of df should be from the label encoder, and you should use the le to revert the column to the previous state
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to revert the column
    :return: The df with the column reverted from label encoder
    """
    # inverse tranform label encoder
    col  = [column]
    df[col] = le.inverse_transform(df[col])
    return df


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_one_hot_encoder
    The columns (one of the method's arguments) are the columns of df that were placed there by the OneHotEncoder.
    You should use the ohe to revert these columns to the previous state (single column) which was present previously
    :param df: Dataset
    :param columns: the one hot encoded columns to be replaced
    :param ohe: the one hot encoder to be used to revert the columns
    :param original_column_name: the original column name which was used before being replaced with the one hot encoded version of it
    :return: The df with the columns reverted from the one hot encoder
    """
    # inverse tranform oh encoder
    df[original_column_name] = pd.DataFrame(ohe.inverse_transform(pd.DataFrame(df[[columns]])),columns= original_column_name)
    return df

