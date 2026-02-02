import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from numpy.typing import ArrayLike

def gower_matrix(X:pd.DataFrame, *, nominal_cols:List[str], ordinal_cols:Dict[str, List[str]], numerical_cols:List[str], as_frame:bool=False) -> np.ndarray:
    """_summary_

    Args:
        X (pd.DataFrame): Data for which to calculate the Gower distance
        nominal_cols (List[str]): A list with the column names of the nominal features in the dataset.
        ordinal_cols (Dict[str, List[str]]): A dictionary whose keys are the names of the columns of ordinal data in the dataset.
                                             The values are the sorted list of possible categories.
        numerical_cols (List[str]): A list containing the column names of the numerical features in the dataset
        as_frame (bool, optional): Whether to return the distance matrix as a DataFrame. Defaults to False.

    Raises:
        TypeError: At the moment only DataFrame are supported as input data
        ValueError: If a numerical column has zero variance
        ValueError: If an ordinal column has only one category

    Returns:
        np.ndarray: _description_
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Currently only apandas.DataFrame is supported. Data is of type: {type(X)}")

    n_samples, _ = X.shape

    # total distance
    total_similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    # total weight matrix
    total_weight_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    X_encoded = X.copy()

    for column in X.columns:
        # Get the column to handle it individually
        column_data = X_encoded.loc[:, column].values
        # Calculate the weight matrix
        weight_matrix = get_weight_matrix_from(column_data)
        # For numeric data types
        if column in numerical_cols:
            # calculate the range
            feature_range = np.nanmax(column_data) - np.nanmin(column_data)
            if feature_range == 0:
                raise ValueError(f"Feature {column} has constant variance. Handle explicitly before passing it to gower_matrix")
            # get the data from the column and create a column vector to use broadcasting
            column_vector = column_data[:, np.newaxis]
            pairwise_difference_matrix = np.abs(column_data - column_vector) # This is a n_samples, n_samples matrix that contains the pairwise differences
            
            # calculate the similarity
            numeric_similarity_matrix = np.zeros_like(pairwise_difference_matrix)
            np.divide(
                pairwise_difference_matrix,
                feature_range,
                out=numeric_similarity_matrix,
                where=weight_matrix > 0
            )
            numeric_similarity_matrix = 1 - numeric_similarity_matrix
    
            numeric_similarity_matrix *= weight_matrix
            
            total_similarity_matrix += numeric_similarity_matrix
            total_weight_matrix += weight_matrix
        
        elif column in nominal_cols:
            column_vector = column_data[:, np.newaxis]
            nominal_similarity_matrix = (column_vector == column_data).astype(np.float64)
            nominal_similarity_matrix *= weight_matrix
            
            total_similarity_matrix += nominal_similarity_matrix
            total_weight_matrix += weight_matrix
        
        elif column in ordinal_cols.keys():
            # calculate the rank
            rank_map = {category : idx for idx, category in enumerate(ordinal_cols[column])}
            column_data_encoded = np.array([rank_map.get(category, np.nan) for category in column_data])

            ordinal_range = np.nanmax(column_data_encoded) - np.nanmin(column_data_encoded)
            if ordinal_range == 0:
                raise ValueError(f"Feature {column} has only one unique value. Handle explicitly before passing it to gower_matrix")
            column_vector = column_data_encoded[:, np.newaxis]
            pairwise_difference_matrix = np.abs(column_data_encoded - column_vector)
            # calculate similarity
            ordinal_similarity_matrix = np.zeros_like(pairwise_difference_matrix)
            np.divide(
                pairwise_difference_matrix,
                ordinal_range,
                out=ordinal_similarity_matrix,
                where=weight_matrix > 0
            )
            ordinal_similarity_matrix = 1 - ordinal_similarity_matrix
            ordinal_similarity_matrix *= weight_matrix

            total_similarity_matrix += ordinal_similarity_matrix
            total_weight_matrix += weight_matrix
        
        else:
            raise ValueError(f"Column {column} not assigned to any data type.")

    # total similarity score
    S = np.zeros_like(total_similarity_matrix)
    np.divide(
        total_similarity_matrix,
        total_weight_matrix,
        out=S,
        where=total_weight_matrix!=0
    )
    # Return the distance matrix
    if as_frame:
        return pd.DataFrame(1 - S)
    return 1 - S
    
def get_weight_matrix_from(column_data:np.ndarray) -> np.ndarray:
    mask = ~pd.isna(column_data)
    mask_as_column_vector = mask[:, np.newaxis]
    return np.logical_and(mask_as_column_vector, mask).astype(np.float64)

if __name__ == "__main__":
    
    Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,],
                     'gender':['M','M','N','M','F','F','F','F'],
                     'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED'],
                     'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0],
                     'has_children':[1,0,1,1,1,0,0,1],
                     'available_credit':[2200,100,22000,1100,2000,100,6000,2200]})
    
    is_result = np.array(
            [[0.        , 0.3590238 , 0.6707398 , 0.31787416, 0.16872811, 0.52622986, 0.59697855, 0.47778758],
             [0.3590238 , 0.        , 0.6964303 , 0.3138769 , 0.523629  , 0.16720603, 0.45600235, 0.65396350],
             [0.6707398 , 0.6964303 , 0.        , 0.6552807 , 0.6728013 , 0.69696970, 0.74042800, 0.81519410],
             [0.31787416, 0.3138769 , 0.6552807 , 0.        , 0.4824794 , 0.48108295, 0.74818605, 0.34332284],
             [0.16872811, 0.523629  , 0.6728013 , 0.4824794 , 0.        , 0.35750175, 0.43237334, 0.31210360],
             [0.52622986, 0.16720603, 0.6969697 , 0.48108295, 0.35750175, 0.        , 0.28987510, 0.48783620],
             [0.59697855, 0.45600235, 0.740428  , 0.74818605, 0.43237334, 0.2898751 , 0.        , 0.57476616],
             [0.47778758, 0.6539635 , 0.8151941 , 0.34332284, 0.3121036 , 0.4878362 , 0.57476616, 0.        ]], dtype=np.float64)
    