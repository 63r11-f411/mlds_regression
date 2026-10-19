import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

QUALITY = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}

QUALITY_COLUMNS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

ORDINAL_COLUMNS_OTHER = {"BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
"BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
"BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
"GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1},
"Functional": {"Typ": 8, "Min1": 7, "Min2": 6, "Mod": 5, "Maj1": 4, "Maj2": 3, "Sev": 2, "Sal": 1},
"PavedDrive": {"Y": 3, "P": 2, "N": 1}
}

OBJECT_INDICATORS = {
    'BsmtQual': ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual'],
    'GarageType': ['GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'GarageFinish'],
    'MasVnrType': ['MasVnrArea'],
    'Fireplaces': ['FireplaceQu'],
    'PoolArea': ['PoolQC']
}


def _add_ordinal_features(df):
    '''
    Преобразует качественные признаки в порядковые.
    '''
    df = df.copy()
    for col in QUALITY_COLUMNS:
        df[col] = df[col].map(QUALITY)
    
    for col in ORDINAL_COLUMNS_OTHER:
        df[col] = df[col].map(ORDINAL_COLUMNS_OTHER[col])
    
    return df


def _nullify_missing_objects(df):
    '''
    Явно приводит к нулю признаки отсутствующих объектов.
    '''
    df = df.copy()
    
    for col in OBJECT_INDICATORS:
        mask = df[col].isna() | df[col].eq(0)
        df.loc[mask, OBJECT_INDICATORS[col]] = df.loc[mask, OBJECT_INDICATORS[col]].fillna(0)

    return df


def process_ordinal_and_missing_features(df):
    df = _add_ordinal_features(df)
    df = _nullify_missing_objects(df)
    return df