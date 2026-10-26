QUALITY = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}

QUALITY_COLUMNS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]

ORDINAL_COLUMNS_OTHER = {"BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
"BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
"BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
"GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1},
"Functional": {"Typ": 8, "Min1": 7, "Min2": 6, "Mod": 5, "Maj1": 4, "Maj2": 3, "Sev": 2, "Sal": 1},
"PavedDrive": {"Y": 3, "P": 2, "N": 1},
"LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
"LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1}
}

DROP_COLUMNS = ['PoolQC', 'Utilities', 'GarageYrBlt'] # 7 бассейнов, почти константа, и обычно совпадает с YearBuilt

ZERO_IF_MISSING_COLUMNS = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                           'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']


def engineer_features(df):
    '''
    Преобразования признакой:
        drop константных/неинформативных колонок
        заполнение пропусков в Functional
        преобразование ordinal колонок
        NaN в площадаях и счётчиках означает отсутствие значения - заполняем нулями
        приведение MSSubclass к строкам
    '''
    df = df.copy()

    df = df.drop(DROP_COLUMNS, axis=1)

    df['Functional'] = df['Functional'].fillna('Typ')

    for col in QUALITY_COLUMNS:
        df[col] = df[col].map(QUALITY)
    
    for col in ORDINAL_COLUMNS_OTHER:
        df[col] = df[col].map(ORDINAL_COLUMNS_OTHER[col])

    df[QUALITY_COLUMNS] = df[QUALITY_COLUMNS].fillna(0)
    df[list(ORDINAL_COLUMNS_OTHER.keys())] = df[list(ORDINAL_COLUMNS_OTHER.keys())].fillna(0)

    for col in ZERO_IF_MISSING_COLUMNS:
        df[col] = df[col].fillna(0)

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    return df