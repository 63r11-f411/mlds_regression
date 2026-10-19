import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from feature_engineering import process_ordinal_and_missing_features

# скошенные вправо площади: log1p приближает распределение ненулевой части к симметричному
LOG_COLUMNS = ['LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'OpenPorchSF',
               'LotFrontage', 'MasVnrArea', 'WoodDeckSF', 'BsmtFinSF1']

# площади, где 0 означает отсутствие объекта
PRESENCE_FLAG_COLUMNS = ['2ndFlrSF', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1', 'TotalBsmtSF',
                         'PoolArea', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'ScreenPorch', 'BsmtFinSF2', 'EnclosedPorch']


def add_log_and_presence_flags(df):
    '''
    Добавляет флаги Has<колонка> (есть ли объект) и логарифмирует скошенные площади.
    Флаги считаются до логарифма, чтобы отделить нули от ненулевой части распределения.
    '''
    df = df.copy()
    for col in PRESENCE_FLAG_COLUMNS:
        df[f'Has{col}'] = (df[col] > 0).astype(int)

    df[LOG_COLUMNS] = np.log1p(df[LOG_COLUMNS])
    return df


def build_scaled_preprocessor():
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, make_column_selector(dtype_include=str)),
        ('num', num_pipeline, make_column_selector(dtype_include=np.number))
    ])

    pipeline = Pipeline([
        ('ordinal_and_missing', FunctionTransformer(process_ordinal_and_missing_features)),
        ('log_and_flags', FunctionTransformer(add_log_and_presence_flags)),
        ('preprocessor', preprocessor)
    ])
    return pipeline