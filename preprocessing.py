import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder
from feature_engineering import engineer_features

# скошенные вправо площади: log1p приближает распределение ненулевой части к симметричному
LOG_COLUMNS = ['LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'OpenPorchSF',
               'LotFrontage', 'MasVnrArea', 'WoodDeckSF', 'BsmtFinSF1']

# признаки, где 0 означает отсутствие объекта
PRESENCE_FLAG_COLUMNS = ['2ndFlrSF', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1', 'TotalBsmtSF',
                         'PoolArea', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'ScreenPorch', 'BsmtFinSF2', 'EnclosedPorch', 'GarageArea', 'Fireplaces']


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
        ('fe', FunctionTransformer(engineer_features)),
        ('log_and_flags', FunctionTransformer(add_log_and_presence_flags)),
        ('preprocessor', preprocessor)
    ])
    return pipeline


def build_unscaled_preprocessor():
    '''
    Препроцессор для деревьев: без масштабирования и логарифма (монотонные преобразования не меняют разбиения),
    категории кодируются порядково вместо OHE, пропуски заполняются значением вне диапазона признака.
    '''
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # -1 вне диапазона всех числовых признаков, поэтому дерево может отделить пропуски одним разбиением
    num_imputer = SimpleImputer(strategy='constant', fill_value=-1)

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, make_column_selector(dtype_include=str)),
        ('num', num_imputer, make_column_selector(dtype_include=np.number))
    ])

    pipeline = Pipeline([
        ('fe', FunctionTransformer(engineer_features)),
        ('preprocessor', preprocessor)
    ])
    return pipeline

