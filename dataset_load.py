import kagglehub

from config import DATA_DIR

path = kagglehub.competition_download("house-prices-advanced-regression-techniques", output_dir=str(DATA_DIR), force_download=True)
print("Path to competition files:", path)
