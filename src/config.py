import os

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_PATH=os.path.join(BASE_DIR,'data','Data.csv')
STOCK_PATH=os.path.join(BASE_DIR,'data','StockPrice.csv')

# Model paths
MODELS_DIR=os.path.join(BASE_DIR,'models')
MODEL_FILE=os.path.join(MODELS_DIR,'FF_Model.pkl')

# Create a new directory if it does not already exist
os.makedirs(MODELS_DIR,exist_ok=True)