from help_func import preproces
from catboost import CatBoostClassifier
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

data = pd.read_parquet('submit_data/test.parquet')
submit = pd.read_csv('submit_data/sample_submission.csv', index_col='id')
data = preproces(data)
classifeir = CatBoostClassifier()
classifeir.load_model('catboost_model.cbm')
submit['score'] = classifeir.predict_proba(data)[:, 1]
submit.to_csv('submission.csv')
print('Финиш')
