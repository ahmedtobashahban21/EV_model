# importing 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
#### spliting data
from sklearn.model_selection import train_test_split 
#### accuracy scoring 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
import warnings 
warnings.filterwarnings('ignore')
# eval machine learning algoritm
!python3 -m pip install -q evalml==0.28.0  
####################################
!pip install fast_ml
from evalml.automl import AutoMLSearch



## use algoritm


# fiting algoritms
EV_model = AutoMLSearch(X_train=X_train , y_train=y_train ,problem_type='regression' ,max_time=600)
EV_model.search()


EV_model.rankings

EV_model.best_pipeline


!pip install pandas -U
import pandas as pd

test_predict =EV_model.best_pipeline.predict(X_test) 
sample['Indoor_temperature_room'] =test_predict 
sample.to_csv('submission.csv' , index=False) 
sample.head()