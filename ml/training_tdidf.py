#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib


# In[2]:


train = pd.read_csv(r"C:\Users\ruksh\Desktop\Full projects\Drug review\data\drugsComTrain_raw.csv")
test = pd.read_csv(r"C:\Users\ruksh\Desktop\Full projects\Drug review\data\drugsComTest_raw.csv")


# In[3]:


X_train=train["review"]
y_train=train["rating"]
X_test=test["review"]
y_test=test["rating"]

MODEL_PATH = r"C:\Users\ruksh\Desktop\Full projects\Drug review\models\td_idf.joblib"


# In[4]:


def pipeline_func (X_train, y_train, X_test, y_test):

    best_score=0
    
    td_idf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1,3)
    )
    
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.001),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
        "XGboost" : xgb.XGBRegressor(
                                        n_estimators=300,
                                        learning_rate=0.05,
                                        max_depth=6,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        objective="reg:squarederror",
                                        n_jobs=-1,
                                        random_state=42,
                                        tree_method="hist",   # good default
                                    )
    }

    for name, model in models.items():
        
        pipeline = Pipeline([
            ("TD-IDF",td_idf),
            ("regressor",model)
        ])
    
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    
        print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

        # saving best model based on R2 score

        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name  = name

    print(f"Best model - {best_name} with R2 of {best_score}")

    joblib.dump(best_model, MODEL_PATH)
    
 


# In[5]:


if __name__ =="__main__":

    pipeline_func (X_train, y_train, X_test, y_test)

