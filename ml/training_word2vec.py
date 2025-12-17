#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib


# In[2]:


train = pd.read_csv(r"C:\Users\ruksh\Desktop\Full projects\Drug review\data\drugsComTrain_raw.csv")
test = pd.read_csv(r"C:\Users\ruksh\Desktop\Full projects\Drug review\data\drugsComTest_raw.csv")

X_train=train["review"]
y_train=train["rating"]
X_test=test["review"]
y_test=test["rating"]

MODEL_PATH = r"C:\Users\ruksh\Desktop\Full projects\Drug review\models\word2vec.joblib"


# In[3]:


def tokenizer (X):
    X_tokens=(X.astype(str)
                .str.lower()
                .str.replace(r'[/\"]',"",regex=True)
                .str.split()
             )
    sentenses = X_tokens.tolist()
    
    return sentenses


# In[4]:


def pipeline(X_train, X_test, y_train, y_test):

    best_score=0
    
    w2v = Word2Vec(
        sentences=tokenizer(X_train),
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1
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

    def doc_vector(tokens, model, vector_size=100):
        vecs = [model.wv[word] for word in tokens if word in model.wv]
        if not vecs:
            return np.zeros(vector_size)
        return np.mean(vecs, axis=0)

    X_train_tokens = tokenizer(X_train)  # same tokenizer
    X_test_tokens = tokenizer(X_test)

    X_train_vecs = np.vstack([doc_vector(tokens, w2v, 100) for tokens in X_train_tokens])
    X_test_vecs  = np.vstack([doc_vector(tokens, w2v, 100) for tokens in X_test_tokens])
    
    for name, model in models.items():
        
        model.fit(X_train_vecs, y_train)
        
        y_pred = model.predict(X_test_vecs)

        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}" )

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    print(f"Best model - {best_name} with R2 - {best_score:.3f}")

    joblib.dump(best_model, MODEL_PATH)


# In[5]:


if __name__=="__main__":

    pipeline(X_train, X_test, y_train, y_test)

    

