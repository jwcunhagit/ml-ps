
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


# In[2]:


train_ds = pd.read_csv('dataset/train.csv')
test_ds = pd.read_csv('dataset/test.csv')


# In[3]:


train_ds.head()


# In[ ]:


X_train = train_ds.iloc[:, 2:58].values
y_train = train_ds.iloc[:, 1].values


# In[73]:


test_ds.head()


# In[86]:


X_test = test_ds.iloc[:, 1:57].values


# In[55]:


#test_ds.describe()


# In[24]:


#test_ds.info()


# In[61]:


#train_ds.columns


# In[57]:


#categorizing which filds should be as category
categorical_features = [a for a in train_ds.columns if a.endswith('cat')]
#categorical_features


# In[58]:


model_vars = [a for a in train_ds.columns if 'id' not in a and 'target' not in a]
#model_vars


# In[ ]:


# https://www.kaggle.com/eoakley/porto-seguro-lightgbm


# In[59]:


model = lgb.LGBMClassifier(n_estimators=150)


# In[68]:


cross_val_score(model, train_ds[model_vars], train_ds.target, cv=5, fit_params=dict(categorical_feature=categorical_features))


# In[38]:


model.fit(train_ds[model_vars], train_ds.target, categorical_feature=categorical_features)


# In[40]:


preds = model.predict_proba(test_ds[model_vars])[:,1]


# In[42]:


sub = pd.DataFrame({'id': test_ds.id, 'target': preds})
sub.head()


# In[43]:


sub.to_csv('output/output_lgb_2.csv', index=False, header=True)


# In[ ]:





# In[29]:


# Xgboost


# In[69]:


from xgboost import XGBClassifier
classifier = XGBClassifier()


# In[84]:


X_train.shape


# In[83]:


y_train.shape


# In[87]:


X_test.shape


# In[ ]:


classifier.fit(train_ds[model_vars], train_ds.target)


# In[ ]:


y_pred = classifier.predict(test_ds[model_vars])
y_pred.shape


# In[ ]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,  fit_params=dict(categorical_feature=categorical_features))
accuracies.mean()
accuracies.std()


# In[ ]:


sub = pd.DataFrame({'id': test_ds.id, 'target': y_pred})
sub.head()


# In[ ]:


sub.to_csv('output/output_xgb_2.csv', index=False, header=True)


# In[ ]:





# In[ ]:


# Data Exploration


# In[28]:


from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


j = sns.jointplot(data=train_ds, x='target', y='ps_calc_20_bin')

