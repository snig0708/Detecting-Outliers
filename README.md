# Detecting-Outliers
Checking for anomalies in the viewership data through measured methods like clustering followed by one-class SVM.

## Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 9)


# In[102]:


## Importing data 

data = pd.read_csv("C:\\Users\\Snigdha.Cheekoty\\Downloads\\daily_revenue.csv")


# In[103]:


type(data)

### Checking the dattype of the input: Pandas dataframe


# In[104]:


data.head(20)
## checking the first 20 records


# In[110]:


## Importing subtted data for EDA
data2 = pd.read_csv("C:\\Users\\Snigdha.Cheekoty\\OneDrive - Serco\\Desktop\\monthlydata.csv")


# In[114]:


data2
## Time-based(Monthly) data for revenue and pageviews


# In[125]:


sns.jointplot(x = "Pageviews", y = "Revenue", data  = data2 , kind = "reg")


# In[ ]:





# In[ ]:





# In[64]:


## importing library for kmeans clustering
from sklearn.cluster import KMeans


# In[65]:


# Obtaining the values and Plotting them
f1 = data['revenue'].values
f2 = data['pageviews'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)


# In[66]:


# Assigning the Number of clusters 
## Choosing the k value after considering an elbow plot
kmeans = KMeans(n_clusters=4)
# Fitting the input data
kmeans = kmeans.fit(X)
# Obtaining the cluster labels
labels = kmeans.predict(X)
# Obtaining the Centroid values
centroids = kmeans.cluster_centers_
print(centroids)


# In[18]:


# Plotting the clusters
plt.scatter(data['revenue'], data['pageviews'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)


# In[107]:


## Importing the data for the black box method: one classsvm ... 
## This data contains the computed "cpm" 
## after manual computation of CPM as per the report instructions, I have obtained the CPM values, shown below
data = pd.read_csv("C:\\Users\\Snigdha.Cheekoty\\OneDrive - Serco\Desktop\\daily_revenue123.csv")


# In[99]:


data.head(20)


# In[141]:


fig, ax = plt.subplots() # creating a figure
fig.set_size_inches(15,35)
sns.stripplot(data = data, y = "site", x = "pageviews")
#### Different sites and the correspondin pageviews (on a daily basis)
## You can see few abnormally high values pertaining to sesonality factors


# In[ ]:





# In[86]:


## Importing the library for svm
from sklearn import svm


# In[89]:


# Preparing the data 
## Instead of using sampling, I have manually partioned the data into training and test sets
## The trainset contains the records prior to 04/01/2017
## and the test set contains records after  04/01/2017
X = data[["revenue", "cpm"]]
train_feature = X.loc[0:4932, :]
train_feature = train_feature.drop('cpm', 1)
Y_1 = X.loc[4932:, "cpm"]
Y_2 = X['cpm']


# In[90]:


# Creating test observations and features

X_test_1 = X.loc[4932:, :].drop('cpm',1)

X_test = X_test_1.append(X_test_2)


# In[92]:


# Setting the hyperparameters for oneclass SVM
#Y_test is used to evaluate the model
oneclass = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
# Used various combination of hyperparameters like linear, rbf, poly, gamma-
Y_1 = X.loc[4320:, 'cpm']
Y_2 = X['cpm']
Y_test= Y_1.append(Y_2)


# In[93]:


#training the model
oneclass.fit(train_feature)


# In[94]:


# Testing the model on the validation set

fraud_pred = oneclass.predict(X_test)


# In[95]:


# Check the number of outliers predicted by the algorithm

unique, counts = np.unique(fraud_pred, return_counts=True)
print (np.asarray((unique, counts)).T)


# In[96]:


#Convert Y-test and fraud_pred to dataframe for ease of operation

Y_test= Y_test.to_frame()
Y_test=Y_test.reset_index()
fraud_pred = pd.DataFrame(fraud_pred)
fraud_pred= fraud_pred.rename(columns={0: 'prediction'})


# In[97]:


fraud_pred[fraud_pred['prediction']==1]=0
fraud_pred[fraud_pred['prediction']==-1]=1


# In[98]:


print(fraud_pred['prediction'].value_counts())
print(sum(fraud_pred['prediction'])/fraud_pred['prediction'].shape[0])


# In[ ]:





# In[ ]:

