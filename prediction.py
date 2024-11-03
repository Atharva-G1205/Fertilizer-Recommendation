# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("D:\FE\First Year Project\PBL3latest"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
data=pd.read_csv("D:\FE\First Year Project\PBL3latest\PBL3\Fertilizer Prediction.csv")
data.head()

# %% [markdown]
# ## Dataset Informations

# %%
data.describe()

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Finding the length of the Dataset

# %%
data.shape

# %% [markdown]
# #### Finding the missing values

# %%
data.isnull().sum()

# %% [markdown]
# **No Missing values detected**

# %% [markdown]
# ### Visualizing the Dataset

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=data,x='Fertilizer Name')

# %% [markdown]
# ### Removing Categorical Variable from the Dataset which is soil type and crop type

# %%
data['Soil Type'].value_counts()

# %%
data['Crop Type'].value_counts()

# %% [markdown]
# ### As Soil Type and Crop Type are Categorical Variable mapping them to a numerical variable for good model accuracy

# %%
soil_dict={
    'Loamy':1,
    'Sandy':2,
    'Clayey':3,
    'Black':4,
    'Red':5
}

crop_dict={
    'Sugarcane':1,
    'Cotton':2,
    'Millets':3,
    'Paddy':4,
    'Pulses':5,
    'Wheat':6,
    'Tobacco':7,
    'Barley':8,
    'Oil seeds':9,
    'Ground Nuts':10,
    'Maize':11

}

# %%
data['Soil_Num']=data['Soil Type'].map(soil_dict)
data['Crop_Num']=data['Crop Type'].map(crop_dict)

# %%
data=data.drop(['Soil Type','Crop Type'],axis=1)
data.head()

# %% [markdown]
# #### Splitting the Dataset into X and Y

# %%
X=data.drop(['Fertilizer Name'],axis=1)
Y=data['Fertilizer Name']

# %% [markdown]
# ### Splitting Dataset into Train and Test for checking the Accuracy

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# #### Model Building

# %% [markdown]
# **Importing all the Classifier Algorithms**

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# create instances of all models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

for name,md in models.items():
    md.fit(X_train,Y_train)
    ypred=md.predict(X_test)

    print(f"the Accuracy of {name} is ",accuracy_score(Y_test,ypred))



# %% [markdown]
# ### Let's Take Decision Tree Classifier for our model building

# %%

classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
ypred=classifier.predict(X_test)
accuracy_score(Y_test,ypred)

# %%
def recommendation(Temperature,Humidity,Moisture,Nitrogen,Phosphorous,Potassium,Soil_Num,Crop_Num):
    features = np.array([[Temperature,Humidity,Moisture,Nitrogen,Phosphorous,Potassium,Soil_Num,Crop_Num]])
    prediction = classifier.predict(features).reshape(1,-1)

    return prediction[0]

# %%
Temperature=22
Humidity=59
Moisture=19
Nitrogen=12
Potassium=7
Phosphorous=10
Soil_Num=4
Crop_Num=8
predict=recommendation(Temperature,Humidity,Moisture,Nitrogen,Phosphorous,Potassium,Soil_Num,Crop_Num)
predict[0]

# %%
### Create a Pickle file using serialization
import pickle
pickle_out = open("Fertclassifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

# %%



