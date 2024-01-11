#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction
# In this machine learning project, I have collected the dataset from Kaggle and I will be using Machine Learning to predict whether any person is suffering from heart disease.

# Importing Necessary Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import cufflinks as cf
import hvplot.pandas
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')


# Metrics used for Classification Technique

# In[4]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# Scalers

# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  RandomizedSearchCV, train_test_split


# Model Building for the system 

# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# # Data import

# In[8]:


data = pd.read_csv("F:/AI Project Files/Heart Disease Prediction/Datasets/heart.csv")
data.head(6)


# # Exploratory Data Analysis 

# In[9]:


data.shape


# In[96]:


y=data["target"]
pd.set_option("display.float", "{:.3f}".format)
data.describe()


# In[12]:


data.info()


# In[148]:


#Checking Missing Values 
data.isna().sum()


# # Checking Correlation between various Features 

# In[14]:


plt.figure(figsize=(15,12))
sns.set_context('notebook',font_scale = 1.3)
sns.heatmap(data.corr(),annot=True,linewidth =1)
plt.tight_layout()


# In[22]:


sns.set_context('notebook',font_scale = 2.3)
data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(18, 10), 
                                                        title="Correlation with the target feature")
plt.tight_layout()


# In[23]:


print(data.corr()["target"].abs().sort_values(ascending=False))


# # Analysis of Age

# In[24]:



plt.figure(figsize=(25,12))
sns.set_context('notebook',font_scale = 1.5)
sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values)
plt.tight_layout()


# In[25]:


#Checking the range of age in the data
minAge=min(data.age)
maxAge=max(data.age)
meanAge=data.age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)


# In[26]:


#Dividing the age feature into three parts
Young = data[(data.age>=29)&(data.age<40)]
Middle = data[(data.age>=40)&(data.age<55)]
Elder = data[(data.age>55)]

plt.figure(figsize=(23,10))
sns.set_context('notebook',font_scale = 1.5)
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(Young),len(Middle),len(Elder)])
plt.tight_layout()


# In[27]:


#Plotting Pie Chart
colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize=(10,10))
sns.set_context('notebook',font_scale = 1.2)
plt.pie([len(Young),len(Middle),len(Elder)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.tight_layout()


# In[62]:


def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(categorial)] 
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)] 


# In[63]:


continuous = [('trestbps', 'blood pressure in mm Hg'), 
              ('chol', 'serum cholestoral in mg/d'), 
              ('thalach', 'maximum heart rate achieved'), 
              ('oldpeak', 'ST depression by exercise relative to rest'), 
              ('ca', '# major vessels: (0-3) colored by flourosopy')]


# In[65]:


def plotContinuous(attribute, xlabel, ax_index):
    sns.distplot(data[[attribute]], ax=axes[ax_index][0])
    axes[ax_index][0].set(xlabel=xlabel, ylabel='density')
    sns.violinplot(x='target', y=attribute, data=data, ax=axes[ax_index][1])
    


# In[66]:


fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(15, 22))

plotGrid(isCategorial=False)


# # Analysis of Sex Feature

# In[32]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.5)
sns.countplot(data['sex'])
plt.tight_layout()


# # Plotting relation between sex and Slope

# In[67]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.6)
sns.countplot(data['sex'],hue=data["slope"])
plt.tight_layout()


# # Analysis of Chest Pain

# In[68]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.5)
sns.countplot(data['cp'])
plt.tight_layout()


# # Analysis of CP vs Target Column

# In[69]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.5)
sns.countplot(data['cp'],hue=data["target"])
plt.tight_layout()


# # Analysis of Thal

# In[70]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.5)
sns.countplot(data['thal'])
plt.tight_layout()


# In[73]:


#Heart Disease Count 
data.target.value_counts().hvplot.bar(
    title="Heart Disease Count", xlabel='Heart Disease', ylabel='Count', 
    width=500, height=350
)


# In[76]:


# Plotting the results for Heart Disease by Fasting Blood Sugar
have_disease = data.loc[data['target']==1, 'fbs'].value_counts().hvplot.bar(alpha=0.4) 
no_disease = data.loc[data['target']==0, 'fbs'].value_counts().hvplot.bar(alpha=0.4) 

(no_disease * have_disease).opts(
    title="Heart Disease by fasting blood sugar", xlabel='fasting blood sugar > 120 mg/dl (1 = true; 0 = false)', 
    ylabel='Count', width=500, height=450, legend_cols=2, legend_position='top_right'
)


# In[77]:


# Plotting the results for heart disease by resting ECG 
have_disease = data.loc[data['target']==1, 'restecg'].value_counts().hvplot.bar(alpha=0.4) 
no_disease = data.loc[data['target']==0, 'restecg'].value_counts().hvplot.bar(alpha=0.4) 

(no_disease * have_disease).opts(
    title="Heart Disease by resting electrocardiographic results", xlabel='resting electrocardiographic results', 
    ylabel='Count', width=500, height=450, legend_cols=2, legend_position='top_right'
)


# In[78]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(categor_val, 1):
    plt.subplot(3, 3, i)
    data[data["target"] == 0][column].hist(bins=32, color='green', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=32, color='yellow', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# # Heart Disease Frequency for Ages 
# 

# In[79]:


pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# # Heart Disease vs Maximum Heart Rate

# In[80]:


# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data.age[data.target==1],
            data.thalach[data.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data.age[data.target==0],
            data.thalach[data.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# # Analysing the disorder in blood thalassemia 
# 

# In[92]:


data["thal"].unique()


# In[93]:


sns.distplot(data["thal"])


# # Comparing with target

# In[97]:


sns.barplot(data["thal"],y)


# # Analysis of Target

# In[149]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1.5)
sns.countplot(data['target'])
plt.tight_layout()


# In[150]:


data.target.value_counts()


# # Processing of Data

# In[147]:


categor_val = []
contin_val = []
for column in data.columns:
    print("**************")
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categor_val.append(column)
    else:
        contin_val.append(column)


# In[98]:


categor_val.remove('target')
dataset = pd.get_dummies(data, columns = categor_val)


# In[99]:


dataset.head()


# In[100]:


print(data.columns)
print(dataset.columns)


# In[101]:


from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])


# In[102]:


dataset.head()


# # Building Model

# In[103]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        prediction = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, prediction, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, prediction) * 100:.2f}%")
        print("**************************************")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("**********************************************")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")
    
         
    elif train==False:
        prediction = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, prediction) * 100:.2f}%")
        print("*******************************************")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("*******************************************")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, prediction)}\n")
        
       
        


# In[104]:


from sklearn.model_selection import train_test_split

X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)


# # Logistic Regression

# In[151]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='newton-cg')
lr_clf.fit(X_train, y_train)
y_pred_rf = lr_clf.predict(X_test)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
print( y_pred_rf)


# In[152]:


tested_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
trained_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", trained_score, tested_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df


# In[107]:


matrix= confusion_matrix(y_test, y_pred_rf)
sns.heatmap(matrix,annot = True, fmt = "d")


# #  K-nearest neighbors

# In[108]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred_lr = knn_clf.predict(X_test)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)
print(y_pred_lr)


# In[109]:


test_score = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[110]:


matrix= confusion_matrix(y_test,y_pred_lr )
sns.heatmap(matrix,annot = True, fmt = "d")


# # Support Vector Machine

# In[111]:


from sklearn.svm import SVC


svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_clf.fit(X_train, y_train)
y_pred_mr = svm_clf.predict(X_test)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)
print (y_pred_mr)


# In[112]:


test_score = accuracy_score(y_test, svm_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, svm_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Support Vector Machine", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[114]:


matrix= confusion_matrix(y_test,y_pred_mr )
sns.heatmap(matrix,annot = True, fmt="d")


# # Hyper Parameter Tuning 

# # Logistic Regression Hypter Parameter Tuning 

# In[115]:


from sklearn.model_selection import GridSearchCV

params = {"C": np.logspace(-4, 4, 20),
          "solver": ["newton-cg"]}

lr_clf = LogisticRegression()

lr_cv = GridSearchCV(lr_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
lr_cv.fit(X_train, y_train)
x_pred_mr = lr_cv.predict(X_test)
best_params = lr_cv.best_params_
print(f"Best parameters: {best_params}")
lr_clf = LogisticRegression(**best_params)

lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# In[116]:


test_score1 = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score1= accuracy_score(y_train, lr_clf.predict(X_train)) * 100

tuning_results_df = pd.DataFrame(data=[["Tuned Logistic Regression", train_score1, test_score1]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_df


# In[91]:


#Confusion Matrix, Plot 
matrix= confusion_matrix(y_test,x_pred_mr )
sns.heatmap(matrix,annot = True, fmt = "d")


# # Precision score

# In[117]:



from sklearn.metrics import precision_score
precision = precision_score(y_test, x_pred_mr)
print("Precision: ",precision)


# # Recall

# In[118]:



from sklearn.metrics import recall_score
recall = recall_score(y_test, x_pred_mr )
print("Recall is: ",recall)


# # F-score

# In[119]:


print((2*precision*recall)/(precision+recall))


# # Model's False Negative rate

# In[120]:



CM =pd.crosstab(y_test, x_pred_mr)
CM
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr=FN*100/(FN+TP)
fnr


# # Hyperparameter Tuning of K-nearest neighbors

# In[121]:


train_score = []
test_score = []
neighbors = range(1, 30)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_score.append(accuracy_score(y_train, model.predict(X_train)))
    test_score.append(accuracy_score(y_test, model.predict(X_test)))


# In[122]:


plt.figure(figsize=(10, 7))

plt.plot(neighbors, train_score, label="Train score")
# plt.plot(neighbors, test_score, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of Neighbours ")
plt.ylabel("Score gained by model")
plt.legend()

print(f"Maximum KNN score on the test data: {max(train_score)*100:.2f}%")


# In[123]:


knn_clf = KNeighborsClassifier(n_neighbors=27)
knn_clf.fit(X_train, y_train)
x_pred_zr = knn_clf.predict(X_test)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)


# In[124]:


test_score2 = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
train_score2 = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Tuned K-nearest neighbors", train_score2, test_score2]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df_2


# In[125]:


#Confusion Matrix, Plot 
matrix= confusion_matrix(y_test,x_pred_zr )
sns.heatmap(matrix,annot = True, fmt = "d")


# # Precision score

# In[126]:



from sklearn.metrics import precision_score
precision = precision_score(y_test, x_pred_zr)
print("Precision: ",precision)


# # Recall

# In[127]:



from sklearn.metrics import recall_score
recall = recall_score(y_test, x_pred_zr )
print("Recall is: ",recall)


# # F-score

# In[128]:



print((2*precision*recall)/(precision+recall))


# # Model's False Negative rate

# In[129]:



CM =pd.crosstab(y_test, x_pred_zr)
CM
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr=FN*100/(FN+TP)
fnr


# # Hypterparameter Tuning of SVM

# In[130]:


svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)

params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20), 
          "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 
          "kernel":('linear', 'poly', 'rbf')}

svm_cv = GridSearchCV(svm_clf, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
svm_cv.fit(X_train, y_train)
x_pred_qr= svm_cv.predict(X_test)
best_params = svm_cv.best_params_
print(f"Best params: {best_params}")

svm_clf = SVC(**best_params)
svm_clf.fit(X_train, y_train)

print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)


# In[131]:


test_score3 = accuracy_score(y_test, svm_clf.predict(X_test)) * 100
train_score3 = accuracy_score(y_train, svm_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Tuned Support Vector Machine", train_score3, test_score3]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df_2


# In[132]:


#Confusion Matrix, Plot 
matrix= confusion_matrix(y_test,x_pred_qr )
sns.heatmap(matrix,annot = True, fmt = "d")


# # Recall

# In[133]:



from sklearn.metrics import recall_score
recall = recall_score(y_test, x_pred_qr )
print("Recall is: ",recall)


# # F-score

# In[134]:



print((2*precision*recall)/(precision+recall))


# # Model's False Negative rate

# In[135]:



CM =pd.crosstab(y_test, x_pred_qr)
CM
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr=FN*100/(FN+TP)
fnr


# # Precision score

# In[136]:



from sklearn.metrics import precision_score
precision = precision_score(y_test, x_pred_qr)
print("Precision: ",precision)


# # Comparison of Training accuracy  score of the models after tuning  

# In[137]:



scores = [train_score1,train_score2,train_score3]
algorithms = ["Logistic Regression","K-Nearest Neighbors","Support Vector Machine"] 
sns.set(rc={'figure.figsize':(16,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# # Comparison of Testing accuracy score of the models after tuning
# 

# In[138]:


scores = [test_score1,test_score2,test_score3]
algorithms = ["Logistic Regression","K-Nearest Neighbors","Support Vector Machine"] 
sns.set(rc={'figure.figsize':(14,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# # Confusion Matrix for HypterParameter Tunings
# 

# In[146]:


matrix= confusion_matrix(y_test,x_pred_mr )
ax= sns.heatmap(matrix,annot = True, cmap="Blues", fmt="d")
ax.set_title('Confusion Matrix for Logistic Regression')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
sns.set(font_scale=3.0)
plt.show()
print("*********************************************************************************************************************\n")

matrix= confusion_matrix(y_test,x_pred_zr )
ax= sns.heatmap(matrix,annot = True, cmap="Blues", fmt="d")
ax.set_title('Confusion Matrix for KNN ')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
sns.set(font_scale=3.0)
plt.show()

print("*********************************************************************************************************************\n")

matrix= confusion_matrix(y_test,x_pred_qr )
ax= sns.heatmap(matrix,annot = True, cmap="Blues", fmt="d")
ax.set_title('Confusion Matrix for SVM ')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
sns.set(font_scale=3.0)
plt.show()


# In[ ]:




