import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

import os, sys, re, pickle, glob
import urllib.request
import zipfile

import IPython.display as ipd
from tqdm import tqdm
import librosa

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder



st.title("Classifying an audio file belonging to 'Potter' or 'StarWars'") #adding the title
st.markdown('The intresting thing about this classification task is if a person cannot remember the lyrics of the song, one way he/she can think of is hum or whistle to the song, by doing this a person can send a audio file to our machine learning model and our model will output the song label.')
    
#load data

st.header('MLEnd Hums and Whistles dataset')
st.text('This Dataset contains 4 features that is extracted from audio file.')

st.markdown('**1. Power**')
st.markdown('**2. Pitch mean.**')
st.markdown('**3. Pitch standard deviation.**')
st.markdown('**4. Fraction of voiced region.**')
   
st.markdown('class label 1 is ** Potter ** and class label 0 is ** StarWars **')

@st.cache
def load_data():
    data=pd.read_csv('/users/rpkarunkumar/Desktop/potter_starwars.csv')
    lowercase=lambda x:str(x).lower()
    data.rename(lowercase,axis='columns',inplace=True)
    return data

data_load_state= st.text('Loading data...')

data=load_data()

data_load_state=st.text('Loading data...done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


sample_path = '/users/rpkarunkumar/Desktop/Webapp dataset/*.wav'
files=glob.glob(sample_path)
st.write('The length of the total data set is ',len(files))

st.header('play the audio file')
a=st.slider("Select any file from 0 to 831",min_value=0,max_value=831,value=110,step=1)
st.audio(files[a])

st.header('Number of data points that belong to each class')
extract_df=pd.DataFrame(data)
classes=extract_df['class'].value_counts().to_frame()
st.bar_chart(data=classes,width=0, height=0, use_container_width=True)
st.text('The given dataset is a Balanced Dataset')

st.header('The statistical summaries for each features')
st.write(extract_df[['power','pitch_mean','pitch_std','voiced_fr']].describe())
st.text('The above table gives the statistical summaries of each feature of the dataset')

st.header('Distribution of features in the Dataset')
st.set_option('deprecation.showPyplotGlobalUse', False)
selected_feature=st.selectbox('Choose the feature to see its PDF',options=['power','pitch_mean','pitch_std','voiced_fr'])
sns.FacetGrid(extract_df,hue='class',size=7).map(sns.distplot,selected_feature).add_legend()
st.pyplot()

#PAIR PLOT
st.header('Pair plot of the dataset')
sns.pairplot(extract_df,hue='class')
st.pyplot()
st.text('As can be seen from the above pair plot that the data is not linearly separable')

#TRAINING THE ML MODEL
st.header('Training the Adaboost classifier')
st.markdown('The Reason for building AdaBoost Classifier was it was observed that AdaBoost Classifier had the highest accuracies for both training and CV data among all the other all the models ')


X = pd.DataFrame(data,columns=['power','pitch_mean','pitch_std','voiced_fr'])
y=pd.Categorical(extract_df['class'])

if st.checkbox('Show features'):
    st.subheader('features data')
    st.write(X)

if st.checkbox('Show Labels'):
    st.subheader('Class Label')
    st.write(y)

encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))


train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state=1,test_size=0.2)


st.header('Hyperparameter tuning using Grid Search CV for AdaBoost classifier')

#grid searching key hyperparameters for adaboost on this classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
# defining the grid of values to search
grid = dict()

n_est=st.multiselect('choose at least two estimators from the options below',options=[10, 50, 100, 500])
grid['n_estimators'] = n_est

learning_r=st.multiselect('choose at least two learning rates',options=[0.0001, 0.001, 0.01, 0.1, 1.0])
grid['learning_rate'] = learning_r

# defining the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# defining the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# defining the grid search
grid_result = grid_search.fit(X, y)
# summarizing the best score and configuration
st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarizing all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    st.write("%f (%f) with: %r" % (mean, stdev, param))


st.header('Building an AdaBoost Classifier With the inputs taken from the user')

from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

n_e=st.selectbox('Select the size of n_estimators',options=[10, 50, 100, 500])
l_r=st.selectbox('select the learning rate',options=[0.0001, 0.001, 0.01, 0.1, 1.0])
classifier = AdaBoostClassifier(n_estimators=n_e,learning_rate=l_r)
classifier.fit(train_X,train_y)

predictions = classifier.predict(test_X)

from sklearn.metrics import classification_report

conf_matr=confusion_matrix(test_y, predictions)
df_cm = pd.DataFrame(conf_matr, columns=np.unique(test_y), index = np.unique(test_y))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (8,6))
sns.set(font_scale=1.0)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
st.text('Confusion Matrix of this Dataset')
st.pyplot()


st.header("Lets test our model")
st.text('Lets take a random sample from the test dataset')

q=st.slider("Select the sample from the test dataset",min_value=0,max_value=166,value=0,step=1)
op=test_X.index[q]
tt=test_X.iloc[q].to_numpy()
test_data=tt.reshape(1,4)

predictions=classifier.predict(test_data)
st.write(predictions)

if predictions==extract_df['class'].iloc[op]:
    st.write('Our model predicted audio file class label correctly')
else:
    st.write('Our model predicted audio file class label incorrectly')

st.header('Conclusions')
st.markdown('1. Some advance featurisation techniques in audio processing with the help of domain expert can be used to improve the model accuracies even more further.')
st.markdown('2. Training the Model with more number of the data, with Deep learning model, which can be used to improve the accuracies, but we should keep in mind that DL model easily overfit.')
