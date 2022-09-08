import streamlit as st
import pandas as pd 
import subprocess 
try : import seaborn
except: subprocess.call(["pip","install","seaborn"])
try : import sklearn
except: subprocess.call(["pip","install","sklearn"])
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

st.title('🎈 Decision Tree Classifier 🎈')
st.sidebar.subheader("Input")
source = st.sidebar.text_input("Souce Data","")
# https://github.com/dataprofessor/data/raw/master/iris.csv
if source :
    st.subheader("Output")
    st.info(f"the github url of your data is : {source}")
    st.subheader("Dataframe Display")
    df = pd.read_csv(source)
    st.write(df)
    figure = sns.pairplot(df, hue=df.keys()[-1])
    st.subheader("Plotting dataset")
    st.pyplot(figure)
    values = st.sidebar.slider(
     'test_size : [0,0.5]',
     min_value = 0.1, max_value=0.5, step = 0.05)
    st.sidebar.write('test_size:', values)
    x = df.drop(df.columns[-1],axis = 1)
    ## one hot encoding 
    x = pd.get_dummies(x,sparse=False,drop_first=True)
    y = df[df.columns[-1]]
    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    stratify = y,  # To fix distribution of y for each splited dataset
                                                    test_size=values,
                                                    random_state=42)
    model = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None,random_state=42).fit(x_train,y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred).T
    
    st.header("Confusion Matrix")

    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=model.classes_)
    st.write(disp.plot().figure_)

    st.subheader(f"Accuracy : {model.score(x_test,y_test):.2f}%")
    
else : 
    st.subheader("Enter Your Input")
    st.warning('Awaiting your input')
