#!/usr/bin/env python
# coding: utf-8

# In[315]:


# #Loading Libraries
# import warnings
# warnings.filterwarnings('ignore')
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.gridspec import GridSpec
# import re
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics


# In[418]:



# df=pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8')
# for cat in ["Hadoop","ETL Developer","HR","Advocate","Arts","Testing",'PMO',"Operations Manager","SAP Developer"]:
#     mask=df.Category!=cat
#     df=df[mask]

# resumeDataSet =df.copy()
# #EDA
# plt.figure(figsize=(15,15))
# plt.xticks(rotation=90)
# sns.countplot(y="Category", data=resumeDataSet)
# plt.savefig('../jobcategory_details.png')
# #Pie-chart
# targetCounts = resumeDataSet['Category'].value_counts().reset_index()['Category']
# targetLabels  = resumeDataSet['Category'].value_counts().reset_index()['index']
# # Make square figures and axes
# plt.figure(1, figsize=(25,25))
# the_grid = GridSpec(2, 2)
# plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
# source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
# plt.savefig('../category_dist.png')
# #Data Preprocessing
# def cleanResume(resumeText):
#     resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
#     resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
#     resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
#     resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
#     resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
#     resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
#     resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
#     return resumeText
# resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
# var_mod = ['Category']
# le = LabelEncoder()
# for i in var_mod:
#     resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
# requiredText = resumeDataSet['cleaned_resume'].values
# requiredTarget = resumeDataSet['Category'].values
# word_vectorizer = TfidfVectorizer(
#     analyzer='word', 
#     sublinear_tf=True,
#     strip_accents='unicode',
#     token_pattern=r'\w{1,}',
#     ngram_range=(1, 1),
#     max_features=10000)
# word_vectorizer.fit(requiredText)
# WordFeatures = word_vectorizer.transform(requiredText)
# #Model Building
# X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
# # print(X_train.shape)
# # print(X_test.shape)
# clf = OneVsRestClassifier(KNeighborsClassifier())
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
# #Results
# print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
# print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
# # print("n Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(y_test, prediction)))


# In[419]:


# resumeDataSet.drop(['Resume'],axis=1,inplace=True)


# In[130]:





# In[318]:


from pdfminer.high_level import extract_text
def load_dict():
    return pickle.load(open("models/dictionary.pickel","rb"))
def load_vectorizer():
    return pickle.load(open("models/tfidf.sav","rb"))
def load_model():
    return pickle.load(open('models/model.sav', 'rb'))
def prediction(resume):
    loaded_model = load_model()
    try:
        output=loaded_model.predict(resume)[0]
        return output
    except:
        return 

def pdfExtracter(pathName):
    return extract_text(pathName)


def main():    
    file_path=input("Enter the file path: ")
    text=pdfExtracter(file_path)
    text=cleanResume(text)
    text=np.array([text])
    vector_form=load_vectorizer().transform(text)
    output=prediction(vector_form)
    dic=load_dict()[output]
    print(dic)
    return dic
pathName="C:/Users/sefineh/Downloads/Sefineh's Resume (1).pdf"
pred=main()


# In[ ]:





# In[14]:


# import pickle
# import os
# if not os.path.exists("models"):
#     os.mkdir("models")
# fileName="models/model.sav"
# pickle.dump(clf,open(fileName,"wb"))
# pickle.dump(word_vectorizer, open("models/tfidf.sav", "wb"))


# In[369]:


job_description="""
Job Description

● Designing and developing high-performance, data-driven systems in a distributed infrastructure to help with credit decision engines.

● Working with large, interesting data sets including historical, real-time and third party data streams

● Making machine learning models and data pipelines available via low-latency and high performance APIs

● Implementing and optimizing ML solutions in collaboration with Data Scientists

● Working closely with the Data Engineering Team to grow the Data Science software infrastructure

● Building mature CI/CD pipelines, monitor and maintain several production deployments

● Designing and building machine learning algorithms, statistical analysis and predictive modeling

Job Requirements

● 3 years experience in working with Statistical/Machine Learning models operating on large-scale datasets in production.

● BSC University degree in Math, Statistics, Computer Science, Engineering preferably masters.

● Proficiency in Python and a statically typed language

● Experience in SQL and shell scripting

● Experience in developing and maintaining microservices in a production environment

● cloud infrastructure Experience

● Experience with the AWS stack is a plus

● Working experience with Docker and Kubernetes and developing Rest APIs with Flask or FastAP

Skills and Knowledge:
• Effective Communication skill, team player and avid learner.
• problem solver, detail-oriented and self-motivated.
• Ability to Work with ambiguity and believe in lean experimentation.
• Passionate about data science and Willing to play an essential role in the future of digital lending.How to Apply

Interested Applicants should submit a well-prepared and updated CV along with an application letter stating the position; No additional document is required at this stage.

Applications should mail to and state the position on the subject of the email."""


# In[370]:





# In[373]:


job=cleanResume(job_description)

job=load_vectorizer().transform([job])
index=prediction(job)
load_dict()[index]


# In[ ]:


candidate=resumeDataSet[resumeDataSet.Category==index]
text=word_vectorizer.transform(candidate.cleaned_resume)
output=cosine_similarity(vector,text)
top=list(zip(output[0],range(84)))
top.sort()
top=top[::-1]
top[:10]


# In[ ]:





# In[410]:





# In[411]:





# In[377]:





# In[413]:





# In[414]:





# In[ ]:





# In[401]:





# In[ ]:





# In[ ]:





# In[ ]:




