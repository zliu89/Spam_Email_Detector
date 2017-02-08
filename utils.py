import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import codecs
import re

cost_benefit = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}

def getDfSummary(input_data, drop_na = True):
    if drop_na:
        output_data = input_data.dropna().describe().T
    else:
        output_data = input_data.describe().T
    output_data['number_nan'] = input_data.isnull().sum()
    output_data['number_distinct'] = list(map(lambda col: len(input_data[col].dropna().unique()), input_data.columns))
    return output_data

def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")

def getScoreForGridSearch(estimator, X, y):
    score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}
    fpr, tpr, thresholds = roc_curve(y, estimator.predict_proba(X)[:,1])
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    return max(scores)

def getScore(truth, pred, score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    return thresholds, scores

def plotScore(truth, pred, lab, score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(thresholds, scores, color=c, label= lab+' (Best score = %0.4f)' % max(scores))
    plt.xlabel('threshold')
    plt.ylabel('score')
    plt.title('Score')
    plt.legend(loc="lower right")

    return thresholds, scores

def readEmailAsDataFrame():

    def cleanText(text):
        # This function takes in a text string and cleans it 
        # by keeping only alphanumeric and common punctuations
        # Returns the cleaned string
        clean_text = text.replace('\n',' ').replace('\r',' ')
        clean_text = re.sub(r'[^a-zA-Z0-9.:!? ]',' ',clean_text)
        return clean_text

    def readEmail(path):
        # This function takes a path to an email text file
        # Returns a tuple of the subject line and the body text
        with codecs.open(path, "r",encoding='utf-8', errors='ignore') as f:
            subject = cleanText(f.readline()[9:])
            body = cleanText(f.read())
            return [subject, body]

    subjects = []
    bodys = []
    spam = []
    # read hams
    for i in range(1, 6+1):
        folder_name = "enron"+str(i)
        for filename in os.listdir(folder_name+"/ham"):
            if filename.endswith(".txt"):
                subject, body = readEmail(folder_name+"/ham/"+filename)
                subjects.append(subject)
                bodys.append(body)
                spam.append(0)
        # read spams
        for filename in os.listdir(folder_name+"/spam"):
            if filename.endswith(".txt"):
                subject, body = readEmail(folder_name+"/spam/"+filename)
                subjects.append(subject)
                bodys.append(body)
                spam.append(1)
    data = pd.DataFrame()
    data['subject'] = subjects
    data['body'] = bodys
    data['spam'] = spam

    # store using pickle
    data.to_pickle("enron_email.df")

    return data

def dataframeFromWord2VecModel(model, X_):
    def makeFeatureVec(words, model, num_features):
        # This function computes the average of the word vectors in a given list of words
        # Returns the resulting vector
        
        feature_vec = np.zeros((num_features,),dtype="float32")
        #
        nwords = 0.
        
        # using set is faster
        index2word_set = set(model.index2word)

        for word in words:
            # only take it if it's in the vocabulary
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])

        feature_vec = np.divide(feature_vec,nwords)
        # if none of the words are in the vocab, return 0 instead of nan
        if np.isnan(feature_vec):
            feature_vec = 0
        return feature_vec

    def getAvgFeatureVecs(sentences, model, num_features):
        # This function computes the average feature vector for each word in the 
        # given sentence.

        feature_vecs = np.zeros((len(reviews),num_features),dtype="float32")
        
        for i, sentence in enumerate(sentences):
            featureVecs[i] = makeFeatureVec(sentence, model, num_features)
            
        return feature_vecs

    # get the feature vectors
    trainDataVecs = getAvgFeatureVecs(X_train, model, model.vector_size)
    testDataVecs = getAvgFeatureVecs(X_test, model, 300)
    # clear invalid values
    trainDataVecs[np.argwhere(np.isnan(trainDataVecs))] = 0
    testDataVecs[np.argwhere(np.isnan(testDataVecs))] = 0

    # compile to dataframe and save to pickle
    w2v_train_DF = pd.DataFrame(trainDataVecs)
    w2v_train_DF['spam'] = pd.Series(Y_train.tolist(), dtype = np.dtype(int))
    w2v_train_DF.to_pickle("w2v_train.df")

    w2v_test_DF = pd.DataFrame(testDataVecs)
    w2v_test_DF['spam'] = pd.Series(Y_test.tolist(), dtype = np.dtype(int))
    w2v_test_DF.to_pickle("w2v_test.df")

    return w2v_train_DF, w2v_test_DF
