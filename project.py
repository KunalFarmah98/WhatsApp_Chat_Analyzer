# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:57:26 2020

@author: Kunal Farmah   DTU/2K17/IT/061
"""

import re
import pandas as pd
from matplotlib import pyplot as plt
import operator 

def import_data(file, path = ''):
    """ Import whatsapp data and transform it to a dataframe """
   
    with open(path + file, encoding = 'utf-8') as outfile:
        raw_text = outfile.readlines()
        messages = {}

        for message in raw_text: 

            # Some messages are not sent by the user, 
            # but are simply comments and therefore need to be removed
            try:
                name = message.split(' - ')[1].split(':')[0]
            except:
                continue

            if name in messages:
                messages[name].append(message)
            else:
                messages[name] = [message]

    # Convert dictionary to dataframe
    df = pd.DataFrame(columns=['Message_Raw', 'User'])

    for name in messages.keys():
        df = df.append(pd.DataFrame({'Message_Raw': messages[name], 'User': name}))

    df.reset_index(inplace=True)

    return df

def clean_message(row):
    
    """ Try to extract name """

    name = row.User + ': '
    
    try:
        return row.Message_Raw.split(name)[1][:-1]
    except:
        return row.Message_Raw
    

def preprocess_data(df):
   
    """ Creates column with only message, not date/name etc (Message_Clean).
     Creates column with only text message, no smileys etc.(Message_Only_Text) """
 
    
    # Create column with only message, not date/name etc.
    df['Message_Clean'] = df.apply(lambda row: clean_message(row), axis = 1)

    # Create column with only text message in lowercase, no emojis, numerals and punctations.
    df['Message_Only_Text'] = df.apply(lambda row: re.sub(r'[^a-zA-Z\' ]+', '', 
                                                          row.Message_Clean.lower()), 
                                       axis = 1)
    
    return df


def preprocess_data_ngram(df):
   
    """ Creates column with only message, not date/name etc (Message_Clean).
     Creates column with only text message, no smileys etc.(Message_Only_Text) """
 
    
    # Create column with only message, not date/name etc.
    df['Message_Clean'] = df.apply(lambda row: clean_message(row), axis = 1)

    # Create column with only text message in lowercase, no emojis, numerals and punctations.
    df['Message_Only_Text'] = df.apply(lambda row: re.sub(r'[^a-zA-Z\', ]+', '', 
                                                          row.Message_Clean.lower()), 
                                       axis = 1)
    
    return df



import nltk
tokensizer = nltk.RegexpTokenizer(r"\w+")
from nltk.corpus import stopwords
nltk.download('stopwords')

# extra words not to be included in unigram model
myStopWords = ['media','omitted','ggwp','gntc',
                   'kya','ttyl','kkrh','han','hai','deleted','message',
                   'mai','nhi','nii','tum','bye','tha','rha','aur','mai','bhi','kunal','farmah','mea','nhn']
Stopwords = stopwords.words('English')
for word in myStopWords:
    Stopwords.append(word)
        
def removeStopwords(df,outFileName):
    
    word_tokens_without_stopwords = []
    stopwords_removed = []
    for words in df['Message_Only_Text']:
        for word in words.split():
            # word with length>2 is considered a keyword
            if len(word)<=2:
                continue
            # if the token is not in stopwords list, add it to final
            # list or add it to removed list
            
            if word not in Stopwords:
                word_tokens_without_stopwords.append(word)
            else:
                stopwords_removed.append(word)
                

    print(stopwords_removed)
    
    # storing list of keywords
    f = open(outFileName,'w')
    f.write(str(word_tokens_without_stopwords))
    f.close()

    return word_tokens_without_stopwords

# counting all keywords (unigrams)
def countKeyWords(wordslist):
    keywords = {}
    for word in wordslist:
        if word in keywords.keys():
            print(word)
            val = keywords[word]
            val = val+1
            keywords[word]=val
        else:
            keywords[word]=1
                
    return keywords


# counting all bigrams 
def countBigrams(wordslist):
    bigrams = {}
    n = len(wordslist)
    removeList = ['fcomma args','arg fcomma', 'long int' , 'args arg', 'missed voice' , 'voice call' , 'media deleted' ,'include bitsstdch']
    for i in range(0,n-1):
        bigram = wordslist[i]+' '+wordslist[i+1]
        if bigram in removeList:
            continue
        if bigram in bigrams.keys():
            print(bigram)
            val = bigrams[bigram]
            val = val+1
            bigrams[bigram]=val
        else:
            bigrams[bigram]=1
                
    return bigrams

""" TRIGRAMS NEEDS A LOT OF EXTRA PROCESSING TO RULE OUT UNNECCESSARY GROUPING
    STOPWORD REMOVAL MAKES THE SITUATION WORSE, PUNCTUATIONS REMOVAL IS NOT RECOMMENDED
    BUT THEN SPLITTING UP THE SENTENCE BECOMES TRICKY """
    
""" TODO: SEPARATE PREPROCESSING FOR TRIGRAMS """ 
# counting all trigrams 
def countTrigrams(wordslist):
    trigrams = {}
    n = len(wordslist)
    removeList = ['cbseresultsnicin for cbse', 'results check cbsenicin', 'cbsenicin cbseresultsnicin for',
                    'check cbsenicin cbseresultsnicin','hmsvcf file attached', 'monica mam hmsvcf', 'cbsenic', 'cbseresult'
                    'mam hmsvcf file','missed voice call','fcomma','http','arg','com','attatched', 'onedrive', 'psychthrillersupernatural'
                    ,'include', 'bits','disassembler','rotations','differ','long int', 'int','fcomma args','arg fcomma', 'long int' , 
                    'args arg', 'missed voice' , 'voice call' , 'media deleted' ,'include bitsstdch',
                    'functionsnamesstringsmemory','aimtheorycodeoutputinferencelearning','cbseresultsnicin'
                    ,'sday','nday','iday','rday','hmsvcf']
                    
    for i in range(0,n-2):
        trigram = wordslist[i]+' '+wordslist[i+1]+' '+wordslist[i+2]
        skip = False
        for val in removeList:  
            if trigram in removeList or operator.contains(trigram,val):
                skip=True
                break
        if skip:
            continue
        if trigram in trigrams.keys():
            print(trigram)
            val = trigrams[trigram]
            val = val+1
            trigrams[trigram]=val
        else:
            trigrams[trigram]=1
                
    return trigrams


def ohe(keywords,keywords1,keywords2,keywords3,keywords4,keywords5):
    OHE = []
    for i in range(0,6):
        row = []
        for key in keywords.keys():
            if(i==0):
                if(key in keywords1.keys()):
                    row.insert(keywords[key],1)
                else:
                    row.insert(keywords[key],0)
            
            elif(i==1):
                if(key in keywords2.keys()):
                    row.insert(keywords[key],1)
                else:
                    row.insert(keywords[key],0)
            
            elif(i==2):
                if(key in keywords3.keys()):
                    row.insert(keywords[key],1)
                else:
                    row.insert(keywords[key],0)
            
            elif(i==3):
                if(key in keywords4.keys()):
                    row.insert(keywords[key],1)
                else:
                    row.insert(keywords[key],0)
            
            elif(i==4):
                if(key in keywords5.keys()):
                    row.insert(keywords[key],1)
                else:
                    row.insert(keywords[key],0)
            
        OHE.append(row)
        
    print(OHE)
    return OHE


def tf(keywords,keywords1,keywords2,keywords3,keywords4,keywords5): 
    TF = []
    for i in range(0,6):
        row = []
        for key in keywords.keys():
            if(i==0):
                if(key in keywords1.keys()):
                    row.insert(keywords[key],keywords1[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==1):
                if(key in keywords2.keys()):
                    row.insert(keywords[key],keywords2[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==2):
                if(key in keywords3.keys()):
                    row.insert(keywords[key],keywords3[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==3):
                if(key in keywords4.keys()):
                    row.insert(keywords[key],keywords4[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==4):
                if(key in keywords5.keys()):
                    row.insert(keywords[key],keywords5[key])
                else:
                    row.insert(keywords[key],0)
            
        TF.append(row)
    
    print(TF)
    return TF


def tfidf(keywords,keywords1,keywords2,keywords3,keywords4,keywords5,IDF):  
    TFIDF = []
    for i in range(0,5):
        row = []
        for key in keywords.keys():
            if(i==0):
                if(key in keywords1.keys()):
                    row.insert(keywords[key],keywords1[key]*IDF[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==1):
                if(key in keywords2.keys()):
                    row.insert(keywords[key],keywords2[key]*IDF[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==2):
                if(key in keywords3.keys()):
                    row.insert(keywords[key],keywords3[key]*IDF[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==3):
                if(key in keywords4.keys()):
                    row.insert(keywords[key],keywords4[key]*IDF[key])
                else:
                    row.insert(keywords[key],0)
            
            elif(i==4):
                if(key in keywords5.keys()):
                    row.insert(keywords[key],keywords5[key]*IDF[key])
                else:
                    row.insert(keywords[key],0)

        TFIDF.append(row)
    
    print(TFIDF)
    return TFIDF



def storeOHE(OHE,keywords,fname,sorted_keys):
    i=1
    # storing One Hot Matrix
    f = open(fname,"w")
    f.write('\t\t')
    
    for k in range(0,len(keywords)):
        f.write(str(k)+'\t')
    f.write('\n\n')
    for row in OHE:
        f.write('Document '+str(i)+'\t')
        for ele in row:
            f.write(str(ele)+'  \t')
        f.write('\n\n')
        i=i+1
        
    f.write('\n\nKeywords (in order):')
    for key in sorted_keys:
        f.write(key+', ')
            
    f.close()
    return f

def storeTF(TF,keywords,fname,sorted_keys):
    # storing TF Matrix
    i=1
    f = open(fname,"w")
    f.write('\t\t')
    
    for k in range(0,len(keywords)):
        f.write(str(k)+'\t')
    f.write('\n\n')
    for row in TF:
        f.write('Document '+str(i)+'\t')
        for ele in row:
            f.write(str(ele)+'  \t')
        f.write('\n\n')
        i=i+1
        
    f.write('\n\nKeywords (in order):')
    for key in sorted_keys:
        f.write(key+', ')
            
    f.close()
    return f


def storeTFIDF(TFIDF,keywords,fname,sorted_keys):
    # storing TF-IDF Matrix
    i=1
    f = open(fname,"w")
    f.write('\t\t')
    
    for k in range(0,len(keywords)):
        f.write(str(k)+'  \t')
    f.write('\n\n')
    
    for row in TFIDF:
        f.write('Document '+str(i)+'  \t')
        for ele in row:
            f.write(str(round(ele,3))+' \t')
        f.write('\n\n')
        i=i+1
        
    f.write('\n\nKeywords (in order):')
    for key in sorted_keys:
        f.write(key+', ')
            
    f.close()
    return f


def calcIDF(keywords, keywords1, keywords2, keywords3, keywords4, keywords5):     
    # calcualting IDF for each keyword
    import math
    Nt = 5
    IDF = {}
    for key in keywords.keys():
        Nw=0
        if key in keywords1.keys():
            Nw = Nw+1
        if key in keywords2.keys():
            Nw = Nw+1
        if key in keywords3.keys():
            Nw = Nw+1
        if key in keywords4.keys():
            Nw = Nw+1
        if key in keywords5.keys():
            Nw = Nw+1
        print(Nt/Nw)
        IDF[key]  = (math.log(Nt/Nw))
    print(IDF)
    return IDF


def assignIndex(list1, list2, list3, list4, list5):
    # assigning an index to all keywords
    i=-1;
    keywords = {}
    
    for (val,count) in list1:
        if val not in keywords.keys():
            i=i+1;
            keywords[val] = i
    
            
    for (val,count) in list2:
        if val not in keywords.keys():
            i=i+1;
            keywords[val] = i
            
    for (val,count) in list3:
        if val not in keywords.keys():
            i=i+1;
            keywords[val] = i
            
    for (val,count) in list4:
        if val not in keywords.keys():
            i=i+1;        
            keywords[val] = i
            
    for (val,count) in list5:
        if val not in keywords.keys():
            i=i+1;
            keywords[val] = i
            
    return keywords
        
def saveMetric(matrix,outFileName, rowName, colName):
    f = open(outFileName,'w+')
    f.write(colName+' : \t')
    for i in range (0,5):
        f.write(str(i+1)+'\t\t\t')
    f.write('\n')
        
    for i in range (0,5):
        f.write(rowName + str(i+1) +' :\t')
        row = matrix[i]
        for j in range (0,5):
            if(row[j]==0.0):
                f.write(str(row[j])+'\t\t\t')
            else:
                f.write(str(row[j])+'\t')
        
        f.write('\n')
        
    f.close()
        
    return f   

import math 

def magnitude(vector):
    s=0
    for val in vector:
        s = s+(val*val)
    return math.sqrt(s)



def plot(x,y,xlabel,ylabel,scale,color,width):
    plt.bar(x,y,color=color,width=width)
    if scale:
        #plt.ylim(min(y)-10,max(y)+10)
        plt.xticks(rotation = 'vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



""" Pre-processing high school messages """

hsdf1 = import_data('high_school_chat_1.txt')
hsdf1 = preprocess_data(hsdf1)

hsdf2 = import_data('high_school_chat_2.txt')
hsdf2 = preprocess_data(hsdf2)

hsdf3 = import_data('high_school_chat_3.txt')
hsdf3 = preprocess_data(hsdf3)

hsdf4 = import_data('high_school_chat_4.txt')
hsdf4 = preprocess_data(hsdf4)

hsdf5 = import_data('high_school_chat_5.txt')
hsdf5 = preprocess_data(hsdf5)


""" Pre-processing college messages """

clgdf1 = import_data('college_chat_1.txt')
clgdf1 = preprocess_data(clgdf1)

clgdf2 = import_data('college_chat_2.txt')
clgdf2 = preprocess_data(clgdf2)

clgdf3 = import_data('college_chat_3.txt')
clgdf3 = preprocess_data(clgdf3)

clgdf4 = import_data('college_chat_4.txt')
clgdf4 = preprocess_data(clgdf4)

clgdf5 = import_data('college_chat_5.txt')
clgdf5 = preprocess_data(clgdf5)



""" .......................... Unigrams ......................... """
""" High School """


wordsHs1 = removeStopwords(hsdf1, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_High_School_1.txt')
wordsHs2 = removeStopwords(hsdf2, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_High_School_2.txt')
wordsHs3 = removeStopwords(hsdf3, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_High_School_3.txt')
wordsHs4 = removeStopwords(hsdf4, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_High_School_4.txt')
wordsHs5 = removeStopwords(hsdf5, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_High_School_5.txt')
            
keywordsHs1=countKeyWords(wordsHs1)
keywordsHs2=countKeyWords(wordsHs2)
keywordsHs3=countKeyWords(wordsHs3)
keywordsHs4=countKeyWords(wordsHs4)
keywordsHs5=countKeyWords(wordsHs5)


topKHs1=[]
topKHs2=[]
topKHs3=[]
topKHs4=[]
topKHs5=[]
topKHs6=[]


for key in keywordsHs1.keys():
    topKHs1.append((key,keywordsHs1[key]))    
for key in keywordsHs2.keys():
    topKHs2.append((key,keywordsHs2[key]))
for key in keywordsHs3.keys():
    topKHs3.append((key,keywordsHs3[key]))
for key in keywordsHs4.keys():
    topKHs4.append((key,keywordsHs4[key]))
for key in keywordsHs5.keys():
    topKHs5.append((key,keywordsHs5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topKHs1 = sorted(topKHs1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKHs1 = topKHs1[0:10]

topKHs2 = sorted(topKHs2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKHs2 = topKHs2[0:10]

topKHs3 = sorted(topKHs3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKHs3= topKHs3[0:10]

topKHs4 = sorted(topKHs4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKHs4 = topKHs4[0:10]

topKHs5 = sorted(topKHs5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKHs5 = topKHs5[0:10]



print('Top 10 Keywords in High School Chat 1 with frequency: '+str(topKHs1))
print('Top 10 Keywords in High School Chat 2 with frequency: '+str(topKHs2))
print('Top 10 Keywords in High School Chat 3 with frequency: '+str(topKHs3))
print('Top 10 Keywords in High School Chat 4 with frequency: '+str(topKHs4))
print('Top 10 Keywords in High School Chat 5 with frequency: '+str(topKHs5))

plot([val[0] for val in topKHs1],[val[1] for val in topKHs1],'High School Chat 1 Keywords','Frequency',False,color='red',width = 0.5)
plot([val[0] for val in topKHs2],[val[1] for val in topKHs2],'High School Chat 2 Keywords','Frequency',False,color='blue',width = 0.5)
plot([val[0] for val in topKHs3],[val[1] for val in topKHs3],'High School Chat 3 Keywords','Frequency',False,color='orange',width = 0.5)
plot([val[0] for val in topKHs4],[val[1] for val in topKHs4],'High School Chat 4 Keywords','Frequency',False,color='green',width = 0.5)
plot([val[0] for val in topKHs5],[val[1] for val in topKHs5],'High School Chat 5 Keywords','Frequency',False,color='purple',width = 0.5)

# assigning index to each keywords
keywordsHS = assignIndex(topKHs1,topKHs2,topKHs3,topKHs4,topKHs5)

# finding IDF vector for the keywords
IDF_HS = calcIDF(keywordsHS,keywordsHs1,keywordsHs2,keywordsHs3,keywordsHs4,keywordsHs5)

# one hot encoding
OHE_HS = ohe(keywordsHS,keywordsHs1,keywordsHs2,keywordsHs3,keywordsHs4,keywordsHs5)
# term frequency
TF_HS = tf(keywordsHS,keywordsHs1,keywordsHs2,keywordsHs3,keywordsHs4,keywordsHs5)
# term frequency, inverse dense frequency
TFIDF_HS = tfidf(keywordsHS,keywordsHs1,keywordsHs2,keywordsHs3,keywordsHs4,keywordsHs5,IDF_HS)


# making a list of keys sorted by index
sorted_keys_HS=list(keywordsHS)

storeOHE(OHE_HS,keywordsHS,'High_School_OHE.txt',sorted_keys_HS)
storeTF(TF_HS,keywordsHS,'High_School_TF.txt',sorted_keys_HS)
storeTFIDF(TFIDF_HS,keywordsHS,'High_School_TFIDF.txt',sorted_keys_HS)

print('\nOHE_HS Matrix')

# printing all 3 matrices
for row in OHE_HS:
    print(row)
    
print('\nTF_HS Matrix')
    
for row in TF_HS:
    print(row)

print('\nTF-IDF_HS Matrix')

for row in TFIDF_HS:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    


""" College """

wordsClg1 = removeStopwords(clgdf1, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_College_1.txt')
wordsClg2 = removeStopwords(clgdf2, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_College_2.txt')
wordsClg3 = removeStopwords(clgdf3, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_College_3.txt')
wordsClg4 = removeStopwords(clgdf4, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_College_4.txt')
wordsClg5 = removeStopwords(clgdf5, 'KUNAL_FARMAH_2K17_IT_61_StopwordsRemoved_College_5.txt')

keywordsClg1=countKeyWords(wordsClg1)
keywordsClg2=countKeyWords(wordsClg2)
keywordsClg3=countKeyWords(wordsClg3)
keywordsClg4=countKeyWords(wordsClg4)
keywordsClg5=countKeyWords(wordsClg5)

f = open('temp.txt','w+')
words = {}
for word in sorted(keywordsClg1.keys()):
    words[word]=1
    
for word in sorted(keywordsClg2.keys()):
    words[word]=1

for word in sorted(keywordsClg3.keys()):
    words[word]=1

for word in sorted(keywordsClg4.keys()):
    words[word]=1

for word in sorted(keywordsClg5.keys()):
    words[word]=1


for word in sorted(words.keys()):
    f.write(word+'\n')
f.close()
    
topKClg1=[]
topKClg2=[]
topKClg3=[]
topKClg4=[]
topKClg5=[]
topKClg6=[]


for key in keywordsClg1.keys():
    topKClg1.append((key,keywordsClg1[key]))    
for key in keywordsClg2.keys():
    topKClg2.append((key,keywordsClg2[key]))
for key in keywordsClg3.keys():
    topKClg3.append((key,keywordsClg3[key]))
for key in keywordsClg4.keys():
    topKClg4.append((key,keywordsClg4[key]))
for key in keywordsClg5.keys():
    topKClg5.append((key,keywordsClg5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topKClg1 = sorted(topKClg1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKClg1= topKClg1[0:10]

topKClg2 = sorted(topKClg2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKClg2 = topKClg2[0:10]

topKClg3 = sorted(topKClg3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKClg3= topKClg3[0:10]

topKClg4 = sorted(topKClg4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKClg4 = topKClg4[0:10]

topKClg5 = sorted(topKClg5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topKClg5 = topKClg5[0:10]



print('Top 10 Keywords in College Chat 1 with frequency: '+str(topKClg1))
print('Top 10 Keywords in College Chat 2 with frequency: '+str(topKClg2))
print('Top 10 Keywords in College Chat 3 with frequency: '+str(topKClg3))
print('Top 10 Keywords in College Chat 4 with frequency: '+str(topKClg4))
print('Top 10 Keywords in College Chat 5 with frequency: '+str(topKClg5))


"""  ..................... Plotting Unigram Keywords ...................................... """

plot([val[0] for val in topKClg1],[val[1] for val in topKClg1],'College Chat 1 Keywords','Frequency',False,color='red',width = 0.5)
plot([val[0] for val in topKClg2],[val[1] for val in topKClg2],'College Chat 2 Keywords','Frequency',False,color='blue',width = 0.5)
plot([val[0] for val in topKClg3],[val[1] for val in topKClg3],'College Chat 3 Keywords','Frequency',False,color='orange',width = 0.5)
plot([val[0] for val in topKClg4],[val[1] for val in topKClg4],'College Chat 4 Keywords','Frequency',False,color='green',width = 0.5)
plot([val[0] for val in topKClg5],[val[1] for val in topKClg5],'College Chat 5 Keywords','Frequency',False,color='purple',width = 0.5)


# assigning an index to all keywords
keywordsClg = assignIndex(topKClg1, topKClg2, topKClg3, topKClg4, topKClg5)
# calcualting IDF for each keyword
IDF_CLG = calcIDF(keywordsClg,keywordsClg1,keywordsClg2,keywordsClg3,keywordsClg4,keywordsClg5)

# one hot encoding
OHE_CLG = ohe(keywordsClg,keywordsClg1,keywordsClg2,keywordsClg3,keywordsClg4,keywordsClg5)
# term frequency
TF_CLG = tf(keywordsClg,keywordsClg1,keywordsClg2,keywordsClg3,keywordsClg4,keywordsClg5)
# term frequency, inverse dense frequency
TFIDF_CLG = tfidf(keywordsClg,keywordsClg1,keywordsClg2,keywordsClg3,keywordsClg4,keywordsClg5,IDF_CLG)

# making a list of keys sorted by index
sorted_keys_CLG=list(keywordsClg)

storeOHE(OHE_CLG,keywordsHS,'College_OHE.txt',sorted_keys_CLG)
storeTF(TF_CLG,keywordsHS,'College_TF.txt',sorted_keys_CLG)
storeTFIDF(TFIDF_CLG,keywordsHS,'College_TFIDF.txt',sorted_keys_CLG)

print('\nOHE_CLG Matrix')

# printing all 3 matrices
for row in OHE_CLG:
    print(row)
    
print('\nTF_CLG Matrix')
    
for row in TF_CLG:
    print(row)

print('\nTF-IDF_CLG Matrix')

for row in TFIDF_CLG:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    
    
 
""" Cosine Similarity in High School and College Keywords in BOW """

""" Hs vs CLG """


train = TFIDF_HS
test = TFIDF_CLG
l1 = len(test[0])
l2 = len(train[0])
# managed different lenghts according to this article
# https://stackoverflow.com/questions/3121217/cosine-similarity-of-vectors-of-different-lengths
l = min(l1,l2)


# finding cosine similarity
similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k]) 
        cosine = s/(trainmag*testmag)
        cosines.append(cosine)
        print('Cosine similarity of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(cosine))
    similarity.append(cosines)
    
# saving the cosine matrix
saveMetric(similarity, 'Cosine_Similarity_Unigram_Matrix.txt' ,'Collge Chat ','High School Chat ')
    

# finding closest documents 
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = similarity[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()

f = open('Cosine_Similarity_Unigram_Matrix.txt','a')
f.write('\n\n')
print('The College Chat Document - {} is closest to the High School Chat Document - {} with a cosine similarity of {}'.
      format(clg+1,hs+1,maxval),file=f)
f.close()


""" Verifying our result by doing reverse """
""" CLG vs HS """
train = TFIDF_CLG
test = TFIDF_HS
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k])
           
        cosine = s/(trainmag*testmag)
        cosines.append(cosine)
        print('Cosine similarity of High_School_Chat{} with College_Chat_{} is '.format(i+1,j+1)+str(cosine))
    similarity.append(cosines)
    

# plotting cosine similarities

plot([1,2,3,4,5],similarity[0],'High School Chats','Unigram Cosine Similarity with College Chat Document 1',False,'red',0.5)
plot([1,2,3,4,5],similarity[1],'High School Chats','Unigram Cosine Similarity with College Chat Document 2',False,'red',0.5)
plot([1,2,3,4,5],similarity[2],'High School Chats','Unigram Cosine Similarity with College Chat Document 3',False,'red',0.5)
plot([1,2,3,4,5],similarity[3],'High School Chats','Unigram Cosine Similarity with College Chat Document 4',False,'red',0.5)
plot([1,2,3,4,5],similarity[4],'High School Chats','Unigram Cosine Similarity with College Chat Document 5',False,'red',0.5)


maxval = 0
hs=0
clg=0
    
for i in range (0,5):
    row = similarity[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()
    

print('The High School Chat Document - {} is closest to the College Chat Document- {} with a cosine similarity of {}'.
      format(hs+1,clg+1,maxval))

""" Manhattan Distance """
""" HS Vs CLG """

train = TFIDF_HS
test = TFIDF_CLG
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

# finding the manhattan distance
manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)

# saving the manhattan matrix
saveMetric(manhattan, 'Manhattan_Distance_Unigram_Matrix.txt' ,'Collge Chat ','High School Chat ')
# plotting manhattan distance
plot([1,2,3,4,5],manhattan[0],'High School Chats','Unigram Manhattan Distance from College Chat Document 1',False,'red',0.5)
plot([1,2,3,4,5],manhattan[1],'High School Chats','Unigram Manhattan Distance from College Chat Document 2',False,'red',0.5)
plot([1,2,3,4,5],manhattan[2],'High School Chats','Unigram Manhattan Distance from College Chat Document 3',False,'red',0.5)
plot([1,2,3,4,5],manhattan[3],'High School Chats','Unigram Manhattan Distance from College Chat Document 4',False,'red',0.5)
plot([1,2,3,4,5],manhattan[4],'High School Chats','Unigram Manhattan Distance from College Chat Document 5',False,'red',0.5)



    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()

f = open('Manhattan_Distance_Unigram_Matrix.txt','a')
f.write('\n\n')
print('The College Chat Document - {} is farthest to the High School Chat Document - {} with a manhattan distance of {}'.
      format(clg+1,hs+1,maxval),file=f)
f.close()


""" Verifying the Result """
""" CLG Vs Hs """

train = TFIDF_CLG
test = TFIDF_HS
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of High_School_Chat_{} with College_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)
    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()

print('The High School Chat Document - {} is farthest to the College Chat Document - {} with a manhattan distance of {}'.
      format(hs+1,clg+1,maxval))



"""     ................................. Bigrams .................................  """

""" Pre-processing high school messages """

hsdf1 = import_data('high_school_chat_1.txt')
hsdf1 = preprocess_data_ngram(hsdf1)

hsdf2 = import_data('high_school_chat_2.txt')
hsdf2 = preprocess_data_ngram(hsdf2)

hsdf3 = import_data('high_school_chat_3.txt')
hsdf3 = preprocess_data_ngram(hsdf3)

hsdf4 = import_data('high_school_chat_4.txt')
hsdf4 = preprocess_data_ngram(hsdf4)

hsdf5 = import_data('high_school_chat_5.txt')
hsdf5 = preprocess_data_ngram(hsdf5)


""" Pre-processing college messages """

clgdf1 = import_data('college_chat_1.txt')
clgdf1 = preprocess_data_ngram(clgdf1)

clgdf2 = import_data('college_chat_2.txt')
clgdf2 = preprocess_data_ngram(clgdf2)

clgdf3 = import_data('college_chat_3.txt')
clgdf3 = preprocess_data_ngram(clgdf3)

clgdf4 = import_data('college_chat_4.txt')
clgdf4 = preprocess_data_ngram(clgdf4)

clgdf5 = import_data('college_chat_5.txt')
clgdf5 = preprocess_data_ngram(clgdf5)

""" High School """


bigramsHs1 = removeStopwords(hsdf1, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_High_School_1.txt')
bigramsHs2 = removeStopwords(hsdf2, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_High_School_2.txt')
bigramsHs3 = removeStopwords(hsdf3, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_High_School_3.txt')
bigramsHs4 = removeStopwords(hsdf4, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_High_School_4.txt')
bigramsHs5 = removeStopwords(hsdf5, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_High_School_5.txt')

bigramsHs1=countBigrams(bigramsHs1)
bigramsHs2=countBigrams(bigramsHs2)
bigramsHs3=countBigrams(bigramsHs3)
bigramsHs4=countBigrams(bigramsHs4)
bigramsHs5=countBigrams(bigramsHs5)
    
topBHs1=[]
topBHs2=[]
topBHs3=[]
topBHs4=[]
topBHs5=[]


for key in bigramsHs1.keys():
    topBHs1.append((key,bigramsHs1[key]))    
for key in bigramsHs2.keys():
    topBHs2.append((key,bigramsHs2[key]))
for key in bigramsHs3.keys():
    topBHs3.append((key,bigramsHs3[key]))
for key in bigramsHs4.keys():
    topBHs4.append((key,bigramsHs4[key]))
for key in bigramsHs5.keys():
    topBHs5.append((key,bigramsHs5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topBHs1 = sorted(topBHs1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBHs1= topBHs1[0:10]

topBHs2 = sorted(topBHs2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBHs2 = topBHs2[0:10]

topBHs3 = sorted(topBHs3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBHs3= topBHs3[0:10]

topBHs4 = sorted(topBHs4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBHs4 = topBHs4[0:10]

topBHs5 = sorted(topBHs5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBHs5 = topBHs5[0:10]



print('Top 10 Bigrams in High School Chat 1 with frequency: '+str(topBHs1))
print('Top 10 Bigrams in High School Chat 2 with frequency: '+str(topBHs2))
print('Top 10 Bigrams in High School Chat 3 with frequency: '+str(topBHs3))
print('Top 10 Bigrams in High School Chat 4 with frequency: '+str(topBHs4))
print('Top 10 Bigrams in High School Chat 5 with frequency: '+str(topBHs5))


plot([val[0] for val in topBHs1],[val[1] for val in topBHs1],'High School Chat 1 Bigrams','Frequency',True,color='red',width = 0.5)
plot([val[0] for val in topBHs2],[val[1] for val in topBHs2],'High School Chat 2 Bigrams','Frequency',True,color='blue',width = 0.5)
plot([val[0] for val in topBHs3],[val[1] for val in topBHs3],'High School Chat 3 Bigrams','Frequency',True,color='orange',width = 0.5)
plot([val[0] for val in topBHs4],[val[1] for val in topBHs4],'High School Chat 4 Bigrams','Frequency',True,color='green',width = 0.5)
plot([val[0] for val in topBHs5],[val[1] for val in topBHs5],'High School Chat 5 Bigrams','Frequency',True,color='purple',width = 0.5)


# assigning an index to all keywords
bigramsHs = assignIndex(topBHs1, topBHs2, topBHs3, topBHs4, topBHs5)
# calculating IDF        
IDF_HS_Bigram =calcIDF(bigramsHs, bigramsHs1, bigramsHs2, bigramsHs3, bigramsHs4, bigramsHs5)

# one hot encoding
OHE_HS_Bigram = ohe(bigramsHs,bigramsHs1,bigramsHs2,bigramsHs3,bigramsHs4,bigramsHs5)
# term frequency
TF_HS_Bigram = tf(bigramsHs,bigramsHs1,bigramsHs2,bigramsHs3,bigramsHs4,bigramsHs5)
# term frequency, inverse dense frequency
TFIDF_HS_Bigram = tfidf(bigramsHs,bigramsHs1,bigramsHs2,bigramsHs3,bigramsHs4,bigramsHs5,IDF_HS_Bigram)


# making a list of keys sorted by index
sorted_keys_HS_bigram=list(bigramsHs)

storeOHE(OHE_HS_Bigram,bigramsHs,'High_School_OHE_Bigram.txt',sorted_keys_HS_bigram)
storeTF(TF_HS_Bigram,bigramsHs,'High_School_TF_Bigram.txt',sorted_keys_HS_bigram)
storeTFIDF(TFIDF_HS_Bigram,bigramsHs,'High_School_TFIDF_Bigram.txt',sorted_keys_HS_bigram)

print('\nOHE_HS Bigram Matrix')

# printing all 3 matrices
for row in OHE_HS_Bigram:
    print(row)
    
print('\nTF_HS Bigram Matrix')
    
for row in TF_HS_Bigram:
    print(row)

print('\nTF-IDF_HS Bigram Matrix')

for row in TFIDF_HS_Bigram:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    
    
""" College """


bigramsClg1 = removeStopwords(clgdf1, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_College_1.txt')
bigramsClg2 = removeStopwords(clgdf2, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_College_2.txt')
bigramsClg3 = removeStopwords(clgdf3, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_College_3.txt')
bigramsClg4 = removeStopwords(clgdf4, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_College_4.txt')
bigramsClg5 = removeStopwords(clgdf5, 'KUNAL_FARMAH_2K17_IT_61_BigramsRemoved_College_5.txt')

bigramsClg1=countBigrams(bigramsClg1)
bigramsClg2=countBigrams(bigramsClg2)
bigramsClg3=countBigrams(bigramsClg3)
bigramsClg4=countBigrams(bigramsClg4)
bigramsClg5=countBigrams(bigramsClg5)
    
topBClg1=[]
topBClg2=[]
topBClg3=[]
topBClg4=[]
topBClg5=[]
topBClg6=[]


for key in bigramsClg1.keys():
    topBClg1.append((key,bigramsClg1[key]))    
for key in bigramsClg2.keys():
    topBClg2.append((key,bigramsClg2[key]))
for key in bigramsClg3.keys():
    topBClg3.append((key,bigramsClg3[key]))
for key in bigramsClg4.keys():
    topBClg4.append((key,bigramsClg4[key]))
for key in bigramsClg5.keys():
    topBClg5.append((key,bigramsClg5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topBClg1 = sorted(topBClg1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBClg1= topBClg1[0:10]

topBClg2 = sorted(topBClg2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBClg2 = topBClg2[0:10]

topBClg3 = sorted(topBClg3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBClg3= topBClg3[0:10]

topBClg4 = sorted(topBClg4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBClg4 = topBClg4[0:10]

topBClg5 = sorted(topBClg5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topBClg5 = topBClg5[0:10]



print('Top 10 Bigrams in College Chat 1 with frequency: '+str(topBClg1))
print('Top 10 Bigrams in College Chat 2 with frequency: '+str(topBClg2))
print('Top 10 Bigrams in College Chat 3 with frequency: '+str(topBClg3))
print('Top 10 Bigrams in College Chat 4 with frequency: '+str(topBClg4))
print('Top 10 Bigrams in College Chat 5 with frequency: '+str(topBClg5))

plot([val[0] for val in topBClg1],[val[1] for val in topBClg1],'College Chat 1 Bigrams','Frequency',True,color='red',width = 0.5)
plot([val[0] for val in topBClg2],[val[1] for val in topBClg2],'College Chat 2 Bigrams','Frequency',True,color='blue',width = 0.5)
plot([val[0] for val in topBClg3],[val[1] for val in topBClg3],'College Chat 3 Bigrams','Frequency',True,color='orange',width = 0.5)
plot([val[0] for val in topBClg4],[val[1] for val in topBClg4],'College Chat 4 Bigrams','Frequency',True,color='green',width = 0.5)
plot([val[0] for val in topBClg5],[val[1] for val in topBClg5],'College Chat 5 Bigrams','Frequency',True,color='purple',width = 0.5)


# assigning an index to all keywords

bigramsClg = assignIndex(topBClg1,topBClg2,topBClg3,topBClg4,topBClg5)
        

        
# calculating IDF for each bigram

IDF_CLG_Bigram = calcIDF(bigramsClg,bigramsClg1,bigramsClg2,bigramsClg3,bigramsClg4,bigramsClg5)

# one hot encoding
OHE_CLG_Bigram = ohe(bigramsClg,bigramsClg1,bigramsClg2,bigramsClg3,bigramsClg4,bigramsClg5)
# term frequency
TF_CLG_Bigram = tf(bigramsClg,bigramsClg1,bigramsClg2,bigramsClg3,bigramsClg4,bigramsClg5)
# term frequency, inverse dense frequency
TFIDF_CLG_Bigram = tfidf(bigramsClg,bigramsClg1,bigramsClg2,bigramsClg3,bigramsClg4,bigramsClg5,IDF_CLG_Bigram)


# making a list of keys sorted by index
sorted_keys_CLG_bigram=list(bigramsClg)

storeOHE(OHE_CLG_Bigram,bigramsClg,'College_OHE_Bigram.txt',sorted_keys_CLG_bigram)
storeTF(TF_CLG_Bigram,bigramsClg,'College_TF_Bigram.txt',sorted_keys_CLG_bigram)
storeTFIDF(TFIDF_CLG_Bigram,bigramsClg,'College_TFIDF_Bigram.txt',sorted_keys_CLG_bigram)

print('\nOHE_CLG Bigram Matrix')

# printing all 3 matrices
for row in OHE_CLG_Bigram:
    print(row)
    
print('\nTF_CLG Bigram Matrix')
    
for row in TF_CLG_Bigram:
    print(row)

print('\nTF-IDF_CLG Bigram Matrix')

for row in TFIDF_CLG_Bigram:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    
    

        
""" Cosine Similarity in High School and College Keywords in BOP """

""" Hs vs CLG """


train = TFIDF_HS_Bigram
test = TFIDF_CLG_Bigram
l1 = len(test[0])
l2 = len(train[0])
# managed different lenghts according to this article
# https://stackoverflow.com/questions/3121217/cosine-similarity-of-vectors-of-different-lengths
l = min(l1,l2)

similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k])
           
        cosine = s/(trainmag*testmag)
        
        cosines.append(cosine)
            
        print('Cosine similarity of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(cosine))
        
    similarity.append(cosines)
    
    
saveMetric(similarity, 'Cosine_Similarity_Bigram_Matrix.txt' ,'Collge Chat ','High School Chat ')

# plotting cosine similarities

plot([1,2,3,4,5],similarity[0],'High School Chats','Bigram Cosine Similarity with College Chat Document 1',False,'green',0.5)
plot([1,2,3,4,5],similarity[1],'High School Chats','Bigram Cosine Similarity with College Chat Document 2',False,'green',0.5)
plot([1,2,3,4,5],similarity[2],'High School Chats','Bigram Cosine Similarity with College Chat Document 3',False,'green',0.5)
plot([1,2,3,4,5],similarity[3],'High School Chats','Bigram Cosine Similarity with College Chat Document 4',False,'green',0.5)
plot([1,2,3,4,5],similarity[4],'High School Chats','Bigram Cosine Similarity with College Chat Document 5',False,'green',0.5)

    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = similarity[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()

f = open('Cosine_Similarity_Bigram_Matrix.txt','a')
f.write('\n\n')

print('The College Chat Document - {} is closest to the High School Chat Document - {} with a cosine similarity of {}'.
      format(clg+1,hs+1,maxval),file = f)
f.close()


""" Verifying our result by doing reverse """
""" CLG vs HS """
train = TFIDF_CLG_Bigram
test = TFIDF_HS_Bigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k])
           
        cosine = s/(trainmag*testmag)
        cosines.append(cosine)
        print('Cosine similarity of High_School_Chat{} with College_Chat_{} is '.format(i+1,j+1)+str(cosine))
    similarity.append(cosines)
    

maxval = 0
hs=0
clg=0
    
for i in range (0,5):
    row = similarity[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()
    

print('The High School Chat Document - {} is closest to the College Chat Document- {} with a cosine similarity of {}'.
      format(hs+1,clg+1,maxval))


""" Manhattan Distance """
""" HS Vs CLG """

train = TFIDF_HS_Bigram
test = TFIDF_CLG_Bigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)
    
saveMetric(manhattan, 'Manhattan_Distance_Bigram_Matrix.txt' ,'Collge Chat ','High School Chat ')

# plotting manhattan distance
plot([1,2,3,4,5],manhattan[0],'High School Chats','Bigram Manhattan Distance from College Chat Document 1',False,'green',0.5)
plot([1,2,3,4,5],manhattan[1],'High School Chats','Bigram Manhattan Distance from College Chat Document 2',False,'green',0.5)
plot([1,2,3,4,5],manhattan[2],'High School Chats','Bigram Manhattan Distance from College Chat Document 3',False,'green',0.5)
plot([1,2,3,4,5],manhattan[3],'High School Chats','Bigram Manhattan Distance from College Chat Document 4',False,'green',0.5)
plot([1,2,3,4,5],manhattan[4],'High School Chats','Bigram Manhattan Distance from College Chat Document 5',False,'green',0.5)


    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()
f = open('Manhattan_Distance_Bigram_Matrix.txt','a')
f.write('\n\n')

print('The College Chat Document - {} is farthest to the High School Chat Document - {} with a manhattan distance of {}'.
      format(clg+1,hs+1,maxval),file = f)
f.close()

""" Verifying the Result """
""" CLG Vs Hs """

train = TFIDF_CLG_Bigram
test = TFIDF_HS_Bigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of High_School_Chat_{} with College_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)
    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()

print('The High School Chat Document - {} is farthest to the College Chat Document - {} with a manhattan distance of {}'.
      format(hs+1,clg+1,maxval))



"""     ................................. Trigrams .................................  """


""" Pre-processing high school messages """

hsdf1 = import_data('high_school_chat_1.txt')
hsdf1 = preprocess_data_ngram(hsdf1)

hsdf2 = import_data('high_school_chat_2.txt')
hsdf2 = preprocess_data_ngram(hsdf2)

hsdf3 = import_data('high_school_chat_3.txt')
hsdf3 = preprocess_data_ngram(hsdf3)

hsdf4 = import_data('high_school_chat_4.txt')
hsdf4 = preprocess_data_ngram(hsdf4)

hsdf5 = import_data('high_school_chat_5.txt')
hsdf5 = preprocess_data_ngram(hsdf5)


""" Pre-processing college messages """

clgdf1 = import_data('college_chat_1.txt')
clgdf1 = preprocess_data_ngram(clgdf1)

clgdf2 = import_data('college_chat_2.txt')
clgdf2 = preprocess_data_ngram(clgdf2)

clgdf3 = import_data('college_chat_3.txt')
clgdf3 = preprocess_data_ngram(clgdf3)

clgdf4 = import_data('college_chat_4.txt')
clgdf4 = preprocess_data_ngram(clgdf4)

clgdf5 = import_data('college_chat_5.txt')
clgdf5 = preprocess_data_ngram(clgdf5)


""" High School """


trigramsHs1 = removeStopwords(hsdf1, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_High_School_1.txt')
trigramsHs2 = removeStopwords(hsdf2, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_High_School_2.txt')
trigramsHs3 = removeStopwords(hsdf3, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_High_School_3.txt')
trigramsHs4 = removeStopwords(hsdf4, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_High_School_4.txt')
trigramsHs5 = removeStopwords(hsdf5, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_High_School_5.txt')

trigramsHs1=countTrigrams(trigramsHs1)
trigramsHs2=countTrigrams(trigramsHs2)
trigramsHs3=countTrigrams(trigramsHs3)
trigramsHs4=countTrigrams(trigramsHs4)
trigramsHs5=countTrigrams(trigramsHs5)
    
topTHs1=[]
topTHs2=[]
topTHs3=[]
topTHs4=[]
topTHs5=[]


for key in trigramsHs1.keys():
    topTHs1.append((key,trigramsHs1[key]))    
for key in trigramsHs2.keys():
    topTHs2.append((key,trigramsHs2[key]))
for key in trigramsHs3.keys():
    topTHs3.append((key,trigramsHs3[key]))
for key in trigramsHs4.keys():
    topTHs4.append((key,trigramsHs4[key]))
for key in trigramsHs5.keys():
    topTHs5.append((key,trigramsHs5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topTHs1 = sorted(topTHs1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTHs1= topTHs1[0:10]

topTHs2 = sorted(topTHs2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTHs2 = topTHs2[0:10]

topTHs3 = sorted(topTHs3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTHs3= topTHs3[0:10]

topTHs4 = sorted(topTHs4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTHs4 = topTHs4[0:10]

topTHs5 = sorted(topTHs5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTHs5 = topTHs5[0:10]



print('Top 10 Trigrams in High School Chat 1 with frequency: '+str(topTHs1))
print('Top 10 Trigrams in High School Chat 2 with frequency: '+str(topTHs2))
print('Top 10 Trigrams in High School Chat 3 with frequency: '+str(topTHs3))
print('Top 10 Trigrams in High School Chat 4 with frequency: '+str(topTHs4))
print('Top 10 Trigrams in High School Chat 5 with frequency: '+str(topTHs5))

plot([val[0] for val in topTHs1],[val[1] for val in topTHs1],'High School Chat 1 Trigrams','Frequency',True,color='red',width = 0.5)
plot([val[0] for val in topTHs2],[val[1] for val in topTHs2],'High School Chat 2 Trigrams','Frequency',True,color='blue',width = 0.5)
plot([val[0] for val in topTHs3],[val[1] for val in topTHs3],'High School Chat 3 Trigrams','Frequency',True,color='orange',width = 0.5)
plot([val[0] for val in topTHs4],[val[1] for val in topTHs4],'High School Chat 4 Trigrams','Frequency',True,color='green',width = 0.5)
plot([val[0] for val in topTHs5],[val[1] for val in topTHs5],'High School Chat 5 Trigrams','Frequency',True,color='purple',width = 0.5)


""" As we can see max frequency is just 2, which makes trigram unsuitable for this project """
""" Lets vrify it here """

# assigning an index to all keywords
trigramsHs = assignIndex(topTHs1,topTHs2,topTHs3,topTHs4,topTHs5)


# calculating IDF for each bigram
IDF_HS_Trigram = calcIDF(trigramsHs,trigramsHs1,trigramsHs2,trigramsHs3,trigramsHs4,trigramsHs5)
# one hot encoding
OHE_HS_Trigram = ohe(trigramsHs,trigramsHs1,trigramsHs2,trigramsHs3,trigramsHs4,trigramsHs5)
# term frequency
TF_HS_Trigram = tf(trigramsHs,trigramsHs1,trigramsHs2,trigramsHs3,trigramsHs4,trigramsHs5)
# term frequency, inverse dense frequency
TFIDF_HS_Trigram = tfidf(trigramsHs,trigramsHs1,trigramsHs2,trigramsHs3,trigramsHs4,trigramsHs5,IDF_HS_Trigram)


# making a list of keys sorted by index
sorted_keys_HS_trigram=list(trigramsHs)

storeOHE(OHE_HS_Trigram,trigramsHs,'High_School_OHE_Trigram.txt',sorted_keys_HS_trigram)
storeTF(TF_HS_Trigram,trigramsHs,'High_School_TF_Trigram.txt',sorted_keys_HS_trigram)
storeTFIDF(TFIDF_HS_Trigram,trigramsHs,'High_School_TFIDF_Trigram.txt',sorted_keys_HS_trigram)

print('\nOHE_HS Trigram Matrix')

# printing all 3 matrices
for row in OHE_HS_Trigram:
    print(row)
    
print('\nTF_HS Trigram Matrix')
    
for row in TF_HS_Trigram:
    print(row)

print('\nTF-IDF_HS Trigram Matrix')

for row in TFIDF_HS_Trigram:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    
    
""" College """


trigramsClg1 = removeStopwords(clgdf1, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_College_1.txt')
trigramsClg2 = removeStopwords(clgdf2, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_College_2.txt')
trigramsClg3 = removeStopwords(clgdf3, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_College_3.txt')
trigramsClg4 = removeStopwords(clgdf4, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_College_4.txt')
trigramsClg5 = removeStopwords(clgdf5, 'KUNAL_FARMAH_2K17_IT_61_TrigramsRemoved_College_5.txt')

trigramsClg1=countTrigrams(trigramsClg1)
trigramsClg2=countTrigrams(trigramsClg2)
trigramsClg3=countTrigrams(trigramsClg3)
trigramsClg4=countTrigrams(trigramsClg4)
trigramsClg5=countTrigrams(trigramsClg5)
    
topTClg1=[]
topTClg2=[]
topTClg3=[]
topTClg4=[]
topTClg5=[]
topTClg6=[]


for key in trigramsClg1.keys():
    topTClg1.append((key,trigramsClg1[key]))    
for key in trigramsClg2.keys():
    topTClg2.append((key,trigramsClg2[key]))
for key in trigramsClg3.keys():
    topTClg3.append((key,trigramsClg3[key]))
for key in trigramsClg4.keys():
    topTClg4.append((key,trigramsClg4[key]))
for key in trigramsClg5.keys():
    topTClg5.append((key,trigramsClg5[key]))


# finding top 10 keywords with preferance to larger length keywords in case of a tie

topTClg1 = sorted(topTClg1,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTClg1= topTClg1[0:10]

topTClg2 = sorted(topTClg2,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTClg2 = topTClg2[0:10]

topTClg3 = sorted(topTClg3,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTClg3= topTClg3[0:10]

topTClg4 = sorted(topTClg4,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTClg4 = topTClg4[0:10]

topTClg5 = sorted(topTClg5,key=lambda x: (x[1], len(x[0])),reverse=True)    
topTClg5 = topTClg5[0:10]



print('Top 10 Trigrams in College Chat 1 with frequency: '+str(topTClg1))
print('Top 10 Trigrams in College Chat 2 with frequency: '+str(topTClg2))
print('Top 10 Trigrams in College Chat 3 with frequency: '+str(topTClg3))
print('Top 10 Trigrams in College Chat 4 with frequency: '+str(topTClg4))
print('Top 10 Trigrams in College Chat 5 with frequency: '+str(topTClg5))

plot([val[0] for val in topTClg1],[val[1] for val in topTClg1],'College Chat 1 Trigrams','Frequency',True,color='red',width = 0.5)
plot([val[0] for val in topTClg2],[val[1] for val in topTClg2],'College Chat 2 Trigrams','Frequency',True,color='blue',width = 0.5)
plot([val[0] for val in topTClg3],[val[1] for val in topTClg3],'College Chat 3 Trigrams','Frequency',True,color='orange',width = 0.5)
plot([val[0] for val in topTClg4],[val[1] for val in topTClg4],'College Chat 4 Trigrams','Frequency',True,color='green',width = 0.5)
plot([val[0] for val in topTClg5],[val[1] for val in topTClg5],'College Chat 5 Trigrams','Frequency',True,color='purple',width = 0.5)


# assigning an index to all keywords

trigramsClg = assignIndex(topTClg1,topTClg2,topTClg3,topTClg4,topTClg5)
        

        
# calculating IDF for each trigram

IDF_CLG_Trigram = calcIDF(trigramsClg,trigramsClg1,trigramsClg2,trigramsClg3,trigramsClg4,trigramsClg5)

# one hot encoding
OHE_CLG_Trigram = ohe(trigramsClg,trigramsClg1,trigramsClg2,trigramsClg3,trigramsClg4,trigramsClg5)
# term frequency
TF_CLG_Trigram = tf(trigramsClg,trigramsClg1,trigramsClg2,trigramsClg3,trigramsClg4,trigramsClg5)
# term frequency, inverse dense frequency
TFIDF_CLG_Trigram = tfidf(trigramsClg,trigramsClg1,trigramsClg2,trigramsClg3,trigramsClg4,trigramsClg5,IDF_CLG_Trigram)


# making a list of keys sorted by index
sorted_keys_CLG_trigram=list(trigramsClg)

storeOHE(OHE_CLG_Trigram,trigramsClg,'College_OHE_Trigram.txt',sorted_keys_CLG_trigram)
storeTF(TF_CLG_Trigram,trigramsClg,'College_TF_Trigram.txt',sorted_keys_CLG_trigram)
storeTFIDF(TFIDF_CLG_Trigram,trigramsClg,'College_TFIDF_Trigram.txt',sorted_keys_CLG_trigram)

print('\nOHE_CLG Trigram Matrix')

# printing all 3 matrices
for row in OHE_CLG_Trigram:
    print(row)
    
print('\nTF_CLG Trigram Matrix')
    
for row in TF_CLG_Trigram:
    print(row)

print('\nTF-IDF_CLG Trigram Matrix')

for row in TFIDF_CLG_Trigram:
    print('[',end='')
    for ele in row:
        print(round(ele,3),end=', ')
    print(']')
    
    

        
""" Cosine Similarity in High School and College Keywords in BOP """

""" Hs vs CLG """


train = TFIDF_HS_Trigram
test = TFIDF_CLG_Trigram
l1 = len(test[0])
l2 = len(train[0])
# managed different lenghts according to this article
# https://stackoverflow.com/questions/3121217/cosine-similarity-of-vectors-of-different-lengths
l = min(l1,l2)

similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k])
           
        cosine = s/(trainmag*testmag)
        
        cosines.append(cosine)
            
        print('Cosine similarity of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(cosine))
        
    similarity.append(cosines)
    
    
saveMetric(similarity, 'Cosine_Similarity_Trigram_Matrix.txt' ,'Collge Chat ','High School Chat ')

# plotting cosine similarities

plot([1,2,3,4,5],similarity[0],'High School Chats','Trigram Cosine Similarity with College Chat Document 1',False,'blue',0.5)
plot([1,2,3,4,5],similarity[1],'High School Chats','Trigram Cosine Similarity with College Chat Document 2',False,'blue',0.5)
plot([1,2,3,4,5],similarity[2],'High School Chats','Trigram Cosine Similarity with College Chat Document 3',False,'blue',0.5)
plot([1,2,3,4,5],similarity[3],'High School Chats','Trigram Cosine Similarity with College Chat Document 4',False,'blue',0.5)
plot([1,2,3,4,5],similarity[4],'High School Chats','Trigram Cosine Similarity with College Chat Document 5',False,'blue',0.5)

    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = similarity[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()

f = open('Cosine_Similarity_Trigram_Matrix.txt','a')
f.write('\n\n')

print('The College Chat Document - {} is closest to the High School Chat Document - {} with a cosine similarity of {}'.
      format(clg+1,hs+1,maxval),file=f)
f.close()


""" Verifying our result by doing reverse """
""" CLG vs HS """
train = TFIDF_CLG_Trigram
test = TFIDF_HS_Trigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

similarity = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    testmag = magnitude(test[i])
    cosines = []
    for j in range (0,5):
        trainmag = magnitude(train[j])
        s=0
        for k in range(0,l):
           s+=(test[i][k]*train[j][k])
           
        cosine = s/(trainmag*testmag)
        cosines.append(cosine)
        print('Cosine similarity of High_School_Chat{} with College_Chat_{} is '.format(i+1,j+1)+str(cosine))
    similarity.append(cosines)
    

maxval = 0
hs=0
clg=0
    
for i in range (0,5):
    row = similarity[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()
    

print('The High School Chat Document - {} is closest to the College Chat Document- {} with a cosine similarity of {}'.
      format(hs+1,clg+1,maxval))


""" Manhattan Distance """
""" HS Vs CLG """

train = TFIDF_HS_Trigram
test = TFIDF_CLG_Trigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of College_Chat_{} with High_School_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)
    
saveMetric(manhattan, 'Manhattan_Distance_Trigram_Matrix.txt' ,'Collge Chat ','High School Chat ')

# plotting manhattan distance
plot([1,2,3,4,5],manhattan[0],'High School Chats','Trigram Manhattan Distance from College Chat Document 1',False,'blue',0.5)
plot([1,2,3,4,5],manhattan[1],'High School Chats','Trigram Manhattan Distance from College Chat Document 2',False,'blue',0.5)
plot([1,2,3,4,5],manhattan[2],'High School Chats','Trigram Manhattan Distance from College Chat Document 3',False,'blue',0.5)
plot([1,2,3,4,5],manhattan[3],'High School Chats','Trigram Manhattan Distance from College Chat Document 4',False,'blue',0.5)
plot([1,2,3,4,5],manhattan[4],'High School Chats','Trigram Manhattan Distance from College Chat Document 5',False,'blue',0.5)

    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('College Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = j
            clg = i
    print()
f = open('Manhattan_Distance_Trigram_Matrix.txt','a')
f.write('\n\n')

print('The College Chat Document - {} is farthest to the High School Chat Document - {} with a manhattan distance of {}'.
      format(clg+1,hs+1,maxval),file=f)

f.close()

""" Verifying the Result """
""" CLG Vs Hs """

train = TFIDF_CLG_Trigram
test = TFIDF_HS_Trigram
l1 = len(test[0])
l2 = len(train[0])
l = min(l1,l2)

manhattan = []
for i in range(0,5):
    print('Test IDF Vector is :'+str(test[i]) )
    distance = []
    for j in range (0,5):
        s=0
        # difference till common length
        for k in range(0,l):
           s+=(abs(test[i][k]-train[j][k]))
        # the remaining elements
        if(l1>l2):
            for k in range(l,l2):
                s+=test[i][k]
        elif(l2>l1):
             for k in range(l,l1):
                s+=train[j][k]
            
                   
        distance.append(s)
            
        print('Manhattan Distance of High_School_Chat_{} with College_Chat_{} is '.format(i+1,j+1)+str(s))
        
    manhattan.append(distance)
    
    
maxval = 0
hs=0
clg=0
for i in range (0,5):
    row = manhattan[i]
    print('High School Chat {} :'.format(i+1))
    for j in range (0,5):
        print(row[j],end=' ')
        if row[j]>maxval:
            maxval = row[j]
            hs = i
            clg = j
    print()

print('The High School Chat Document - {} is farthest to the College Chat Document - {} with a manhattan distance of {}'.
      format(hs+1,clg+1,maxval))



