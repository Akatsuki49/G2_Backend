import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import numpy as np
import math

#nltk.download('punkt')

def trank(Text):

    def clean(text):
        text = text.lower()
        printable = set(string.printable)
        text = filter(lambda x: x in printable, text)
        text = "".join(list(text))
        return text

    Cleaned_text = clean(Text)
    # print(Cleaned_text)
    text = word_tokenize(Cleaned_text)


    POS_tag = nltk.pos_tag(text)

    # print ("Tokenized Text with POS tags: \n")
    # print (POS_tag)

    wordnet_lemmatizer = WordNetLemmatizer()
    adjective_tags = ['JJ','JJR','JJS']

    lemmatized_text = []

    for word in POS_tag:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
    
    POS_tag = nltk.pos_tag(lemmatized_text)

    stopwords = []

    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])

    punctuations = list(str(string.punctuation))

    stopwords = stopwords + punctuations

    stopword_file = open("long_stopwords.txt", "r")
#Source = https://www.ranks.nl/stopwords

    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = []
    stopwords_plus = stopwords + lots_of_stopwords
    stopwords_plus = set(stopwords_plus)

    processed_text = []
    for word in lemmatized_text:
        if word not in stopwords_plus:
            processed_text.append(word)

    vocabulary = list(set(processed_text))

    vocab_len = len(vocabulary)

    weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

    score = np.zeros((vocab_len),dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0,vocab_len):
        score[i]=1
        for j in range(0,vocab_len):
            if j==i:
                weighted_edge[i][j]=0
            else:
                for window_start in range(0,(len(processed_text)-window_size)):
                    
                    window_end = window_start+window_size
                    
                    window = processed_text[window_start:window_end]
                    
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        
                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])
                        
                        # index_of_x is the absolute position of the xth term in the window 
                        # (counting from 0) 
                        # in the processed_text
                        
                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                            covered_coocurrences.append([index_of_i,index_of_j])
    
    inout = np.zeros((vocab_len),dtype=np.float32)

    for i in range(0,vocab_len):
        for j in range(0,vocab_len):
            inout[i]+=weighted_edge[i][j]

    MAX_ITERATIONS = 50
    d=0.85
    threshold = 0.0001 #convergence threshold

    for iter in range(0,MAX_ITERATIONS):
        prev_score = np.copy(score)
        
        for i in range(0,vocab_len):
            
            summation = 0
            for j in range(0,vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j]/inout[j])*score[j]
                    
            score[i] = (1-d) + d*(summation)
        
        if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
            print("Converging at iteration "+str(iter)+"....")
            break
    
    phrases = []

    phrase = " "
    for word in lemmatized_text:
        
        if word in stopwords_plus:
            if phrase!= " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase+=str(word)
            phrase+=" "
        
    
    unique_phrases = []

    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)
    
    for word in vocabulary:
    #print word
        for phrase in unique_phrases:
            if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
                unique_phrases.remove([word])   
    
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score=0
        keyword = ''
        for word in phrase:
            keyword += str(word)
            keyword += " "
            phrase_score+=score[vocabulary.index(word)]
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())

    sorted_index = np.flip(np.argsort(phrase_scores),0)

    keywords_num = 20

    keywords_list = []

    for i in range(0,keywords_num):
        keywords_list.append(str(keywords[sorted_index[i]]))
    
    keywords_string = ", ".join(keywords_list)

    return keywords_string