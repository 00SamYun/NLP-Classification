from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
import numpy as np

train_set = {'In every scene, you are my star, @MichelleObama! Happy birthday, baby!':1,
             '🎂 Happy Birthday to me 🎂':1,
             'America, I’m honored that you have chosen me to lead our great country.':1,
             'We did it, @JoeBiden.':1,
             'Congratulations to the Astronauts that left Earth today. Good choice':1,
             'To lose Gianna is even more heartbreaking to us as parents.':0,
             '[It is with immeasurable grief that we confirm the passing of Chadwick Boseman...]':0,
             'Always in my heart @Harry_Styles . Yours sincerely, Louis':1,
             'teamwork makes the dream work !':1,
             'from the bottom of my heart, i am so so sorry. i dont have words.':0,
             'Today was a sad day :(':0,
             'Watching this made me cry from the bottom of my heart. @titanic':0,
             'I dont agree with what you said.':0,
             'Merry christmas and a happy new year!! #2020':1,
             'Good job and thank you! @nurses @doctors':1,
             'Im sorry I dont speak Spanish :/':0,
             'This year was filled with grief and losses...':0,
             'We hope the year gets better #hope #2020':1,
             'You can do this!':1,
             'Congrats for winning this award! #winner @winner':1
            }

STOPWORDS = stopwords.words('english')
porter = PorterStemmer()

def preprocess(datum):
    tokens = []
    wrds = [x[:-1] if x[-1] in punctuation else x for x in datum.split()]
    for wrd in wrds:
        if wrd.isalpha():
            wrd = wrd.lower()
            if wrd not in STOPWORDS:
                token = porter.stem(wrd)
                tokens.append(token)
    return tokens

def get_freq_dict(data):
    good_tokens = []
    bad_tokens = []
    all_tokens = []
    for inp,out in data:
        tokens = preprocess(inp)
        if out == 1:
            good_tokens += tokens
        else:
            bad_tokens += tokens
        all_tokens += tokens

    freq_dict = {token:[good_tokens.count(token),bad_tokens.count(token)] for token in all_tokens}

    return freq_dict

def get_vector(tokens,freq_dict):
    vector = np.zeros(3)
    vector[0] = 1
    vector[1] = sum([freq_dict[token][0] for token in tokens])
    vector[2] = sum([freq_dict[token][1] for token in tokens])

    return vector

def get_matrix(data):
    matrix_X = np.zeros(((len(train_set)), 3))
    for i in range(len(matrix_X)):
        tokens = preprocess(list(train_set.keys())[i])
        vector = get_vector(tokens, freq_dict)
        matrix_X[i] = vector

    return matrix_X

freq_dict = get_freq_dict(train_set.items())

matrix_X = get_matrix(train_set)

print(matrix_X)
