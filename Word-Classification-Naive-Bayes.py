'''
Classify Words Based On Their Positivity Or Negativity
1. Create a training and testing dataset - input sentences mapped onto positive or negative sentiments
2. Preprocess the data
3. Create a frequency dictionary
4. Calculate the conditional probability of each word using laplacian smoothing
5. Calculate the lambda of the ratio of each word
6. Determine the sentiment of the word
'''

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from math import log

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

def get_cond_prob(word,freq_dict):
    cond_prob = []
    for i in range(2):
        num = freq_dict[word][i] + 1
        den = sum([freq_dict[rec][i] for rec in freq_dict]) + len(freq_dict)
        cond_prob.append(num/den)

    return cond_prob

def get_lambda(word, cond_prob):
    return log(cond_prob[word][0]/cond_prob[word][1],10)

def get_sentiment(word,cond_prob):
    if get_lambda(word, cond_prob) > 0:
        return 1
    elif get_lambda(word, cond_prob) < 0:
        return -1
    else:
        return 0

freq_dict = get_freq_dict(train_set.items())
cond_prob = {token:get_cond_prob(token,freq_dict) for token in freq_dict}

positive = []
negative = []
neutral = []

for token in cond_prob:
    sentiment = get_sentiment(token,cond_prob)
    if sentiment == 1:
        positive.append(token)
    elif sentiment == 0:
        neutral.append(token)
    else:
        negative.append(token)

print('Positive words: \n {}\n'.format(positive))
print('Negative words: \n {}\n'.format(negative))
print('Neutral words: \n {}\n'.format(neutral))
