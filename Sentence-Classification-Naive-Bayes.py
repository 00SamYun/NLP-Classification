from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from math import log

STOPWORDS = stopwords.words('english')
porter = PorterStemmer()

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
             'Watching this made me cry from the bottom of my heart. @titanic':0
            }

test_set = {'I dont agree with what you said.':0,
             'Merry christmas and a happy new year!! #2020':1,
             'Good job and thank you! @nurses @doctors':1,
             'Im sorry I dont speak Spanish :/':0,
             'This year was filled with grief and losses...':0,
             'We hope the year gets better #hope #2020':1,
             'You can do this!':1,
             'Congrats for winning this award! #winner @winner':1
            }

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

def get_cond_prob(word):
    cond_prob = []
    for i in range(2):
        num = freq_dict[word][i] + 1
        den = sum([freq_dict[rec][i] for rec in freq_dict]) + len(freq_dict)
        cond_prob.append(num/den)

    return cond_prob

def get_dct(data):
    get_lambda = lambda token:log(get_cond_prob(token)[0]/get_cond_prob(token)[1],10)

    dct = {token:get_lambda(token) for token in freq_dict}

    return dct

def get_prior(data):
    pos = sum(val == 1 for val in data.values())
    neg = sum(val == 0 for val in data.values())
    try:
        return log(pos/neg,10)
    except:
        print('No negative data')

def test(data):
    categories = {'positive':[],'negative':[],'neutral':[]}
    for datum in data:
        score = 0
        tokens = preprocess(datum)
        for token in tokens:
            if token in lambda_dict:
                score += lambda_dict[token]
        score += log_prior

        if score > 0:
            categories['positive'].append(datum)
        elif score < 0:
            categories['negative'].append(datum)
        else:
            categories['neutral'].append(datum)

    return categories

freq_dict = get_freq_dict(train_set.items())
lambda_dict = get_dct(train_set.items())
log_prior = get_prior(train_set)

prediction = test(test_set)

for k,v in prediction.items():
    print(k)
    print(v, '\n')
