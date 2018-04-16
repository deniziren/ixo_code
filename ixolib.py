import json
from wordcloud import WordCloud
import numpy as np
import pandas as pd

def removeAdditionalStopwords(text):
    text_file = open("additional_stopwords.txt", "r")
    stopwords = text_file.readlines()
    for a in stopwords:
        text = text.replace(a.lower().rstrip(), '')
    return text


def isRelevant(jobTitle, jobTitleInCorpus):
    relevant = False
    #if fuzz.partial_ratio(jobTitle, jobTitleInCorpus) > 64:
    if jobTitle.lower() in jobTitleInCorpus.lower():
        relevant = True
    return relevant

def getWordcloud(jobTitle):
    # load the master JSON file.
    data = json.load(open('job_descriptions_ultimate/job_descriptions_ultimate.json'))
    print("master file contains " + str(len(data)) + " records.")

    textCloud = ""
    # titleToSearch = "accountant"
    titleToSearch = jobTitle
    titleWords = titleToSearch.split(' ') # we will remove these words from the corpus because we don't want them to appear in the word cloud.

    i = 0
    for v in data:
        if isRelevant(titleToSearch, v['title']):
            desc = v['description'].lower()
            for a in titleWords:
                desc = desc.replace(a.lower(), '')
            desc = removeAdditionalStopwords(desc.lower()) # we may want to remove standard recruiting terminology
            textCloud = textCloud + desc
            i = i + 1

    print(str(i) + " occurrences are found.")            
    # print text
    # Generate a word cloud image
    if len(textCloud) < 1:
        imageFilePath="not_enough_data.png"
    else:
        wordcloud = WordCloud(background_color='white').generate(textCloud)
		 # Display the generated image:
		 # the matplotlib way:
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
	
        imageFilePath = "clouds/wcloud_" + jobTitle.replace(' ', '_') + ".png"
        plt.savefig("static/" + imageFilePath)
		#plt.show()
    return imageFilePath, str(i)



def important_features2HTMLList(job_title):
	html = "<UL>"
	list = jobtitle2properties(jobtitle=job_title)
	for i in list:
		html = html + "<LI>" + str(i) + "</LI>"
	html = html + "</UL>"
	return html

def jobtitle2properties(database_file = 'HR_dataset/HRdatabase.csv', jobtitle = 'Area Sales Manager'):
    '''
    given a job title and
    a database with assesments of employees 
    finds out the properties of those employees that have functioned well 
    in the position of job title
    
    function is dependent on the database structure
    can be improved by giving as input the columns 
    where the assesment scores are relevant
    
    laura.astola@accenture.com
    '''
    # works currently with only a csv formatted as the database_file      
    df = pd.read_csv(database_file)
    # select only those person that excel in the particular given position
    # in this case the area sales manager
    seldf = df[(df['Position'].str.lower() == jobtitle) \
       & ((df['Performance Score']=='Exceeds') | (df['Performance Score']=='Exceptional'))]
    # select the properties in which the persons score highest
    important_properties = []
    for i in seldf.index:
        important_properties.append(list(df.iloc[i,6:].sort_values(ascending = False).index)[0:3])
    if len(important_properties)==0:
        sorted_joined_list = ['sorry, not enough info in the database']
    # properties that are common to these persons probably deserve to be mentioned first
    else:
        ip = list(np.concatenate(important_properties))
        sorted_joined_list = sorted(ip,key=ip.count,reverse=True)
        unique_titles = []
        for i,w in enumerate(sorted_joined_list):
            if (w not in unique_titles) and len(unique_titles) <6:
                unique_titles.append(w)
        sorted_joined_list = unique_titles
        print(len(unique_titles))
        print(sorted_joined_list)
    # output a list of 3 characteristics per person wherein the persons
    # succesful in the jobtitle-position excel
    return sorted_joined_list

def mbti_classify(textfile, model_EI, model_SN, model_TF, model_PJ, max_len = 500):
    '''
    GIVEN A TEXT FILE AND THE 4 TRAINED CLASSIFIER MODELS, 
    ASSIGNS THE TEXT FILE 
    A PERSONALITY Myer-Briggs trait indicator.
    laura.astola@accenture.com
    '''
    from keras.models import load_model
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import np_utils
    from keras.models import Model
    MAX_NB_WORDS=20000
    
    texts = []
    with open(textfile) as f:
        t = f.read()
        texts.append(t)   
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789', lower=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)    
    data = pad_sequences(sequences, maxlen=max_len)
    classification_result = []
    scores = []
    model = load_model(model_EI)
    score = model.predict(data)[0][1]
    scores.append(score)
    if score > 0.5:
        classification_result.append('E')
    else:
        classification_result.append('I')

    model = load_model(model_SN)
    score = model.predict(data)[0][1]
    scores.append(score)
    if score > 0.5:
        classification_result.append('S')
    else:
        classification_result.append('N')
        
    model = load_model(model_TF)
    score = model.predict(data)[0][1]
    scores.append(score)
    if score > 0.5:
        classification_result.append('T')
    else:
        classification_result.append('F')
        
    model = load_model(model_PJ)
    score = model.predict(data)[0][1]
    scores.append(score)
    if score > 0.5:
        classification_result.append('P')
    else:
        classification_result.append('J')
        
    return scores, ''.join(classification_result)

def plot_mbti(scores):
    '''
    input: a list of 4 scores,
    for example: scores = [.9,.3,.2,.1]
    extrovert (1) -introvert (0),
    sensing (1)-intuitive (0),
    thinking (1)-feeling (0),
    perceiving (1)-judging (0).
    output: mbti-visualization  
    laura.astola@accenture.com
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import cm
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
    #fig=plt.figure(figsize=(10,10))plot_mbti(scores):
    gs = gridspec.GridSpec(6,6)
    ax1 = plt.subplot(gs[0,0:5])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlim([-1,16])
    plt.ylim([-0.5,0.5])
    ax2 = plt.subplot(gs[1:6,0:5])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlim([-0.2,1.2])
    plt.ylim([-0.2,1.2])
    ax3 = plt.subplot(gs[1:6,5])

    ax2.set_xlabel('Introvert $\leftrightarrow $  Extravert',fontsize=25)
    ax2.set_ylabel('Intuitive $\leftrightarrow $ Sensing',fontsize=25)
    
    ax2.add_artist(Ellipse((scores[0], scores[1]), scores[2]*.3+.05, (1-scores[2])*.3+0.05, facecolor= cm.PiYG(scores[3]), alpha = 1.0, edgecolor='k'))
    cmap = cm.PiYG
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                    orientation='vertical',ticks=[])
    cb1.set_label('Thinking $\leftrightarrow $  Feeling',fontsize=25, rotation=270, labelpad=20)
    
    ax1.set_xlabel('Judging $\leftrightarrow $  Perceiving',fontsize=25)
    ax1.xaxis.set_label_position('top') 

    for j in np.linspace(0.0, 1., num=15): 
        ax1.add_artist(Ellipse((j*15, 0), j*.5+.05, (1-j)*.2+0.05, facecolor= 'none', edgecolor='k'))
    plt.show()
    
    return ax2


def add_mbti(baseax, scores):
    '''add a list of 4 scores,
    to an existing plot
    for example: scores = [.6,.1,.5,.8]
    laura.astola@accenture.com
     '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import cm
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
   
    baseax.add_artist(Ellipse((scores[0], scores[1]), scores[2]*.3+.05, (1-scores[2])*.3+0.05, facecolor= cm.PiYG(scores[3]), alpha = 1.0, edgecolor='k'))
   
    plt.show()
    
    return

def top_five_skills(csv_file,list_of_columns,list_of_names):
    '''
    given a database (csv_file)
    a team (list_of_names) and 
    the columns with numeric assesments of skills (list_of_columns)
    computes the top five skills that the memebers of the team shares
    laura.astola@accenture.com
    '''
    import pandas as pd
    df = pd.read_csv(csv_file)
    bestskills=[]
    for person in list_of_names:
        features=df[df['Employee Name']==person][list_of_columns]
        skills = [x for x in list_of_columns if list(np.argsort(features)[x]<10)[0]]
        bestskills.append(skills)
    flatlist = [x for sublist in bestskills for x in sublist]
    skills = set(flatlist)
    frequencylist=[]
    for s in skills:
        counter = 0
        for list in bestskills:
            if s in list:
                counter +=1        
        frequencylist.append([s,counter])
    frequencylist.sort(key=lambda x: x[1], reverse = True)
    return [x[0] for x in frequencylist[:5]]
        
    
    
