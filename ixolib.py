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
    data = json.load(open('job_descriptions_ultimate.json'))
    print "master file contains " + str(len(data)) + " records."

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

    print str(i) + " occurrences are found."            
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
    return imageFilePath

def important_features2HTMLList(job_title):
	html = "<UL>"
	list = jobtitle2properties(jobtitle=job_title)
	for i in list:
		html = html + "<LI>" + str(i) + "</LI>"
	html = html + "</UL>"
	return html

def jobtitle2properties(database_file = 'HR_dataset/HRdatabase.csv', jobtitle = 'Area Sales Manager'):
    # works currently with only a csv formatted as the database_file      
    df = pd.read_csv(database_file)
    # select only those person that excel in the particular given position
    # in this case the area sales manager
    seldf = df[(df['Position'] == jobtitle) \
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
        print len(unique_titles)
        print sorted_joined_list
    # output a list of 3 characteristics per person wherein the persons
    # succesful in the jobtitle-position excel
    return sorted_joined_list


