from flask import Flask, redirect, url_for, request, render_template
from ixolib import getWordcloud, important_features2HTMLList
app = Flask(__name__)


@app.route('/vacancy/<job_title>')
def hello_name(job_title):

	return render_template('hello.html', jobtitle = job_title, wordcloudpath = getWordcloud(job_title), important_features_list=important_features2HTMLList(job_title))
   
if __name__ == '__main__':
	app.debug=True
	#app.run(host='0.0.0.0',port=1403)
	app.run(host='localhost',port=4000)