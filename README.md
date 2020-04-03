# Disaster-Response-System
![disaster](https://upliftconnect.com/wp-content/uploads/2018/02/DISASTER-FEATURE-2.jpg)

### Table of Contents
1. [Installations](#libraries)
2. [Project Motivation](#introduction)
3. [Files](#files)
4. [ETL Pipeline](#ETL)
5. [ML Pipeline](#ML)
6. [Flask Web App](#Flask)
7. [Plot Demos](#demo1)
8. [Classification Demo](#demo2)
9. [Licensing, Authors, and Acknowledgements](#licensing)


## 1. Installations <a name="libraries"></a>
Besides the libraries included in the Anaconda distribution for Python 3.6 the following libraries have been included in this project:
* `nltk` 
* `sqlalchemy` 




In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

## 2.Project Motivation<a name="introduction"></a>

Natural disasters kill on average 60,000 people per year which makes disaster reponse agencies very important. These agencies work closely with people and wait for the information on any disaster occuring.People generally communicate on social platforms disussing their needs and asking for help.The whole idea of the project is keep track of all the relevant information and to identify the people who are in need of help during any disaster by categorizing the tweets or any text messages sent during that critical time. we want a machine learning model which debugs the message and categories its category, which can help that specific disaster responce agencies in providing help.

to automate a machine learning model which can debug the text messages, which 
[Figure 8](https://www.figure-eight.com/) helps companies transform they data by providing human annotators and machine learning to annotate data at all scales.
Disaster response is one of events that greatly benefits from data and machine learning modeling. In this project I propose an approach to social media messages annotation.
NLP allows the extraction of great significance in text, understanding how a model classifies and predicts needed responses in disaster cases provides good understanding of the power of words in functional responses.

In this project I will be using a data set containing messages that were sent during disaster events and build a classifier to identify messages or events in need of attention or relief.
The data cleaning and model building will be using pipelines, automating repetitive steps, and preventing data leakage.

The best performing machine learning model will be deployed as a web app where the user can test their own tentative messages to see how they would be classified with the models I selected and trained. 
Through the web app the user can also consult visualizations of the clean and transformed data.


## 3. Files <a name="files"></a>
### A Figure 8 dataset
Data was downloaded from [Figure 8](https://www.figure-eight.com/dataset/combined-disaster-response-data/).

#### 4. ETL Pipeline <a name="ETL"></a>

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

#### 5. ML Pipeline <a name="ML"></a>

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

#### 6. Flask Web App <a name="Flask"></a>

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python:
data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

To run ML pipeline that trains classifier and saves python 
models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

Run the following command in the app's directory to run your web app 
python run.py

Go to http://0.0.0.0:3001/

##### 7. Plot Demos <a name="demo1"></a>
![Plot demos](https://github.com/sousablde/Disaster-Response-Pipeline/blob/master/Images/plots_demo.gif)

##### 8. Classification Demo <a name="demo2"></a>
![Classification demos](https://github.com/sousablde/Disaster-Response-Pipeline/blob/master/Images/classification_demo.gif)


Notebooks
ETL Pipeline Prep.ipynb - jupyter notebook for data exploration and cleaning
ML Pipeline Preparation - jupyter notebook for model selection and evaluation



## 9. Licensing, Authors, Acknowledgements<a name="licensing"></a>
I am greatly thankful for the incredible challenge provided by Udacity.
