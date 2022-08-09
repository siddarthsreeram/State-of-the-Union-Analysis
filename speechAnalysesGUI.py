#Written by Siddarth Sreeram
#Background Information/Data from:
#Presidential Ratings: https://global.nielsen.com/news-center/2022/audience-ratings-for-the-2022-presidential-state-of-the-union/
#Speeches: https://www.kaggle.com/datasets/rtatman/state-of-the-union-corpus-1989-2017
#Note: set current directory to that of file folder using "cd [directory]" command
#Tested on Python 3.10.5

# import necessary packages
from cProfile import label
from ctypes import Structure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from sklearn.metrics import r2_score
import scipy as spi
from tkinter import simpledialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGraphicsView, QLabel, QPlainTextEdit, QPushButton, QTextBrowser
from PyQt5 import uic
import sys
import seaborn as sn
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import nest_asyncio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import textstat
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import wordcloud

nest_asyncio.apply()


#use pandas to read CSV data - **Adjust to location of CSV on one's computer**
df=pd.read_csv("presidentialRatings.csv")

#create an np array of the TV ratings
ratingsData = pd.read_csv("presidentialRatings.csv", encoding='utf-8')
ratingsList = ratingsData["Rating"].to_list()
scaledRatingsList=[value -(min(ratingsList)-1) for value in ratingsList]
sizeScaledRatingsList=[value *150 for value in scaledRatingsList]

yearsList=[]
#populates x list with values of years not including inagural years
for i in range (2009-2001):
    yearsList.append(i+2001)
for i in range (2001-1993):
    yearsList.append(i+1993)
for i in range (2017-2009):
    yearsList.append(i+2009)
for i in range (2021-2017):
    yearsList.append(i+2017)
yearsList.remove(1993)
yearsList.remove(2001)
yearsList.remove(2009)
yearsList.remove(2017)

#creates list needed for data
easeList=[]
sentimentList=[]

#creates functions for removal of file details
def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

#creates word cloud creation function
def wordCloudCmd(string1, self):
    string1=self.box.currentText()
    file1=str(string1.split(" ")[0])
    file2="_"+str(rchop(remove_prefix((string1.split(" "))[1],"("),")"))+".txt"
    file3="PresTxtList\\"+file1+file2

    data2 = (open((file3), "r", encoding='utf-8')).read()
    comment_words = ''
    stopwords = set(STOPWORDS)

    # split the value
    tokens = data2.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

    #creates the WordCloud
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color='white',
        stopwords=stopwords,
        min_font_size=10).generate(comment_words)
    
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

#main PyQt5 window
class UI(QMainWindow): 
    def __init__(self):
        super(UI, self).__init__() 

        #Load the UI file
        uic.loadUi("speechAnalysesGUI2.ui", self)

        #define widgets
        self.label = self.findChild(QLabel, "label")
        self.backgroundTxt = self.findChild(QTextBrowser, "textBrowser")
        self.button1 = self.findChild(QPushButton, "pushButton")
        self.button2 = self.findChild(QPushButton, "pushButton_2")
        self.box = self.findChild(QComboBox, "comboBox")

        #iterate over files
        for filename in os.listdir('PresTxtList'):
            f = os.path.join('PresTxtList', filename)
            # checking if it is a file and adding reading ease value to list
            if os.path.isfile(f):
                sentence = open(f, mode="r", encoding="utf-8").read()
                easeList.append(textstat.flesch_reading_ease(sentence))
                def sentiment_scores(sentence):

                    # Create a SentimentIntensityAnalyzer object.
                    sid_obj = SentimentIntensityAnalyzer()

                    # polarity_scores method of SentimentIntensityAnalyzer
                    # object gives a sentiment dictionary.
                    # which contains pos, neg, neu, and compound scores.
                    sentiment_dict = sid_obj.polarity_scores(sentence)
                    sentimentList.append(sentiment_dict['pos'] * 100)

                    f2 = rchop((remove_prefix(f, ("PresTxtList\\"))), ".txt")
                    flist = f2.split("_")

                    self.box.addItem(str(flist[0]+ " ("+str(flist[1]))+")")
                    
                sentiment_scores(sentence)

        #button works
        self.button1.clicked.connect(self.ease)
        self.button2.clicked.connect(self.sentiment)
        self.box.currentTextChanged.connect(self.text_changed)

        #show app
        self.setStyleSheet("background-color: dodgerblue;")
        self.show()

    #checks for change of choice in drop down box
    def text_changed(self, s):
        wordCloudCmd(self.box.currentText(), self)
        
    #function for reading ease button
    def ease(self):
        #create the bubble graph
        plt.figure()
        plt.scatter(yearsList, easeList, s=sizeScaledRatingsList, c=scaledRatingsList, cmap="Reds", alpha=0.5)
        plt.xlabel("Year")
        plt.ylabel("Flesch Reading Ease")
        plt.title("State of the Union Speeches' Flesch Reading Ease Over Time as it relates to TV Rating")
        plt.colorbar(label="TV Rating (values translated such that \nthe smallest rating value was set to 1)", orientation="horizontal")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    #function for positivity sentiment button
    def sentiment(self):
        #create the bubble graph
        plt.figure()
        labels = ['Tiny', 'Small', 'Medium', 'Large', 'Huge']
        plt.scatter(yearsList, sentimentList, s=sizeScaledRatingsList, c=scaledRatingsList, cmap="Reds", alpha=0.5, label=labels[i])
        plt.xlabel("Year")
        plt.ylabel("Positivity Sentiment (%)")
        plt.title("State of the Union Speeches' Positivity Sentiment (%) Over Time as it relates to TV Rating")
        plt.colorbar(label="TV Rating (values translated such that \nthe smallest rating value was set to 1)", orientation="horizontal")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

#initialize the app

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
