# TODO Make code framework for bot, I will import the libraries and base code into here by Friday (hopefully). - EK

    ### TEST & EXAMPLE CODE ###
    ### INSTALLED LIBRARIES ###
"""import sklearn as sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import KaggleDatasetAdapter.PANDAS
from KaggleDatasetAdapter.PANDAS import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "kazanova/sentiment140",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

    ### TEST CODE FOR PANDA ###
#ser = pd.Series()
#print("Panda Series: ", ser)

#data = np.array(['a','p','p','l','e'])

#ser = pd.Series(data)
#print("Panda Series:\n ", ser)

    ### TEST CODE FOR SCIKIT ###
df = pd.read_csv('training.1600000.processed.noemoticon.csv.zip', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
print(df.head())"""

# 1. pip install NRClex
# 2. python -m textblob.download_corpora


from nrclex import NRCLex
import nltk
import product_review_ex as pr
import twitter_posts_ex as tp
import customer_survey_ex as cs

#used to avoid the already installed package message
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


def dataAnalysis(data):
  print(data)
  text1 = NRCLex(data)
  topE = text1.top_emotions

  print("\nAnalysis Result")
  print("-----------------------")
  print('{0:15}: {1}'.format("EMOTION","RATING"))
  print("-----------------------")
  for i in range(len(topE)):
    #print(topE[i][0],":\t\t\t",topE[i][1])
    print('{0:15}: {1}'.format(topE[i][0].capitalize(),round(topE[i][1],2)))



def main():
   #Return highest emotions.

  #userData = pr.questionnaire()
  userData = tp.tweet1

  # userData= cs.survey()
  # userData = "Coding Python is a fun experience." # for testing only

  dataAnalysis(userData)
  userData = tp.tweet2
  dataAnalysis(userData)
  userData= tp.tweet3
  dataAnalysis(userData)



if __name__ == "__main__":
  main()




