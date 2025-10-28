# TODO Make code framework for bot, I will import the libraries and base code into here by Friday (hopefully). - EK

    ### INSTALLED LIBRARIES ###
pip install kagglehub[pandas-datasets]
import sklearn as sklearn
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
print(df.head())