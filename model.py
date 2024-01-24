#import model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model():
    #load dataset
    data = pd.read_csv('data2.csv')
    #separate dataset
    X = data[['3A', '7A']]
    y = data['strength']
    #split dataset
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
    #inisiasion model
    model= LinearRegression()
    #train model
    model.fit(X_train, y_train)
    #save dataset
    joblib.dump(model, 'model.pkl')

if __name__=="__main__":
    train_and_save_model()