from flask import Flask
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


app = Flask(__name__)

@app.route("/teamdata/<picked_season>")
def getData(picked_season):
    df_game_data = pd.read_csv('static/csv/traindata' + picked_season + ".csv", sep='\t', encoding='utf-8', index_col=0)
    df_all_teams = pd.read_csv('static/csv/teamelo' + picked_season + ".csv", sep='\t', encoding='utf-8', index_col=0)

    X = df_game_data.drop(columns=['GAME_ID', 'MATCHUP', 'WL'])
    y = df_game_data['WL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, test_size=.3, shuffle=False)
 
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)
    
    # Piechart for win and loss %
    wrong = 1.00 - score
    predictions_percentage = {
        'W': score,
        'L': wrong
    }

    # Scatterplot for wins to ELO
    scatter_dict = dict(zip(df_all_teams['ELO'], df_all_teams['Wins']))

    nested_dict = {
        'predictions': predictions_percentage,
        'scatter_data': scatter_dict
    }
    
    return nested_dict