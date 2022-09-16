from flask import Flask
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import leaguestandingsv3


app = Flask(__name__)
CORS(app)

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


# Below is how the elo system assigns the elo to the teams based off performance
# NONE OF THIS IS CURRENTLY IN USE FOR THE API
def elo():
    previous_season = '2013-14'
    picked_season = '2014-15'
    all_games = leaguegamefinder.LeagueGameFinder(league_id_nullable='00', season_nullable=picked_season, season_type_nullable='Regular Season').get_data_frames()[0]
    df_game_data = pd.DataFrame(all_games)
    df_game_data = df_game_data.dropna()
    df_game_data = df_game_data.drop_duplicates(subset=['GAME_ID'], keep='last')
    df_game_data = df_game_data.drop(columns=['SEASON_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA'
        , 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'TEAM_NAME', 'TEAM_ID'])

    df_game_data['LEFT_ELO'] = 0
    df_game_data['RIGHT_ELO'] = 0
    df_game_data = df_game_data.iloc[::-1]

    team_dict = teams.get_teams()
    last_season_team_WL = leaguestandingsv3.LeagueStandingsV3(season=previous_season).get_data_frames()[0]
    last_season_team_WL = pd.DataFrame(last_season_team_WL)
    print('last season WL')
    print(last_season_team_WL[['TeamName', 'WinPCT', 'TeamSlug']])

    df_all_teams = pd.DataFrame(team_dict)
    df_all_teams = df_all_teams.drop(columns=['id', 'nickname', 'city', 'state', 'year_founded'])
    df_all_teams['ELO'] = 0
    df_all_teams['Wins'] = 1
    df_all_teams['Losses'] = 1
    df_all_teams['Win Streak'] = 1
    df_all_teams['Lose Streak'] = 1

    print(df_all_teams)
    print('Simulating season and adjusting elo...')
    for index, row in last_season_team_WL.iterrows():
        for i, team in df_all_teams.iterrows():
            if row.loc['TeamName'] in df_all_teams.loc[i, 'full_name']:
                df_all_teams.loc[i, 'ELO'] = (df_all_teams.loc[i, 'ELO'] + (row.loc['WINS'] * 5)) - (row.loc['LOSSES'] * 5)

    for index, row in df_game_data.iterrows():

        home_or_away = df_game_data.loc[index, 'MATCHUP']

        if df_game_data.loc[index, 'WL'] == 'L':
            losing_team = home_or_away[0:3]
            winning_team = home_or_away[-3:]
            recent_performance = 0

            for i in range(0, len(team_dict)):
                if df_all_teams.loc[i, 'abbreviation'] == losing_team:
                    if df_game_data.loc[index, 'LEFT_ELO'] == 0:

                        if df_all_teams.loc[i, 'Lose Streak'] > 0:
                            recent_performance = 75 * df_all_teams.loc[i, 'Lose Streak']

                        df_game_data.loc[index, 'LEFT_ELO'] = df_all_teams.loc[i, 'ELO'] - recent_performance
                        df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] - 50 - round((35 * ((df_all_teams.loc[i, 'Wins']) / (df_all_teams.loc[i, 'Wins']
                                                                                            + df_all_teams.loc[i, 'Losses']))), 2)
                        df_all_teams.loc[i, 'Losses'] = df_all_teams.loc[i, 'Losses'] + 1
                        df_all_teams.loc[i, 'Lose Streak'] += 1
                        df_all_teams.loc[i, 'Win Streak'] = 0

                elif df_all_teams.loc[i, 'abbreviation'] == winning_team:
                    if df_game_data.loc[index, 'RIGHT_ELO'] == 0:

                        if df_all_teams.loc[i, 'Win Streak'] > 0:
                            recent_performance = 75 * df_all_teams.loc[i, 'Win Streak']

                        df_game_data.loc[index, 'RIGHT_ELO'] = df_all_teams.loc[i, 'ELO'] + recent_performance
                        df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] + 50 + round((35 * ((df_all_teams.loc[i, 'Losses']) / (df_all_teams.loc[i, 'Wins']
                                                                                            + df_all_teams.loc[i, 'Losses']))), 2)
                        df_all_teams.loc[i, 'Wins'] = df_all_teams.loc[i, 'Wins'] + 1
                        df_all_teams.loc[i, 'Win Streak'] += 1
                        df_all_teams.loc[i, 'Lose Streak'] = 0

        else:
            winning_team = home_or_away[0:3]
            losing_team = home_or_away[-3:]
            recent_performance = 0

            for i in range(0, len(team_dict)):
                if df_all_teams.loc[i, 'abbreviation'] == winning_team:
                    if df_game_data.loc[index, 'LEFT_ELO'] == 0:

                        if df_all_teams.loc[i, 'Win Streak'] > 0:
                            recent_performance = 75 * df_all_teams.loc[i, 'Win Streak']

                        df_game_data.loc[index, 'LEFT_ELO'] = df_all_teams.loc[i, 'ELO'] + recent_performance
                        df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] + 50 + round((35 * ((df_all_teams.loc[i, 'Losses']) / (df_all_teams.loc[i, 'Wins']
                                                                                            + df_all_teams.loc[i, 'Losses']))), 2)
                        df_all_teams.loc[i, 'Wins'] = df_all_teams.loc[i, 'Wins'] + 1
                        df_all_teams.loc[i, 'Win Streak'] += 1
                        df_all_teams.loc[i, 'Lose Streak'] = 0

                elif df_all_teams.loc[i, 'abbreviation'] == losing_team:
                    if df_game_data.loc[index, 'RIGHT_ELO'] == 0:

                        if df_all_teams.loc[i, 'Lose Streak'] > 0:
                            recent_performance = 75 * df_all_teams.loc[i, 'Lose Streak']

                        df_game_data.loc[index, 'RIGHT_ELO'] = df_all_teams.loc[i, 'ELO'] - recent_performance
                        df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] - 50 - round((35 * ((df_all_teams.loc[i, 'Wins']) / (df_all_teams.loc[i, 'Wins']
                                                                                            + df_all_teams.loc[i, 'Losses']))), 2)
                        df_all_teams.loc[i, 'Losses'] = df_all_teams.loc[i, 'Losses'] + 1
                        df_all_teams.loc[i, 'Lose Streak'] += 1
                        df_all_teams.loc[i, 'Win Streak'] = 0