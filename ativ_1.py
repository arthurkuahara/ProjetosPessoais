import pandas as pd
import plotly.express as px

df = pd.read_excel('TrainingSet_2023_02_08.xlsx')
df['TG'] = df['HS'] + df['AS']
df = df[df['HS'] >= 0]
df = df[df['AS'] >= 0]
df = df[df['HS'] <= 10]
df = df[df['AS'] <= 10]
df = df[abs(df['GD']) <= 10]
df = df[df['TG'] <= 20]

fig = px.histogram(df, x='HS', nbins=10, title='Gols do time da casa')
fig.show()
fig = px.histogram(df, x='AS', nbins=10, title='Gols do time fora')
fig.show()
fig = px.histogram(df, x='GD', nbins=10, title='Diferença de gols')
fig.show()
fig = px.histogram(df, x='TG', nbins=10, title='Gols totais')
fig.show()

df = pd.read_excel('TrainingSet_2023_02_08.xlsx')
df = df[df['Lge'] == 'ENG1']
df = df[df['Sea'] == '21-22']
df['TG'] = df['HS'] + df['AS']
df = df[df['HS'] >= 0]
df = df[df['AS'] >= 0]
df = df[df['HS'] <= 10]
df = df[df['AS'] <= 10]
df = df[abs(df['GD']) <= 10]
df = df[df['TG'] <= 20]

fig = px.histogram(df, x='HS', nbins=10, title='Gols do time da casa')
fig.show()
fig = px.histogram(df, x='AS', nbins=10, title='Gols do time fora')
fig.show()
fig = px.histogram(df, x='GD', nbins=20, title='Diferença de gols')
fig.show()
fig = px.histogram(df, x='TG', nbins=20, title='Gols totais')
fig.show()

# Não necessariamente podemos indicar nada sobre a qualidade ofensiva da PL, pois a qualidade defensiva da PL também
# é melhor do que a média.

#PL 21-22

teams = {}
for index, row in df.iterrows():
    home_team = row['HT']
    away_team = row['AT']
    home_score = row['HS']
    away_score = row['AS']
    if home_team not in teams:
        teams[home_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[home_team]['games'] += 1
    teams[home_team]['goals_for'] += home_score
    teams[home_team]['goals_against'] += away_score
    if home_score > away_score:
        teams[home_team]['points'] += 3
        teams[home_team]['wins'] += 1
    elif home_score == away_score:
        teams[home_team]['points'] += 1
        teams[home_team]['draws'] += 1
    else:
        teams[home_team]['losses'] += 1
    if away_team not in teams:
        teams[away_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[away_team]['games'] += 1
    teams[away_team]['goals_for'] += away_score
    teams[away_team]['goals_against'] += home_score
    if away_score > home_score:
        teams[away_team]['points'] += 3
        teams[away_team]['wins'] += 1
    elif away_score == home_score:
        teams[away_team]['points'] += 1
        teams[away_team]['draws'] += 1
    else:
        teams[away_team]['losses'] += 1

df_teams = pd.DataFrame.from_dict(teams, orient='index')
df_teams.index.name = 'Team'
df_teams = df_teams.reset_index()
df_teams.columns = ['Team', 'Points', 'Games', 'Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against']
df_teams['Goal Difference'] = df_teams['Goals For'] - df_teams['Goals Against']

df_teams = df_teams.sort_values(['Points', 'Wins', 'Goal Difference', 'Goals For'], ascending=[False, False, False, False]).reset_index(drop = True)

df_teams

# se quiser calcular apenas para a primeira metade, basta rodar o comando 
# df_half = df.head(len(df)//2) antes de executar o código

teams = {}
df_half = df.head(len(df)// 2)
for index, row in df_half.iterrows():
    home_team = row['HT']
    away_team = row['AT']
    home_score = row['HS']
    away_score = row['AS']
    if home_team not in teams:
        teams[home_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[home_team]['games'] += 1
    teams[home_team]['goals_for'] += home_score
    teams[home_team]['goals_against'] += away_score
    if home_score > away_score:
        teams[home_team]['points'] += 3
        teams[home_team]['wins'] += 1
    elif home_score == away_score:
        teams[home_team]['points'] += 1
        teams[home_team]['draws'] += 1
    else:
        teams[home_team]['losses'] += 1
    if away_team not in teams:
        teams[away_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[away_team]['games'] += 1
    teams[away_team]['goals_for'] += away_score
    teams[away_team]['goals_against'] += home_score
    if away_score > home_score:
        teams[away_team]['points'] += 3
        teams[away_team]['wins'] += 1
    elif away_score == home_score:
        teams[away_team]['points'] += 1
        teams[away_team]['draws'] += 1
    else:
        teams[away_team]['losses'] += 1

df_teams = pd.DataFrame.from_dict(teams, orient='index')
df_teams.index.name = 'Team'
df_teams = df_teams.reset_index()
df_teams.columns = ['Team', 'Points', 'Games', 'Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against']
df_teams['Goal Difference'] = df_teams['Goals For'] - df_teams['Goals Against']

df_teams = df_teams.sort_values(['Points', 'Wins', 'Goal Difference', 'Goals For'], ascending=[False, False, False, False]).reset_index(drop = True)
df_teams

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam


epl = df.copy()
epl = epl.rename(columns={'HS': 'HomeGoals', 'AS': 'AwayGoals', 'HT': 'HomeTeam', 'AT': 'AwayTeam'})
epl.head()

epl = epl[:-10]
epl.mean()

goal_model_data = pd.concat([epl[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

home_team = 'Manchester City'
away_team = 'Tottenham Hotspur'

home_score_rate=poisson_model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team,
                                       'home':1},index=[1]))
away_score_rate=poisson_model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team,
                                       'home':0},index=[1]))
print(home_team + ' against ' + away_team + ' expect to score: ' + str(home_score_rate))
print(away_team + ' against ' + home_team + ' expect to score: ' + str(away_score_rate))

#Lets just get a result
home_goals=np.random.poisson(home_score_rate)
away_goals=np.random.poisson(away_score_rate)
print(home_team + ': ' + str(home_goals[0]))
print(away_team + ': '  + str(away_goals[0]))

# Code to caluclate the goals for the match.
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                           'opponent': awayTeam, 'home': 1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                           'opponent': homeTeam, 'home': 0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

#Fill in the matrix
max_goals=5
score_matrix=simulate_match(poisson_model, home_team, away_team,max_goals)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
pos=ax.imshow(score_matrix, extent=[-0.5,max_goals+0.5,-0.5,max_goals+0.5], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of outcome')
plt.xlim((-0.5,5.5))
plt.ylim((-0.5,5.5))
plt.tight_layout()
ax.set_xlabel('Goals scored by ' + away_team)
ax.set_ylabel('Goals scored by ' + home_team)
plt.show()

#Home, draw, away probabilities
homewin=np.sum(np.tril(score_matrix, -1))
draw=np.sum(np.diag(score_matrix))
awaywin=np.sum(np.triu(score_matrix, 1))

# Nesse caso, a função que iremos usar é simulate_match(foot_model, homeTeam, awayTeam, max_goals=10)
# o modelo é foot_model
import random

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                           'opponent': awayTeam, 'home': 1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                           'opponent': homeTeam, 'home': 0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    

    prob_sum0 = sum(team_pred[0])
    probabilities_norm0 = [p / prob_sum0 for p in team_pred[0]]

    prob_sum1 = sum(team_pred[1])
    probabilities_norm1 = [p / prob_sum1 for p in team_pred[1]]
    
    home_score = np.random.choice([0,1,2,3,4,5,6,7,8,9,10], p = probabilities_norm0)
    
    #away_score = team_pred[1].index(max(team_pred[1]))
    
    away_score = np.random.choice([0,1,2,3,4,5,6,7,8,9,10], p = probabilities_norm1)
    
    return home_score, away_score

epl

epl2 = epl.copy()
epl2['SimulatedHG'] = ''
epl2['SimulatedAG'] = ''
epl2['SimulatedTG'] = ''
epl2['SimulatedGD'] = ''

for index, row in epl2.iterrows():
    
    homeTeam = row['HomeTeam']
    awayTeam = row['AwayTeam']
    
    home_score, away_score = simulate_match(poisson_model, homeTeam, awayTeam, max_goals=10)
    epl2.at[index,'SimulatedHG'] = home_score
    epl2.at[index,'SimulatedAG'] = away_score
    epl2.at[index,'SimulatedTG'] = home_score + away_score
    epl2.at[index,'SimulatedGD'] = home_score - away_score
    
    
    
    
teams = {}
for index, row in epl2.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_score = row['SimulatedHG']
    away_score = row['SimulatedAG']
    if home_team not in teams:
        teams[home_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[home_team]['games'] += 1
    teams[home_team]['goals_for'] += home_score
    teams[home_team]['goals_against'] += away_score
    if home_score > away_score:
        teams[home_team]['points'] += 3
        teams[home_team]['wins'] += 1
    elif home_score == away_score:
        teams[home_team]['points'] += 1
        teams[home_team]['draws'] += 1
    else:
        teams[home_team]['losses'] += 1
    if away_team not in teams:
        teams[away_team] = {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0}
    teams[away_team]['games'] += 1
    teams[away_team]['goals_for'] += away_score
    teams[away_team]['goals_against'] += home_score
    if away_score > home_score:
        teams[away_team]['points'] += 3
        teams[away_team]['wins'] += 1
    elif away_score == home_score:
        teams[away_team]['points'] += 1
        teams[away_team]['draws'] += 1
    else:
        teams[away_team]['losses'] += 1

df_teams = pd.DataFrame.from_dict(teams, orient='index')
df_teams.index.name = 'Team'
df_teams = df_teams.reset_index()
df_teams.columns = ['Team', 'Points', 'Games', 'Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against']
df_teams['Goal Difference'] = df_teams['Goals For'] - df_teams['Goals Against']

df_teams = df_teams.sort_values(['Points', 'Wins', 'Goal Difference', 'Goals For'], ascending=[False, False, False, False]).reset_index(drop = True)
df_teams


# Apenas uma das conclusões que podemos ver é que o modelo subestima times realmente bons, como o Manchester City, 
# que possuem banco suficiente para
# jogar todas as competições ao mesmo tempo. O modelo não leva em consideração que os times podem poupar 
# jogadores por exemplo - isso faz com que times com banco ruim (como o próprio Chelsea, em 21-22) tenham vantagem.
