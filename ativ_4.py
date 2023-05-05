_spadl_cfg = {
    'length': 105,
    'width': 68,
    'penalty_box_length': 16.5,
    'penalty_box_width': 40.3,
    'six_yard_box_length': 5.5,
    'six_yard_box_width': 18.3,
    'goal_widht': 7.32,
    'penalty_spot_distance': 11,
    'goal_width': 7.3,
    'goal_length': 2,
    'origin_x': 0,
    'origin_y': 0,
    'circle_radius': 9.15,
}


def goalangle(actions, cfg=_spadl_cfg):
    dx = cfg['length'] - actions['start_x']
    dy = cfg['width'] / 2 - actions['start_y']
    angledf = pd.DataFrame()
    angledf['shot_angle'] = np.arctan(
        cfg['goal_width']
        * dx
        / (dx ** 2 + dy ** 2 - (cfg['goal_width'] / 2) ** 2)
    )
    angledf.loc[angledf['shot_angle'] < 0, 'shot_angle'] += np.pi
    angledf.loc[(actions['start_x'] >= cfg['length']), 'shot_angle'] = 0
    # Ball is on the goal line
    angledf.loc[
        (actions['start_x'] == cfg['length'])
        & (
            actions['start_y'].between(
                cfg['width'] / 2 - cfg['goal_width'] / 2,
                cfg['width'] / 2 + cfg['goal_width'] / 2,
            )
        ),
        'shot_angle',
    ] = np.pi
    return angledf
  
 


shots = spadl[spadl['type_name'] == 'shot'].reset_index(drop = True)
shots['angle'] = goalangle(shots)

shots



#shots = spadl[spadl['type_name'] == 'shot']

# Vamos criar uma coluna contendo a angulação entre o começo e o final do chute, para ver se o jogador tinha ângulo
# para o arremate

# Também iremos usar a qualidade do chute medida nas colunas bodypart_name, a distância do gol (medida pelo start_x 
# e pelo start_y) - considerando o gol como estando na posição x = 100, y = 34

def distance_between_points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

shots['distance'] = shots.apply(lambda row: distance_between_points(row['start_x'], row['start_y'], 100, 34), axis=1)
shots['goal'] = shots['result_name'].apply(lambda x: 1 if x == 'success' else 0)
# Também iremos usar o action_id como métrica da qualidade do arremate, se foi uma cabeçada, uma cobrança de falta, etc.

# Portanto, as colunas que usaremos como features são a parte do corpo utilizada, o tipo de ação, o ângulo do chute
# e a distância do gol como features no nosso modelo.

shots


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = shots.copy()

df['action_id'] = df['action_id'].astype(str)
# Selecionar apenas as colunas necessárias
df = df[['angle', 'distance', 'action_id', 'bodypart_id', 'goal']]

# Separar em dados de treino e teste
X = df[['angle', 'distance', 'action_id', 'bodypart_id']]
y = df['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Treinar o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Calcular e imprimir as acurácias do modelo
print('Acurácia no conjunto de treino:', model.score(X_train, y_train))
print('Acurácia no conjunto de teste:', model.score(X_test, y_test))



shots['xG'] = model.predict_proba(shots[['angle', 'distance', 'action_id', 'bodypart_id']])[:, 1]
shots_grouped = shots.groupby('player_name').agg({'goal': 'sum', 'xG': 'sum'})
top_10_xG = shots_grouped.sort_values('xG', ascending=False).head(10)

shots['diff'] = shots['goal'] - shots['xG']
diff_grouped = shots.groupby('player_name')['diff'].sum()

top_10_diff = diff_grouped.abs().sort_values(ascending=False).head(10)


chunk_size = 1000
start_index = 3000
end_index = 4000

chunk_df = spadl.iloc[start_index:end_index].copy()
xTModel = xt.ExpectedThreat(l=25, w=16)
xTModel.fit(chunk_df)


prog_actions = xt.get_successful_move_actions(spadl)
prog_actions["xT_value"] = xTModel.rate(prog_actions)

actions_grouped = prog_actions.groupby('player_name')['xT_value'].sum()
actions_grouped.sort_values(ascending = False).head(10)
