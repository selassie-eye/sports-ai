import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score 

_width = 31

_szn = '17-18'
_szns = [
            '17-18',
            '16-17',
            '15-16',
            '14-15',
            '13-14',
            '12-13',
            '11-12',
            '10-11',
            '09-10',
            '08-09',
            '07-08',
            '06-07',
            '05-06',
            '04-05',
            '03-04',
            '02-03',
            '01-02',
            '00-01',
            '99-00',
            '98-99',
            '97-98'
        ]
_labels = ['Ht', 'No.', 'Wt', 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', '2P', '2PA',
       '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK',
       'TOV', 'PF', 'PTS', 'FG%', '2P%', '3P%', 'eFG%', 'FT%', 'TS%', 'Attendance',
       'Result']

_teams = {
    'Atlanta Hawks': 'ATL', 
    'Boston Celtics': 'BOS', 
    'Brooklyn Nets': 'BRK',
    'Charlotte Bobcats': 'CBO',
    'Charlotte Hornets': 'CHO', 
    'Chicago Bulls': 'CHI', 
    'Cleveland Cavaliers': 'CLE', 
    'Dallas Mavericks': 'DAL', 
    'Denver Nuggets': 'DEN', 
    'Detroit Pistons': 'DET', 
    'Golden State Warriors': 'GSW', 
    'Houston Rockets': 'HOU', 
    'Indiana Pacers': 'IND', 
    'Los Angeles Clippers': 'LAC', 
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL', 
    'Minnesota Timberwolves': 'MIN',
    'New Jersey Nets': 'NJN',
    'New Orleans Pelicans': 'NOP',
    'New Orleans/Oklahoma City Hornets': 'NOK',
    'New Orleans Hornets': 'NOH',
    'New York Knicks': 'NYK', 
    'Oklahoma City Thunder': 'OKC', 
    'Orlando Magic': 'ORL', 
    'Philadelphia 76ers': 'PHI', 
    'Phoenix Suns': 'PHO', 
    'Portland Trail Blazers': 'POR', 
    'Sacramento Kings': 'SAC', 
    'San Antonio Spurs': 'SAS',
    'Seattle SuperSonics': 'SEA',
    'Toronto Raptors': 'TOR', 
    'Utah Jazz': 'UTA',
    'Vancouver Grizzlies': 'VAN',
    'Washington Wizards': 'WAS'        
}

def split_data(dataset, testSize):
    "Splits dataset into training and test sets. Output shape: (X_train, X_test, y_train, y_test)"
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)
    return X_train, X_test, y_train, y_test

def scale_data(train = None, test = None):
    "Scales the dataset using a StandardScaler. Resizes data that uses a single feature"
    from sklearn.preprocessing import StandardScaler
    
    
    if (train is not None):
        sc_train = StandardScaler()
        if (train.shape[1] <= 1): train = train.reshape(-1, 1)
        train = sc_train.fit_transform(train)
    if (test is not None):
        sc_test = StandardScaler()
        if (test.shape[1] <= 1): test = test.reshape(-1, 1)
        test = sc_test.fit_transform(test)
        
    if (train is None): return test, sc_test
    elif (test is None): return train, sc_train
    elif (test is None and train is None): return None
    else: return train, test, sc_train, sc_test
    
# Testing

#sc = StandardScaler()


#data_orl = bb_team_data('ORL')
#data_gsw = bb_team_data('GSW')
#
#data_combined = pd.DataFrame(
#        np.vstack((data_orl.values, data_gsw.values)), 
#        columns=data_orl.columns
#    )

#data = bb_make_dataset()
#sched_mia = bb_schedule('MIA')
#
#rost_orl = bb_roster_stats('ORL')
#rost_mia = bb_roster_stats('MIA')
#rost_was = bb_roster_stats('WAS')
#
#avg_was = bb_average(rost_was)
#avg_orl = bb_average()
#
#dstat = bb_dstat(rost_orl, rost_mia)

# Team ranking by sum of averages - Data Preprocessing
class Team:
    def __init__(self, name, roster):
        self.name = name
        self.roster = roster.drop(columns=['Tm'])
        self.som = self.roster.iloc[:, 3:].mean().to_frame().transpose()
        
    def dstat(self, opp): return self.som - opp.som
    
def bb_build_szns():
    lgs = {}
    dat = None
    for s in _szns: 
        lgs[s] = bb_build_league(s)
        dtmp = bb_league_data(lgs[s])
#        for i, d in enumerate(dtmp.values):
#            if d != d: dtmp.iloc[i] = 0
        if dat is None: dat = dtmp
        else: dat = bb_merge_data(dat, dtmp)
    return lgs, dat.dropna()

def bb_build_league(szn=_szn):
    uri = '../Data/' + szn
    players = pd.read_csv(uri + '/players.csv')
    schedule = pd.read_csv(uri + '/schedule.csv')
    
    league = {}
    for t in _teams: league[_teams[t]] = Team(_teams[t], players[players.Tm == _teams[t]])
    
    schedule.insert(schedule.shape[1], 'hwin', schedule['PTS.1'] > schedule.PTS)
    league['schedule'] = schedule
    
    return league

def bb_league_data(league={}):
    if 'schedule' in league:
        buf = []
        for t in league['schedule'][['Home/Neutral', 'Visitor/Neutral', 'Attend.', 'hwin']].values:
            if _teams[t[0]] and _teams[t[1]] in league:
                dstat = league[_teams[t[0]]].dstat(league[_teams[t[1]]])
                dstat.insert(dstat.shape[1], 'attn', t[2])
                dstat.insert(dstat.shape[1], 'win', t[3])
                buf.append(dstat)
            
        col = list(buf[0].columns)
        data_r = []
        for g in buf:
            if len(data_r) == 0: data_r = g.values
            else: data_r = np.vstack((data_r, g.values))
            
        return pd.DataFrame(data_r, columns=col)
    else: return None
    
def bb_merge_data(top=pd.DataFrame(), bot=pd.DataFrame()):
    if not top.empty and not bot.empty:
        if np.array_equal(top.columns, bot.columns):
            labels = list(top.columns)
            return pd.DataFrame(np.vstack((top.values, bot.values)), columns=labels)
        else: return top
    else: return top
    
def bb_cross_val(machine, x, y, _cv=100):
    #K-Fold Cross Validation
    cvs = cross_val_score(machine, x, list(y.values), cv=_cv)
    avg = cvs.mean()
    dev = cvs.std()
    return {"avg": avg, "std dev": dev}
        
#nba_1516 = bb_build_league('15-16')
#data_1516 = bb_league_data(nba_1516)
#labels_1516 = list(data_1516.columns)
#
#nba_1617 = bb_build_league('16-17')
#data_1617 = bb_league_data(nba_1617)
#labels_1617 = list(data_1617.columns)
#    
#nba_1718 = bb_build_league()
#data_1718 = bb_league_data(nba_1718)
#labels_1718 = list(data_1718.columns)

nba, data = bb_build_szns()

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

train_x, test_x, train_y, test_y = split_data(data, 1/4)
train_x_sc, test_x_sc, scx_train, scx_test = scale_data(train_x, test_x)

train_y_lda = []
for ty in train_y.values: train_y_lda.append(1 if ty else 0)

# Feature Extraction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
train_x_lda = lda.fit_transform(train_x_sc, train_y_lda)
test_x_lda = lda.transform(test_x_sc)
ex_var = lda.explained_variance_ratio_

# Dimensionality Reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
train_x_pca = pca.fit_transform(train_x_sc)
test_x_pca = pca.transform(test_x_sc)
ex_var = pca.explained_variance_ratio_


# ~~ Algorithm Training ~~

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(train_x_lda, list(train_y.values))
pred_knn = knn.predict(test_x_lda)
acc_knn = len((pred_knn == test_y)[(pred_knn == test_y) == True])/len(pred_knn == test_y)

knn_cvs = bb_cross_val(knn, train_x_lda, train_y)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rfc = RandomForestClassifier(criterion='entropy')
rfc.fit(train_x_lda, list(train_y.values))
pred_rfc = rfc.predict(test_x_lda)
acc_rfc = len((pred_rfc == test_y)[(pred_rfc == test_y) == True])/len(pred_rfc == test_y)

rfc_cvs = bb_cross_val(rfc, train_x_lda, train_y)

# Initializing ANN and  layer size
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import backend

ann = Sequential()
u = ((x.shape[1] if len(x.shape) > 1 else 1) + (y.shape[1] if len(y.shape) > 1 else 1))/2

# Adding input layer, first hidden layer
ann.add(Dense(
    int(u),
    input_dim=int(test_x_lda.shape[1]),
    kernel_initializer='uniform',
    activation='relu'
))

# Adding subsequent layers
ann.add(Dense(
    int(u),
    kernel_initializer='uniform',
    activation='relu'
))

# Adding output layer
ann.add(Dense(
    int(1),
    kernel_initializer='uniform',
    activation='relu'
))

# Compiling the ANN
ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fitting the training data to the ANN
ann.fit(
    train_x_lda, train_y,
    batch_size=10,
    epochs=100
)

prob_ann = ann.predict(test_x_lda)
pred_ann = (y_prob >= 0.5)

acc_v = []
for i, p in enumerate(pred_ann): acc_v.append(p[0] == test_y.values[i])
acc_v = pd.DataFrame(acc_v)

acc_ann = len(acc_v[acc_v[0] == True])/len(acc_v)

ann_cvs = bb_cross_val(ann, train_x_lda, train_y)

# XGBoost

from xgboost import XGBClassifier

# XGBoost Fitting
xgb = XGBClassifier()
xgb.fit(train_x_lda, train_y.values)


# Prediction
pred_xgb = xgb.predict(test_x_lda)
_acc_xgb = []
for i, p in enumerate(pred_xgb): _acc_xgb.append(p == test_y.values[i])
_acc_xgb = pd.DataFrame(_acc_xgb)

acc_xgb = len(_acc_xgb[_acc_xgb[0] == True])/len(_acc_xgb)

# k-Fold Cross Validation
xgb_csv = bb_cross_val(xgb, train_x_lda, train_y)






