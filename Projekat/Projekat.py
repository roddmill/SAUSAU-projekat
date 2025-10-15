# BIBLIOTEKE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

# POCETNO PRETPROCESIRANJE PODATAKA

# Iz dalje analize izbacujemo url, unikatan je za svaki clanak i ocigledno je da nece imati uticaja na dalji tok ucenja
podaci = pd.read_csv("OnlineNewsPopularity.csv").drop(columns=["url"])

# Uklanjanje vodećih i pratecih razmaka u imenima kolona
podaci.columns = podaci.columns.str.strip()

# Provera da li imamo "rupe" u nekoj od kolona
print("Provera rupa")
print(podaci.isnull().sum())
print("---------------------------------")
# Rezultat: nemamo!

# Kolone sa procentualnim udelom (0–1)
udeo_kolone = [
    'nuniquetokens', 'nnonstopwords', 'nnonstopuniquetokens',
    'globalsubjectivity', 'globalsentimentpolarity',
    'globalratepositivewords', 'globalratenegativewords',
    'ratepositivewords', 'ratenegativewords',
    'avgpositivepolarity', 'minpositivepolarity', 'maxpositivepolarity',
    'avgnegativepolarity', 'minnegativepolarity', 'maxnegativepolarity',
    'titlesubjectivity', 'titlesentimentpolarity', 'abstitlesubjectivity', 'abstitlesentimentpolarity',
    'LDA00', 'LDA01', 'LDA02', 'LDA03', 'LDA04'
]

# Kolone koje su celobrojne
broj_kolone = [
    'numhrefs', 'numselfhrefs', 'numimgs', 'numvideos', 'numkeywords',
    'shares', 'selfreferenceminshares', 'selfreferencemaxshares', 'selfreferenceavgsharess',
    'ntokenstitle', 'ntokenscontent', 'kwminmin', 'kwmaxmin', 'kwavgmin',
    'kwminmax', 'kwmaxmax', 'kwavgmax', 'kwminavg', 'kwmaxavg', 'kwavgavg', 'timedelta'
]

# Kolone koje su binarne (0 ili 1)
binarne_kolone = [
    'datachannelislifestyle', 'datachannelisentertainment', 'datachannelisbus',
    'datachannelissocmed', 'datachannelistech', 'datachannelisworld',
    'weekdayismonday', 'weekdayistuesday', 'weekdayiswednesday', 'weekdayisthursday',
    'weekdayisfriday', 'weekdayissaturday', 'weekdayisunday', 'isweekend'
]

# Obrada nepostojecih vrednsti - Popunjavamo sa medijanom
podaci = podaci.fillna(podaci.median())

podaci['shares'] = np.log1p(podaci['shares'])

# Provera validnosti
udeo_neispravni = {}
for i in udeo_kolone:
    if i in podaci.columns:
        n = podaci[(podaci[i] < 0) | (podaci[i] > 1)].shape[0]
        if n > 0:
            udeo_neispravni[i] = n

broj_neispravni = {}
for kolona in broj_kolone:
    if kolona in podaci.columns:
        # Oznaci vrednosti ciji rezultat pri deljenju sa 1 nije 0
        neispravni = podaci[kolona][~(podaci[kolona] % 1 == 0)]
        if not neispravni.empty:
            broj_neispravni[kolona] = len(neispravni)

binarni_neispravni = {}
for i in binarne_kolone:
    if i in podaci.columns:
        n = podaci[~podaci[i].isin([0, 1])].shape[0]
        if n > 0:
            binarni_neispravni[i] = n

print("Provera gresaka u datasetu")
print("Nevalidni - udeo:", udeo_neispravni)
print("Nevalidni - broj:", broj_neispravni)
print("Nevalidni - binarne:", binarni_neispravni)
print("Nedostajuće vrednosti (NaN):")
print("---------------------------------")

podaci = podaci.fillna(podaci.median())

print(f"Broj preostalih redova posle uklanjanja outlier-a: {len(podaci)}")

#EKSPLORATIVNA ANALIZA SKUPA

# Necitljiva korelaciona matrica
sns.heatmap(podaci.corr(), annot=True)
plt.rcParams['figure.figsize'] = (16,12)
plt.show()

# Matricu cemo razdeliti na vise citljivih blokova

# Definisacemo da je broj atributa po bloku 15 jer daje poprilicno optimalnu citljivost
n = 15
kolone = podaci.columns.tolist()

# Petlja kroz sve atribute podeljene u grupe
for i in range(0, len(kolone), n):
    grupa = kolone[i:i + n]
    plt.figure(figsize=(8, 6))
    sns.heatmap(podaci[grupa].corr(), annot=True, fmt = ".1f", cmap='coolwarm', annot_kws={"size":6})
    plt.title(f"Heatmap korelacija: {grupa[0]} - {grupa[-1]}")
    plt.tight_layout()
    plt.show()

#Uklanjanje visoke korelacije, posmatramo samo trougao iznad dijagonale
prag = 0.8
korelacije = podaci.corr().abs()
gornji = korelacije.where(np.triu(np.ones(korelacije.shape), k=1).astype(bool))
za_brisanje = [col for col in gornji.columns if any(gornji[col] > prag)]
print("Atributi koji se uklanjaju zbog visoke korelacije:", za_brisanje)
podaci = podaci.drop(columns=za_brisanje)

# Uklanjanje ekstremnih vrednosti pomocu z-score-a
z_skorovi = np.abs(stats.zscore(podaci.select_dtypes(include=[np.number])))
podaci = podaci[(z_skorovi < 3).all(axis=1)]

# Skaliranje podataka tako da ne bivaju zanemareni prilikom treniranja, zadrzavamo komponente koje sadrze 95% varijanse
numericke_kolone = podaci.select_dtypes(include=['number']).columns.tolist()
numericke_kolone.remove('shares')
scaler = StandardScaler()

scaled = scaler.fit_transform(podaci[numericke_kolone])
pca = PCA(n_components=0.95)
pca_df = pd.DataFrame(pca.fit_transform(scaled), columns=[f'PC{i+1}' for i in range(pca.n_components_)])
pca_df['shares'] = podaci['shares'].values

# Pregled korelacije preostalih atributa

cols = podaci.columns.tolist()

for i in range(0, len(cols), n):
    block = cols[i:i+n]
    plt.figure(figsize=(10, 8))
    sns.heatmap(podaci[block].corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size":7})
    plt.title(f"Korelaciona matrica atributa: {block[0]} - {block[-1]}")
    plt.tight_layout()
    plt.show()

# ODABIR I TRENIRANJE MODELA

# Sablon za modele

X = pca_df.drop(columns=['shares'])
y = pca_df['shares']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linearna regresija - 0.1093

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R2 skor modela linearne regresije: {r2:.4f}")

# Ridge regresija - 0.1094

ridge = Ridge()

param_grid = {'alpha': np.logspace(-4, 4, 50)}

# Cross validacija, nece biti spominjano u ostalim modelima
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolja vrednost alpha: {grid_search.best_params_['alpha']}")
print(f"R2 skor najboljeg Ridge modela: {r2:.4f}")

# Lasso regresija - 0.1095

lasso = Lasso(max_iter=10000)

param_grid = {'alpha': np.logspace(-4, 1, 50)}

grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolja vrednost alpha za Lasso: {grid_search.best_params_['alpha']}")
print(f"R2 skor najboljeg Lasso modela: {r2:.4f}")

# Decision tree regresija - 0.0619

dt = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolji hiperparametri Decision Tree: {grid_search.best_params_}")
print(f"R2 skor najboljeg Decision Tree modela: {r2:.4f}")

# Random forrest regresija - 0.1073

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolji hiperparametri Random Forest: {grid_search.best_params_}")
print(f"R2 skor najboljeg Random Forest modela: {r2:.4f}")

feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)
print("Top 10 najvažnijih atributa:")
print(top_features)

# Gradient Boost - 0.1116

gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_gbr = grid_search.best_estimator_
y_pred = best_gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolji hiperparametri Gradient Boosting: {grid_search.best_params_}")
print(f"R2 skor najboljeg Gradient Boosting modela: {r2:.4f}")

#XGBoost - 0.1219

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]    
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Najbolji hiperparametri XGBoost: {grid_search.best_params_}")
print(f"R2 skor najboljeg XGBoost modela: {r2:.4f}")


# ANALIZA REZULTATA PREDIKCIJE

# Trazenje 10 najboljih atributa koristeci Random Forrest regresiju
X_rf = podaci.drop(columns=['shares'])
y_rf = podaci['shares']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)

importances = rf.feature_importances_
features = X_rf.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

top_n = 10
top_features = importance_df.head(top_n)

# Plot
plt.figure(figsize=(10, 6))
top_features.plot(kind='bar')
plt.title('Top 10 najvažnijih atributa')
plt.ylabel('Važnost')
plt.show()

# Treniranje najgoreg i najboljeg modela na trening skupu koji sacunjavaju samo najbolji atributi

# Sablon
top_features_list = top_features['Feature'].tolist()

X_top = podaci[top_features_list]
y_top = podaci['shares']

X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y_top, test_size=0.2, random_state=42)

# Decision tree - -0.8747
dt_top = DecisionTreeRegressor(random_state=42)
dt_top.fit(X_train_top, y_train_top)

y_pred_top = dt_top.predict(X_test_top)
r2_top = r2_score(y_test_top, y_pred_top)

print(f"R2 skor Decision Tree modela na top {top_n} atributa: {r2_top:.4f}")

# XG Boost - 0.1193

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, seed=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train_top, y_train_top)

best_xgb = grid.best_estimator_

y_pred = best_xgb.predict(X_test_top)
r2 = r2_score(y_test_top, y_pred)

print(f"Najbolji hiperparametri: {grid.best_params_}")
print(f"R2 score na odabranim atributima: {r2:.4f}")

