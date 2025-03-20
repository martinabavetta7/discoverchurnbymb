import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE, SelectFromModel
from scipy.stats import chi2_contingency
import plotly.express as px
# 📂 Caricamento del dataset
file_path = r"C:\Users\HP\Downloads\Churn_Modelling.csv"
df = pd.read_csv(file_path)

# 🔍 Esplorazione dei dati
print(df.head()) #Mostrare le prime 5 righe del mio dataset
print(df.isnull().sum())# Controllo valori nulli
print(df.info()) #Informazioni sul dataset
df[df.duplicated()]
label_encoder=LabelEncoder()
# 🚀 Preprocessing
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-Hot Encoding per Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Encoding per Gender
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# 📊 Definizione delle variabili
X = df.drop('Exited', axis=1)
y = df['Exited']


# 🎯 Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔢 Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🌲 Modello Random Forest
rf= RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
conf_matrix=confusion_matrix(y_test, y_pred_rf)
class_report=classification_report(y_test, y_pred_rf)
accuracy=accuracy_score(y_test, y_pred_rf)
print(conf_matrix)
print(class_report)
print(accuracy)

importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]
names=[X.columns[i] for i in indices]
plt.figure(figsize=(10,6))
plt.title("Feature Importance Random Forest")
plt.barh(range (X.shape[1]), importances[indices])
plt.yticks(range(X.shape[1]), names)
plt.show()

log_reg=LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg=log_reg.predict(X_test)
conf_matrix_log_reg=confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg=classification_report(y_test, y_pred_log_reg)
accuracy_log_reg=accuracy_score(y_test, y_pred_log_reg)
print(conf_matrix_log_reg, class_report_log_reg, accuracy_log_reg)

svm_model=SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm=svm_model.predict(X_test)
conf_matrix_svm=confusion_matrix(y_test,y_pred_svm)
class_report_svm=classification_report(y_test,y_pred_svm)
accuracy_svm=accuracy_score(y_test, y_pred_svm)
print(conf_matrix_svm, class_report_svm, accuracy_svm)

knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn=knn_model.predict(X_test)
conf_matrix_knn=confusion_matrix(y_test, y_pred_knn)
class_report_knn=classification_report(y_test,y_pred_knn)
accuracy_knn=accuracy_score(y_test, y_pred_knn)
print(conf_matrix_knn, class_report_knn, accuracy_knn)

gbm_model=GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)
y_pred_gbm=gbm_model.predict(X_test)
conf_matrix_gbm=confusion_matrix(y_test, y_pred_gbm)
class_report_gbm=classification_report(y_test, y_pred_gbm)
accuracy_gbm=accuracy_score(y_test, y_pred_gbm)
print(conf_matrix_gbm,class_report_gbm, accuracy_gbm)
importances=gbm_model.feature_importances_
indices=np.argsort(importances)[::-1]
names=[X.columns[i] for i in indices]
plt.figure(figsize=(10,6))
plt.title("Feature Importance GBM")
plt.barh(range (X.shape[1]), importances[indices])
plt.yticks(range(X.shape[1]), names)
plt.show()

# Definire il grid di iperparametri per Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Inizializzare il modello RandomForest
rf = RandomForestClassifier(random_state=42)

# Inizializzare GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)

# Eseguire il grid search
grid_search_rf.fit(X_train, y_train)

# Stampa i migliori parametri trovati
print(f"Migliori parametri per Random Forest: {grid_search_rf.best_params_}")

# Previsioni con il modello ottimizzato
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
y_pred_rf_tuned = best_rf.predict(X_test)

# Matrice di confusione per Random Forest con tuning
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusione con tuning (Random Forest):")
print(conf_matrix_best_rf)

# Definire il grid di iperparametri per Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

# Inizializzare il modello GradientBoosting
gb = GradientBoostingClassifier(random_state=42)

# Inizializzare GridSearchCV
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)

# Eseguire il grid search
grid_search_gb.fit(X_train, y_train)
print(f"Migliori parametri per Gradient Boosting: {grid_search_gb.best_params_}")

best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
conf_matrix_best_gb = confusion_matrix(y_test, y_pred_gb)
print("Confusione con tuning (Gradient Boosting):")
print(conf_matrix_best_gb)
# Funzione per disegnare la matrice di confusione
def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=["Non Exited", "Exited"], yticklabels=["Non Exited", "Exited"])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Matrice di confusione per Random Forest senza tuning
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(conf_matrix_rf, title="Random Forest (No Tuning)")

# Matrice di confusione per Gradient Boosting senza tuning
conf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)
plot_confusion_matrix(conf_matrix_gbm, title="Gradient Boosting (No Tuning)")

# Matrice di confusione per Random Forest con tuning
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(conf_matrix_best_rf, title="Random Forest (With Tuning)")

# Matrice di confusione per Gradient Boosting con tuning
conf_matrix_best_gb = confusion_matrix(y_test, y_pred_gb)
plot_confusion_matrix(conf_matrix_best_gb, title="Gradient Boosting (With Tuning)")

# ⚖️ Applichiamo SMOTE solo ai dati di training
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 🧐 Controllo della distribuzione prima e dopo SMOTE
print("Distribuzione classi prima di SMOTE:", Counter(y_train))
print("Distribuzione classi dopo SMOTE:", Counter(y_train_smote))

# 🌲 Random Forest con SMOTE
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_rf_smote = rf_smote.predict(X_test)

# 📊 Confusion Matrix RF con SMOTE
conf_matrix_rf_smote = confusion_matrix(y_test, y_pred_rf_smote)
print("\nConfusion Matrix - Random Forest con SMOTE:\n", conf_matrix_rf_smote)
print("\nClassification Report - Random Forest con SMOTE:\n", classification_report(y_test, y_pred_rf_smote))

# 📈 Visualizzazione grafica Confusion Matrix RF con SMOTE
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_rf_smote, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - RF con SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# 🚀 Gradient Boosting con SMOTE
gbm_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_smote.fit(X_train_smote, y_train_smote)
y_pred_gbm_smote = gbm_smote.predict(X_test)
y_pred_proba = gbm_smote.predict_proba(X_test)[:, 1]
# Ottieni le probabilità previste dal modello y_pred_proba = gbm_smote.predict_proba(X_test)[:, 1]

 # Calcola la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

 # Calcola l'AUC
auc = roc_auc_score(y_test, y_pred_proba)

 # Visualizza la curva ROC
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Linea diagonale casuale
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC-GBM SMOTE')
plt.legend()
plt.show()

print(f'AUC: {auc:.2f}')
# 📊 Confusion Matrix GBM con SMOTE
conf_matrix_gbm_smote = confusion_matrix(y_test, y_pred_gbm_smote)
print("\nConfusion Matrix - Gradient Boosting con SMOTE:\n", conf_matrix_gbm_smote)
print("\nClassification Report - Gradient Boosting con SMOTE:\n", classification_report(y_test, y_pred_gbm_smote))
print("Migliori parametri:", grid_search_gb)

# Valutazione Incrociata
scores = cross_val_score(grid_search_gb, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print("AUC con validazione incrociata:", scores.mean())
# 📈 Visualizzazione grafica Confusion Matrix GBM
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_gbm_smote, annot=True, fmt='d', cmap="Oranges")
plt.title("Confusion Matrix - GBM con SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🌀 Funzione per testare i metodi
def test_smote_variants(X_train, y_train, X_test, y_test, smote_type="SMOTE-Tomek"):
    if smote_type == "SMOTE-Tomek":
        smote = SMOTETomek(random_state=42)
    elif smote_type == "SMOTE-ENN":
        smote = SMOTEENN(random_state=42)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 🌲 Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test)

    print(f"📊 Performance RF con {smote_type}:")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("-" * 50)

    # 🚀 Gradient Boosting
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm.fit(X_resampled, y_resampled)
    y_pred_gbm = gbm.predict(X_test)

    print(f"📊 Performance GBM con {smote_type}:")
    print(confusion_matrix(y_test, y_pred_gbm))
    print(classification_report(y_test, y_pred_gbm))
    print("Accuracy:", accuracy_score(y_test, y_pred_gbm))
    print("=" * 50)

# 🔍 Testiamo le due varianti
test_smote_variants(X_train, y_train, X_test, y_test, smote_type="SMOTE-Tomek")
test_smote_variants(X_train, y_train, X_test, y_test, smote_type="SMOTE-ENN")
#FEATURE ENGINEERING
df['BalanceZero']= (df['Balance']==0).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 75, 85, 95], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95'])
df['BalanceToSalaryRatio']=df['Balance']/df['EstimatedSalary']
df['ProductUsage']= df['NumOfProducts'] * df['IsActiveMember']
df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 2, 5, 7, 10], labels=['0-2', '3-5', '6-7', '8-10'])
label_encoder=LabelEncoder()
df['Gender']=label_encoder.fit_transform(df['Gender'])

df['Male_Germany']=df['Gender']* df['Geography_Germany']
df['Male_Spain']=df['Gender']* df['Geography_Spain']
df=pd.get_dummies(df, columns=['AgeGroup', 'TenureGroup'], drop_first=True)
features=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain','BalanceZero','BalanceToSalaryRatio','ProductUsage','Male_Germany','Male_Spain'] + [col for col in df.columns if 'AgeGroup_' in col or 'TenureGroup_' in col]
X=df[features]
y=df['Exited']
X_train, X_test, y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Gradient Boosting con SMOTE
gbm_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_smote.fit(X_train_smote, y_train_smote)


# Visualizzazione dell'importanza delle features
if isinstance(X_train_smote, np.ndarray):
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=features)
else:
    X_train_smote_df = X_train_smote

feature_importances = gbm_smote.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train_smote_df.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances - GBM con SMOTE')
plt.tight_layout()
plt.show()

print(feature_importance_df)

# XGBoost con SMOTE per confrontarlo con GBM con SMOTE
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)

# Valutazione del modello XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
conf_matrix_xgb=confusion_matrix(y_test, y_pred_xgb)
print(conf_matrix_xgb)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap="Oranges")
plt.title("Confusion Matrix - XGB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Calcola le metriche di valutazione per XGBoost
print("XGBoost Metrics:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba_xgb))
# Calcola la matrice di confusione e visualizza la curva ROC per XGBoost
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 2. Calcola FPR, TPR e soglie
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# 3. Calcola l'AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 4. Traccia la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Linea diagonale per un modello casuale
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.title('Curva ROC - XGBoost')
plt.legend(loc='lower right')
plt.show()
# Visualizzazione dell'importanza delle feature per XGBoost
feature_importances_xgb = xgb_model.feature_importances_
feature_importance_df_xgb = pd.DataFrame({'Feature': X_train_smote_df.columns, 'Importance': feature_importances_xgb})
feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances - XGBoost')
plt.tight_layout()
plt.show()

print(feature_importance_df_xgb) 
# Definisci la griglia dei parametri per XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],  # Aggiungi subsample
    'colsample_bytree': [0.7, 0.8, 0.9]  # Aggiungi colsample_bytree
}

# Inizializza il modello XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Esegui la ricerca a griglia con GridSearchCV
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='roc_auc')
grid_search_xgb.fit(X_train_smote, y_train_smote)

# Stampa i migliori parametri trovati
print("Migliori parametri XGBoost:", grid_search_xgb.best_params_)

# Valutazione incrociata con i migliori parametri
scores_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print("AUC con validazione incrociata XGBoost:", scores_xgb.mean())


# Stampa l'importanza in ordine decrescente
# Soglia di Importanza GBM SMOTE per eventuale implementazione
threshold = 0.02  # Modifica la soglia se necessario
sfm = SelectFromModel(gbm_smote, threshold=threshold)
sfm.fit(X_train_smote, y_train_smote)
X_train_selected_threshold = sfm.transform(X_train_smote)
X_test_selected_threshold = sfm.transform(X_test)

# Selezione Ricorsiva delle Features (RFE) di GBM per eventuale implementazione
rfe = RFE(gbm_smote, n_features_to_select=10) # Modifica il numero di features se necessario
rfe.fit(X_train_smote, y_train_smote)
X_train_selected_rfe = rfe.transform(X_train_smote)
X_test_selected_rfe = rfe.transform(X_test)

# Valutazione delle Prestazioni
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

print("Prestazioni del modello originale:")
evaluate_model(gbm_smote, X_train_smote, y_train_smote, X_test, y_test)

print("\nPrestazioni del modello con selezione basata sulla soglia:")
evaluate_model(GradientBoostingClassifier(n_estimators=100, random_state=42), X_train_selected_threshold, y_train_smote, X_test_selected_threshold, y_test)

print("\nPrestazioni del modello con RFE:")
evaluate_model(GradientBoostingClassifier(n_estimators=100, random_state=42), X_train_selected_rfe, y_train_smote, X_test_selected_rfe, y_test)
# 3. Ottimizzazione del Modello XGBoost
# Definisci la griglia dei parametri per XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],  # Aggiungi subsample
    'colsample_bytree': [0.7, 0.8, 0.9]  # Aggiungi colsample_bytree
}

# Inizializza il modello XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Esegui la ricerca a griglia con GridSearchCV
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='roc_auc')
grid_search_xgb.fit(X_train_smote, y_train_smote)

# Stampa i migliori parametri trovati
print("Migliori parametri XGBoost:", grid_search_xgb.best_params_)

# Valutazione incrociata con i migliori parametri
scores_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print("AUC con validazione incrociata XGBoost:", scores_xgb.mean())

# Calcoliamo la correlazione con il target
correlation_matrix = df.corr()
target_correlation = correlation_matrix["Exited"].sort_values(ascending=False)

# Visualizziamo le correlazioni con il target
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlation.index, y=target_correlation.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Correlazione tra le feature e il target (Exited)")
plt.xlabel("Feature")
plt.ylabel("Correlazione con Exited")
plt.show()
# Calcola la correlazione tra tutte le feature e il target 'Exited'
target_correlation = df.corr()["Exited"].sort_values(ascending=False)

# Stampiamo la correlazione in ordine decrescente
print(target_correlation)
# Lista delle nuove feature
new_features = ['BalanceZero', 'BalanceToSalaryRatio', 'ProductUsage']

# Stampiamo la correlazione solo per le nuove feature
print(target_correlation.loc[new_features])

contingency_table = pd.crosstab(df['Geography_Germany'], df['Exited'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Test del chi-quadrato tra Geografia e Churn:")
print("Chi2:", chi2)
print("P-value:", p)
# Calcola il tasso di abbandono per ciascun paese
churn_rate_by_country = df[['Geography_Germany', 'Geography_Spain', 'Exited']].groupby(
    ['Geography_Germany', 'Geography_Spain']
)['Exited'].mean().reset_index()

# Crea una nuova colonna 'Country' per identificare i paesi
churn_rate_by_country['Country'] = pd.Series(['France'] * len(churn_rate_by_country))
churn_rate_by_country.loc[churn_rate_by_country['Geography_Germany'] == 1, 'Country'] = 'Germany'
churn_rate_by_country.loc[churn_rate_by_country['Geography_Spain'] == 1, 'Country'] = 'Spain'

# Crea la mappa utilizzando plotly.express
fig = px.choropleth(
    churn_rate_by_country,
    locations='Country',
    locationmode='country names',
    color='Exited',
    color_continuous_scale='OrRd',
    title='Tasso di Abbandono per Paese'
)
# Aggiungi la legenda
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Tasso di Abbandono",
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
    )
)
fig.show()
fig.write_html("mappa_churn.html")
# 1. Conteggio dei Clienti per Numero di Prodotti
product_counts = df['NumOfProducts'].value_counts().sort_index()
print("Conteggio dei clienti per numero di prodotti:\n", product_counts)

# 2. Percentuali di Clienti per Numero di Prodotti
product_percentages = df['NumOfProducts'].value_counts(normalize=True).sort_index() * 100
print("\nPercentuali di clienti per numero di prodotti:\n", product_percentages)

# 3. Tasso di Abbandono per Numero di Prodotti
product_churn_rate = df.groupby('NumOfProducts')['Exited'].mean() * 100
print("\nTasso di abbandono per numero di prodotti:\n", product_churn_rate)

# Definisci le fasce d'età
bins = [18, 30, 40, 50, 60, 100]  # Puoi personalizzare queste fasce
labels = ['18-29', '30-39', '40-49', '50-59', '60+']  # Puoi personalizzare queste etichette

# Crea la colonna 'AgeGroup'
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Verifica la creazione della colonna
print(df.head())

# Crea la tabella pivot
pivot_table = pd.pivot_table(df, 
                               values='Exited', 
                               index='AgeGroup', 
                               aggfunc=['count', 'mean'])

# Rinomina le colonne
pivot_table.columns = ['Count', 'Churn Rate']

# Stampa la tabella pivot
print(pivot_table)

# Crea il grafico a linee
plt.figure(figsize=(10, 6))
plt.plot(pivot_table.index, pivot_table['Churn Rate'], marker='o')
plt.xlabel('Fascia d\'età')
plt.ylabel('Tasso di abbandono')
plt.title('Tasso di abbandono per fascia d\'età')
plt.grid(True)
plt.show()


# Calcola il tasso di abbandono per numero di prodotti
product_churn = df.groupby('NumOfProducts')['Exited'].mean().reset_index()

# Crea il grafico a linee
plt.figure(figsize=(8, 5))
plt.plot(product_churn['NumOfProducts'], product_churn['Exited'], marker='o', linestyle='-')
plt.title('Tasso di Abbandono per Numero di Prodotti')
plt.xlabel('Numero di Prodotti')
plt.ylabel('Tasso di Abbandono')
plt.xticks(product_churn['NumOfProducts'])  # Assicura che i tick siano interi
plt.grid(True)
plt.show()


# Calcola il tasso di abbandono per stato di membro attivo
active_churn = df.groupby('IsActiveMember')['Exited'].mean().reset_index()

# Crea il grafico a linee
plt.figure(figsize=(6, 4))
plt.plot(active_churn['IsActiveMember'], active_churn['Exited'], marker='o', linestyle='-')
plt.title('Tasso di Abbandono per Stato di Membro Attivo')
plt.xlabel('Membro Attivo (0 = No, 1 = Sì)')
plt.ylabel('Tasso di Abbandono')
plt.xticks([0, 1])  # Assicura che i tick siano 0 e 1
plt.grid(True)
plt.show()

import pickle
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder


file_path = r"C:\Users\HP\Downloads\Churn_Modelling.csv"
df = pd.read_csv(file_path)

# 🔍 Esplorazione dei dati
print(df.head()) #Mostrare le prime 5 righe del mio dataset
print(df.isnull().sum())# Controllo valori nulli
print(df.info()) #Informazioni sul dataset
df[df.duplicated()]
label_encoder=LabelEncoder()
# 🚀 Preprocessing
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-Hot Encoding per Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Encoding per Gender
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# 📊 Definizione delle variabili
X = df.drop('Exited', axis=1)
y = df['Exited']


# 🎯 Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔢 Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
gbm_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_smote.fit(X_train_smote, y_train_smote)
y_pred_gbm_smote = gbm_smote.predict(X_test)
# Salva il modello
with open('gbm_smote_model.pkl', 'wb') as f:
    pickle.dump(gbm_smote, f)

print("Modello gbm_smote salvato come gbm_smote_model.pkl")

# Salva lo scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler salvato come scaler.pkl")
