import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)


df = pd.read_csv('D:/programming_workspace/WGU/d208/d206_final.csv')

df_analysis = df[['readmis','vitamin_d_level', 'full_meals_eaten', 'high_bp', 'complication_risk',
'arthritis', 'diabetes', 'hyperlipidemia', 'back_pain', 'anxiety', 'allergic_rhinitis', 'reflux_esophagitis',  'initial_stay_length']]

df_cat = df[['readmis', 'high_bp', 'complication_risk', 'arthritis', 'diabetes', 'hyperlipidemia', 'back_pain', 'anxiety',
'allergic_rhinitis', 'reflux_esophagitis']]

df_num = df[['vitamin_d_level', 'full_meals_eaten', 'vit_d_sups_taken', 'initial_stay_length']]

label = "charge"

y = df.avg_daily_charge
X = df_num.assign(const = 1)

model = sm.OLS(y,X)
results = model.fit()

print(results.summary())
df['prediction'] = results.fittedvalues
print(df[['avg_daily_charge', 'prediction']])


for i in df_cat:
    df_analysis = pd.get_dummies(df_analysis, dtype = int, columns = [i], prefix = i, drop_first = True)



X = df_analysis.assign(const = 1)



model = sm.OLS(y,X)
results = model.fit()


print(f'Dropped: Services, Overweight, dr_visits MLR: {results.summary()}')


df['prediction'] = results.fittedvalues
print(df[['avg_daily_charge', 'prediction']])

MAE = mean_absolute_error(df[['avg_daily_charge']], df[['prediction']])
print('MAE:', MAE)


RSE = results.resid.std(ddof=X.shape[1])
print('RSE: ', RSE)

residuals = df[['avg_daily_charge']]- df[['prediction']]
plot = sns.regplot(x=residuals, y=df[['prediction']])
plt.show()