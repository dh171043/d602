import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv('D:/programming_workspace/WGU/d208/d206_final.csv')

df = df[['children', 'age', 'gender', 'readmis','vitamin_d_level', 'dr_visits', 'full_meals_eaten', 'vit_d_sups_taken','multiple_soft_drinks', 'high_bp','stroke',
  'complication_risk', 'overweight', 'arthritis', 'diabetes', 'hyperlipidemia', 'back_pain', 'anxiety', 'allergic_rhinitis', 'reflux_esophagitis', 
'asthma', 'services', 'initial_stay_length', 'avg_daily_charge']]

df = df.astype({'overweight':bool, 'anxiety':bool})

# df_cat = df[['gender', 'readmis', 'multiple_soft_drinks',
# 'high_bp', 'stroke', 'complication_risk', 'overweight', 'arthritis', 'diabetes', 'hyperlipidemia', 'back_pain', 'anxiety', 'allergic_rhinitis', 'reflux_esophagitis', 
# 'asthma', 'services']]

# df_num = df[['children', 'age','vitamin_d_level', 'dr_visits', 'full_meals_eaten', 'vit_d_sups_taken', 'initial_stay_length']]

exploratory_variable = "avg_daily_charge"

def unistats(df):
    import pandas as pd

    for col in df: 
        if pd.api.types.is_bool_dtype(df[col]):
            for row in col:
                if row == 0:
                    row = 'False'
                else:
                    row = 'True'
    output_df = pd.DataFrame(columns=['count', 'missing', 'unique', 'dtype', 'mean', 'mode', 'min', 'q1', 'q2', 'q3', 'max', 'std', 'skew', 'kurt'])

    for col in df:
        if pd.api.types.is_bool_dtype(df[col]):
            output_df.loc[col]= [df[col].count(),df[col].isnull().sum(),df[col].nunique(),df[col].dtype, df[col].mean(),df[col].mode().values[0],'-','-',df[col].median(),'-','-',df[col].std(),df[col].skew(),df[col].kurt()]
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            output_df.loc[col]= [df[col].count(),df[col].isnull().sum(),df[col].nunique(),df[col].dtype, df[col].mean(),df[col].mode().values[0],df[col].min(),df[col].quantile(0.25),df[col].median(),df[col].quantile(0.75),df[col].max(),df[col].std(),df[col].skew(),df[col].kurt()]

        else:
            output_df.loc[col]= [df[col].count(),df[col].isnull().sum(),df[col].nunique(),df[col].dtype, '-',df[col].mode().values[0],'-','-','-','-','-','-','-','-']

    output_df.to_csv('C:/Users/donovan.a.hall/Desktop/practiceData/python/myenv/d208.linear/d208_univariate.csv')
    return output_df

# print(unistats(df))

#numeric to numeric: correlation
#numeric to categorical: ANOVA (3+ groups) or t-tests (2 groups)
#categorical to categorical: chi square

def hscdt(df, col, label):
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if col == 'multiple_soft_drinks':
        print(f'missing data: {df[col].isnull().values.ravel().sum()}')
        print(f'corrcoef: {np.corrcoef(df[col])}')
        print(f'unique values for bool: {df[col].unique()}')
        print(f'unique values for numeric: {df[label].unique().sum()}')

        cross_tab = pd.crosstab(df[col], df[label])
        plt.scatter(df[col], df[label])
        plt.show()

    if col != label:
        model = ols(formula = (label + '~' + col), data = df).fit()
        white_test = het_white(model.resid, model.model.exog)
        bp_test = het_breuschpagan(model.resid, model.model.exog)

        output_df = pd.DataFrame(columns = ['LM stat', "LM p", 'F stat', 'F stat p']) 
        output_df.loc['White'] = white_test
        output_df.loc['Br-Pa'] = bp_test



    return output_df.round(3)


# print(hscdt(df, 'multiple_soft_drinks', 'avg_daily_charge'))

def bar_chart(df, col, label):
    import pandas as pd
    from scipy import stats
    from matplotlib import pyplot as plt

    means = df.groupby(col)[label].mean().round(2)

    g = plt.bar(df.groupby(col).groups.keys(), means)
    plt.title(label + 'by ' + col)
    plt.xlabel(col)
    plt.ylabel(label)

    groups = df[col].unique()
    df_grouped = df.groupby(col)
    group_labels = []
    for group in groups:
        g_list = df_grouped.get_group(group)
        group_labels.append(g_list[label])

    
    oneway = stats.f_oneway(*group_labels)

    unique_groups = df[col].unique()
    print(f'this is unique_groups {unique_groups}')
    ttests = []

    print(unique_groups.dtype)

    for i, group in enumerate(unique_groups):
        for i2, group_2 in enumerate(unique_groups):
            if i2>i:
                type_1 = df[df[col] == group]
                type_2 = df[df[col] == group_2]
                t, p = stats.ttest_ind(type_1[label], type_2[label])
                ttests.append([group, group_2, t.round(4), p.round(4)])

    p_threshold = 0.05/len(ttests)

    textstr = '     ANOVA' + '\n'
    textstr += 'F:       ' + str(oneway[0].round(2)) + '\n'
    textstr += 'p-value  ' + str(oneway[1].round(2)) + '\n'
    textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'


    for ttest in ttests:
        if ttest[3] <= p_threshold:
            textstr += ttest[0] + '-' + ttest[1] + ': t=' + str(ttest[2]) + ', p=' + str(ttest[3]) + '\n'
    plt.text(0.75, 0.1, textstr, fontsize = 12, transform = plt.gcf().transFigure, verticalalignment='bottom', horizontalalignment='right')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()


def scatter(col, label):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd

    if pd.api.types.is_bool_dtype(df[col]):
        sns.set(color_codes = True)
        g = sns.jointplot(data = df, x = col, y = label, height = 10)
    elif pd.api.types.is_numeric_dtype(df[col]):
        m, b, r, p, err = stats.linregress(df[col], df[label])
        textstr = 'y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
        textstr += 'r2 = ' + str(round(r**2, 6)) + '\n'
        textstr += 'p = '+ str(round(p, 2)) + '\n'
        textstr += str(col) + ' skew = ' + str(round(df[col].skew(), 2)) + '\n'
        textstr += str(label) + ' skew = ' + str(round(df[label].skew(), 2)) + '\n'
        textstr += str(hscdt(pd.DataFrame(df[label]).join(pd.DataFrame(df[col])), col, label))

        sns.set(color_codes = True)
        g = sns.jointplot(data = df, x = col, y = label, kind = 'reg', height = 10)
        g.ax_joint.text(0.05, 0.99, textstr, transform = g.ax_joint.transAxes, fontsize = 12, verticalalignment = 'top')
    else:
        sns.set(color_codes = True)
        g = sns.jointplot(data = df, x = col, y = label, height = 10)
    plt.show()


def bivstats(df, label):
    from scipy import stats
    import pandas as pd
    import numpy as np

    output_df = pd.DataFrame(columns=['stat', '+/-', 'effect_size', 'p-value'])
    grouped_values = []

    for col in df:
        if col != label:
            print(col)
            if pd.api.types.is_bool_dtype(df[col]):
                [print('this is bool')]
                groups = df[col].unique()
                for group in groups:
                    grouped_values.append(df[df[col]==group][label])
                f, p = stats.f_oneway(*grouped_values)
                output_df.loc[col] = 'F', np.sign(f), round(f, 3), round(p, 3)
            elif pd.api.types.is_numeric_dtype(df[col]):
                r, p = stats.pearsonr(df[label], df[col])
                output_df.loc[col] = 'r', np.sign(r), round(r,3), round(p, 3)
            else:
                print(f"Data type of '{col}' column before conversion: {df[col].dtype}") # Debugging
                groups = df[col].unique()
                for group in groups:
                    grouped_values.append(df[df[col]==group][label])
                f, p = stats.f_oneway(*grouped_values)
                output_df.loc[col] = 'F', np.sign(f), round(f, 3), round(p, 3)
                #bar_chart(df, col, label)
            #scatter(col, label)
    
    return output_df.sort_values(by = ['effect_size', 'stat'], ascending = [False, False])


print(bivstats(df, 'avg_daily_charge'))


print(f'second round!! {bivstats(df,'readmis')}')


#limit of bivariate analysis: cannot determine true correlation (intercorrelation is a problem)




