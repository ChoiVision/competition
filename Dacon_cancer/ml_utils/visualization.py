import matplotlib.pyplot as plt
import seaborn as sns

def show_hist_by_target(df, columns):
    cond_0 = (df['N_category'] == 0)
    cond_1 = (df['N_category'] == 1)
    
    for column in columns:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), squeeze=False)
        sns.violinplot(x='N_category', y=column, data=df, ax=axs[0][0] )
        sns.distplot(df[cond_0][column], ax=axs[0][1], label='0', color='blue')
        sns.distplot(df[cond_1][column], ax=axs[0][1], label='1', color='red')

def show_cateogry_by_target(df, columns):
    for col in columns:
        chart= sns.catplot(x= col, y= 'N_category', data= df, kind= 'train')
        chart.set_xticklabels(rotation= 65)

        