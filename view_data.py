import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('dataset/train.csv')
df_train['SalePrice'].describe()
plt.figure("test")
sns.distplot(df_train['SalePrice']);
plt.show()
