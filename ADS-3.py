import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

API_Data = pd.read_excel(r"C:\Users\Jalaluddin Shaik\Downloads\API_19_DS2_en_excel_v2_5455559.xlsx")

def choose_data(filename):
    path=r"C:\Users\Jalaluddin Shaik\Downloads"
    API_Data = pd.read_excel(path + "/" + filename)
    API_Data_Archive = API_Data.copy()
    API_Data1 = API_Data[2:]
    API_Data1.columns = API_Data1.iloc[0]
    API_Data = API_Data1[1:]
    
    
    API_Data2 = pd.melt(API_Data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                        var_name="Year", value_name="Value")
    
    Country_As_Columns = API_Data2.pivot(index = ['Indicator Name', 'Indicator Code', 'Year'], columns = 'Country Name', 
                                      values = 'Value').reset_index()
    
    Date_As_Columns = API_Data2.pivot(index = ['Country Name', 'Country Code', 'Indicator Name'], columns = 'Year', 
                                      values = 'Value').reset_index()
    return Date_As_Columns , Country_As_Columns

def cleaning(data):
    columns = data.columns
    country_list = data['Country Name'].unique()
    Indicator_list = data['Indicator Name'].unique()
    
    data=data.sort_values(['Country Name','Indicator Name'])
    Clean_Data = pd.DataFrame()
    
    for i in Indicator_list:
        data_i = data[(data['Indicator Name']==i)]
        data_i['Value'] = data_i['Value'].fillna(data_i['Value'].mean())

    Clean_Data = Clean_Data.append(data_i)
    
    return Clean_Data  
    

Date_As_Columns , Country_As_Columns = choose_data('API_19_DS2_en_excel_v2_5455559.xlsx')

Date_As_Columns
Date_As_Columns.describe()
Country_As_Columns.describe()

Dates_Country_choosen= Date_As_Columns
[(Date_As_Columns['Country Name']=='India')|(Date_As_Columns['Country Name']=='Australia')
                 |(Date_As_Columns['Country Name']=='Japan')|(Date_As_Columns['Country Code']=='USA')
                 |(Date_As_Columns['Country Name']=='United Kingdom')|(Date_As_Columns['Country Name']=='Germany')
                 |(Date_As_Columns['Country Name']=='China')|(Date_As_Columns['Country Name']=='Brazil')]
Dates_Country_choosen

API_Data2 = pd.melt(Dates_Country_choosen, id_vars=['Country Name', 'Country Code', 'Indicator Name'], 
                        var_name="Year", value_name="Value")
API_Data2

# API_Data2['Indicator Name'].unique()
data_1 = API_Data2[API_Data2['Indicator Name'] =='CO2 emissions (metric tons per capita)'].rename(columns={'Indicator Name':'CO2 emissions','Value':'CO2 emissions value'})
data_2 = API_Data2[API_Data2['Indicator Name'] =='Access to electricity (% of population)'].rename(columns={'Indicator Name':'Access to electricity','Value':'Access to electricity value'})
data_3 = API_Data2[API_Data2['Indicator Name'] =='Electric power consumption (kWh per capita)'].rename(columns={'Indicator Name':'power consumption','Value':'power consumption value'})
data_4 = API_Data2[API_Data2['Indicator Name'] =='Renewable energy consumption (% of total final energy consumption)'].rename(columns={'Indicator Name':'Renewable energy consumption','Value':'Renewable energy consumption value'})
merge_1 = data_1.merge(data_2,on =['Country Name','Country Code','Year'])
merge_2 = merge_1.merge(data_3,on=['Country Name','Country Code','Year'])
merge_3 = merge_2.merge(data_4,on=['Country Name','Country Code','Year'])
merge_4 = merge_3.dropna()
merge_5 = merge_4[['Country Name','Year','CO2 emissions value','Access to electricity value',
         'power consumption value','Renewable energy consumption value']]

Kmeans_data = merge_5[['power consumption value',
       'CO2 emissions value']]
merge_5 = merge_5[['Country Name', 'Year', 'power consumption value',
       'CO2 emissions value']]

kmeans = KMeans(n_clusters=3)
kmeans.fit(Kmeans_data)

kmeans.cluster_centers_

cluster = kmeans.fit_predict(Kmeans_data)
merge_5['Cluster'] = cluster
output_cluster_2010 = merge_5[merge_5['Year']==2010]
sns.scatterplot(data=output_cluster_2010, x='power consumption value', y='CO2 emissions value', hue='Cluster')

output_cluster_2010[output_cluster_2010['Cluster']==1].sort_values('power consumption value')

merge_3[['Country Name','Year','CO2 emissions value','Access to electricity value',
         'power consumption value','Renewable energy consumption value']].head(3)
merge_3.head(3).reset_index()

scatter_matrix(merge_4,figsize=(11,11))
plt.show()

labels = np.asarray(merge_5['Country Name'])
labels

le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
df_selected1 = merge_5.drop(['Country Name'], axis=1)

df_features = df_selected1.to_dict(orient='records')

vec = DictVectorizer()
features = vec.fit_transform(df_features).toarray()

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=0)

sns.set(style="ticks", color_codes=True)
sns.pairplot(merge_4, hue="Country Name", palette="husl")


