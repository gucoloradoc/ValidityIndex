import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_df= pd.read_csv('data/all_data.csv')
base_df['attribute']=base_df['attribute'].astype('category')
unive_ranking=pd.read_csv('data/search_keys2.csv')

####cleaning countries (If we want 1 attribute per country)
countries=base_df[base_df['attribute']=='Countries']['value'].astype('category').cat.categories.values
corrupt_countries=pd.read_csv('output/countries_corrupt.csv').drop(columns='Unnamed: 0')
countries_df=base_df[base_df['attribute']=='Countries']
countries_df.drop(countries_df[countries_df['value'].isin(corrupt_countries['corrupt'])].index)

##Subareas
subarea_df=base_df[base_df['attribute']=='SubArea']

#Doctypes
doctype_df=base_df[base_df['attribute']=='DocType']

#Access Type
access_type_df=base_df[base_df['attribute']=='Access_type']

#Affiliations
affiliations_df=base_df[base_df['attribute']=='Affiliations']

#Creation of the dataframe to work
data_to_work=pd.DataFrame(unive_ranking['university'])

###Access_type
#The count of each access type is the attribute, per year
access_type_att=[i+str(j) for j in range(2014,2019) for i in ['open_', 'other_']]

for acc_atrr in access_type_att:
    data_to_work.insert(column=acc_atrr, value=np.nan, loc=len(data_to_work.columns))

for year in range(2014,2019):
    for uni in data_to_work['university']:
        try:
            data_to_work.loc[data_to_work['university']==uni,['open_'+str(year),'other_'+str(year)]]=access_type_df[(access_type_df['university']==uni) & (access_type_df['year']==year)]['count'].values
        except:
            print(uni)
            print(year)

### Affiliations
#mean(log()) and var(log()) for each year overall affiliations
aff_attrs=[var+str(i) for i in range(2014,2019) for var in ['aff_mean_l_','aff_sd_l_']]

for aff_atrr in aff_attrs:
    data_to_work.insert(column=aff_atrr, value=0, loc=len(data_to_work.columns))

for year in range(2014,2019):
    for uni in data_to_work['university']:
        try:
            y_mean=np.mean(np.log(affiliations_df[(affiliations_df['university']==uni) & (affiliations_df['year']==year)]['count']))
            y_std=np.std(np.log(affiliations_df[(affiliations_df['university']==uni) & (affiliations_df['year']==year)]['count']))
            data_to_work.loc[data_to_work['university']==uni,['aff_mean_l_'+str(year),'aff_sd_l_'+str(year)]]=[y_mean,y_std]
        except:
            print(uni)
            print(year)

## Doctype
doctps=doctype_df['value'].unique()
doct_attr=[dty+'_'+str(year) for dty in doctps for year in range(2014,2019)]

for doc_attr in doct_attr:
    data_to_work.insert(column=doc_attr, value=0, loc=len(data_to_work.columns))

for year in range(2014,2019):
    for uni in data_to_work['university']:
        for dtyp in doctps:
            try:
                docy_c=doctype_df[(doctype_df['value']==dtyp) & (doctype_df['university']==uni) & (doctype_df['year']==year)]['count'].values
                data_to_work.loc[data_to_work['university']==uni,dtyp+'_'+str(year)]=docy_c
            except:
                data_to_work.loc[data_to_work['university']==uni,dtyp+'_'+str(year)]=0
                #print('Error in uni {1}, year {2}, doc type {3}'.format(uni, year, dtyp))

##Subareas
subareas_df=base_df[base_df['attribute']=='SubArea']
subareas=subareas_df['value'].unique()
sub_attrs=[sub+' '+str(year) for sub in subareas for year in range(2014,2019)]

for sub_attr in sub_attrs:
    data_to_work.insert(column=sub_attr, value=0, loc=len(data_to_work.columns))

for year in range(2014,2019):
    for uni in data_to_work['university']:
        for suba in subareas:
            try:
                docy_c=subareas_df[(subareas_df['value']==suba) & (subareas_df['university']==uni) & (subareas_df['year']==year)]['count'].values
                data_to_work.loc[data_to_work['university']==uni,suba+' '+str(year)]=docy_c
            except:
                data_to_work.loc[data_to_work['university']==uni,suba+' '+str(year)]=0
                #print('Error in uni {1}, year {2}, doc type {3}'.format(uni, year, dtyp))

data_to_work.to_csv('output/output_att.csv')