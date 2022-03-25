from optparse import Values
from pandas.core.base import DataError, PandasObject
from pandas.core.dtypes.missing import isna, isnull
from pandas.io.parsers import read_csv
import streamlit as st 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pickle 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations
from docplex.mp.model import Model
import cplex

import warnings 
warnings.filterwarnings("ignore")

#title
st.title('Product Cross-Sell-Up-Sell Predictor')
image=Image.open('LinkedIn Cover.jpg')
st.image(image,use_column_width=True)
#data=st.file_uploader('Upload The Dataset',type=['csv'])
def main():
    st.subheader('Product cross-selling analysis')
    data=st.file_uploader('Upload The Dataset',type=['csv'],key='0')
    #d=pd.read_csv(data, encoding='ISO 8859-1')
    #st.success('Data Successfully Uploaded')
    #d=pd.read_csv(data, encoding='ISO 8859-1')
    #st.write('Raw data:',d.head(10))
    if data is not None:
        data=pd.read_csv(data, encoding='ISO 8859-1')
        st.success('Data Successfully Uploaded')
        st.write('Raw data:',data.head(10))
        #d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
        #d['PRODUCTS'] = d['PRODUCTS'].str.strip()
        #d['RECEIPT'] = d['RECEIPT'].astype('str')
        #d = d[~d['RECEIPT'].str.contains('C')]
        #d['Unique Id']=d['STOCKIEST CODE']+'_'+d['MONTH']
        x=['DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        #option=st.selectbox('Select The Depot:',x)
        master_data_frame=pd.DataFrame()

        st.title('National Level')
        basket_n = (data
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
        def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
        basket_sets1 = basket_n.applymap(encode_units)
        frequent_itemsets1 = apriori(basket_sets1, min_support=0.07, use_colnames=True)
        #st.write(frequent_itemsets.head())
        rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
        rules1["antecedents"] = rules1["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
        rules1["consequents"] = rules1["consequents"].apply(lambda x: list(x)[0]).astype("unicode") 
        #st.write(rules1.head())
        c11=rules1['antecedents'].unique()
        cl1=list(c11)
        sb1=st.selectbox('select product',cl1,key='1')
        for i in cl1:
            if sb1==i:
                r11=(rules1.loc[rules1['antecedents']==i])
                r11=r11.rename(columns={'consequents':'Recommendations'},inplace=False)
                st.subheader('Recommendations')
                st.write(r11[['Recommendations','confidence']])
                

        st.title('Depot Level')
        option=st.selectbox('Select The Depot:',x)
        for i in x:
            if option==i:
                data['STOCKIEST NAME'] = data['STOCKIEST NAME'].str.strip()
                data['PRODUCTS'] = data['PRODUCTS'].str.strip()
                basket = (data[data['DEPOT']==i]
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
                def encode_units(x):
                    if int(x) <= int(0):
                        return 0
                    if int(x) >= int(1):
                        return 1
                
                basket_sets = basket.applymap(encode_units)
                frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
                rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
                rules['Depot']=i
                #st.write(rules)
                #master_data_frame=master_data_frame.append(rules)
                c1=rules['antecedents'].unique()
                cl=list(c1)
                sb=st.selectbox('select product',cl)
                for i in cl:
                    if sb==i:
                        r1=(rules.loc[rules['antecedents']==i])
                        r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                        st.subheader('Recommendations')
                        st.write(r1[['Recommendations','confidence']].head())
                        # dfr=(r1['Recommendations'].unique())
                        # k=pd.DataFrame(dfr)
                        # k.columns= ['Recommendations']
                        # st.write(k['Recommendations'].unique())
        #st.write(master_data_frame)
        # st.title('National Level')
        # basket_n = (data
        #             .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
        #             .sum().unstack().reset_index().fillna(0)
        #             .set_index('Unique Id'))
        # def encode_units(x):
        #         if int(x) <= int(0):
        #             return 0
        #         if int(x) >= int(1):
        #             return 1
        # basket_sets1 = basket_n.applymap(encode_units)
        # frequent_itemsets1 = apriori(basket_sets1, min_support=0.07, use_colnames=True)
        # #st.write(frequent_itemsets.head())
        # rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
        # rules1["antecedents"] = rules1["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
        # rules1["consequents"] = rules1["consequents"].apply(lambda x: list(x)[0]).astype("unicode") 
        # #st.write(rules1.head())
        # c11=rules1['antecedents'].unique()
        # cl1=list(c11)
        # sb1=st.selectbox('select product',cl1,key='1')
        # for i in cl1:
        #     if sb1==i:
        #         r11=(rules1.loc[rules1['antecedents']==i])
        #         r11=r11.rename(columns={'consequents':'Recommendations'},inplace=False)
        #         st.write(r11['Recommendations'].unique())


        # st.title('verify your plan')
        # for i in x:
        # #if option==i:
        #     data['STOCKIEST NAME'] = data['STOCKIEST NAME'].str.strip()
        #     data['PRODUCTS'] = data['PRODUCTS'].str.strip()
        #     basket = (data[data['DEPOT']==i]
        #         .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
        #         .sum().unstack().reset_index().fillna(0)
        #         .set_index('Unique Id'))
        #     def encode_units(x):
        #         if int(x) <= int(0):
        #             return 0
        #         if int(x) >= int(1):
        #             return 1
      
        #     basket_sets = basket.applymap(encode_units)
        #     frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
        #     rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        #     rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
        #     rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
        #     rules['Depot']=i
        #     #st.write(rules)
        #     master_data_frame=master_data_frame.append(rules)



        #     basket_n = (data
        #             .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
        #             .sum().unstack().reset_index().fillna(0)
        #             .set_index('Unique Id'))
        #     def encode_units(x):
        #         if int(x) <= int(0):
        #             return 0
        #         if int(x) >= int(1):
        #             return 1
        #     basket_sets1 = basket_n.applymap(encode_units)
        #     frequent_itemsets1 = apriori(basket_sets1, min_support=0.07, use_colnames=True)
            
        #     #st.write(frequent_itemsets1)
        #     rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
        #     rules1['Depot']='National'
        #     rules1["antecedents"] = rules1["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
        #     rules1["consequents"] = rules1["consequents"].apply(lambda x: list(x)[0]).astype("unicode") 
        #     #st.write(rules1)
        #     m_t=master_data_frame.append(rules1)
        # #st.write(m_t)
        # #st.download_button(label='Download File',data=m_t.to_csv(),file_name='Master_table.csv',mime='csv')


        # data_v=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
        # if data_v is not None:
            
        #     data_v1=pd.read_csv(data_v, encoding='ISO 8859-1')
        #     st.success('Data Successfully Uploaded')
        #     st.write('Raw data:',data_v1.head(10))
        #     sub_master=pd.DataFrame()
        #     x1=['National','DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        #     depot=st.selectbox('Select The Depot:',x1,key='1')
        #     #st.write(data_v1['Recommendations'].to_list())
        #     for row in data_v1['Products'].to_list():

        #         product=row
        #         filter_df=master_data_frame[(master_data_frame['Depot']==depot) & (master_data_frame['antecedents']==product)]
        #         #st.write(depot)
        #         #st.write(product)
        #         #st.write(filter_df)
        #         sub_master=sub_master.append(filter_df)
        #     #x1=['DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        # #options=st.selectbox('Select The Depot:',x1,key='1')
        # #st.write(pd.__version__)
        # # for i in x1:
        # #     if options==i:
        # #         r1f=(m_t.loc[m_t['Depot']==i])
        # #         r1f=r1f.rename(columns={'consequents':'Recommendations'},inplace=False)
        # #         st.write(r1f)
            
        #     #st.write(sub_master)
        #         sub_master=sub_master.sort_values(by='confidence',ascending=False)
        #         sub_master=sub_master.drop_duplicates(subset={'antecedents','consequents'},keep='first')
        #     #st.write(sub_master)
        #         rec_list=sub_master['consequents'].to_list()
        #         f_list=[]
        #     #st.write(rec_list)
        #         for item in rec_list:
        #             if item not in data_v1['Products'].to_list():
                    
        #             # st.write(item)
        #                 f_list.append(item)
        #         # else:
        #             # st.write('no match',item)
        #     st.subheader('Products you might want to add to your plan:')    
        #     # st.write(set(f_list))
        #     s=set(f_list)
        #     s=list(s)
        #     w=pd.DataFrame(s)
        #     # w.rename({'0':'Recommendations'})
        #     st.write(w)
        #     # dataframe=pd.DataFrame(s)
        #     # st.write(dataframe.unique())

        
        st.title('Stockiest level recommendation')
        lst_depot=['DELHI','AMBALA','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        depot_level=st.selectbox('Select the Depot:',list(lst_depot))
        
        for i in lst_depot:
            if depot_level==i:
                filtered_depot=data[data["DEPOT"]==i]
                # st.write(filtered_depot.head())
        lst_stockiest=list(filtered_depot['STOCKIEST NAME'].unique())
        # st.write(lst_stockiest)
        stockiest_level=st.selectbox("Select the Stockiest:",lst_stockiest)
        for i in lst_stockiest:
            if stockiest_level==i:
                basket = (data[data['STOCKIEST NAME'] == i]
                            .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                            .sum().unstack().reset_index().fillna(0)
                            .set_index('Unique Id'))
                # st.write(basket.head())
                def encode_units(x):
                    if int(x) <= int(0):
                        return 0
                    if int(x) >= int(1):
                        return 1
                
                basket_sets = basket.applymap(encode_units)
                # st.write(basket_sets.head())
                frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
                # st.write(frequent_itemsets.head())
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
                rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
                # st.write(rules)
                lst_antecedents=list(rules['antecedents'].unique())
        # st.write(lst_stockiest)
                antecedent_level=st.selectbox("Select the product:",lst_antecedents)
                for i in lst_antecedents:
                    if antecedent_level==i:
                        rules_k=rules[rules['antecedents']==i]
                        rules_final=rules_k[['consequents','confidence','support']]
                        
                st.write(rules_final.head())
            # sub_master['consequents']
                
        st.title('verify your plan')
        for i in x:
        #if option==i:
            data['STOCKIEST NAME'] = data['STOCKIEST NAME'].str.strip()
            data['PRODUCTS'] = data['PRODUCTS'].str.strip()
            basket = (data[data['DEPOT']==i]
                .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                .sum().unstack().reset_index().fillna(0)
                .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
      
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules['Depot']=i
            #st.write(rules)
            master_data_frame=master_data_frame.append(rules)



            basket_n = (data
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets1 = basket_n.applymap(encode_units)
            frequent_itemsets1 = apriori(basket_sets1, min_support=0.07, use_colnames=True)
            
            #st.write(frequent_itemsets1)
            rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
            rules1['Depot']='National'
            rules1["antecedents"] = rules1["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules1["consequents"] = rules1["consequents"].apply(lambda x: list(x)[0]).astype("unicode") 
            #st.write(rules1)
            m_t=master_data_frame.append(rules1)
        #st.write(m_t)
        #st.download_button(label='Download File',data=m_t.to_csv(),file_name='Master_table.csv',mime='csv')


        data_v=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
        if data_v is not None:
            
            data_v1=pd.read_csv(data_v, encoding='ISO 8859-1')
            st.success('Data Successfully Uploaded')
            st.write('Raw data:',data_v1.head(10))
            sub_master=pd.DataFrame()
            x1=['DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
            depot=st.selectbox('Select The Depot:',x1,key='1')
            #st.write(data_v1['Recommendations'].to_list())
            for row in data_v1['Products'].to_list():

                product=row
                filter_df=master_data_frame[(master_data_frame['Depot']==depot) & (master_data_frame['antecedents']==product)]
                #st.write(depot)
                #st.write(product)
                #st.write(filter_df)
                sub_master=sub_master.append(filter_df)
            #x1=['DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        #options=st.selectbox('Select The Depot:',x1,key='1')
        #st.write(pd.__version__)
        # for i in x1:
        #     if options==i:
        #         r1f=(m_t.loc[m_t['Depot']==i])
        #         r1f=r1f.rename(columns={'consequents':'Recommendations'},inplace=False)
        #         st.write(r1f)
            
            #st.write(sub_master)
                sub_master=sub_master.sort_values(by='confidence',ascending=False)
                sub_master=sub_master.drop_duplicates(subset={'antecedents','consequents'},keep='first')
            #st.write(sub_master)
                rec_list=sub_master['consequents'].to_list()
                f_list=[]
            #st.write(rec_list)
                for item in rec_list:
                    if item not in data_v1['Products'].to_list():
                    
                    # st.write(item)
                        f_list.append(item)
                # else:
                    # st.write('no match',item)
            st.subheader('Products you might want to add to your plan:')    
            # st.write(set(f_list))
            s=set(f_list)
            s=list(s)
            w=pd.DataFrame(s)
            w.rename(columns={'0':'Recommendations'},inplace=True)
            st.title('Products')
            st.write(w)
            # dataframe=pd.DataFrame(s)
            # dataframe.rename({'0':'Recommendations'})
            # st.write(dataframe.unique())
        
        
        
        
        
        
        
        
        
        
        
        
        if data_v is not None:
            st.title('Discount Optimization')
        # lst_depot_do=['DELHI','AMBALA','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        # depot_level_do=st.selectbox('Select the Depot:',list(x1),key=2)
            for i in x1:
                if depot==i:
                    filtered_depot_do=data[data["DEPOT"]==i]

            lst_stockiest_do=list(filtered_depot_do['STOCKIEST NAME'].unique())
            # stockiest_level_do=st.selectbox("Select the Stockiest:",lst_stockiest_do,key=3)
            class_a=[lst_stockiest_do[0],lst_stockiest_do[1],lst_stockiest_do[2],lst_stockiest_do[3],lst_stockiest_do[4],lst_stockiest_do[5],lst_stockiest_do[6],lst_stockiest_do[7],lst_stockiest_do[8],lst_stockiest_do[9],lst_stockiest_do[10],lst_stockiest_do[11],lst_stockiest_do[12],lst_stockiest_do[13],lst_stockiest_do[14],lst_stockiest_do[15],lst_stockiest_do[16],lst_stockiest_do[17],lst_stockiest_do[18],lst_stockiest_do[19],lst_stockiest_do[20],lst_stockiest_do[21],lst_stockiest_do[22]]
            class_b=[lst_stockiest_do[7],lst_stockiest_do[8],lst_stockiest_do[9],lst_stockiest_do[10],lst_stockiest_do[11],lst_stockiest_do[12],lst_stockiest_do[13]]
            class_c=[lst_stockiest_do[14],lst_stockiest_do[15],lst_stockiest_do[16],lst_stockiest_do[17],lst_stockiest_do[18],lst_stockiest_do[19],lst_stockiest_do[20],lst_stockiest_do[21],lst_stockiest_do[22]]
            # st.write(class_a)
            products=s
            sb=st.selectbox('Select  a product from the above recommendation',products)
            if sb=='Androanagen Tablets 10S':
                base_price_p1=1200
            elif sb=='Es Body Wash 200ML':
                base_price_p1=1540
            elif sb=='Aknay Bar 100GM':
                base_price_p1=2765
            elif sb=='Androanagen Solution 100Ml':
                base_price_p1=3215
            elif sb=='Acnemoist Cream 60GM':
                base_price_p1=1230
            elif sb=='Banatan Cream 50GM':
                base_price_p1=1300
            elif sb=='Canthex 10 Capsules':
                base_price_p1=750
            elif sb=='Triobloc Cream 25GM':
                base_price_p1=1543
            elif sb=="""Nixiyax 15 'S""":
                base_price_p1=589
            elif sb=='Melawash 100 ML':
                base_price_p1=960
            elif sb=='Zinikam Cream 200 G':
                base_price_p1=1150


            sku=[100,200,300,400,500,600,700,800,900,1000]
            sku1=st.selectbox('Select the quantity : ',sku)
            # product_box2=st.selectbox('Select the 2nd product',products)
            # sku2=st.selec("Enter the number of units ordered  : ",10,500,10,10,key=2)
            # base_price_p1=st.number_input("Enter the base Price: ",250,5000,250,10,key=3)
            # base_price_p2=st.number_input("Enter the base Price: ",250,5000,250,10,key=4)
            m=Model(name='Revenue optimization')
            a1_disc=m.continuous_var(name='Product discount',lb=20)
            # a2_disc=m.continuous_var(name='second Product discount',lb=20)
            # for i in products:
            #     if product_box1==i:
            #         a1_disc=m.continuous_var(name=i +  '_discount',lb=20)
            # for i in products:
            #     if product_box2==i:
            #         a2_disc=m.continuous_var(name=i +  '_discount',lb=20)
            
            for i in lst_stockiest_do:
                if i in class_a:
                    if sku1<=100:
                        a1_constraints=m.add_constraint(a1_disc<=(12*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(8*base_price_p1)/100)
                    
                    

                    # elif sku1>100:
                    #     a1_constraints=m.add_constraint(a1_disc<=(18*base_price_p1)/100)
                    #     a11_constraints=m.add_constraint(a1_disc>=(10*base_price_p1)/100)
                    elif sku1<=200:
                        a1_constraints=m.add_constraint(a1_disc<=(30*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(18*base_price_p1)/100)
                    elif sku1<=300:
                        a1_constraints=m.add_constraint(a1_disc<=(35*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21*base_price_p1)/100)
                    elif sku1<=400:
                        a1_constraints=m.add_constraint(a1_disc<=(35.25*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.25*base_price_p1)/100)
                    elif sku1<=500:
                        a1_constraints=m.add_constraint(a1_disc<=(35.50*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.50*base_price_p1)/100)
                    elif sku1<=600:
                        a1_constraints=m.add_constraint(a1_disc<=(35.75*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.75*base_price_p1)/100)
                    elif sku1<=700:
                        a1_constraints=m.add_constraint(a1_disc<=(36*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21*base_price_p1)/100)
                    elif sku1<=800:
                        a1_constraints=m.add_constraint(a1_disc<=(36.25*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.25*base_price_p1)/100)
                    elif sku1<=900:
                        a1_constraints=m.add_constraint(a1_disc<=(36.50*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.50*base_price_p1)/100)
                    elif sku1<=1000:
                        a1_constraints=m.add_constraint(a1_disc<=(39.75*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(27.75*base_price_p1)/100)
                    # if sku2<=100:
                    #     a2_constraints=m.add_constraint(a2_disc<=(12*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a2_disc>=(8*base_price_p2)/100)
                    # if sku2>100:
                    #     a2_constraints=m.add_constraint(a2_disc<=(18*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a2_disc>=(10*base_price_p2)/100)
                    # if sku2>=200:
                    #     a2_constraints=m.add_constraint(a1_disc<=(35*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a1_disc>=(20*base_price_p1)/100)
                    # if sku2>=300:
                    #     a2_constraints=m.add_constraint(a1_disc<=(30*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a1_disc>=(18*base_price_p1)/100)
            a_total_constraints=m.add_constraint(m.sum([a1_disc,0])<=(50*base_price_p1)/100)
            
            for i in lst_stockiest_do:
                if i in class_b:
                    if sku1<=100:
                        a1_constraints=m.add_constraint(a1_disc<=(8*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(6*base_price_p1)/100)
                    # elif sku1>=100:
                    #     a1_constraints=m.add_constraint(a1_disc<=(16*base_price_p1)/100)
                    #     a11_constraints=m.add_constraint(a1_disc>=(8*base_price_p1)/100)
                    elif sku1<=200:
                        a1_constraints=m.add_constraint(a1_disc<=(25*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(15*base_price_p1)/100)
                    # elif sku1<=400:
                    #     a1_constraints=m.add_constraint(a1_disc<=(35.25*base_price_p1)/100)
                    #     a11_constraints=m.add_constraint(a1_disc>=(21.25*base_price_p1)/100)
                    elif sku1<=300:
                        a1_constraints=m.add_constraint(a1_disc<=(22*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(15*base_price_p1)/100)
                    elif sku1<=400:
                        a1_constraints=m.add_constraint(a1_disc<=(29.25*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.25*base_price_p1)/100)
                    elif sku1<=500:
                        a1_constraints=m.add_constraint(a1_disc<=(28.50*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.50*base_price_p1)/100)
                    elif sku1<=600:
                        a1_constraints=m.add_constraint(a1_disc<=(29.75*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.75*base_price_p1)/100)
                    elif sku1<=700:
                        a1_constraints=m.add_constraint(a1_disc<=(36*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21*base_price_p1)/100)
                    elif sku1<=800:
                        a1_constraints=m.add_constraint(a1_disc<=(31.25*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.25*base_price_p1)/100)
                    elif sku1<=900:
                        a1_constraints=m.add_constraint(a1_disc<=(33.50*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(21.50*base_price_p1)/100)
                    elif sku1<=1000:
                        a1_constraints=m.add_constraint(a1_disc<=(35.75*base_price_p1)/100)
                        a11_constraints=m.add_constraint(a1_disc>=(27.75*base_price_p1)/100)
                    # if sku2<=100
                    #     a2_constraints=m.add_constraint(a2_disc<=(10*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a2_disc>=(6*base_price_p2)/100)
                    # if sku2>100:
                    #     a2_constraints=m.add_constraint(a2_disc<=(16*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a2_disc>=(8*base_price_p2)/100)
                    # if sku2>200:
                    #     a2_constraints=m.add_constraint(a1_disc<=(28*base_price_p2)/100)
                    #     a21_constraints=m.add_constraint(a1_disc>=(16*base_price_p1)/100)
            # a_total_constraints=m.add_constraint(m.sum([a1_disc,a2_disc])<=(50*base_price_p1)/100)
            # base_price_p1=st.number_input("Enter the base Price: ",250,5000,250,10)
            # base_price_p2=st.number_input("Enter the base Price: ",250,5000,250,10)
            # if sb=='Androanagen Tablets 10S':
            #     base_price_p1=1200
            # elif sb=='Es Body Wash 200ML':
            #     base_price_p1=1540
            # elif sb=='Aknay Bar 100GM':
            #     base_price_p1=2765
            # elif sb=='Androanagen Solution 100Ml':
            #     base_price_p1=3215
            # elif sb=='Acnemoist Cream 60GM':
            #     base_price_p1=1230
            # elif sb=='Banatan Cream 50GM':
            #     base_price_p1=1300
            # elif sb=='Canthex 10 Capsules':
            #     base_price_p1=750
            # elif sb=='Triobloc Cream 25GM':
            #     base_price_p1=1543
            # elif sb=="""Nixiyax 15 'S""":
            #     base_price_p1=589
            # for key,Values in dict.items():
                
            m.maximize(sku1*(base_price_p1-a1_disc))
            sol1 = m.solve()
            solution=str(sol1)
            solution=solution.replace('solution for:'," ")
            # st.write(solution)
            st.write('Base price is Rs'+' ',base_price_p1)
            st.write(solution)





if __name__ == '__main__':
    main()