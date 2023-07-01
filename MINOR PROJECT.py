#!/usr/bin/env python
# coding: utf-8

# # RECOMMENDER SYSTEMS
Data set Information :
This is a transnational data set that contains all the transaction occurring between 01/12/2010 and 09/12/2011 for a UK based and registred non-store online retail. the company mainly sells unique all occasion gifts. Many customers of the company are wholesalers.
Attributes used in the data set :

InvoiceNo: Invoice number Nominal a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c' ,  it indicates a cancellation.

StockCode: Product (item) code. Nominal a 5-digit integral number uniquely assigned to each distinct product

Description: Product (item) name . Nominal.

Quantity: The quantities of each product (item) per transaction .  Numeric.

InvoiceDate: Invoice date and time. Numeric- the day and time when each transaction was generated.

UnitPrice: Unit price. Numeric product price per unit in sterling pound 

CustomerID: Customer number. Nominal-a 5-digit integral number uniquely assigned to each customer.

Country: Country name .  Nominal- the name of the country where each customer resides.

# Data pre-processing
we will first take a quick look data set in the segment that we are going to evalute.
. Import the necessary libraries and data set 

# In[1]:


#install openpyxl using the command given below
get_ipython().system('pip install openpyxl')


# In[2]:


import numpy as np ,pandas as pd ,re, scipy as sp ,scipy.stats


# In[3]:


pd.options.mode.chained_assignment = None


# In[4]:


datasetURL = 'https://query.data.world/s/5einhbnpmkhxwyrtghfgwmx5dlqaun'


# In[5]:


df1 = pd.read_excel(datasetURL)


# In[6]:


df1.size


# In[7]:



original_df = df1.copy()
original_df.head()


# In[8]:


df1.isnull().sum().sort_values(ascending =False)


# In[9]:


df1=df1.dropna(axis =0)


# In[10]:


df1.isnull().sum()


# In[11]:


# formating date and time
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'], format = '%d/%m/%Y %H:%M')


# In[12]:


##checking the string 
df1['Description'] = df1['Description'].str.replace('.' ,'').str.upper().str.strip()


# In[13]:


df1['Description'].replace('\s+' , ' ' , regex = True)


# In[14]:


df1['InvoiceNo'] = df1['InvoiceNo'].astype(str).str.upper()


# In[15]:


df1['StockCode'] = df1['StockCode'].str.upper()


# In[16]:


df1['Country'] = df1['Country'].str.upper()


# In[17]:


df1.head()


# In[18]:


df1.drop(df1[(df1.Quantity > 0) & (df1.InvoiceNo.str.contains('C')== True)].index ,inplace = True)


# In[19]:


df1.drop(df1[(df1.Quantity < 0) & (df1.InvoiceNo.str.contains('C')== False)].index ,inplace = True)


# In[20]:


df1.drop(df1[df1.Description.str.contains('?' , regex = False)==True].index , inplace =  True)


# In[21]:


df1.drop(df1[df1.UnitPrice == 0].index , inplace =  True)


# In[22]:


for index  , value in df1.StockCode[df1.Description.isna() == True].items():
    if pd.notna(df1.Description [df1.StockCode == value]).sum() != 0: 
               df1.Description[index] = df1.Description[df1.Stockcode == value].mode()[0]
    else:
                df1.drop(index = index , inplace = True)


# In[23]:


df1['Description'] = df1[ 'Description'].astype(str)


# In[24]:


## Adding desired Features
df1['FinalPrice'] = df1['Quantity']*df1['UnitPrice']
df1['InvoiceMonth'] = df1['InvoiceDate'].apply(lambda x : x.strftime('%B'))
df1['Day of Week'] = df1['InvoiceDate'].dt.day_name()
df1.shape


# Exploratory data analysis
In this section we will visualize the data to have a clear vision and gain insights into it.
we import our data in this section and update to determine so that we can work on the data from the time series .

In the first plot of the subplot below , we can see the top 20 goods purchased by client with respect to price and in the most quantities.
# In[25]:


#import necessary libraries and the cleaned dataset.
import pandas as pd , numpy as np ,matplotlib.pyplot as plt , seaborn as sns


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


cleaned_data =df1


# In[28]:


cleaned_data.index = pd.to_datetime(cleaned_data.index , format = '%Y-%m-%d %H:%M')


# In[29]:


#top 20 products by quantity  and final price
sns.set_style('whitegrid')
TopTwenty = cleaned_data.groupby('Description')['Quantity'].agg('sum').sort_values(ascending = False)[0:20]


# In[30]:


Top20Price = cleaned_data.groupby('Description')['FinalPrice'].agg('sum').sort_values(ascending = False)[0:20]


# In[31]:


##creating subplot
fig ,axs = plt.subplots(nrows = 2 ,ncols = 1 ,figsize = (12,12))
plt.subplots_adjust(hspace = 0.3)
fig.suptitle('Best Selling Products By Amount and Value' , fontsize = 15 , x=0.4 , y = 0.98)
sns.barplot(x=TopTwenty.values , y = Top20Price.index ,ax=axs[0]).set(xlabel = 'Total amount of sales')
axs[0].set_title('By Amount' , size = 12 ,fontweight = 'bold')
sns.barplot(x = Top20Price.values , y = Top20Price.index , ax = axs[1]).set(xlabel = 'Total values of sales')
axs[1].set_title('By value' , size = 12 , fontweight = 'bold')
plt.show


# In[32]:


## now let us find out the items that were returned the most, and customers with corrosponding country

ReturnedItems = cleaned_data[cleaned_data.Quantity < 0].groupby('Description')['Quantity'].sum()


# In[33]:


ReturnedItems = ReturnedItems.abs().sort_values(ascending = False)[0:10]


# In[34]:


ReturnCust = cleaned_data[cleaned_data.Quantity < 0 ].groupby(['CustomerID' , 'Country'])['Quantity'].sum()


# In[35]:


ReturnCust = ReturnCust.abs().sort_values(ascending = False)[0:10]


# In[36]:


## creating subplot

fig ,[ ax1 ,ax2] = plt.subplots(nrows = 2 ,ncols = 1 , figsize = (12 ,10))
ReturnedItems.sort_values().plot(kind = 'barh' , ax= ax1).set_title('Most Returned Items' , fontsize = 15)


ReturnCust.sort_values().plot(kind = 'barh' , ax= ax2).set_title('Customers With Most Returns' , fontsize = 15)



ax1.set(xlabel = 'Quantity')
ax2.set(xlabel = 'Quantity')
plt.subplots_adjust(hspace = 0.4)
plt.show()


# In[37]:


## pie chart of week sold

cleaned_data.groupby('Day of Week')['FinalPrice'].sum().plot(kind = 'pie' , autopct = '%.3f%%' ,figsize = (7,7)).set(ylabel = '')

plt.title('Percentage of Sales Values of Day of Week' , fontsize = 17)
plt.show()


# # MODEL BUILDING

# In[38]:


##Creating a excel file for the cleaned_data

cleaned_data.to_excel('Online_Retail_data.xlsx')


# In[39]:


##loading the excel file cleaned data into a dataframe
final_data = pd.read_excel('Online_Retail_data.xlsx' , index_col = 0)


# In[40]:


final_data.head()


# In[41]:


## convert stockcode into  to string
final_data ['StockCode'] = final_data[ 'StockCode'].astype(str)


# In[42]:


#list of unique customers
Customers = final_data['CustomerID'].unique().tolist()


# In[43]:


len(Customers)


# In[44]:


import random
random.shuffle(Customers)


# In[45]:


Customers_train = [Customers[i] for i in range  (round(0.9*len(Customers)))]


# In[46]:


train_df = final_data[final_data['CustomerID'].isin(Customers_train)]

validation_df = final_data [ ~final_data['CustomerID'].isin(Customers_train)]


# In[47]:


#build sequence of purchases made by customer
get_ipython().system('pip install tqdm')
from tqdm import tqdm
purchases_train =[]
for i in tqdm (Customers_train):                          ## we could have used tqdm(train_df)??
    temp = train_df [train_df['CustomerID']== i]['StockCode'].tolist()
    purchases_train.append(temp)
    purchases_val =[]
    


# In[48]:


for i in tqdm (validation_df['CustomerID'].unique()):                          ## we could have used tqdm(train_df)??
    temp = validation_df [validation_df['CustomerID']== i]['StockCode'].tolist()
    purchases_val.append(temp)


# In[49]:


get_ipython().system('pip  install gensim')
from gensim.models import Word2Vec


# In[50]:


model = Word2Vec(window = 10 ,sg =1 , hs = 0 ,negative = 10 ,alpha = 0.03 , min_alpha =0.0007 , seed  = 14)


# In[51]:


model.build_vocab(purchases_train , progress_per = 200)


# In[52]:


model.train(purchases_train , total_examples = model.corpus_count , epochs = 10 ,report_delay = 1)


# In[53]:


model.init_sims (replace = True)


# In[54]:


model.save('my_Word2Vec_2.model')


# In[55]:


# RECOMMENDING PRODUCTS
product = train_df[['StockCode' , 'Description']]

#REMOVE DUPLICATE
product.drop_duplicates(inplace =True , subset = 'StockCode' ,keep ='last')


# In[56]:


##create product id and product description dictionary
product_dict = product.groupby('StockCode')['Description'].apply(list).to_dict()


# In[57]:


product_dict['84796A']


# In[58]:


##Lets Create a Function which will take product vector(v) as input and return top six similar product

def get_Similar_item(v ,n=6):
    ms = model.wv.most_similar(positive=[v], topn = n+1 )[1:]
    my_ms = []
    for j in ms:
        pair = (product_dict[j[0]][0] , j[1])
        my_ms.append(pair)
        
    return my_ms
    
    


# In[59]:


get_Similar_item(model['84796A'])

The my_aggr_vec function takes in a list of product IDs as its input. It then iterates over this list and checks whether each product ID exists in a pre-trained word embeddings model model.

If the ID exists in model, the corresponding vector representation of the product is retrieved using model.wv[i], where i is the current product ID being processed. This vector representation is appended to a list p_vec.

If the ID does not exist in model, the function continues to the next iteration without adding anything to p_vec.

Once all the product IDs have been processed, the function returns the average of all the vector representations present in p_vec. This is done using np.mean(p_vec, axis=0) which computes the element-wise mean of all vectors along the 0th axis.

The get_Similar_item function takes in a vector representation aggr_vec and a parameter topn as its input. It then calculates the cosine similarity between the input vector aggr_vec and all other vector representations in the word embeddings model model.

The function then returns the top n most similar vectors based on cosine similarity, along with their corresponding product IDs. This is done using the similar_by_vector method of the model.wv object, which returns a list of tuples containing the IDs and cosine similarity scores of the top n similar products. The function then returns this list of tuples as its output.




# In[60]:


## create a function that takes a list of product IDs and return a 100_dimensional Vector which is the mean of the product vector in input list
def my_aggr_vec(products):
    p_vec = []
    for i in products:
        try:
            p_vec.append(model.wv[i])
        except KeyError:
            continue
    return np.mean(p_vec, axis=0)

def get_Similar_item(aggr_vec, topn=5):
    similar_items = model.wv.similar_by_vector(aggr_vec, topn=topn)
    return similar_items


# In[61]:


get_Similar_item( my_aggr_vec(purchases_val[5]))

The code get_Similar_item( my_aggr_vec(purchases_val[5])) is returning a list of tuples, where each tuple represents a similar item and its similarity score with the input vector.

The first item in the list is ('nan', 0.9995766282081604), which means that the most similar item to the input vector is labeled 'nan' and has a similarity score of 0.9995766282081604. However, this label 'nan' is not informative, and it's likely that there is some issue with the data that is causing it to appear.
# In[62]:


get_Similar_item( my_aggr_vec(purchases_val[2][-10:]))

The code get_Similar_item( my_aggr_vec(purchases_val[5][-10:])) is calculating the most similar items to the last 10 items in the purchase history of the 6th customer.

The output will be a list of tuples, where each tuple represents a similar item and its similarity score with the input vector. The first item in the list is likely to be the same 'nan' label as before, followed by other items and their similarity scores.

The exact output will depend on the content of the purchase history for the 6th customer, as well as the word embeddings used to calculate the similarity scores.
# In[ ]:




