#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel("C:\\Users\\ROHIT\\Desktop\\PORTFOLIO PROJECTS\\Tablaeu\\Data Set\\superstore_USA.xlsx")
data


# In[5]:


data.describe()


# In[6]:


data.info()


# In[38]:


data.fillna(method="ffill", inplace = True)


# In[39]:


data.isnull().sum()


# # Product Category Distribution

# In[19]:


ax = sns.countplot(data = data, x="Product Category")
ax.bar_label(ax.containers[0])


# In[ ]:


#from the above chart we have analysed that:
#number of Office supplies are highest in number then come Technology and then Furniture.


# # Sales and profit according to customer segment

# In[58]:


gb =data.groupby('Customer Segment').sum(numeric_only=True)
gb


# In[67]:


customer_segment = data['Customer Segment']
sales = data['Sales']
profit = data['Profit']
grouped_df = data.groupby('Customer Segment').sum(numeric_only = True)
fig, ax = plt.subplots(figsize=(10,9))
index = range(len(grouped_df))
bar_width = 0.35

sales_bars = ax.bar(index, grouped_df['Sales'], bar_width, label='Sales')
profit_bars = ax.bar([i + bar_width for i in index], grouped_df['Profit'], bar_width, label='Profit')

for bar in sales_bars + profit_bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2 - 0.2, yval, round(yval, 2), va='bottom')

ax.set_xlabel('Customer Segment')
ax.set_ylabel('Amount')
ax.set_title('Sales and Profit by Customer Segment')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(grouped_df.index)
ax.legend()

ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.show()



# In[ ]:


#from above the chart we can analyse that:
#Corporates are the biggest buyers and also the most profitable segment


# In[40]:


grouped_data = data.groupby('Product Category')['Sales'].sum().reset_index()
categories = grouped_data['Product Category'].tolist()
sales = grouped_data['Sales'].tolist()
fig, ax = plt.subplots()
outer_circle = ax.pie(sales, labels=categories, autopct=lambda pct:'{:.1f}%'.format(pct), startangle=90, wedgeprops=dict(width=0.3), radius = 1)
inner_circle = ax.pie([1], radius=0.7, colors='white')
ax.axis('equal')
plt.title('Total Sales Distribution by Product Categories')
plt.show()


# In[ ]:


#from above donut chart we can see our sales are in highest in Technology category and lowest in office supplies. 
#we should focus more on office supplies and improve its sales.


# In[41]:


grouped_data = data.groupby(['Product Category', 'Product Sub-Category'])['Sales'].sum().reset_index()
pivot_data = grouped_data.pivot_table(index='Product Sub-Category', columns='Product Category', values='Sales', aggfunc='sum')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="plasma", ax=ax)
plt.xlabel('Product Category')
plt.ylabel('Product Sub-Category')
plt.title('Sales of Product Sub Categories by Product Category')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#from above we can see the sales of each product-sub category.


# # Top 10 States according to sales

# In[20]:


gb = data.groupby('State or Province')['Sales'].sum().reset_index()
sr = gb.sort_values(by='Sales', ascending=False).head(10)
plt.figure(figsize=(16,9))
bars = plt.bar(sr['State or Province'], sr['Sales'])
plt.xlabel('States', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.title('Top 10 Sales by State', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
formatter = FuncFormatter(lambda x, _: '₹{:,.0f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)
plt.yticks(fontsize=12)
plt.tight_layout()
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, '{:,.0f}'.format(height),
            ha='center', va='bottom', fontsize=10)
plt.show()



# In[ ]:


#from above plot we can see the top 10 performing states according to sales


# # Sales Trend Over Time

# In[11]:


sd = data[['Order Date', 'Sales']].copy()
sd['Order Date'] = pd.to_datetime(sd['Order Date'])
sd.set_index('Order Date', inplace = True)
plt.figure(figsize=(16,9))
plt.plot(sd.resample('M').sum(), marker='o', linestyle='-')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
cursor = mplcursors.cursor(hover=True)
cursor.connect('add', lambda sel: sel.annotation.set_text(f'{sel.artist.get_label()}: {sel.target[1]}'))
plt.show()


# In[ ]:


#from above plot you can clearly see sales trend over time

