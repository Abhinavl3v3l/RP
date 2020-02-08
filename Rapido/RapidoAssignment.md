
# Part 1. Exploratory data analysis


## Data Overview

Getting the know and understand of data.


```python
#importing libs

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
sb.set(style="ticks")

#Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#References - https://github.com/uber/h3-py
#           - https://github.com/uber/h3-py-notebooks
from h3 import h3
```


```python
# Reading Excel File
rapido = pd.read_excel("Rapido Data Analyst Assignment DataSet (1).xlsx") # customer_id as index

# First Look at the data
rapido.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
      <td>17.442705</td>
      <td>78.387878</td>
      <td>17.457829</td>
      <td>78.399056</td>
      <td>2.806</td>
      <td>12.609667</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID002</td>
      <td>CUST_002</td>
      <td>1546709309524</td>
      <td>17.490189</td>
      <td>78.415512</td>
      <td>17.450548</td>
      <td>78.367294</td>
      <td>11.991</td>
      <td>24.075200</td>
      <td>119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID003</td>
      <td>CUST_003</td>
      <td>1546709331857</td>
      <td>17.370108</td>
      <td>78.515045</td>
      <td>17.377041</td>
      <td>78.517921</td>
      <td>1.322</td>
      <td>8.708300</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID004</td>
      <td>CUST_004</td>
      <td>1546709358403</td>
      <td>17.439314</td>
      <td>78.443001</td>
      <td>17.397131</td>
      <td>78.516586</td>
      <td>11.822</td>
      <td>24.037550</td>
      <td>121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID005</td>
      <td>CUST_005</td>
      <td>1546709386884</td>
      <td>17.432325</td>
      <td>78.381966</td>
      <td>17.401625</td>
      <td>78.400032</td>
      <td>6.978</td>
      <td>16.120867</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>



#### Dimensions Information


```python
# Dimension of Data
print('Dataframe Dimensions \n MEASURES    :',rapido.shape[0],' \n DIMENSIONS  :',rapido.shape[1])
```

    Dataframe Dimensions 
     MEASURES    : 44587  
     DIMENSIONS  : 10
    

There are 44587 rows and 10 Columns 


```python
i = 1
col_List = rapido.columns.tolist() 

# print('\nDimension :\n',Col_list )
for col in col_List:
    print(str(i)+'. '+col)
    i += 1
```

    1. trip_id
    2. customer_id
    3. timestamp
    4. pick_lat
    5. pick_lng
    6. drop_lat
    7. drop_lng
    8. travel_distance
    9. travel_time
    10. trip_fare
    

Serial No.|Column names|Description
---       |---         |---
1 | trip_id        | Unique identifier for trips
2 | customer_id    | Unique identifier for customers
3 | timestamp      | Time stamp of the trip in Epoch format(**in ms**)
4 | pick_lat       | Latitude of the pickup location
5 | pick_lng       | Longitude of the pickup location
6 | drop_lat       | Latitude of the drop location
7 | drop_lng       | Longitude of the drop location
8 | travel_distance| Distance of trip measured in KMs
9 | travel_time    | Duration of the trip measured in Minutes
10| trip_fare      | Trip fare calculated in Rupees

#### Unique Values


```python
print ('\nThere are no missing values:\n', rapido.isnull().sum())
```

    
    There are no missing values:
     trip_id            0
    customer_id        0
    timestamp          0
    pick_lat           0
    pick_lng           0
    drop_lat           0
    drop_lng           0
    travel_distance    0
    travel_time        0
    trip_fare          0
    dtype: int64
    

**No missing value found**

---

#### Distinct Values


```python
print ('\nUnique values :  \n',rapido.nunique())
```

    
    Unique values :  
     trip_id            44587
    customer_id        19139
    timestamp          44586
    pick_lat           29677
    pick_lng           19902
    drop_lat           29962
    drop_lng           20530
    travel_distance    11756
    travel_time        43980
    trip_fare            284
    dtype: int64
    


```python
print('Variables Data Types\n',rapido.dtypes) 
```

    Variables Data Types
     trip_id             object
    customer_id         object
    timestamp            int64
    pick_lat           float64
    pick_lng           float64
    drop_lat           float64
    drop_lng           float64
    travel_distance    float64
    travel_time        float64
    trip_fare            int64
    dtype: object
    


```python
rapido.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.458700e+04</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
      <td>44587.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.546632e+12</td>
      <td>17.427919</td>
      <td>78.435542</td>
      <td>17.427891</td>
      <td>78.434897</td>
      <td>5.094359</td>
      <td>15.819835</td>
      <td>53.872833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.518684e+08</td>
      <td>0.030049</td>
      <td>0.053333</td>
      <td>0.037722</td>
      <td>0.054965</td>
      <td>3.365008</td>
      <td>23.462865</td>
      <td>33.430462</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.546368e+12</td>
      <td>17.330339</td>
      <td>78.308258</td>
      <td>12.921696</td>
      <td>77.548103</td>
      <td>-1.000000</td>
      <td>0.022750</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.546503e+12</td>
      <td>17.405327</td>
      <td>78.386562</td>
      <td>17.405660</td>
      <td>78.385010</td>
      <td>2.744500</td>
      <td>8.428525</td>
      <td>36.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.546611e+12</td>
      <td>17.432136</td>
      <td>78.438866</td>
      <td>17.431213</td>
      <td>78.438164</td>
      <td>4.299000</td>
      <td>13.126250</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.546772e+12</td>
      <td>17.446777</td>
      <td>78.480839</td>
      <td>17.446907</td>
      <td>78.480255</td>
      <td>6.679500</td>
      <td>20.111167</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.546886e+12</td>
      <td>17.529791</td>
      <td>78.600647</td>
      <td>17.736155</td>
      <td>78.634804</td>
      <td>52.801000</td>
      <td>4134.388700</td>
      <td>1670.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido.skew()
```




    timestamp            0.135380
    pick_lat             0.003127
    pick_lng             0.165726
    drop_lat           -38.196997
    drop_lng             0.081117
    travel_distance      1.706639
    travel_time        124.499745
    trip_fare            7.562344
    dtype: float64



##### OBSERVATIONS

1. We can clearly avoide pick and drop longitude and latitude respectively, they do not signify measure but geographical locations
2. Travel Distance(in KM) - Needs more observation 
    - Column looks funny. 
    - Minimum Travel Distance of -1. Clear Anomaly
    
3. Travel Times(in minutes)  - 
    - Maximum time travel on bike is 4134 : 68.9 Hours or 2 Days, 20 Hours, 53 Minutes and 59 Seconds. 
    - Minimum time traveled on bike is  0.02275 or 1.365 seconds

There are clear errors in data which need to be rectified

---

# Problem Statements
### Part 1. Exploratory data analysis
Perform an exploratory data analysis on the given dataset and share your findings.

### Part 2. Metric calculation
What is the average duration between the 1st trip and the 2nd trip of customers? Note: Consider only
the customers who have done 2 or more trips.

### Part 3. Model building
Build a model to predict trip_fare using travel_distance and travel_time. Measure the accuracy of the
model and use the model to predict trip_fare for a trip with travel_distance of 3.5 kms and travel_time

### Part 4. Top Hex clusters
Top 5 pairs of hex (resolution=8) clusters where most of the trips happened? You can refer to the
library listed below to get hexid for a given latitude and longitude.
-  https://github.com/uber/h3-py

Expected output

Rank Hex | pair (source_hexid, destination_hexid) |  Total trips
---|---|---
1 | 883c8e4159fffff to 88754e6499fffff |34
2 | 883db66b55fffff to 883c8e4159fffff | 28

## EDA - Exploratory Data Analysis

- `Part 1` of problem statement deals with entrire data. 
- Each problem statement deals with different slices of dataset.
- So to keep it simple each part of problem statement has its own EDA.


```python
##  correlation between variables indicates that as one variable changes in value,
##  the other variable tends to change in a specific direction.

rapido_corr = rapido
# rapido_corr = rapido[['travel_distance', 'travel_time', 'trip_fare']]
corr = rapido_corr.corr()

def hmap(corr,strr='HeatMap'):
    ax = sb.heatmap(
        corr,
        annot=True,
        cmap='RdYlGn',
        linewidths=0.2,
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True
        )
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.title(strr)
    plt.show()
hmap(corr,'Correlation between all columns of Data,(WITHERRORS)')
```


![png](output_19_0.png)


Based on correlation matrix above
There is positive correlation between 
    - Distance travelled and trip's fare
    - Time travelled and trip's fare
    



```python
corr = rapido[['travel_distance','travel_time','trip_fare']].corr()
hmap(corr,'Correlation between Travel Time, Travel Distance and Trip Fair')


```


![png](output_21_0.png)



```python
ax = sb.pairplot(rapido[
    ['travel_distance','travel_time','trip_fare']
])
plt.title('Pair Plot - Travel Time, Travel Distance and Trip Fair')
plt.show()
```


![png](output_22_0.png)



### NOTE: THESE ARE OBSERVATIONS WITHOUT ERROR HANDLING
For example, we have a `travel_time` greater than `4000`

# What is the average duration between the 1st trip and the 2nd trip of customers? Note: Consider only the customers who have done 2 or more trips.

For this problem the only needed columns are `trip_id`, `customer_id` and `timestamp`.
Since other measure are irrelavent and have no correlations, as can be seen in corelation matrix above.


```python
#Not altering the orginial dataset.
rapidodt = rapido.copy()
rapidodt = rapidodt[['trip_id','customer_id','timestamp']]
rapidodt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID002</td>
      <td>CUST_002</td>
      <td>1546709309524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID003</td>
      <td>CUST_003</td>
      <td>1546709331857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID004</td>
      <td>CUST_004</td>
      <td>1546709358403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID005</td>
      <td>CUST_005</td>
      <td>1546709386884</td>
    </tr>
  </tbody>
</table>
</div>




```python
# One Customer can have more than one trip, all trips have unique ID.
rapidodt.loc[rapidodt['customer_id']== 'CUST_001']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
    </tr>
    <tr>
      <th>1389</th>
      <td>ID1390</td>
      <td>CUST_001</td>
      <td>1546871083066</td>
    </tr>
    <tr>
      <th>1866</th>
      <td>ID1867</td>
      <td>CUST_001</td>
      <td>1546873649099</td>
    </tr>
    <tr>
      <th>4983</th>
      <td>ID4984</td>
      <td>CUST_001</td>
      <td>1546765642873</td>
    </tr>
    <tr>
      <th>5269</th>
      <td>ID5270</td>
      <td>CUST_001</td>
      <td>1546769557670</td>
    </tr>
    <tr>
      <th>5528</th>
      <td>ID5529</td>
      <td>CUST_001</td>
      <td>1546773038937</td>
    </tr>
    <tr>
      <th>33261</th>
      <td>ID33262</td>
      <td>CUST_001</td>
      <td>1546676690138</td>
    </tr>
    <tr>
      <th>36625</th>
      <td>ID36626</td>
      <td>CUST_001</td>
      <td>1546702978281</td>
    </tr>
  </tbody>
</table>
</div>



## Data Manipulation



```python
# Calculating number of trips per customer
rapido_series = rapidodt['customer_id'].value_counts()

# Which is same as number of unique cutomers
print(rapido_series.shape[0] == rapido['customer_id'].nunique()) 
```

    True
    


```python
# Trips/Customer Calculation

rapido_series_index = rapido_series.index
rapido_series = rapido_series.tolist()
rapido2 = pd.DataFrame(data = rapido_series_index,columns=['Customer_ID'])
rapido2['trip_count'] = rapido_series
print('No of unique customer : ',len(rapido2))
rapido2.head()
```

    No of unique customer :  19139
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer_ID</th>
      <th>trip_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST_279</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST_4119</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST_3100</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST_1120</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST_1237</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



Trips Per Customer Tableau visualization. 
Total Customer : 44587

![Trips per Customer](images/tripcount.png)

**Customer 279 is an MVP(MVC)**


```python
# Scatter plot for top 20 TRIPS/CUSTOMERS
ax = sb.relplot(y="trip_count", x="Customer_ID", data=rapido2.head(20));
sb.set(style="whitegrid")
fig = plt.gcf()
fig.set_size_inches(10,8)
ax.set_xticklabels(
        rotation=45,
        horizontalalignment='right'
    );
plt.title('SCATTER PLOT: TRIPS/CUSTOMERS')
plt.show()


tc  = (rapido2['trip_count'])
ax = sb.distplot(tc,hist=False);
fig.set_size_inches(10,8)
plt.title('Spread: TRIPS/CUSTOMERS')
plt.show()


print('Skewness of(POSITIVE TAIL)' , tc.skew(), ' Kurtoise of(VERY POINTY) : ',tc.kurt())


```


![png](output_32_0.png)



![png](output_32_1.png)


    Skewness of(POSITIVE TAIL) 3.159787080150149  Kurtoise of(VERY POINTY) :  22.815107790445268
    


```python
# Customers with more then 1 trip.

rapido2_filtered = rapido2[rapido2['trip_count']>=2]
rapido_filtered_customer_list = rapido2_filtered['Customer_ID'].tolist()
print('There are ',len(rapido_filtered_customer_list),' customer with more then one trip')

rapidodt['customer_id'].astype(str)

# Filtering DataSet : Only have values of customer with more than one trip.
rapidodt = rapidodt[rapidodt.customer_id.isin(rapido_filtered_customer_list)] 


```

    There are  9130  customer with more then one trip
    


```python
#ORIGINAL DATASET
print(rapido.shape)
rapido.head() 
```

    (44587, 10)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
      <td>17.442705</td>
      <td>78.387878</td>
      <td>17.457829</td>
      <td>78.399056</td>
      <td>2.806</td>
      <td>12.609667</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID002</td>
      <td>CUST_002</td>
      <td>1546709309524</td>
      <td>17.490189</td>
      <td>78.415512</td>
      <td>17.450548</td>
      <td>78.367294</td>
      <td>11.991</td>
      <td>24.075200</td>
      <td>119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID003</td>
      <td>CUST_003</td>
      <td>1546709331857</td>
      <td>17.370108</td>
      <td>78.515045</td>
      <td>17.377041</td>
      <td>78.517921</td>
      <td>1.322</td>
      <td>8.708300</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID004</td>
      <td>CUST_004</td>
      <td>1546709358403</td>
      <td>17.439314</td>
      <td>78.443001</td>
      <td>17.397131</td>
      <td>78.516586</td>
      <td>11.822</td>
      <td>24.037550</td>
      <td>121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID005</td>
      <td>CUST_005</td>
      <td>1546709386884</td>
      <td>17.432325</td>
      <td>78.381966</td>
      <td>17.401625</td>
      <td>78.400032</td>
      <td>6.978</td>
      <td>16.120867</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtered Dataset of Customer with more then 1 trip.

print(rapidodt.shape)
rapidodt.head()
```

    (34578, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID003</td>
      <td>CUST_003</td>
      <td>1546709331857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID004</td>
      <td>CUST_004</td>
      <td>1546709358403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID005</td>
      <td>CUST_005</td>
      <td>1546709386884</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ID006</td>
      <td>CUST_006</td>
      <td>1546709396752</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initial a new dataframe which will store value of first two trips of customer 
rapidodt_top2 = pd.DataFrame()

# Of all the customer with trips more than one, filter the top 2 and save in the new dataframe
for customer_id in rapido_filtered_customer_list:
    df_temp = rapidodt.loc[rapidodt['customer_id']== customer_id].sort_values('timestamp').head(2)
    rapidodt_top2 = rapidodt_top2.append(df_temp)
```


```python
#Data Time Conversion

rapido_customer_avg = pd.DataFrame(data=rapido_filtered_customer_list,columns=['customer_id'])
rapidodt_top2['timestamp'] = pd.to_datetime(rapidodt_top2['timestamp'],unit='ms')
rapidodt_top2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17350</th>
      <td>ID17351</td>
      <td>CUST_279</td>
      <td>2019-01-02 05:59:14.312</td>
    </tr>
    <tr>
      <th>17583</th>
      <td>ID17584</td>
      <td>CUST_279</td>
      <td>2019-01-02 06:33:58.294</td>
    </tr>
    <tr>
      <th>19645</th>
      <td>ID19646</td>
      <td>CUST_4119</td>
      <td>2019-01-02 11:47:13.914</td>
    </tr>
    <tr>
      <th>20432</th>
      <td>ID20433</td>
      <td>CUST_4119</td>
      <td>2019-01-02 13:01:56.379</td>
    </tr>
    <tr>
      <th>15398</th>
      <td>ID15399</td>
      <td>CUST_3100</td>
      <td>2019-01-02 02:48:03.038</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Time Difference between first two trip of all customer with trip count more than 1

l = []
for customer_id in rapido_filtered_customer_list:
    temp = rapidodt_top2.loc[rapidodt['customer_id']== customer_id] 
    var = temp.iloc[1]['timestamp'] - temp.iloc[0]['timestamp']
    l.append(var)
```


```python
print('Minimum time between trip by a customer : ',np.min(l),'and max difference is : ' ,np.max(l))
print('\nAverage duration between the 1st trip and the 2nd trip of customers : ',np.mean(l))
```

    Minimum time between trip by a customer :  0 days 00:01:41.770000 and max difference is :  5 days 14:43:14.647000
    
    Average duration between the 1st trip and the 2nd trip of customers :  1 days 00:47:09.691580
    

### Average duration between the 1st trip and the 2nd trip of customers :  **1 days 47 minutes**


---

# Part 3. Model building
Build a model to predict trip_fare using travel_distance and travel_time. Measure the accuracy of the
model and use the model to predict trip_fare for a trip with travel_distance of 3.5 kms and travel_time
of 15 minutes.


Train Data = ['travel_distance','travel_time']


Test Data =[3.5,15]


Serial No. |Column names      | Description
---|---|---
1 |         travel_distance   | Distance of trip measured in KMs
2 |         travel_time       | Duration of the trip measured in Minutes
3 |         trip_fare         | Trip fare calculated in Rupees


### EDA and Error Handling

`Time Related Error`


```python
rapido_model = rapido[['travel_distance', 'travel_time', 'trip_fare']]
print(rapido_model.sort_values('travel_distance').head(10))

```

           travel_distance  travel_time  trip_fare
    12636             -1.0    11.838667        959
    145               -1.0    19.247367        959
    17206             -1.0    11.014850        959
    42371              0.0     0.311767         20
    152                0.0     0.242750         20
    23053              0.0     0.182883         20
    35669              0.0     0.092817         20
    155                0.0     0.261333         20
    19505              0.0     0.366333         20
    6262               0.0     0.078567         20
    

As seen above there are a lot of errors in the data.
There are 3 negative distance values which make no sense.

For example, A customer travel  `-1 KM` in `11 minutes ` and was charged `959 rupees`. 

There 3 many entries with `0 KM ` distance travelled. 

**DELETING ALL SUCH ANOMALIES**

---

`Fare Related Error`


```python
corr = rapido_model.sort_values('trip_fare',ascending=False).head().corr()
hmap(corr,'Correlation between Travel Time, Travel Distance and Trip Fair')

sb.pairplot(rapido_model[
    ['travel_distance','travel_time','trip_fare']
])

```


![png](output_45_0.png)





    <seaborn.axisgrid.PairGrid at 0x168d3d74a20>




![png](output_45_2.png)


**This contradicts the original data set**


```python
corr = rapido[['travel_distance','travel_time','trip_fare']].corr()
hmap(corr,'Correlation between Travel Time, Travel Distance and Trip Fair')
rapido.sort_values('trip_fare',ascending=False).head()


```


![png](output_47_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42530</th>
      <td>ID42531</td>
      <td>CUST_18770</td>
      <td>1546519406927</td>
      <td>17.425972</td>
      <td>78.340454</td>
      <td>17.441185</td>
      <td>78.391289</td>
      <td>9.4</td>
      <td>267.825483</td>
      <td>1670</td>
    </tr>
    <tr>
      <th>27304</th>
      <td>ID27305</td>
      <td>CUST_2196</td>
      <td>1546858611537</td>
      <td>17.404274</td>
      <td>78.492340</td>
      <td>17.411062</td>
      <td>78.493576</td>
      <td>2.7</td>
      <td>10.949667</td>
      <td>959</td>
    </tr>
    <tr>
      <th>17206</th>
      <td>ID17207</td>
      <td>CUST_7151</td>
      <td>1546407689604</td>
      <td>17.433880</td>
      <td>78.384483</td>
      <td>17.441532</td>
      <td>78.362556</td>
      <td>-1.0</td>
      <td>11.014850</td>
      <td>959</td>
    </tr>
    <tr>
      <th>145</th>
      <td>ID146</td>
      <td>CUST_142</td>
      <td>1546712267334</td>
      <td>17.396492</td>
      <td>78.479980</td>
      <td>17.368525</td>
      <td>78.533218</td>
      <td>-1.0</td>
      <td>19.247367</td>
      <td>959</td>
    </tr>
    <tr>
      <th>36239</th>
      <td>ID36240</td>
      <td>CUST_6416</td>
      <td>1546699878502</td>
      <td>17.397224</td>
      <td>78.480911</td>
      <td>17.428328</td>
      <td>78.450958</td>
      <td>1.0</td>
      <td>1.596217</td>
      <td>959</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido_model.sort_values('trip_fare',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42530</th>
      <td>9.4</td>
      <td>267.825483</td>
      <td>1670</td>
    </tr>
    <tr>
      <th>27304</th>
      <td>2.7</td>
      <td>10.949667</td>
      <td>959</td>
    </tr>
    <tr>
      <th>17206</th>
      <td>-1.0</td>
      <td>11.014850</td>
      <td>959</td>
    </tr>
    <tr>
      <th>145</th>
      <td>-1.0</td>
      <td>19.247367</td>
      <td>959</td>
    </tr>
    <tr>
      <th>36239</th>
      <td>1.0</td>
      <td>1.596217</td>
      <td>959</td>
    </tr>
  </tbody>
</table>
</div>



First 4 entries are clear anomalies in dataset, since there is no relation between the distance, time or fare
Hence removing them from the dataset makes sense. 



```python
print('9th row from the last was fared wrong. Need to remove ')
rapido_model.sort_values('trip_fare',ascending=False).head()

```

    9th row from the last was fared wrong. Need to remove 
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42530</th>
      <td>9.4</td>
      <td>267.825483</td>
      <td>1670</td>
    </tr>
    <tr>
      <th>27304</th>
      <td>2.7</td>
      <td>10.949667</td>
      <td>959</td>
    </tr>
    <tr>
      <th>17206</th>
      <td>-1.0</td>
      <td>11.014850</td>
      <td>959</td>
    </tr>
    <tr>
      <th>145</th>
      <td>-1.0</td>
      <td>19.247367</td>
      <td>959</td>
    </tr>
    <tr>
      <th>36239</th>
      <td>1.0</td>
      <td>1.596217</td>
      <td>959</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering
rapido_model = rapido_model[~(rapido_model['trip_fare'] == 20) & (rapido_model['travel_distance'] > .9)]

rapido_model = rapido_model[rapido_model['travel_distance'] > 0]
```

Most people travelled between distance 0 and 10 KM 

> Observation - Customers prefer rapido for short trips.


```python

# Speed : 10 - 120
# Trip-Fare : < 522
rapido_model['speed'] = rapido_model['travel_distance']/(rapido_model['travel_time']/60)
rapido_model.sort_values(['travel_distance'],ascending=False).head(20)

# Choosing value of speed between 10 and 120
rapido_model = rapido_model[rapido_model['speed'].between(10,120)]

## values above 522 are anomalies as the data do not add up.
rapido_model = rapido_model[rapido_model['trip_fare']<522] 
rapido_model.sort_values('travel_time',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32164</th>
      <td>25.690</td>
      <td>142.484950</td>
      <td>275</td>
      <td>10.817985</td>
    </tr>
    <tr>
      <th>18078</th>
      <td>26.676</td>
      <td>122.900400</td>
      <td>314</td>
      <td>13.023229</td>
    </tr>
    <tr>
      <th>14422</th>
      <td>20.544</td>
      <td>117.289150</td>
      <td>239</td>
      <td>10.509412</td>
    </tr>
    <tr>
      <th>40951</th>
      <td>18.706</td>
      <td>108.660950</td>
      <td>199</td>
      <td>10.329010</td>
    </tr>
    <tr>
      <th>17500</th>
      <td>19.886</td>
      <td>107.459217</td>
      <td>226</td>
      <td>11.103375</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(rapido_model['travel_distance'].value_counts(normalize= True,bins=10))

ax = sb.distplot(rapido_model['travel_distance'],hist=False,color="g", kde_kws={"shade": True},norm_hist=False);
plt.title('Distance Distribution')
plt.show()
```

    (0.854, 5.486]      0.631750
    (5.486, 10.07]      0.284227
    (10.07, 14.655]     0.064532
    (14.655, 19.239]    0.015164
    (19.239, 23.824]    0.003233
    (23.824, 28.409]    0.000808
    (28.409, 32.993]    0.000238
    (42.162, 46.747]    0.000024
    (37.578, 42.162]    0.000024
    (32.993, 37.578]    0.000000
    Name: travel_distance, dtype: float64
    


![png](output_54_1.png)



```python
#Checking for missing Values
print ('\nThere are no missing values:\n', rapido_model.isnull().sum())
```

    
    There are no missing values:
     travel_distance    0
    travel_time        0
    trip_fare          0
    speed              0
    dtype: int64
    


```python
print(rapido_model['travel_time'].value_counts(normalize= True,bins=10))

ax = sb.distplot(rapido_model['travel_time'],hist=False,color="r", kde_kws={"shade": True});
plt.title('Time Distribution')
plt.show()
```

    (0.732, 15.035]       0.579340
    (15.035, 29.196]      0.329459
    (29.196, 43.358]      0.070926
    (43.358, 57.519]      0.015711
    (57.519, 71.68]       0.003185
    (71.68, 85.841]       0.000879
    (85.841, 100.002]     0.000261
    (100.002, 114.163]    0.000166
    (114.163, 128.324]    0.000048
    (128.324, 142.485]    0.000024
    Name: travel_time, dtype: float64
    


![png](output_56_1.png)



```python
print(rapido_model['speed'].value_counts(normalize= True,bins=10))

ax = sb.distplot(rapido_model['speed'],hist=False, kde_kws={"shade": True});
plt.title('Speed Distribution')
plt.show()
```

    (9.894, 20.886]      0.550556
    (20.886, 31.769]     0.401074
    (31.769, 42.652]     0.043711
    (42.652, 53.535]     0.003565
    (53.535, 64.418]     0.000523
    (75.301, 86.184]     0.000190
    (64.418, 75.301]     0.000166
    (97.067, 107.95]     0.000095
    (107.95, 118.833]    0.000071
    (86.184, 97.067]     0.000048
    Name: speed, dtype: float64
    


![png](output_57_1.png)



```python
# print(rapido_model['travel_time'].value_counts(normalize= True,bins=2))

# ax = sb.distplot(rapido_model['travel_time'],hist=False);
# rapido_model2 = rapido_model[rapido_model['travel_time']<61 ]
rapido_model = rapido_model.sort_values('travel_time',ascending=False)
rapido_model.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32164</th>
      <td>25.690</td>
      <td>142.484950</td>
      <td>275</td>
      <td>10.817985</td>
    </tr>
    <tr>
      <th>18078</th>
      <td>26.676</td>
      <td>122.900400</td>
      <td>314</td>
      <td>13.023229</td>
    </tr>
    <tr>
      <th>14422</th>
      <td>20.544</td>
      <td>117.289150</td>
      <td>239</td>
      <td>10.509412</td>
    </tr>
    <tr>
      <th>40951</th>
      <td>18.706</td>
      <td>108.660950</td>
      <td>199</td>
      <td>10.329010</td>
    </tr>
    <tr>
      <th>17500</th>
      <td>19.886</td>
      <td>107.459217</td>
      <td>226</td>
      <td>11.103375</td>
    </tr>
  </tbody>
</table>
</div>



DATA AFTER ERROR HANDLING Exploring Filtered DataSet


```python
corr = rapido_model[['travel_distance','trip_fare']].corr()
hmap(corr,'Correlation between  Travel Distance and Trip Fair')


corr = rapido_model[['travel_time','trip_fare']].corr()
hmap(corr,'Correlation between Travel Time and Trip Fair')
```


![png](output_60_0.png)



![png](output_60_1.png)


After Data filtering we see an increase in correlation between travel metrics.

Since there is a high correlation between Travel Distance and  Trip Fare, of 0.91. Trip Fare is in a linear relationship with distance.


```python
# Splitting into two 
train_data = rapido_model[['travel_distance','travel_time']]
labels = rapido_model['trip_fare']
ax = sb.pairplot(rapido_model[
    ['travel_distance','travel_time','trip_fare']
])
plt.title("Correlation Visualized")
plt.show()
```


![png](output_62_0.png)



```python
X_train,X_test,Y_train,Y_test = train_test_split(train_data,labels,random_state=0)

sgd = SGDClassifier()
sgd.fit(X_train,Y_train)
predictions   = sgd.predict(X_test)
```


```python
X_test_2 = pd.DataFrame([[3.5,15.0]],columns=['travel_distance','travel_time'])
X_test_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction = sgd.predict(X_test_2)
prediction[0]
```




    39




```python
rapido_model[rapido_model['travel_time'] == 3.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



# Part 4. Top Hex clusters
Top 5 pairs of hex (resolution=8) clusters where most of the trips happened? You can refer to the
library listed below to get hexid for a given latitude and longitude.
-  https://github.com/uber/h3-py

Expected output

Rank Hex | pair (source_hexid, destination_hexid) |  Total trips
---|---|---
1 | 883c8e4159fffff to 88754e6499fffff |34
2 | 883db66b55fffff to 883c8e4159fffff | 28




### Errors in Data
This image show the Error in drop location.Exluding all location outside Hyderabad
![Error In Geo Data](images/Error.png)

### Error Handling
Based on findings from Tablue Worksheet.
Removing Entries or locations which are outside Hyderabad.
Remove 4 Drop Entries from spatial part of data with Ids.
 - ID25712
 - ID35159
 - ID39114
 - ID30988


```python
rapido[rapido['drop_lat'] == 17.4829464]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39113</th>
      <td>ID39114</td>
      <td>CUST_3041</td>
      <td>1546492009096</td>
      <td>17.415167</td>
      <td>78.425278</td>
      <td>17.482946</td>
      <td>78.032783</td>
      <td>3.647</td>
      <td>13.02025</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido[rapido['drop_lat'] == 17.3520184]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35158</th>
      <td>ID35159</td>
      <td>CUST_4572</td>
      <td>1546692501613</td>
      <td>17.394796</td>
      <td>78.435707</td>
      <td>17.352018</td>
      <td>78.230537</td>
      <td>2.836</td>
      <td>6.5822</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido[rapido['drop_lat'] == 12.9216957]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25711</th>
      <td>ID25712</td>
      <td>CUST_1551</td>
      <td>1546845012421</td>
      <td>17.446897</td>
      <td>78.388855</td>
      <td>12.921696</td>
      <td>77.548103</td>
      <td>4.183</td>
      <td>11.418383</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido[rapido['drop_lat'] == 17.7361546]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30987</th>
      <td>ID30988</td>
      <td>CUST_4562</td>
      <td>1546657274827</td>
      <td>17.405493</td>
      <td>78.450882</td>
      <td>17.736155</td>
      <td>78.479485</td>
      <td>11.17</td>
      <td>35.80665</td>
      <td>116</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido_hex_geo = rapido.copy()
rapido_hex_geo  = rapido_hex_geo[(rapido_hex_geo['drop_lat'] != 12.9216957) &  (rapido_hex_geo['drop_lat'] != 17.3520184) & (rapido_hex_geo['drop_lat'] != 17.4829464) & (rapido_hex_geo['drop_lat'] != 17.7361546)]
```


```python
rapido_hex_geo.head() # REMOVE LATER
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>customer_id</th>
      <th>timestamp</th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>travel_distance</th>
      <th>travel_time</th>
      <th>trip_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID001</td>
      <td>CUST_001</td>
      <td>1546709270211</td>
      <td>17.442705</td>
      <td>78.387878</td>
      <td>17.457829</td>
      <td>78.399056</td>
      <td>2.806</td>
      <td>12.609667</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID002</td>
      <td>CUST_002</td>
      <td>1546709309524</td>
      <td>17.490189</td>
      <td>78.415512</td>
      <td>17.450548</td>
      <td>78.367294</td>
      <td>11.991</td>
      <td>24.075200</td>
      <td>119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID003</td>
      <td>CUST_003</td>
      <td>1546709331857</td>
      <td>17.370108</td>
      <td>78.515045</td>
      <td>17.377041</td>
      <td>78.517921</td>
      <td>1.322</td>
      <td>8.708300</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID004</td>
      <td>CUST_004</td>
      <td>1546709358403</td>
      <td>17.439314</td>
      <td>78.443001</td>
      <td>17.397131</td>
      <td>78.516586</td>
      <td>11.822</td>
      <td>24.037550</td>
      <td>121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID005</td>
      <td>CUST_005</td>
      <td>1546709386884</td>
      <td>17.432325</td>
      <td>78.381966</td>
      <td>17.401625</td>
      <td>78.400032</td>
      <td>6.978</td>
      <td>16.120867</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>



After the fix 
![Filtered the Geo Data](images/Fixed.png)


```python
#Uncomment to check changes
#rapido_hex_geo[rapido_hex_geo['drop_lat'] == 12.9216957]
# rapido_hex_geo[rapido_hex_geo['drop_lat'] == 17.3520184]
# rapido_hex_geo[rapido_hex_geo['drop_lat'] == 17.4829464]
# rapido_hex_geo[rapido_hex_geo['drop_lat'] == 17.7361546]
# OR 
#rapido_hex_geo.shape
```


```python
rapido_hex_geo = rapido.copy()
resolution=8

rapido_hex_geo = rapido_hex_geo[['pick_lat','pick_lng','drop_lat','drop_lng']]
rapido_hex_geo["hex_id_pick"] = rapido_hex_geo.apply(lambda row: h3.geo_to_h3(row["pick_lat"], row["pick_lng"], resolution), axis = 1)
rapido_hex_geo["hex_id_drop"] = rapido_hex_geo.apply(lambda row: h3.geo_to_h3(row["pick_lat"], row["pick_lng"], resolution), axis = 1)
```


```python
rapido_hex_geo.head() # REMOVE LATER
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pick_lat</th>
      <th>pick_lng</th>
      <th>drop_lat</th>
      <th>drop_lng</th>
      <th>hex_id_pick</th>
      <th>hex_id_drop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.442705</td>
      <td>78.387878</td>
      <td>17.457829</td>
      <td>78.399056</td>
      <td>8860a259b9fffff</td>
      <td>8860a259b9fffff</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.490189</td>
      <td>78.415512</td>
      <td>17.450548</td>
      <td>78.367294</td>
      <td>8860b19695fffff</td>
      <td>8860b19695fffff</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.370108</td>
      <td>78.515045</td>
      <td>17.377041</td>
      <td>78.517921</td>
      <td>8860a25b4dfffff</td>
      <td>8860a25b4dfffff</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.439314</td>
      <td>78.443001</td>
      <td>17.397131</td>
      <td>78.516586</td>
      <td>8860a25915fffff</td>
      <td>8860a25915fffff</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.432325</td>
      <td>78.381966</td>
      <td>17.401625</td>
      <td>78.400032</td>
      <td>8860a25995fffff</td>
      <td>8860a25995fffff</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rapido_hex_geo.groupby(['hex_id_pick','hex_id_drop']).agg({i:'value_counts' for i in rapido_hex_geo.columns[5:]})
rapido_hex_geo2 = rapido_hex_geo.groupby(["hex_id_pick", "hex_id_drop"]).size().reset_index(name="Trip")
rapido_hex_geo2 = rapido_hex_geo2.sort_values('Trip',ascending=False)
```


```python
rapido_hex_geo2.head() # REMOVE LATER
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hex_id_pick</th>
      <th>hex_id_drop</th>
      <th>Trip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>236</th>
      <td>8860a259b9fffff</td>
      <td>8860a259b9fffff</td>
      <td>1288</td>
    </tr>
    <tr>
      <th>237</th>
      <td>8860a259bbfffff</td>
      <td>8860a259bbfffff</td>
      <td>933</td>
    </tr>
    <tr>
      <th>220</th>
      <td>8860a25995fffff</td>
      <td>8860a25995fffff</td>
      <td>933</td>
    </tr>
    <tr>
      <th>49</th>
      <td>8860a24a65fffff</td>
      <td>8860a24a65fffff</td>
      <td>798</td>
    </tr>
    <tr>
      <th>53</th>
      <td>8860a24a6dfffff</td>
      <td>8860a24a6dfffff</td>
      <td>720</td>
    </tr>
  </tbody>
</table>
</div>




```python
rapido_hex_geo2["Hex pair (source_hexid, destination_hexid)"] = rapido_hex_geo2["hex_id_pick"].astype(str) + ' to ' + rapido_hex_geo2["hex_id_drop"]
rapido_hex_geo2 = rapido_hex_geo2.head()
```


```python
rapido_hex_geo3= rapido_hex_geo2[['Hex pair (source_hexid, destination_hexid)','Trip']]
rapido_hex_geo3 = rapido_hex_geo3.reset_index(drop = True)
```


```python
rapido_hex_geo3.index = np.arange(1,len(rapido_hex_geo3)+1)
rapido_hex_geo3.index.name='Rank'
rapido_hex_geo3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hex pair (source_hexid, destination_hexid)</th>
      <th>Trip</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8860a259b9fffff to 8860a259b9fffff</td>
      <td>1288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8860a259bbfffff to 8860a259bbfffff</td>
      <td>933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8860a25995fffff to 8860a25995fffff</td>
      <td>933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8860a24a65fffff to 8860a24a65fffff</td>
      <td>798</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8860a24a6dfffff to 8860a24a6dfffff</td>
      <td>720</td>
    </tr>
  </tbody>
</table>
</div>


