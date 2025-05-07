## Part 1: Quick (and dirty) EDA

**TIP: Do this data exploration in a "scrap" file so you can explore quickly and messily.**

_We are going to use this dataset (`input_data2/housing_train.csv`) for the regression and ML assignments, and likely on the prediction contest. The general focus will be on modelling the **Sale Price** (`v_SalePrice`)._

You should do the usual data exploration. 
- Sample basics: What is the unit of observation? What time spans are covered?
- Look for outliers, missing values, or data errors
- Note what variables are continuous or discrete numbers, which variables are categorical variables (and whether the categorical ordering is meaningful)     
- You should read up on what all the variables mean from the documentation in the data folder.
- Visually explore the relationship between `v_Sale_Price` and other variables.
  - For continuous variables - take note of whether the relationship seems linear or quadratic or polynomial
  - For categorical variables - maybe try a box plot for the various levels?
  - Take notes about what you find    

(Delete this cell that contains these instructions before submission, so that your submission starts with the "EDA" section below this.)      

## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('input_data2/housing_train.csv')

# Display basic info
df.shape, df.columns
```




    ((1941, 81),
     Index(['parcel', 'v_MS_SubClass', 'v_MS_Zoning', 'v_Lot_Frontage',
            'v_Lot_Area', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour',
            'v_Utilities', 'v_Lot_Config', 'v_Land_Slope', 'v_Neighborhood',
            'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
            'v_Overall_Qual', 'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add',
            'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd',
            'v_Mas_Vnr_Type', 'v_Mas_Vnr_Area', 'v_Exter_Qual', 'v_Exter_Cond',
            'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
            'v_BsmtFin_Type_1', 'v_BsmtFin_SF_1', 'v_BsmtFin_Type_2',
            'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_Heating',
            'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_1st_Flr_SF',
            'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
            'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
            'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_Kitchen_Qual',
            'v_TotRms_AbvGrd', 'v_Functional', 'v_Fireplaces', 'v_Fireplace_Qu',
            'v_Garage_Type', 'v_Garage_Yr_Blt', 'v_Garage_Finish', 'v_Garage_Cars',
            'v_Garage_Area', 'v_Garage_Qual', 'v_Garage_Cond', 'v_Paved_Drive',
            'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch',
            'v_Screen_Porch', 'v_Pool_Area', 'v_Pool_QC', 'v_Fence',
            'v_Misc_Feature', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_Sale_Type',
            'v_Sale_Condition', 'v_SalePrice'],
           dtype='object'))




```python
df.head()
df.info()
df.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1941 entries, 0 to 1940
    Data columns (total 81 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   parcel             1941 non-null   object 
     1   v_MS_SubClass      1941 non-null   int64  
     2   v_MS_Zoning        1941 non-null   object 
     3   v_Lot_Frontage     1620 non-null   float64
     4   v_Lot_Area         1941 non-null   int64  
     5   v_Street           1941 non-null   object 
     6   v_Alley            136 non-null    object 
     7   v_Lot_Shape        1941 non-null   object 
     8   v_Land_Contour     1941 non-null   object 
     9   v_Utilities        1941 non-null   object 
     10  v_Lot_Config       1941 non-null   object 
     11  v_Land_Slope       1941 non-null   object 
     12  v_Neighborhood     1941 non-null   object 
     13  v_Condition_1      1941 non-null   object 
     14  v_Condition_2      1941 non-null   object 
     15  v_Bldg_Type        1941 non-null   object 
     16  v_House_Style      1941 non-null   object 
     17  v_Overall_Qual     1941 non-null   int64  
     18  v_Overall_Cond     1941 non-null   int64  
     19  v_Year_Built       1941 non-null   int64  
     20  v_Year_Remod/Add   1941 non-null   int64  
     21  v_Roof_Style       1941 non-null   object 
     22  v_Roof_Matl        1941 non-null   object 
     23  v_Exterior_1st     1941 non-null   object 
     24  v_Exterior_2nd     1941 non-null   object 
     25  v_Mas_Vnr_Type     769 non-null    object 
     26  v_Mas_Vnr_Area     1923 non-null   float64
     27  v_Exter_Qual       1941 non-null   object 
     28  v_Exter_Cond       1941 non-null   object 
     29  v_Foundation       1941 non-null   object 
     30  v_Bsmt_Qual        1891 non-null   object 
     31  v_Bsmt_Cond        1891 non-null   object 
     32  v_Bsmt_Exposure    1889 non-null   object 
     33  v_BsmtFin_Type_1   1891 non-null   object 
     34  v_BsmtFin_SF_1     1940 non-null   float64
     35  v_BsmtFin_Type_2   1891 non-null   object 
     36  v_BsmtFin_SF_2     1940 non-null   float64
     37  v_Bsmt_Unf_SF      1940 non-null   float64
     38  v_Total_Bsmt_SF    1940 non-null   float64
     39  v_Heating          1941 non-null   object 
     40  v_Heating_QC       1941 non-null   object 
     41  v_Central_Air      1941 non-null   object 
     42  v_Electrical       1940 non-null   object 
     43  v_1st_Flr_SF       1941 non-null   int64  
     44  v_2nd_Flr_SF       1941 non-null   int64  
     45  v_Low_Qual_Fin_SF  1941 non-null   int64  
     46  v_Gr_Liv_Area      1941 non-null   int64  
     47  v_Bsmt_Full_Bath   1939 non-null   float64
     48  v_Bsmt_Half_Bath   1939 non-null   float64
     49  v_Full_Bath        1941 non-null   int64  
     50  v_Half_Bath        1941 non-null   int64  
     51  v_Bedroom_AbvGr    1941 non-null   int64  
     52  v_Kitchen_AbvGr    1941 non-null   int64  
     53  v_Kitchen_Qual     1941 non-null   object 
     54  v_TotRms_AbvGrd    1941 non-null   int64  
     55  v_Functional       1941 non-null   object 
     56  v_Fireplaces       1941 non-null   int64  
     57  v_Fireplace_Qu     1001 non-null   object 
     58  v_Garage_Type      1836 non-null   object 
     59  v_Garage_Yr_Blt    1834 non-null   float64
     60  v_Garage_Finish    1834 non-null   object 
     61  v_Garage_Cars      1940 non-null   float64
     62  v_Garage_Area      1940 non-null   float64
     63  v_Garage_Qual      1834 non-null   object 
     64  v_Garage_Cond      1834 non-null   object 
     65  v_Paved_Drive      1941 non-null   object 
     66  v_Wood_Deck_SF     1941 non-null   int64  
     67  v_Open_Porch_SF    1941 non-null   int64  
     68  v_Enclosed_Porch   1941 non-null   int64  
     69  v_3Ssn_Porch       1941 non-null   int64  
     70  v_Screen_Porch     1941 non-null   int64  
     71  v_Pool_Area        1941 non-null   int64  
     72  v_Pool_QC          13 non-null     object 
     73  v_Fence            365 non-null    object 
     74  v_Misc_Feature     63 non-null     object 
     75  v_Misc_Val         1941 non-null   int64  
     76  v_Mo_Sold          1941 non-null   int64  
     77  v_Yr_Sold          1941 non-null   int64  
     78  v_Sale_Type        1941 non-null   object 
     79  v_Sale_Condition   1941 non-null   object 
     80  v_SalePrice        1941 non-null   int64  
    dtypes: float64(11), int64(26), object(44)
    memory usage: 1.2+ MB





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
      <th>v_MS_SubClass</th>
      <th>v_Lot_Frontage</th>
      <th>v_Lot_Area</th>
      <th>v_Overall_Qual</th>
      <th>v_Overall_Cond</th>
      <th>v_Year_Built</th>
      <th>v_Year_Remod/Add</th>
      <th>v_Mas_Vnr_Area</th>
      <th>v_BsmtFin_SF_1</th>
      <th>v_BsmtFin_SF_2</th>
      <th>...</th>
      <th>v_Wood_Deck_SF</th>
      <th>v_Open_Porch_SF</th>
      <th>v_Enclosed_Porch</th>
      <th>v_3Ssn_Porch</th>
      <th>v_Screen_Porch</th>
      <th>v_Pool_Area</th>
      <th>v_Misc_Val</th>
      <th>v_Mo_Sold</th>
      <th>v_Yr_Sold</th>
      <th>v_SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1941.000000</td>
      <td>1620.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1923.000000</td>
      <td>1940.000000</td>
      <td>1940.000000</td>
      <td>...</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58.088614</td>
      <td>69.301235</td>
      <td>10284.770222</td>
      <td>6.113344</td>
      <td>5.568264</td>
      <td>1971.321999</td>
      <td>1984.073158</td>
      <td>104.846074</td>
      <td>436.986598</td>
      <td>49.247938</td>
      <td>...</td>
      <td>92.458011</td>
      <td>49.157135</td>
      <td>22.947965</td>
      <td>2.249871</td>
      <td>16.249871</td>
      <td>3.386399</td>
      <td>52.553838</td>
      <td>6.431221</td>
      <td>2006.998454</td>
      <td>182033.238022</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.946015</td>
      <td>23.978101</td>
      <td>7832.295527</td>
      <td>1.401594</td>
      <td>1.087465</td>
      <td>30.209933</td>
      <td>20.837338</td>
      <td>184.982611</td>
      <td>457.815715</td>
      <td>169.555232</td>
      <td>...</td>
      <td>127.020523</td>
      <td>70.296277</td>
      <td>65.249307</td>
      <td>22.416832</td>
      <td>56.748086</td>
      <td>43.695267</td>
      <td>616.064459</td>
      <td>2.745199</td>
      <td>0.801736</td>
      <td>80407.100395</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1470.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>13100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7420.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1965.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2006.000000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9450.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>361.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2007.000000</td>
      <td>161900.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11631.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>168.000000</td>
      <td>735.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2008.000000</td>
      <td>215000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>164660.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2008.000000</td>
      <td>2009.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>742.000000</td>
      <td>1012.000000</td>
      <td>407.000000</td>
      <td>576.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2008.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>




```python
print("Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False).head(10))
```

    Missing Values:
    v_Pool_QC          1928
    v_Misc_Feature     1878
    v_Alley            1805
    v_Fence            1576
    v_Mas_Vnr_Type     1172
    v_Fireplace_Qu      940
    v_Lot_Frontage      321
    v_Garage_Cond       107
    v_Garage_Qual       107
    v_Garage_Finish     107
    dtype: int64


## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

_Note: If you are using VS Code, these might not display correctly. Add a "\\" in front of the underscores in the variable names, so `\text{v_Lot_Area}` becomes `\text{v\_Lot\_Area}`._

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v\_Lot\_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v\_Lot\_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v\_Lot\_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v\_Lot\_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v\_Yr\_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v\_Yr\_Sold==2007})+ \beta_2 * (\text{v\_Yr\_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? One student in five years has beat that. 
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
# Create log-transformed variables
df['log_v_Lot_Area'] = np.log(df['v_Lot_Area'])
df['log_v_SalePrice'] = np.log(df['v_SalePrice'])
```


```python
# Create dummy variables for years
df['Yr_2007'] = (df['v_Yr_Sold'] == 2007).astype(int)
df['Yr_2008'] = (df['v_Yr_Sold'] == 2008).astype(int)


```


```python
# Convert v_Kitchen_Qual to a numeric score
df['Kitchen_Qual_Num'] = df['v_Kitchen_Qual'].map({
    'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
})
```


```python
# Model 1
X1 = sm.add_constant(df['v_Lot_Area'])
model1 = sm.OLS(df['v_SalePrice'], X1).fit()

# Model 2
X2 = sm.add_constant(df['log_v_Lot_Area'])
model2 = sm.OLS(df['v_SalePrice'], X2).fit()

# Model 3
X3 = sm.add_constant(df['v_Lot_Area'])
model3 = sm.OLS(df['log_v_SalePrice'], X3).fit()

# Model 4
X4 = sm.add_constant(df['log_v_Lot_Area'])
model4 = sm.OLS(df['log_v_SalePrice'], X4).fit()


# Model 5
X5 = sm.add_constant(df['v_Yr_Sold'])
model5 = sm.OLS(df['log_v_SalePrice'], X5).fit()

# Model 6
X6 = sm.add_constant(df[['Yr_2007', 'Yr_2008']])
model6 = sm.OLS(df['log_v_SalePrice'], X6).fit()

# Model 7
# Using: v_Overall_Qual, v_Gr_Liv_Area, v_Garage_Cars, v_Total_Bsmt_SF, Kitchen_Qual_Num
custom_vars = ['v_Overall_Qual', 'v_Gr_Liv_Area', 'v_Garage_Cars', 'v_Total_Bsmt_SF', 'Kitchen_Qual_Num', 'log_v_SalePrice']
df_model7 = df[custom_vars].dropna()  

X7 = sm.add_constant(df_model7.drop(columns='log_v_SalePrice'))
y7 = df_model7['log_v_SalePrice']
model7 = sm.OLS(y7, X7).fit()
```


```python
#Results
results_table = summary_col(
    results=[model1, model2, model3, model4, model5, model6, model7],
    model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7'],
    stars=True,
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'No. observations': lambda x: f"{int(x.nobs)}"
    }
)

print(results_table)
```

    
    ====================================================================================================
                        Model 1         Model 2      Model 3    Model 4   Model 5   Model 6    Model 7  
    ----------------------------------------------------------------------------------------------------
    const            154789.5502*** -327915.8023*** 11.8941*** 9.4051*** 22.2932   12.0229*** 10.4265***
                     (2911.5906)    (30221.3471)    (0.0146)   (0.1511)  (22.9368) (0.0161)   (0.0223)  
    v_Lot_Area       2.6489***                      0.0000***                                           
                     (0.2252)                       (0.0000)                                            
    log_v_Lot_Area                  56028.1700***              0.2883***                                
                                    (3315.1392)                (0.0166)                                 
    v_Yr_Sold                                                            -0.0051                        
                                                                         (0.0114)                       
    Yr_2007                                                                        0.0256               
                                                                                   (0.0222)             
    Yr_2008                                                                        -0.0103              
                                                                                   (0.0228)             
    v_Overall_Qual                                                                            0.1145*** 
                                                                                              (0.0046)  
    v_Gr_Liv_Area                                                                             0.0002*** 
                                                                                              (0.0000)  
    v_Garage_Cars                                                                             0.0996*** 
                                                                                              (0.0067)  
    v_Total_Bsmt_SF                                                                           0.0001*** 
                                                                                              (0.0000)  
    Kitchen_Qual_Num                                                                          0.0837*** 
                                                                                              (0.0083)  
    R-squared        0.0666         0.1284          0.0646     0.1350    0.0001    0.0014     0.8114    
    R-squared Adj.   0.0661         0.1279          0.0641     0.1345    -0.0004   0.0004     0.8109    
    No. observations 1941           1941            1941       1941      1941      1941       1939      
    R-squared        0.067          0.128           0.065      0.135     0.000     0.001      0.811     
    ====================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 



```python
# Print beta1 coefficients for Models 1-6
print("Model 1: β₁ (v_Lot_Area) =", model1.params['v_Lot_Area'])
print("Model 2: β₁ (log_v_Lot_Area) =", model2.params['log_v_Lot_Area'])
print("Model 3: β₁ (v_Lot_Area) =", model3.params['v_Lot_Area'])
print("Model 4: β₁ (log_v_Lot_Area) =", model4.params['log_v_Lot_Area'])
print("Model 5: β₁ (v_Yr_Sold) =", model5.params['v_Yr_Sold'])
print("Model 6: β₁ (Yr_2007) =", model6.params['Yr_2007'])
print("Model 6: β₁ (Yr_2008) =", model6.params['Yr_2008'])
```

    Model 1: β₁ (v_Lot_Area) = 2.648935000718187
    Model 2: β₁ (log_v_Lot_Area) = 56028.16996046539
    Model 3: β₁ (v_Lot_Area) = 1.3092338465836234e-05
    Model 4: β₁ (log_v_Lot_Area) = 0.28826331962292995
    Model 5: β₁ (v_Yr_Sold) = -0.005114348195957956
    Model 6: β₁ (Yr_2007) = 0.025590319971642246
    Model 6: β₁ (Yr_2008) = -0.010281565074488656



```python
beta_model2 = model2.params['log_v_Lot_Area']
print("Model 2: β₁ (log_v_Lot_Area) =", beta_model2)
```

    Model 2: β₁ (log_v_Lot_Area) = 56028.16996046539


# Interpretation for Model 2:
A one-unit increase in the natural log of v_Lot_Area (which roughly corresponds 
to multiplying the lot area by 2.718) is associated with an increase in the sale price 
of approximately $56,028. This shows a strong positive relationship when using the log transformation.


```python
# Print beta coefficient with extra decimal precision
beta_model3 = model3.params['v_Lot_Area']
print("Model 3: β₁ (v_Lot_Area) =", format(beta_model3, '.6f'))
```

    Model 3: β₁ (v_Lot_Area) = 0.000013


# Interpretation for Model 3:
For each additional square foot of lot area, the natural log of the sale price increases by approximately 0.000013.
While the coefficient looks small due to the large scale of lot area, over large changes this effect accumulates.


```python
# Print R² values for Models 1-4
print("Model 1 R² =", round(model1.rsquared, 4))
print("Model 2 R² =", round(model2.rsquared, 4))
print("Model 3 R² =", round(model3.rsquared, 4))
print("Model 4 R² =", round(model4.rsquared, 4))


```

    Model 1 R² = 0.0666
    Model 2 R² = 0.1284
    Model 3 R² = 0.0646
    Model 4 R² = 0.135


Among Models 1-4, Model 4 (log(Sale Price) ~ log(v_Lot_Area)) best explains the data because 
it has the highest R². This specification captures a multiplicative (elasticity) relationship 
between lot area and sale price, improving the model's fit.


```python
beta_model5 = model5.params['v_Yr_Sold']
print("Model 5: β₁ (v_Yr_Sold) =", beta_model5)
```

    Model 5: β₁ (v_Yr_Sold) = -0.005114348195957956


# Interpretation for Model 5:
A one-unit increase in the sale year is associated with a -0.0051 change in the natural log of sale price.
This implies that for each additional year, the sale price changes by approximately -0.51% 
(after exponentiating the coefficient, if interpreted in percentage terms).




```python
alpha_model6 = model6.params['const']
beta_2007 = model6.params['Yr_2007']
beta_2008 = model6.params['Yr_2008']

print("Model 6: Constant (α) =", alpha_model6)
print("Model 6: β₁ for (Yr_2007) =", beta_2007)
print("Model 6: β₁ for (Yr_2008) =", beta_2008)
```

    Model 6: Constant (α) = 12.022869210751953
    Model 6: β₁ for (Yr_2007) = 0.025590319971642246
    Model 6: β₁ for (Yr_2008) = -0.010281565074488656




# Interpretation for Model 6:
- The constant (α) of 12.0229 represents the expected log sale price for the base year 
  (i.e., the year not captured by the dummies, likely 2006).
- The coefficient for the 2007 dummy (β₁ ≈ 0.0256) indicates that houses sold in 2007 have, on average, 
  a log sale price that is 2.56% higher than the base year.
- The coefficient for the 2008 dummy (β₁ ≈ -0.0103) indicates that houses sold in 2008 have, on average, 
  a log sale price that is about -1.03% different (lower if negative) than the base year.




```python
print("Model 5 R² =", model5.rsquared)
print("Model 6 R² =", model6.rsquared)
```

    Model 5 R² = 0.00010327268770837783
    Model 6 R² = 0.0014364742956374243


# Comparison of  R² of Models 5 and 6
The R² of Model 6 is slightly higher than that of Model 5 because Model 6 uses dummy variables for 
specific years (2007 and 2008), which can capture non-linear differences in sale prices across years more flexibly than a simple linear trend in Model 5.



# Model 7 includes the following variables 
 - v_Overall_Qual
 - v_Gr_Liv_Area
 - v_Garage_Cars
 - v_Total_Bsmt_SF
 - Kitchen_Qual_Num

Model 7 R² = 0.8113945626518312

# Interpretation:
Model 7, which uses five predictors related to house quality and size, achieves an R² of approximately 0.8114. 
This means that these five variables explain about 81.1% of the variation in the log of sale price.


