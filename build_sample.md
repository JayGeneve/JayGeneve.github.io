```python
import os
import pandas as pd
import numpy as np
import re
import fnmatch
import yfinance as yf
import time
from bs4 import BeautifulSoup
from tqdm import tqdm  # Progress bar for loops
import requests
from requests_html import HTMLSession
from utils.near_regex import *  # Contextual sentiment analysis
import glob
from sec_edgar_downloader import Downloader
from tqdm import tqdm 
import shutil
from zipfile import ZipFile

# Ensure output directory exists
os.makedirs("output", exist_ok=True)
```


```python
sp500_file = 'inputs/sp500.csv'

# get it if we haven't 
if not os.path.exists(sp500_file):
    # 2022 dec version of page
    url = 'https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1130173030'
    pd.read_html(url)[0].to_csv(sp500_file,index=False)

# load and look at it
sp500 = pd.read_csv(sp500_file) 

sp500.head(10)
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
      <th>5-Day Return (%)</th>
      <th>BHR Positive Count</th>
      <th>BHR Negative Count</th>
      <th>LM Positive Count</th>
      <th>LM Negative Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
      <td>-4.894901642693785</td>
      <td>1611.0</td>
      <td>1834.0</td>
      <td>257.0</td>
      <td>1295.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
      <td>-0.15424593099550296</td>
      <td>704.0</td>
      <td>654.0</td>
      <td>100.0</td>
      <td>344.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
      <td>3.2793970775906254</td>
      <td>901.0</td>
      <td>1007.0</td>
      <td>157.0</td>
      <td>466.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
      <td>2.596676113214846</td>
      <td>1045.0</td>
      <td>1124.0</td>
      <td>333.0</td>
      <td>751.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
      <td>5.594433474732396</td>
      <td>1227.0</td>
      <td>1038.0</td>
      <td>376.0</td>
      <td>687.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>2022-02-25</td>
      <td>Ticker\nATVI   NaN\nATVI   NaN\ndtype: float64</td>
      <td>1219.0</td>
      <td>1517.0</td>
      <td>302.0</td>
      <td>860.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>2022-02-17</td>
      <td>2.935404363087299</td>
      <td>1124.0</td>
      <td>1064.0</td>
      <td>326.0</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>2022-01-21</td>
      <td>3.650650986923855</td>
      <td>1269.0</td>
      <td>1143.0</td>
      <td>536.0</td>
      <td>710.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADP</td>
      <td>ADP</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Data Processing &amp; Outsourced Services</td>
      <td>Roseland, New Jersey</td>
      <td>1981-03-31</td>
      <td>8670</td>
      <td>1949</td>
      <td>0000008670-22-000038</td>
      <td>2022-08-03</td>
      <td>3.335790342299098</td>
      <td>905.0</td>
      <td>787.0</td>
      <td>231.0</td>
      <td>468.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAP</td>
      <td>Advance Auto Parts</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Automotive Retail</td>
      <td>Raleigh, North Carolina</td>
      <td>2015-07-09</td>
      <td>1158449</td>
      <td>1932</td>
      <td>0001158449-22-000037</td>
      <td>2022-02-15</td>
      <td>-11.590306782126817</td>
      <td>678.0</td>
      <td>679.0</td>
      <td>140.0</td>
      <td>465.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dl = Downloader("Lehigh", "jag325@lehigh.edu", "10k_files")
```


```python
if not os.path.exists('10k_files/10k_files.zip'):
    
    for cik in tqdm(sp500['CIK']): # start with a small subset while we figure things out, remove later
         
        firm_folder = f'10k_files/sec-edgar-filings/{str(cik).zfill(10)}/'  # str(cik).zfill(10)   means that CIK 1234 becomes 0000001234

        # if I haven't downloaded any HTML for this firm (len=0 files on this pattern), do so
        # you could make this more precise and only look for filings during 2022 (if you were downloading across many years)
        
        if len(glob.glob(firm_folder + '/10-K/*/*.html')) == 0:
            
            dl.get("10-K", cik, 
                   limit=1,                  # get the latest filing within window
                   after="2022-01-01",       # does this download filings ON 1/1 or nah? (check)
                   before="2022-12-31",      # does this download filings ON 12/31 or nah? (check)
                   download_details =True    # download the html 
            ) 
    
        # delete the txt files as we go!!!
        # files are of the form: folder/10-K/*/*.txt
        for txt_f in glob.glob(firm_folder + '/10-K/*/*.txt'):
            os.remove(txt_f)    
    
        # pause if there is a problem and the SEC is mad at my spider
        # unneeded! sec-edgar-dl does it for us 

```

    100%|██████████| 509/509 [00:00<00:00, 1813.11it/s]



```python
sp500.head(10)
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
      <th>5-Day Return (%)</th>
      <th>BHR Positive Count</th>
      <th>BHR Negative Count</th>
      <th>LM Positive Count</th>
      <th>LM Negative Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
      <td>-4.894901642693785</td>
      <td>1611.0</td>
      <td>1834.0</td>
      <td>257.0</td>
      <td>1295.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
      <td>-0.15424593099550296</td>
      <td>704.0</td>
      <td>654.0</td>
      <td>100.0</td>
      <td>344.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
      <td>3.2793970775906254</td>
      <td>901.0</td>
      <td>1007.0</td>
      <td>157.0</td>
      <td>466.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
      <td>2.596676113214846</td>
      <td>1045.0</td>
      <td>1124.0</td>
      <td>333.0</td>
      <td>751.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
      <td>5.594433474732396</td>
      <td>1227.0</td>
      <td>1038.0</td>
      <td>376.0</td>
      <td>687.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>2022-02-25</td>
      <td>Ticker\nATVI   NaN\nATVI   NaN\ndtype: float64</td>
      <td>1219.0</td>
      <td>1517.0</td>
      <td>302.0</td>
      <td>860.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>2022-02-17</td>
      <td>2.935404363087299</td>
      <td>1124.0</td>
      <td>1064.0</td>
      <td>326.0</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>2022-01-21</td>
      <td>3.650650986923855</td>
      <td>1269.0</td>
      <td>1143.0</td>
      <td>536.0</td>
      <td>710.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADP</td>
      <td>ADP</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Data Processing &amp; Outsourced Services</td>
      <td>Roseland, New Jersey</td>
      <td>1981-03-31</td>
      <td>8670</td>
      <td>1949</td>
      <td>0000008670-22-000038</td>
      <td>2022-08-03</td>
      <td>3.335790342299098</td>
      <td>905.0</td>
      <td>787.0</td>
      <td>231.0</td>
      <td>468.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAP</td>
      <td>Advance Auto Parts</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Automotive Retail</td>
      <td>Raleigh, North Carolina</td>
      <td>2015-07-09</td>
      <td>1158449</td>
      <td>1932</td>
      <td>0001158449-22-000037</td>
      <td>2022-02-15</td>
      <td>-11.590306782126817</td>
      <td>678.0</td>
      <td>679.0</td>
      <td>140.0</td>
      <td>465.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
files = glob.glob('10k_files/sec-edgar-filings/*/10-K/*/*.html')
print(files)
f'We have {len(files)} HTML files for {len(sp500["CIK"])} firms'


```

    ['10k_files/sec-edgar-filings/0000882095/10-K/0000882095-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000943819/10-K/0000943819-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001166691/10-K/0001166691-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000723254/10-K/0000723254-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000909832/10-K/0000909832-22-000021/primary-document.html', '10k_files/sec-edgar-filings/0000046080/10-K/0000046080-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000016732/10-K/0000016732-22-000093/primary-document.html', '10k_files/sec-edgar-filings/0001001082/10-K/0001558370-22-001816/primary-document.html', '10k_files/sec-edgar-filings/0000100517/10-K/0000100517-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001478242/10-K/0001478242-22-000041/primary-document.html', '10k_files/sec-edgar-filings/0001039684/10-K/0001039684-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0000726728/10-K/0000726728-22-000046/primary-document.html', '10k_files/sec-edgar-filings/0000091419/10-K/0000091419-22-000049/primary-document.html', '10k_files/sec-edgar-filings/0000011544/10-K/0000011544-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001711269/10-K/0001711269-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000106640/10-K/0000106640-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0001418135/10-K/0001418135-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000813828/10-K/0000813828-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000718877/10-K/0001628280-22-003992/primary-document.html', '10k_files/sec-edgar-filings/0001652044/10-K/0001652044-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000014693/10-K/0000014693-22-000069/primary-document.html', '10k_files/sec-edgar-filings/0001037540/10-K/0001656423-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000354950/10-K/0000354950-22-000070/primary-document.html', '10k_files/sec-edgar-filings/0000217346/10-K/0000217346-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001047862/10-K/0001047862-22-000039/primary-document.html', '10k_files/sec-edgar-filings/0001868275/10-K/0001868275-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0001521332/10-K/0001521332-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000006951/10-K/0000006951-22-000043/primary-document.html', '10k_files/sec-edgar-filings/0001524472/10-K/0001524472-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001038357/10-K/0001038357-22-000039/primary-document.html', '10k_files/sec-edgar-filings/0000029989/10-K/0000029989-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0001781335/10-K/0001781335-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001688568/10-K/0001688568-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000874716/10-K/0000874716-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001821825/10-K/0001821825-22-000002/primary-document.html', '10k_files/sec-edgar-filings/0001020569/10-K/0001020569-22-000035/primary-document.html', '10k_files/sec-edgar-filings/0001109357/10-K/0001109357-22-000039/primary-document.html', '10k_files/sec-edgar-filings/0000087347/10-K/0001564590-22-002421/primary-document.html', '10k_files/sec-edgar-filings/0000077360/10-K/0000077360-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000018230/10-K/0000018230-22-000050/primary-document.html', '10k_files/sec-edgar-filings/0001467858/10-K/0001467858-22-000034/primary-document.html', '10k_files/sec-edgar-filings/0000027419/10-K/0000027419-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000009389/10-K/0001558370-22-001251/primary-document.html', '10k_files/sec-edgar-filings/0001300514/10-K/0001300514-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001701605/10-K/0001701605-22-000050/primary-document.html', '10k_files/sec-edgar-filings/0000033213/10-K/0000033213-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000788784/10-K/0001628280-22-003860/primary-document.html', '10k_files/sec-edgar-filings/0000712515/10-K/0000712515-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000947484/10-K/0000947484-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0000813672/10-K/0000813672-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000920522/10-K/0000920522-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0001370637/10-K/0001370637-22-000024/primary-document.html', '10k_files/sec-edgar-filings/0000765880/10-K/0001628280-22-002117/primary-document.html', '10k_files/sec-edgar-filings/0000842023/10-K/0001558370-22-013935/primary-document.html', '10k_files/sec-edgar-filings/0000035527/10-K/0000035527-22-000119/primary-document.html', '10k_files/sec-edgar-filings/0001385157/10-K/0001558370-22-017931/primary-document.html', '10k_files/sec-edgar-filings/0000732717/10-K/0000732717-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0001032208/10-K/0001032208-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001071739/10-K/0001071739-22-000071/primary-document.html', '10k_files/sec-edgar-filings/0001442145/10-K/0001437749-22-004083/primary-document.html', '10k_files/sec-edgar-filings/0000354190/10-K/0001564590-22-005714/primary-document.html', '10k_files/sec-edgar-filings/0001601712/10-K/0001601712-22-000053/primary-document.html', '10k_files/sec-edgar-filings/0000002969/10-K/0000002969-22-000054/primary-document.html', '10k_files/sec-edgar-filings/0000920148/10-K/0000920148-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0001126328/10-K/0001104659-22-020401/primary-document.html', '10k_files/sec-edgar-filings/0000916076/10-K/0001564590-22-005965/primary-document.html', '10k_files/sec-edgar-filings/0000040545/10-K/0000040545-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001024478/10-K/0001024478-22-000093/primary-document.html', '10k_files/sec-edgar-filings/0000882184/10-K/0000882184-22-000184/primary-document.html', '10k_files/sec-edgar-filings/0000914208/10-K/0000914208-22-000319/primary-document.html', '10k_files/sec-edgar-filings/0000891103/10-K/0000891103-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0000064803/10-K/0000064803-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001466258/10-K/0001466258-22-000031/primary-document.html', '10k_files/sec-edgar-filings/0000352915/10-K/0001564590-22-006717/primary-document.html', '10k_files/sec-edgar-filings/0001137774/10-K/0001137774-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000070858/10-K/0000070858-22-000062/primary-document.html', '10k_files/sec-edgar-filings/0000078003/10-K/0000078003-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000002488/10-K/0000002488-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0001045810/10-K/0001045810-22-000036/primary-document.html', '10k_files/sec-edgar-filings/0001318605/10-K/0000950170-22-000796/primary-document.html', '10k_files/sec-edgar-filings/0000707549/10-K/0000707549-22-000107/primary-document.html', '10k_files/sec-edgar-filings/0000884887/10-K/0000884887-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001065280/10-K/0001065280-22-000036/primary-document.html', '10k_files/sec-edgar-filings/0000912595/10-K/0000950170-22-001423/primary-document.html', '10k_files/sec-edgar-filings/0001604778/10-K/0001604778-22-000029/primary-document.html', '10k_files/sec-edgar-filings/0000764478/10-K/0000764478-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001137789/10-K/0001137789-22-000055/primary-document.html', '10k_files/sec-edgar-filings/0000055785/10-K/0000055785-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000004127/10-K/0000004127-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0001060391/10-K/0001060391-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000031462/10-K/0001558370-22-002059/primary-document.html', '10k_files/sec-edgar-filings/0000051143/10-K/0001558370-22-001584/primary-document.html', '10k_files/sec-edgar-filings/0001002910/10-K/0001002910-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000036270/10-K/0001564590-22-005400/primary-document.html', '10k_files/sec-edgar-filings/0000910606/10-K/0000950170-22-001418/primary-document.html', '10k_files/sec-edgar-filings/0000024741/10-K/0001437749-22-003247/primary-document.html', '10k_files/sec-edgar-filings/0000779152/10-K/0000779152-22-000076/primary-document.html', '10k_files/sec-edgar-filings/0000072903/10-K/0000072903-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000004977/10-K/0000004977-22-000058/primary-document.html', '10k_files/sec-edgar-filings/0000277948/10-K/0000277948-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000916365/10-K/0000916365-22-000049/primary-document.html', '10k_files/sec-edgar-filings/0000024545/10-K/0000024545-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000076334/10-K/0000076334-22-000034/primary-document.html', '10k_files/sec-edgar-filings/0001590895/10-K/0001590895-22-000061/primary-document.html', '10k_files/sec-edgar-filings/0000109380/10-K/0000109380-22-000072/primary-document.html', '10k_files/sec-edgar-filings/0001324404/10-K/0001324404-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001283699/10-K/0001283699-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0000915389/10-K/0000915389-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000927628/10-K/0000927628-22-000106/primary-document.html', '10k_files/sec-edgar-filings/0001783180/10-K/0001783180-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000940944/10-K/0000940944-22-000042/primary-document.html', '10k_files/sec-edgar-filings/0001374310/10-K/0001558370-22-001386/primary-document.html', '10k_files/sec-edgar-filings/0001090727/10-K/0001090727-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001699150/10-K/0001628280-22-003991/primary-document.html', '10k_files/sec-edgar-filings/0000018926/10-K/0000018926-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000804328/10-K/0000804328-22-000021/primary-document.html', '10k_files/sec-edgar-filings/0000746515/10-K/0001564590-22-010381/primary-document.html', '10k_files/sec-edgar-filings/0001120193/10-K/0001120193-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001013237/10-K/0001013237-22-000159/primary-document.html', '10k_files/sec-edgar-filings/0001141391/10-K/0001141391-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0001618921/10-K/0001618921-22-000064/primary-document.html', '10k_files/sec-edgar-filings/0000060667/10-K/0000060667-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000723125/10-K/0000723125-22-000048/primary-document.html', '10k_files/sec-edgar-filings/0001000228/10-K/0001000228-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0000320335/10-K/0000320335-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000065984/10-K/0000065984-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0001364742/10-K/0001564590-22-007117/primary-document.html', '10k_files/sec-edgar-filings/0001551152/10-K/0001551152-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000875320/10-K/0000875320-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001156375/10-K/0001156375-22-000076/primary-document.html', '10k_files/sec-edgar-filings/0001393311/10-K/0001393311-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001326160/10-K/0001326160-22-000072/primary-document.html', '10k_files/sec-edgar-filings/0000315213/10-K/0000315213-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0000086312/10-K/0000086312-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000027904/10-K/0000027904-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0001093557/10-K/0001093557-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0001051470/10-K/0001051470-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000101778/10-K/0000101778-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0000105770/10-K/0001628280-22-003342/primary-document.html', '10k_files/sec-edgar-filings/0001474735/10-K/0001437749-22-004080/primary-document.html', '10k_files/sec-edgar-filings/0001274494/10-K/0001274494-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001410636/10-K/0001410636-22-000048/primary-document.html', '10k_files/sec-edgar-filings/0000858470/10-K/0000858470-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001373715/10-K/0001373715-22-000024/primary-document.html', '10k_files/sec-edgar-filings/0001285785/10-K/0001618034-22-000004/primary-document.html', '10k_files/sec-edgar-filings/0000879169/10-K/0001558370-22-000902/primary-document.html', '10k_files/sec-edgar-filings/0000766704/10-K/0000766704-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001280452/10-K/0001437749-22-004460/primary-document.html', '10k_files/sec-edgar-filings/0000052988/10-K/0000052988-22-000111/primary-document.html', '10k_files/sec-edgar-filings/0000109198/10-K/0000109198-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001687229/10-K/0001687229-22-000002/primary-document.html', '10k_files/sec-edgar-filings/0000004962/10-K/0000004962-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001048911/10-K/0000950170-22-012762/primary-document.html', '10k_files/sec-edgar-filings/0000049196/10-K/0000049196-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000005272/10-K/0001104659-22-024701/primary-document.html', '10k_files/sec-edgar-filings/0001390777/10-K/0001390777-22-000043/primary-document.html', '10k_files/sec-edgar-filings/0001601046/10-K/0001601046-22-000161/primary-document.html', '10k_files/sec-edgar-filings/0001034054/10-K/0001034054-22-000002/primary-document.html', '10k_files/sec-edgar-filings/0000006201/10-K/0000006201-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0000091576/10-K/0000091576-22-000029/primary-document.html', '10k_files/sec-edgar-filings/0001095073/10-K/0001095073-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000277135/10-K/0000277135-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000731802/10-K/0000731802-22-000037/primary-document.html', '10k_files/sec-edgar-filings/0000021076/10-K/0000021076-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0001053507/10-K/0001053507-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0000097476/10-K/0000097476-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000859737/10-K/0000859737-22-000022/primary-document.html', '10k_files/sec-edgar-filings/0001140859/10-K/0001140859-22-000098/primary-document.html', '10k_files/sec-edgar-filings/0000935703/10-K/0000935703-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0000107263/10-K/0000107263-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000936340/10-K/0000936340-22-000077/primary-document.html', '10k_files/sec-edgar-filings/0000773840/10-K/0000773840-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0000084839/10-K/0000084839-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0001551182/10-K/0001551182-22-000004/primary-document.html', '10k_files/sec-edgar-filings/0001013871/10-K/0001013871-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000012208/10-K/0000012208-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000872589/10-K/0001804220-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001063761/10-K/0001558370-22-001845/primary-document.html', '10k_files/sec-edgar-filings/0001755672/10-K/0001755672-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000945841/10-K/0000945841-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000851968/10-K/0000851968-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0000849399/10-K/0000849399-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001050915/10-K/0001050915-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000033185/10-K/0000033185-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0001037646/10-K/0001037646-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000064040/10-K/0000064040-22-000055/primary-document.html', '10k_files/sec-edgar-filings/0001754301/10-K/0001628280-22-022584/primary-document.html', '10k_files/sec-edgar-filings/0000900075/10-K/0000900075-22-000050/primary-document.html', '10k_files/sec-edgar-filings/0000740260/10-K/0000740260-22-000057/primary-document.html', '10k_files/sec-edgar-filings/0001123360/10-K/0001123360-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001179929/10-K/0001179929-22-000025/primary-document.html', '10k_files/sec-edgar-filings/0000885725/10-K/0000885725-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001099219/10-K/0001099219-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0000759944/10-K/0000759944-22-000025/primary-document.html', '10k_files/sec-edgar-filings/0001633917/10-K/0001633917-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000723531/10-K/0000950170-22-012734/primary-document.html', '10k_files/sec-edgar-filings/0001012100/10-K/0001628280-22-003294/primary-document.html', '10k_files/sec-edgar-filings/0001413447/10-K/0001413447-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000883241/10-K/0000883241-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0000075362/10-K/0001564590-22-006237/primary-document.html', '10k_files/sec-edgar-filings/0000097210/10-K/0001193125-22-049828/primary-document.html', '10k_files/sec-edgar-filings/0001108524/10-K/0001108524-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000896878/10-K/0000896878-22-000028/primary-document.html', '10k_files/sec-edgar-filings/0000012927/10-K/0000012927-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000702165/10-K/0000702165-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001133421/10-K/0001133421-22-000004/primary-document.html', '10k_files/sec-edgar-filings/0000045012/10-K/0000045012-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000820027/10-K/0000820027-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0001175454/10-K/0001628280-22-004531/primary-document.html', '10k_files/sec-edgar-filings/0001403568/10-K/0001558370-22-004330/primary-document.html', '10k_files/sec-edgar-filings/0000764180/10-K/0000764180-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000783325/10-K/0000107815-22-000116/primary-document.html', '10k_files/sec-edgar-filings/0000079282/10-K/0000950170-22-001654/primary-document.html', '10k_files/sec-edgar-filings/0000899051/10-K/0000899051-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0000721371/10-K/0000721371-22-000058/primary-document.html', '10k_files/sec-edgar-filings/0000004447/10-K/0001628280-22-004524/primary-document.html', '10k_files/sec-edgar-filings/0001022671/10-K/0001558370-22-002377/primary-document.html', '10k_files/sec-edgar-filings/0001393612/10-K/0001393612-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000827052/10-K/0000827052-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001637459/10-K/0001637459-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0001094285/10-K/0001094285-22-000049/primary-document.html', '10k_files/sec-edgar-filings/0000104169/10-K/0000104169-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000031791/10-K/0000031791-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0001506307/10-K/0001506307-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0001757898/10-K/0001757898-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000798354/10-K/0000798354-22-000004/primary-document.html', '10k_files/sec-edgar-filings/0001262039/10-K/0001262039-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000040533/10-K/0000040533-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001035267/10-K/0001035267-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0001489393/10-K/0001489393-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001585689/10-K/0001585689-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001590955/10-K/0001564590-22-005562/primary-document.html', '10k_files/sec-edgar-filings/0000079879/10-K/0000079879-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000008670/10-K/0000008670-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000073124/10-K/0000073124-22-000071/primary-document.html', '10k_files/sec-edgar-filings/0000713676/10-K/0000713676-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0001058290/10-K/0001058290-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000701985/10-K/0000701985-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001043277/10-K/0001043277-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000016918/10-K/0000016918-22-000069/primary-document.html', '10k_files/sec-edgar-filings/0000101829/10-K/0000101829-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001174922/10-K/0001174922-22-000031/primary-document.html', '10k_files/sec-edgar-filings/0001110803/10-K/0001110803-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000004281/10-K/0000004281-22-000004/primary-document.html', '10k_files/sec-edgar-filings/0000874766/10-K/0000874766-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000766421/10-K/0000766421-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000874761/10-K/0000874761-22-000022/primary-document.html', '10k_files/sec-edgar-filings/0001163165/10-K/0001562762-22-000031/primary-document.html', '10k_files/sec-edgar-filings/0000818479/10-K/0000818479-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0000014272/10-K/0000014272-22-000051/primary-document.html', '10k_files/sec-edgar-filings/0000875045/10-K/0000875045-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001730168/10-K/0001730168-22-000118/primary-document.html', '10k_files/sec-edgar-filings/0000020286/10-K/0000020286-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000029905/10-K/0000029905-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000943452/10-K/0001628280-22-002997/primary-document.html', '10k_files/sec-edgar-filings/0000313616/10-K/0000313616-22-000061/primary-document.html', '10k_files/sec-edgar-filings/0000093556/10-K/0000093556-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0000316709/10-K/0000316709-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0001001250/10-K/0001001250-22-000122/primary-document.html', '10k_files/sec-edgar-filings/0000920760/10-K/0001628280-22-001450/primary-document.html', '10k_files/sec-edgar-filings/0001501585/10-K/0001501585-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001389170/10-K/0001564590-22-006563/primary-document.html', '10k_files/sec-edgar-filings/0000051644/10-K/0000051644-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000827054/10-K/0000827054-22-000094/primary-document.html', '10k_files/sec-edgar-filings/0000946581/10-K/0001628280-22-014580/primary-document.html', '10k_files/sec-edgar-filings/0001403161/10-K/0001403161-22-000081/primary-document.html', '10k_files/sec-edgar-filings/0001043604/10-K/0001043604-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0001579241/10-K/0001579241-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000899689/10-K/0000899689-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001751788/10-K/0001751788-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000007084/10-K/0000007084-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000732712/10-K/0000732712-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000882835/10-K/0000882835-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000056873/10-K/0001558370-22-004595/primary-document.html', '10k_files/sec-edgar-filings/0001408198/10-K/0001564590-22-004803/primary-document.html', '10k_files/sec-edgar-filings/0000091440/10-K/0000091440-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001732845/10-K/0000950170-22-025444/primary-document.html', '10k_files/sec-edgar-filings/0000063754/10-K/0000063754-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000922224/10-K/0000922224-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000731766/10-K/0000731766-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000864749/10-K/0000864749-22-000044/primary-document.html', '10k_files/sec-edgar-filings/0000021344/10-K/0000021344-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000878927/10-K/0001564590-22-006303/primary-document.html', '10k_files/sec-edgar-filings/0000832101/10-K/0000832101-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000040987/10-K/0000040987-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001402057/10-K/0001402057-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0000063908/10-K/0000063908-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000036104/10-K/0001193125-22-048709/primary-document.html', '10k_files/sec-edgar-filings/0000833444/10-K/0000833444-22-000043/primary-document.html', '10k_files/sec-edgar-filings/0001419612/10-K/0001178913-22-000760/primary-document.html', '10k_files/sec-edgar-filings/0000051253/10-K/0000051253-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000023217/10-K/0001437749-22-017530/primary-document.html', '10k_files/sec-edgar-filings/0000814453/10-K/0000814453-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0001564708/10-K/0001564708-22-000265/primary-document.html', '10k_files/sec-edgar-filings/0001705696/10-K/0001705696-22-000046/primary-document.html', '10k_files/sec-edgar-filings/0001725057/10-K/0000950170-22-002143/primary-document.html', '10k_files/sec-edgar-filings/0000010456/10-K/0001628280-22-003432/primary-document.html', '10k_files/sec-edgar-filings/0000073309/10-K/0001564590-22-007679/primary-document.html', '10k_files/sec-edgar-filings/0000051434/10-K/0000051434-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0001555280/10-K/0001555280-22-000078/primary-document.html', '10k_files/sec-edgar-filings/0000719739/10-K/0000719739-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0001035443/10-K/0001035443-22-000040/primary-document.html', '10k_files/sec-edgar-filings/0001075531/10-K/0001075531-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000055067/10-K/0001628280-22-003345/primary-document.html', '10k_files/sec-edgar-filings/0000895421/10-K/0000895421-22-000400/primary-document.html', '10k_files/sec-edgar-filings/0001059556/10-K/0001059556-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000715957/10-K/0001564590-22-006589/primary-document.html', '10k_files/sec-edgar-filings/0001048286/10-K/0001628280-22-002666/primary-document.html', '10k_files/sec-edgar-filings/0001413329/10-K/0001413329-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0001100682/10-K/0001100682-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000106040/10-K/0000106040-22-000055/primary-document.html', '10k_files/sec-edgar-filings/0001140536/10-K/0000950170-22-001932/primary-document.html', '10k_files/sec-edgar-filings/0001682852/10-K/0001682852-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0000202058/10-K/0000202058-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0001156039/10-K/0001156039-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000097745/10-K/0000097745-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000092230/10-K/0000092230-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000821189/10-K/0000821189-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0001335258/10-K/0001335258-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000096943/10-K/0000096943-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0000072331/10-K/0000072331-22-000185/primary-document.html', '10k_files/sec-edgar-filings/0000313927/10-K/0001564590-22-005528/primary-document.html', '10k_files/sec-edgar-filings/0000908255/10-K/0000908255-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000728535/10-K/0001437749-22-004457/primary-document.html', '10k_files/sec-edgar-filings/0000047217/10-K/0000047217-22-000068/primary-document.html', '10k_files/sec-edgar-filings/0000764622/10-K/0000764622-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0001281761/10-K/0001281761-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0001666700/10-K/0001666700-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000877212/10-K/0000877212-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0000077476/10-K/0000077476-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001170010/10-K/0001170010-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0000021665/10-K/0000021665-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0001739940/10-K/0001739940-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000906107/10-K/0001564590-22-005566/primary-document.html', '10k_files/sec-edgar-filings/0000006281/10-K/0000006281-22-000250/primary-document.html', '10k_files/sec-edgar-filings/0001037038/10-K/0001037038-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0000310764/10-K/0000310764-22-000028/primary-document.html', '10k_files/sec-edgar-filings/0000038777/10-K/0000038777-22-000198/primary-document.html', '10k_files/sec-edgar-filings/0001534701/10-K/0001534701-22-000078/primary-document.html', '10k_files/sec-edgar-filings/0001352010/10-K/0001352010-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0000100493/10-K/0000100493-22-000097/primary-document.html', '10k_files/sec-edgar-filings/0000062996/10-K/0000062996-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000092380/10-K/0000092380-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001103982/10-K/0001103982-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0000001800/10-K/0001104659-22-025141/primary-document.html', '10k_files/sec-edgar-filings/0000789570/10-K/0000789570-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001037868/10-K/0001037868-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000080424/10-K/0000080424-22-000064/primary-document.html', '10k_files/sec-edgar-filings/0001099800/10-K/0001099800-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000822416/10-K/0000822416-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000831001/10-K/0000831001-22-000036/primary-document.html', '10k_files/sec-edgar-filings/0001014473/10-K/0001014473-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000320193/10-K/0000320193-22-000108/primary-document.html', '10k_files/sec-edgar-filings/0000886982/10-K/0001193125-22-052682/primary-document.html', '10k_files/sec-edgar-filings/0000915912/10-K/0000915912-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000769397/10-K/0000769397-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0001467373/10-K/0001467373-22-000295/primary-document.html', '10k_files/sec-edgar-filings/0000048465/10-K/0000048465-22-000051/primary-document.html', '10k_files/sec-edgar-filings/0000815097/10-K/0000815097-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000318154/10-K/0000318154-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001136869/10-K/0001564590-22-007160/primary-document.html', '10k_files/sec-edgar-filings/0000820313/10-K/0001558370-22-000961/primary-document.html', '10k_files/sec-edgar-filings/0000062709/10-K/0000062709-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000074208/10-K/0000074208-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001091667/10-K/0001091667-22-000024/primary-document.html', '10k_files/sec-edgar-filings/0001396009/10-K/0001396009-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001596532/10-K/0001596532-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0001341439/10-K/0001564590-22-023675/primary-document.html', '10k_files/sec-edgar-filings/0001158449/10-K/0001158449-22-000037/primary-document.html', '10k_files/sec-edgar-filings/0000831259/10-K/0000831259-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000823768/10-K/0001558370-22-001179/primary-document.html', '10k_files/sec-edgar-filings/0001136893/10-K/0001136893-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000037996/10-K/0000037996-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001613103/10-K/0001613103-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000050863/10-K/0000050863-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001086222/10-K/0001086222-22-000058/primary-document.html', '10k_files/sec-edgar-filings/0000352541/10-K/0000352541-22-000020/primary-document.html', '10k_files/sec-edgar-filings/0000915913/10-K/0000915913-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000106535/10-K/0001564590-22-005707/primary-document.html', '10k_files/sec-edgar-filings/0000906163/10-K/0000906163-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001070750/10-K/0000950170-22-001965/primary-document.html', '10k_files/sec-edgar-filings/0001065088/10-K/0001065088-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000092122/10-K/0000092122-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0000745732/10-K/0000745732-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0001067983/10-K/0001564590-22-007322/primary-document.html', '10k_files/sec-edgar-filings/0000310158/10-K/0000310158-22-000003/primary-document.html', '10k_files/sec-edgar-filings/0001383312/10-K/0001383312-22-000037/primary-document.html', '10k_files/sec-edgar-filings/0001013462/10-K/0001013462-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001097149/10-K/0001097149-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000906345/10-K/0000906345-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000857005/10-K/0000950170-22-025211/primary-document.html', '10k_files/sec-edgar-filings/0001113169/10-K/0001113169-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000829224/10-K/0000829224-22-000058/primary-document.html', '10k_files/sec-edgar-filings/0000866787/10-K/0001558370-22-015239/primary-document.html', '10k_files/sec-edgar-filings/0001278021/10-K/0000950170-22-001811/primary-document.html', '10k_files/sec-edgar-filings/0001792044/10-K/0001792044-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000004904/10-K/0000004904-22-000024/primary-document.html', '10k_files/sec-edgar-filings/0000879101/10-K/0001437749-22-004700/primary-document.html', '10k_files/sec-edgar-filings/0001659166/10-K/0001659166-22-000054/primary-document.html', '10k_files/sec-edgar-filings/0001571949/10-K/0001571949-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001748790/10-K/0001748790-22-000024/primary-document.html', '10k_files/sec-edgar-filings/0001000697/10-K/0001193125-22-051509/primary-document.html', '10k_files/sec-edgar-filings/0000032604/10-K/0000032604-22-000041/primary-document.html', '10k_files/sec-edgar-filings/0001336920/10-K/0001336920-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001041061/10-K/0001041061-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000796343/10-K/0000796343-22-000032/primary-document.html', '10k_files/sec-edgar-filings/0001463101/10-K/0001463101-22-000016/primary-document.html', '10k_files/sec-edgar-filings/0000010795/10-K/0001628280-22-030686/primary-document.html', '10k_files/sec-edgar-filings/0000060086/10-K/0000060086-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000059558/10-K/0000059558-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000080661/10-K/0000080661-22-000046/primary-document.html', '10k_files/sec-edgar-filings/0001045609/10-K/0001564590-22-004436/primary-document.html', '10k_files/sec-edgar-filings/0001324424/10-K/0001324424-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000091142/10-K/0000091142-22-000028/primary-document.html', '10k_files/sec-edgar-filings/0000927066/10-K/0000927066-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0001645590/10-K/0001645590-22-000071/primary-document.html', '10k_files/sec-edgar-filings/0000319201/10-K/0000319201-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000927653/10-K/0000927653-22-000051/primary-document.html', '10k_files/sec-edgar-filings/0000865752/10-K/0001104659-22-028182/primary-document.html', '10k_files/sec-edgar-filings/0001260221/10-K/0001260221-22-000065/primary-document.html', '10k_files/sec-edgar-filings/0001101239/10-K/0001628280-22-003171/primary-document.html', '10k_files/sec-edgar-filings/0001130310/10-K/0001130310-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000066740/10-K/0000066740-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000072971/10-K/0000072971-22-000096/primary-document.html', '10k_files/sec-edgar-filings/0001067701/10-K/0001067701-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001596783/10-K/0001596783-22-000129/primary-document.html', '10k_files/sec-edgar-filings/0000858877/10-K/0000858877-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0001267238/10-K/0001267238-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001057352/10-K/0001057352-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000200406/10-K/0000200406-22-000022/primary-document.html', '10k_files/sec-edgar-filings/0000320187/10-K/0000320187-22-000038/primary-document.html', '10k_files/sec-edgar-filings/0000100885/10-K/0001437749-22-002494/primary-document.html', '10k_files/sec-edgar-filings/0001841666/10-K/0001784031-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000037785/10-K/0000037785-22-000025/primary-document.html', '10k_files/sec-edgar-filings/0000797468/10-K/0000797468-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000711404/10-K/0000711404-22-000053/primary-document.html', '10k_files/sec-edgar-filings/0000315293/10-K/0001628280-22-003180/primary-document.html', '10k_files/sec-edgar-filings/0000034903/10-K/0000034903-22-000023/primary-document.html', '10k_files/sec-edgar-filings/0000019617/10-K/0000019617-22-000272/primary-document.html', '10k_files/sec-edgar-filings/0001138118/10-K/0001138118-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0001539838/10-K/0001539838-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000896159/10-K/0000896159-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0000072741/10-K/0000072741-22-000015/primary-document.html', '10k_files/sec-edgar-filings/0001513761/10-K/0001558370-22-002516/primary-document.html', '10k_files/sec-edgar-filings/0001065696/10-K/0001065696-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000093410/10-K/0000093410-22-000019/primary-document.html', '10k_files/sec-edgar-filings/0000898173/10-K/0000898173-22-000012/primary-document.html', '10k_files/sec-edgar-filings/0001097864/10-K/0001628280-22-002416/primary-document.html', '10k_files/sec-edgar-filings/0001002047/10-K/0000950170-22-011708/primary-document.html', '10k_files/sec-edgar-filings/0001018724/10-K/0001018724-22-000005/primary-document.html', '10k_files/sec-edgar-filings/0001289490/10-K/0001628280-22-004274/primary-document.html', '10k_files/sec-edgar-filings/0000789019/10-K/0001564590-22-026876/primary-document.html', '10k_files/sec-edgar-filings/0000315189/10-K/0001558370-22-018703/primary-document.html', '10k_files/sec-edgar-filings/0001031296/10-K/0001031296-22-000013/primary-document.html', '10k_files/sec-edgar-filings/0000068505/10-K/0000068505-22-000010/primary-document.html', '10k_files/sec-edgar-filings/0000096021/10-K/0000096021-22-000151/primary-document.html', '10k_files/sec-edgar-filings/0001510295/10-K/0001510295-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000103379/10-K/0000103379-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0001707925/10-K/0001628280-22-004180/primary-document.html', '10k_files/sec-edgar-filings/0000029534/10-K/0001558370-22-003921/primary-document.html', '10k_files/sec-edgar-filings/0000936468/10-K/0000936468-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0001090012/10-K/0001564590-22-005321/primary-document.html', '10k_files/sec-edgar-filings/0001022079/10-K/0001022079-22-000027/primary-document.html', '10k_files/sec-edgar-filings/0000028412/10-K/0000028412-22-000067/primary-document.html', '10k_files/sec-edgar-filings/0000075677/10-K/0000950170-22-001913/primary-document.html', '10k_files/sec-edgar-filings/0001035002/10-K/0001035002-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000008818/10-K/0001193125-22-049910/primary-document.html', '10k_files/sec-edgar-filings/0000034088/10-K/0000034088-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000860731/10-K/0000860731-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0000059478/10-K/0000059478-22-000068/primary-document.html', '10k_files/sec-edgar-filings/0001286681/10-K/0000950170-22-002426/primary-document.html', '10k_files/sec-edgar-filings/0001306830/10-K/0001306830-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0000026172/10-K/0000026172-22-000008/primary-document.html', '10k_files/sec-edgar-filings/0000040704/10-K/0001193125-22-185257/primary-document.html', '10k_files/sec-edgar-filings/0001744489/10-K/0001744489-22-000213/primary-document.html', '10k_files/sec-edgar-filings/0000089800/10-K/0000089800-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001326801/10-K/0001326801-22-000018/primary-document.html', '10k_files/sec-edgar-filings/0000049826/10-K/0000049826-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000753308/10-K/0000753308-22-000014/primary-document.html', '10k_files/sec-edgar-filings/0000811156/10-K/0000811156-22-000048/primary-document.html', '10k_files/sec-edgar-filings/0001090872/10-K/0001090872-22-000026/primary-document.html', '10k_files/sec-edgar-filings/0001111711/10-K/0001111711-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0001004980/10-K/0001004980-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000093751/10-K/0000093751-22-000424/primary-document.html', '10k_files/sec-edgar-filings/0000749251/10-K/0000749251-22-000006/primary-document.html', '10k_files/sec-edgar-filings/0000815556/10-K/0000815556-22-000009/primary-document.html', '10k_files/sec-edgar-filings/0000049071/10-K/0000049071-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0000860730/10-K/0001193125-22-046707/primary-document.html', '10k_files/sec-edgar-filings/0001058090/10-K/0001058090-22-000011/primary-document.html', '10k_files/sec-edgar-filings/0001121788/10-K/0000950170-22-001303/primary-document.html', '10k_files/sec-edgar-filings/0001297996/10-K/0001558370-22-002195/primary-document.html', '10k_files/sec-edgar-filings/0001437107/10-K/0001437107-22-000031/primary-document.html', '10k_files/sec-edgar-filings/0001164727/10-K/0001164727-22-000007/primary-document.html', '10k_files/sec-edgar-filings/0000720005/10-K/0000720005-22-000066/primary-document.html', '10k_files/sec-edgar-filings/0001048695/10-K/0001048695-22-000033/primary-document.html', '10k_files/sec-edgar-filings/0001679273/10-K/0001558370-22-011121/primary-document.html', '10k_files/sec-edgar-filings/0000047111/10-K/0000047111-22-000017/primary-document.html', '10k_files/sec-edgar-filings/0001116132/10-K/0001116132-22-000018/primary-document.html']





    'We have 498 HTML files for 509 firms'




```python
for cik in tqdm(sp500["CIK"], desc="Assigning Accession Numbers"):
    cik_str = str(cik).zfill(10)
    firm_folder = f'10k_files/sec-edgar-filings/{cik_str}/10-K/'

    try:
        html_files = glob.glob(firm_folder + '*/primary-document.html')

        if not html_files:
            print(f"⚠️ No HTML filings found for CIK {cik_str}. Skipping...")
            continue

        # ✅ Get accession number from folder name
        accession_number = os.path.basename(os.path.dirname(html_files[0]))

        # ✅ Update sp500 DataFrame
        sp500.loc[sp500["CIK"] == cik, "Accession Number"] = accession_number
        print(f"✅ CIK {cik_str}: Accession Number -> {accession_number}")

    except Exception as e:
        print(f"❌ Error for CIK {cik_str}: {e}")

    # ✅ Delay slightly to avoid hammering any services (SEC or filesystem)
    time.sleep(0.1)
```


```python
sp500
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>0001041061-22-000009</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>0000877212-22-000026</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>0001564590-22-007160</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>0000109380-22-000072</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>0001555280-22-000078</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 10 columns</p>
</div>




```python
import re
import time
from requests_html import HTMLSession
from tqdm import tqdm

session = HTMLSession()
session.headers.update({'User-Agent': 'Jeremiah Geneve jag325@lehigh.edu'})

# 🔄 Iterate over downloaded HTML files and extract filing dates
for item in tqdm(files, desc="Fetching Filing Dates"):
    segment = item.split('/')  # Split the path into segments
    cik = segment[-4]
    accession_number = segment[-2]
    url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}-index.html'

    try:
        r = session.get(url)
        r.raise_for_status()  # Ensure request was successful

        # ✅ Extract Filing Date from the SEC index page
        date_element = r.html.find(
            '#contentDiv > div:nth-child(1) > div.formContent > div:nth-child(1) > div:nth-child(2)',
            first=True
        )

        filing_date = date_element.text.strip() if date_element else None

        # ✅ Update `sp500` DataFrame with the correct Filing Date
        sp500.loc[sp500["CIK"].astype(str).str.zfill(10) == cik, "Filing Dates"] = filing_date
        sp500.loc[sp500["CIK"].astype(str).str.zfill(10) == cik, "Accession Number"] = accession_number

        print(f"✅ CIK {cik}: Filing Date -> {filing_date}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching filing date for CIK {cik}: {e}")

    # ✅ Add a delay to prevent rate-limiting
    time.sleep(0.5)
```


```python
sp500.head(60)
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>2022-02-17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>2022-01-21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADP</td>
      <td>ADP</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Data Processing &amp; Outsourced Services</td>
      <td>Roseland, New Jersey</td>
      <td>1981-03-31</td>
      <td>8670</td>
      <td>1949</td>
      <td>0000008670-22-000038</td>
      <td>2022-08-03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAP</td>
      <td>Advance Auto Parts</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Automotive Retail</td>
      <td>Raleigh, North Carolina</td>
      <td>2015-07-09</td>
      <td>1158449</td>
      <td>1932</td>
      <td>0001158449-22-000037</td>
      <td>2022-02-15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AES</td>
      <td>AES Corporation</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Independent Power Producers &amp; Energy Traders</td>
      <td>Arlington, Virginia</td>
      <td>1998-10-02</td>
      <td>874761</td>
      <td>1981</td>
      <td>0000874761-22-000022</td>
      <td>2022-02-28</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AFL</td>
      <td>Aflac</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Life &amp; Health Insurance</td>
      <td>Columbus, Georgia</td>
      <td>1999-05-28</td>
      <td>4977</td>
      <td>1955</td>
      <td>0000004977-22-000058</td>
      <td>2022-02-23</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A</td>
      <td>Agilent Technologies</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Santa Clara, California</td>
      <td>2000-06-05</td>
      <td>1090872</td>
      <td>1999</td>
      <td>0001090872-22-000026</td>
      <td>2022-12-21</td>
    </tr>
    <tr>
      <th>13</th>
      <td>APD</td>
      <td>Air Products and Chemicals</td>
      <td>reports</td>
      <td>Materials</td>
      <td>Industrial Gases</td>
      <td>Allentown, Pennsylvania</td>
      <td>1985-04-30</td>
      <td>2969</td>
      <td>1940</td>
      <td>0000002969-22-000054</td>
      <td>2022-11-22</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AKAM</td>
      <td>Akamai</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Internet Services &amp; Infrastructure</td>
      <td>Cambridge, Massachusetts</td>
      <td>2007-07-12</td>
      <td>1086222</td>
      <td>1998</td>
      <td>0001086222-22-000058</td>
      <td>2022-02-28</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ALK</td>
      <td>Alaska Air Group</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Airlines</td>
      <td>SeaTac, Washington</td>
      <td>2016-05-13</td>
      <td>766421</td>
      <td>1985</td>
      <td>0000766421-22-000009</td>
      <td>2022-02-11</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ALB</td>
      <td>Albemarle Corporation</td>
      <td>reports</td>
      <td>Materials</td>
      <td>Specialty Chemicals</td>
      <td>Charlotte, North Carolina</td>
      <td>2016-07-01</td>
      <td>915913</td>
      <td>1994</td>
      <td>0000915913-22-000027</td>
      <td>2022-02-22</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ARE</td>
      <td>Alexandria Real Estate Equities</td>
      <td>reports</td>
      <td>Real Estate</td>
      <td>Office REITs</td>
      <td>Pasadena, California</td>
      <td>2017-03-20</td>
      <td>1035443</td>
      <td>1994</td>
      <td>0001035443-22-000040</td>
      <td>2022-01-31</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ALGN</td>
      <td>Align Technology</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Supplies</td>
      <td>Tempe, Arizona</td>
      <td>2017-06-19</td>
      <td>1097149</td>
      <td>1997</td>
      <td>0001097149-22-000011</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ALLE</td>
      <td>Allegion</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>New York City, New York</td>
      <td>2013-12-02</td>
      <td>1579241</td>
      <td>1908</td>
      <td>0001579241-22-000019</td>
      <td>2022-02-15</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LNT</td>
      <td>Alliant Energy</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Electric Utilities</td>
      <td>Madison, Wisconsin</td>
      <td>2016-07-01</td>
      <td>352541</td>
      <td>1917</td>
      <td>0000352541-22-000020</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ALL</td>
      <td>Allstate</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Property &amp; Casualty Insurance</td>
      <td>Northfield Township, Illinois</td>
      <td>1995-07-13</td>
      <td>899051</td>
      <td>1931</td>
      <td>0000899051-22-000015</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>22</th>
      <td>GOOGL</td>
      <td>Alphabet Inc. (Class A)</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Media &amp; Services</td>
      <td>Mountain View, California</td>
      <td>2014-04-03</td>
      <td>1652044</td>
      <td>1998</td>
      <td>0001652044-22-000019</td>
      <td>2022-02-02</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GOOG</td>
      <td>Alphabet Inc. (Class C)</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Media &amp; Services</td>
      <td>Mountain View, California</td>
      <td>2006-04-03</td>
      <td>1652044</td>
      <td>1998</td>
      <td>0001652044-22-000019</td>
      <td>2022-02-02</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MO</td>
      <td>Altria</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Tobacco</td>
      <td>Richmond, Virginia</td>
      <td>1957-03-04</td>
      <td>764180</td>
      <td>1985</td>
      <td>0000764180-22-000019</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AMZN</td>
      <td>Amazon</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Internet &amp; Direct Marketing Retail</td>
      <td>Seattle, Washington</td>
      <td>2005-11-18</td>
      <td>1018724</td>
      <td>1994</td>
      <td>0001018724-22-000005</td>
      <td>2022-02-04</td>
    </tr>
    <tr>
      <th>26</th>
      <td>AMCR</td>
      <td>Amcor</td>
      <td>reports</td>
      <td>Materials</td>
      <td>Paper Packaging</td>
      <td>Warmley, Bristol, United Kingdom</td>
      <td>2019-06-07</td>
      <td>1748790</td>
      <td>2019 (1860)</td>
      <td>0001748790-22-000024</td>
      <td>2022-08-18</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AMD</td>
      <td>AMD</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Semiconductors</td>
      <td>Santa Clara, California</td>
      <td>NaN</td>
      <td>2488</td>
      <td>1969</td>
      <td>0000002488-22-000016</td>
      <td>2022-02-03</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AEE</td>
      <td>Ameren</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Multi-Utilities</td>
      <td>St. Louis, Missouri</td>
      <td>1991-09-19</td>
      <td>1002910</td>
      <td>1902</td>
      <td>0001002910-22-000038</td>
      <td>2022-02-23</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AAL</td>
      <td>American Airlines Group</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Airlines</td>
      <td>Fort Worth, Texas</td>
      <td>2015-03-23</td>
      <td>6201</td>
      <td>1934</td>
      <td>0000006201-22-000026</td>
      <td>2022-02-22</td>
    </tr>
    <tr>
      <th>30</th>
      <td>AEP</td>
      <td>American Electric Power</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Electric Utilities</td>
      <td>Columbus, Ohio</td>
      <td>1957-03-04</td>
      <td>4904</td>
      <td>1906</td>
      <td>0000004904-22-000024</td>
      <td>2022-02-24</td>
    </tr>
    <tr>
      <th>31</th>
      <td>AXP</td>
      <td>American Express</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Consumer Finance</td>
      <td>New York City, New York</td>
      <td>1976-06-30</td>
      <td>4962</td>
      <td>1850</td>
      <td>0000004962-22-000008</td>
      <td>2022-02-11</td>
    </tr>
    <tr>
      <th>32</th>
      <td>AIG</td>
      <td>American International Group</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Property &amp; Casualty Insurance</td>
      <td>New York City, New York</td>
      <td>1980-03-31</td>
      <td>5272</td>
      <td>1919</td>
      <td>0001104659-22-024701</td>
      <td>2022-02-17</td>
    </tr>
    <tr>
      <th>33</th>
      <td>AMT</td>
      <td>American Tower</td>
      <td>reports</td>
      <td>Real Estate</td>
      <td>Specialized REITs</td>
      <td>Boston, Massachusetts</td>
      <td>2007-11-19</td>
      <td>1053507</td>
      <td>1995</td>
      <td>0001053507-22-000017</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>34</th>
      <td>AWK</td>
      <td>American Water Works</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Water Utilities</td>
      <td>Camden, New Jersey</td>
      <td>2016-03-04</td>
      <td>1410636</td>
      <td>1886</td>
      <td>0001410636-22-000048</td>
      <td>2022-02-16</td>
    </tr>
    <tr>
      <th>35</th>
      <td>AMP</td>
      <td>Ameriprise Financial</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Asset Management &amp; Custody Banks</td>
      <td>Minneapolis, Minnesota</td>
      <td>2005-10-03</td>
      <td>820027</td>
      <td>1894</td>
      <td>0000820027-22-000016</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>36</th>
      <td>ABC</td>
      <td>AmerisourceBergen</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Distributors</td>
      <td>Conshohocken, Pennsylvania</td>
      <td>2001-08-30</td>
      <td>1140859</td>
      <td>1985</td>
      <td>0001140859-22-000098</td>
      <td>2022-11-22</td>
    </tr>
    <tr>
      <th>37</th>
      <td>AME</td>
      <td>Ametek</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Electrical Components &amp; Equipment</td>
      <td>Berwyn, Pennsylvania</td>
      <td>2013-09-23</td>
      <td>1037868</td>
      <td>1930</td>
      <td>0001037868-22-000009</td>
      <td>2022-02-22</td>
    </tr>
    <tr>
      <th>38</th>
      <td>AMGN</td>
      <td>Amgen</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Biotechnology</td>
      <td>Thousand Oaks, California</td>
      <td>1992-01-02</td>
      <td>318154</td>
      <td>1980</td>
      <td>0000318154-22-000010</td>
      <td>2022-02-16</td>
    </tr>
    <tr>
      <th>39</th>
      <td>APH</td>
      <td>Amphenol</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Components</td>
      <td>Wallingford, Connecticut</td>
      <td>2008-09-30</td>
      <td>820313</td>
      <td>1932</td>
      <td>0001558370-22-000961</td>
      <td>2022-02-09</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ADI</td>
      <td>Analog Devices</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Semiconductors</td>
      <td>Wilmington, Massachusetts</td>
      <td>1999-10-12</td>
      <td>6281</td>
      <td>1965</td>
      <td>0000006281-22-000250</td>
      <td>2022-11-22</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ANSS</td>
      <td>Ansys</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>Canonsburg, Pennsylvania</td>
      <td>2017-06-19</td>
      <td>1013462</td>
      <td>1969</td>
      <td>0001013462-22-000005</td>
      <td>2022-02-23</td>
    </tr>
    <tr>
      <th>42</th>
      <td>AON</td>
      <td>Aon</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Insurance Brokers</td>
      <td>London, UK</td>
      <td>1996-04-23</td>
      <td>315293</td>
      <td>1982 (1919)</td>
      <td>0001628280-22-003180</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>43</th>
      <td>APA</td>
      <td>APA Corporation</td>
      <td>reports</td>
      <td>Energy</td>
      <td>Oil &amp; Gas Exploration &amp; Production</td>
      <td>Houston, Texas</td>
      <td>1997-07-28</td>
      <td>1841666</td>
      <td>1954</td>
      <td>0001784031-22-000009</td>
      <td>2022-02-22</td>
    </tr>
    <tr>
      <th>44</th>
      <td>AAPL</td>
      <td>Apple Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>Cupertino, California</td>
      <td>1982-11-30</td>
      <td>320193</td>
      <td>1977</td>
      <td>0000320193-22-000108</td>
      <td>2022-10-28</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AMAT</td>
      <td>Applied Materials</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Semiconductor Equipment</td>
      <td>Santa Clara, California</td>
      <td>1995-03-16</td>
      <td>6951</td>
      <td>1967</td>
      <td>0000006951-22-000043</td>
      <td>2022-12-16</td>
    </tr>
    <tr>
      <th>46</th>
      <td>APTV</td>
      <td>Aptiv</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Auto Parts &amp; Equipment</td>
      <td>Dublin, Ireland</td>
      <td>2012-12-24</td>
      <td>1521332</td>
      <td>1994</td>
      <td>0001521332-22-000010</td>
      <td>2022-02-07</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ACGL</td>
      <td>Arch Capital Group</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Reinsurance</td>
      <td>Hamilton, Bermuda</td>
      <td>2022-11-01</td>
      <td>947484</td>
      <td>1995</td>
      <td>0000947484-22-000015</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ANET</td>
      <td>Arista Networks</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Communications Equipment</td>
      <td>Santa Clara, California</td>
      <td>2018-08-28</td>
      <td>1596532</td>
      <td>2004</td>
      <td>0001596532-22-000026</td>
      <td>2022-02-15</td>
    </tr>
    <tr>
      <th>49</th>
      <td>AJG</td>
      <td>Arthur J. Gallagher &amp; Co.</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Insurance Brokers</td>
      <td>Rolling Meadows, Illinois</td>
      <td>2016-05-31</td>
      <td>354190</td>
      <td>1927</td>
      <td>0001564590-22-005714</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>50</th>
      <td>AIZ</td>
      <td>Assurant</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Multi-line Insurance</td>
      <td>New York City, New York</td>
      <td>2007-04-10</td>
      <td>1267238</td>
      <td>1892</td>
      <td>0001267238-22-000006</td>
      <td>2022-02-22</td>
    </tr>
    <tr>
      <th>51</th>
      <td>T</td>
      <td>AT&amp;T</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Integrated Telecommunication Services</td>
      <td>Dallas, Texas</td>
      <td>1983-11-30 (1957-03-04)</td>
      <td>732717</td>
      <td>1983 (1885)</td>
      <td>0000732717-22-000015</td>
      <td>2022-02-16</td>
    </tr>
    <tr>
      <th>52</th>
      <td>ATO</td>
      <td>Atmos Energy</td>
      <td>reports</td>
      <td>Utilities</td>
      <td>Gas Utilities</td>
      <td>Dallas, Texas</td>
      <td>2019-02-15</td>
      <td>731802</td>
      <td>1906</td>
      <td>0000731802-22-000037</td>
      <td>2022-11-14</td>
    </tr>
    <tr>
      <th>53</th>
      <td>ADSK</td>
      <td>Autodesk</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Francisco, California</td>
      <td>1989-12-01</td>
      <td>769397</td>
      <td>1982</td>
      <td>0000769397-22-000019</td>
      <td>2022-03-14</td>
    </tr>
    <tr>
      <th>54</th>
      <td>AZO</td>
      <td>AutoZone</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Specialty Stores</td>
      <td>Memphis, Tennessee</td>
      <td>1997-01-02</td>
      <td>866787</td>
      <td>1979</td>
      <td>0001558370-22-015239</td>
      <td>2022-10-24</td>
    </tr>
    <tr>
      <th>55</th>
      <td>AVB</td>
      <td>AvalonBay Communities</td>
      <td>reports</td>
      <td>Real Estate</td>
      <td>Residential REITs</td>
      <td>Arlington, Virginia</td>
      <td>2007-01-10</td>
      <td>915912</td>
      <td>1978</td>
      <td>0000915912-22-000005</td>
      <td>2022-02-25</td>
    </tr>
    <tr>
      <th>56</th>
      <td>AVY</td>
      <td>Avery Dennison</td>
      <td>reports</td>
      <td>Materials</td>
      <td>Paper Packaging</td>
      <td>Glendale, California</td>
      <td>1987-12-31</td>
      <td>8818</td>
      <td>1990</td>
      <td>0001193125-22-049910</td>
      <td>2022-02-23</td>
    </tr>
    <tr>
      <th>57</th>
      <td>BKR</td>
      <td>Baker Hughes</td>
      <td>reports</td>
      <td>Energy</td>
      <td>Oil &amp; Gas Equipment &amp; Services</td>
      <td>Houston, Texas</td>
      <td>2017-07-07</td>
      <td>1701605</td>
      <td>2017</td>
      <td>0001701605-22-000050</td>
      <td>2022-02-11</td>
    </tr>
    <tr>
      <th>58</th>
      <td>BALL</td>
      <td>Ball Corporation</td>
      <td>reports</td>
      <td>Materials</td>
      <td>Metal &amp; Glass Containers</td>
      <td>Broomfield, Colorado</td>
      <td>1984-10-31</td>
      <td>9389</td>
      <td>1880</td>
      <td>0001558370-22-001251</td>
      <td>2022-02-16</td>
    </tr>
    <tr>
      <th>59</th>
      <td>BAC</td>
      <td>Bank of America</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Diversified Banks</td>
      <td>Charlotte, North Carolina</td>
      <td>1976-06-30</td>
      <td>70858</td>
      <td>1998 (1923 / 1874)</td>
      <td>0000070858-22-000062</td>
      <td>2022-02-22</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(sp500.columns)
print(sp500.shape)
```

    Index(['Symbol', 'Security', 'SEC filings', 'GICS Sector', 'GICS Sub-Industry',
           'Headquarters Location', 'Date first added', 'CIK', 'Founded',
           'Accession Number', 'Filing Dates'],
          dtype='object')
    (503, 11)



```python
sp500["Filing Dates"] = pd.to_datetime(sp500["Filing Dates"], errors="coerce")

# ✅ Filter rows with a valid filing date
valid_tickers = sp500.loc[sp500["Filing Dates"].notna(), "Symbol"].tolist()

valid_tickers
```




    ['MMM',
     'AOS',
     'ABT',
     'ABBV',
     'ACN',
     'ATVI',
     'ADM',
     'ADBE',
     'ADP',
     'AAP',
     'AES',
     'AFL',
     'A',
     'APD',
     'AKAM',
     'ALK',
     'ALB',
     'ARE',
     'ALGN',
     'ALLE',
     'LNT',
     'ALL',
     'GOOGL',
     'GOOG',
     'MO',
     'AMZN',
     'AMCR',
     'AMD',
     'AEE',
     'AAL',
     'AEP',
     'AXP',
     'AIG',
     'AMT',
     'AWK',
     'AMP',
     'ABC',
     'AME',
     'AMGN',
     'APH',
     'ADI',
     'ANSS',
     'AON',
     'APA',
     'AAPL',
     'AMAT',
     'APTV',
     'ACGL',
     'ANET',
     'AJG',
     'AIZ',
     'T',
     'ATO',
     'ADSK',
     'AZO',
     'AVB',
     'AVY',
     'BKR',
     'BALL',
     'BAC',
     'BBWI',
     'BAX',
     'BDX',
     'WRB',
     'BRK.B',
     'BBY',
     'BIO',
     'TECH',
     'BIIB',
     'BLK',
     'BK',
     'BA',
     'BKNG',
     'BWA',
     'BXP',
     'BSX',
     'BMY',
     'AVGO',
     'BR',
     'BRO',
     'BF.B',
     'CHRW',
     'CDNS',
     'CZR',
     'CPT',
     'CPB',
     'COF',
     'CAH',
     'KMX',
     'CCL',
     'CARR',
     'CTLT',
     'CAT',
     'CBOE',
     'CBRE',
     'CDW',
     'CE',
     'CNC',
     'CNP',
     'CDAY',
     'CF',
     'CRL',
     'SCHW',
     'CHTR',
     'CVX',
     'CMG',
     'CB',
     'CHD',
     'CI',
     'CINF',
     'CTAS',
     'CSCO',
     'C',
     'CFG',
     'CLX',
     'CME',
     'CMS',
     'KO',
     'CTSH',
     'CL',
     'CMCSA',
     'CMA',
     'CAG',
     'COP',
     'ED',
     'STZ',
     'CEG',
     'COO',
     'CPRT',
     'GLW',
     'CTVA',
     'CSGP',
     'COST',
     'CTRA',
     'CCI',
     'CSX',
     'CMI',
     'CVS',
     'DHI',
     'DHR',
     'DRI',
     'DVA',
     'DE',
     'DAL',
     'XRAY',
     'DVN',
     'DXCM',
     'FANG',
     'DLR',
     'DFS',
     'DISH',
     'DIS',
     'DG',
     'DLTR',
     'D',
     'DPZ',
     'DOV',
     'DOW',
     'DTE',
     'DUK',
     'DD',
     'DXC',
     'EMN',
     'ETN',
     'EBAY',
     'ECL',
     'EIX',
     'EW',
     'EA',
     'ELV',
     'LLY',
     'EMR',
     'ENPH',
     'ETR',
     'EOG',
     'EPAM',
     'EQT',
     'EFX',
     'EQIX',
     'EQR',
     'ESS',
     'EL',
     'ETSY',
     'RE',
     'EVRG',
     'ES',
     'EXC',
     'EXPE',
     'EXPD',
     'EXR',
     'XOM',
     'FFIV',
     'FDS',
     'FAST',
     'FRT',
     'FDX',
     'FITB',
     'FSLR',
     'FE',
     'FIS',
     'FISV',
     'FLT',
     'FMC',
     'F',
     'FTNT',
     'FTV',
     'FOXA',
     'FOX',
     'BEN',
     'FCX',
     'GRMN',
     'IT',
     'GEN',
     'GNRC',
     'GD',
     'GE',
     'GIS',
     'GM',
     'GPC',
     'GILD',
     'GL',
     'GPN',
     'GS',
     'HAL',
     'HIG',
     'HAS',
     'HCA',
     'PEAK',
     'HSIC',
     'HSY',
     'HES',
     'HPE',
     'HLT',
     'HOLX',
     'HD',
     'HON',
     'HRL',
     'HST',
     'HWM',
     'HPQ',
     'HUM',
     'HBAN',
     'HII',
     'IBM',
     'IEX',
     'IDXX',
     'ITW',
     'ILMN',
     'INCY',
     'IR',
     'INTC',
     'ICE',
     'IP',
     'IPG',
     'IFF',
     'INTU',
     'ISRG',
     'IVZ',
     'INVH',
     'IQV',
     'IRM',
     'JBHT',
     'JKHY',
     'J',
     'JNJ',
     'JCI',
     'JPM',
     'JNPR',
     'K',
     'KDP',
     'KEY',
     'KEYS',
     'KMB',
     'KIM',
     'KMI',
     'KLAC',
     'KHC',
     'KR',
     'LHX',
     'LH',
     'LRCX',
     'LW',
     'LVS',
     'LDOS',
     'LEN',
     'LNC',
     'LIN',
     'LYV',
     'LKQ',
     'LMT',
     'L',
     'LOW',
     'LUMN',
     'LYB',
     'MTB',
     'MRO',
     'MPC',
     'MKTX',
     'MAR',
     'MMC',
     'MLM',
     'MAS',
     'MA',
     'MTCH',
     'MKC',
     'MCD',
     'MCK',
     'MDT',
     'MRK',
     'META',
     'MET',
     'MTD',
     'MGM',
     'MCHP',
     'MU',
     'MSFT',
     'MAA',
     'MRNA',
     'MHK',
     'MOH',
     'TAP',
     'MDLZ',
     'MPWR',
     'MNST',
     'MCO',
     'MS',
     'MOS',
     'MSI',
     'MSCI',
     'NDAQ',
     'NTAP',
     'NFLX',
     'NWL',
     'NEM',
     'NWSA',
     'NWS',
     'NEE',
     'NKE',
     'NI',
     'NDSN',
     'NSC',
     'NTRS',
     'NOC',
     'NCLH',
     'NRG',
     'NUE',
     'NVDA',
     'NVR',
     'NXPI',
     'ORLY',
     'OXY',
     'ODFL',
     'OMC',
     'ON',
     'OKE',
     'ORCL',
     'OGN',
     'OTIS',
     'PCAR',
     'PKG',
     'PARA',
     'PH',
     'PAYX',
     'PAYC',
     'PYPL',
     'PNR',
     'PEP',
     'PKI',
     'PFE',
     'PCG',
     'PM',
     'PSX',
     'PNW',
     'PXD',
     'PNC',
     'POOL',
     'PPG',
     'PPL',
     'PFG',
     'PG',
     'PGR',
     'PLD',
     'PRU',
     'PEG',
     'PTC',
     'PSA',
     'PHM',
     'QRVO',
     'PWR',
     'QCOM',
     'DGX',
     'RL',
     'RJF',
     'RTX',
     'O',
     'REG',
     'REGN',
     'RF',
     'RSG',
     'RMD',
     'RHI',
     'ROK',
     'ROL',
     'ROP',
     'ROST',
     'RCL',
     'SPGI',
     'CRM',
     'SBAC',
     'SLB',
     'STX',
     'SEE',
     'SRE',
     'NOW',
     'SHW',
     'SPG',
     'SWKS',
     'SJM',
     'SNA',
     'SEDG',
     'SO',
     'LUV',
     'SWK',
     'SBUX',
     'STT',
     'STLD',
     'STE',
     'SYK',
     'SIVB',
     'SYF',
     'SNPS',
     'SYY',
     'TMUS',
     'TROW',
     'TTWO',
     'TPR',
     'TRGP',
     'TGT',
     'TEL',
     'TDY',
     'TFX',
     'TER',
     'TSLA',
     'TXN',
     'TXT',
     'TMO',
     'TJX',
     'TSCO',
     'TT',
     'TDG',
     'TRV',
     'TRMB',
     'TFC',
     'TYL',
     'TSN',
     'USB',
     'UDR',
     'ULTA',
     'UNP',
     'UAL',
     'UPS',
     'URI',
     'UNH',
     'UHS',
     'VLO',
     'VTR',
     'VRSN',
     'VRSK',
     'VZ',
     'VRTX',
     'VFC',
     'VTRS',
     'VICI',
     'V',
     'VNO',
     'VMC',
     'WAB',
     'WBA',
     'WMT',
     'WBD',
     'WM',
     'WAT',
     'WEC',
     'WFC',
     'WELL',
     'WST',
     'WDC',
     'WRK',
     'WY',
     'WHR',
     'WMB',
     'WTW',
     'GWW',
     'WYNN',
     'XEL',
     'XYL',
     'YUM',
     'ZBRA',
     'ZBH',
     'ZION',
     'ZTS']




```python
import yfinance as yf
import pandas as pd
import time
from datetime import timedelta
from tqdm import tqdm

# ✅ Define function to fetch stock data
def get_stock_data(tickers, start_date, end_date, retries=3):
    stock_data = None
    for attempt in range(retries):
        try:
            print(f"📡 Fetching stock data (Attempt {attempt+1}/{retries})...")
            stock_data = yf.download(tickers, start=start_date, end=end_date).filter(like="Close").droplevel(0, axis=1)
            return stock_data
        except Exception as e:
            print(f"⚠️ Error fetching stock data: {e}")
            time.sleep(5)  # Wait before retrying
    return None  # Return None if all attempts fail

# ✅ Convert Filing Dates to datetime format
sp500["Filing Dates"] = pd.to_datetime(sp500["Filing Dates"], errors="coerce")

# ✅ Filter valid rows
valid_rows = sp500.dropna(subset=["Filing Dates"]).copy()

# ✅ Get tickers and define date range
tickers = valid_rows["Symbol"].unique().tolist()
start_date = valid_rows["Filing Dates"].min() - timedelta(days=2)
end_date = valid_rows["Filing Dates"].max() + timedelta(days=10)

# ✅ Fetch stock data
stock_data = get_stock_data(tickers, start_date, end_date)

# ✅ Initialize return column
valid_rows["5-Day Return (%)"] = None

# ✅ Compute 5-day aggregate return per stock
def calculate_5_day_return(row):
    symbol = row["Symbol"]
    filing_date = row["Filing Dates"]

    if stock_data is not None and symbol in stock_data.columns and filing_date in stock_data.index:
        try:
            filing_price = stock_data.loc[filing_date, symbol]
            future_date = stock_data.index[stock_data.index.get_loc(filing_date) + 5]  # Get price 5 days ahead
            future_price = stock_data.loc[future_date, symbol]
            return ((future_price - filing_price) / filing_price) * 100
        except Exception:
            return None  # Handle missing prices safely
    return None

valid_rows["5-Day Return (%)"] = valid_rows.apply(calculate_5_day_return, axis=1)

# ✅ Merge returns with main dataset
sp500 = sp500.merge(valid_rows[["Symbol", "Filing Dates", "5-Day Return (%)"]], 
                    on=["Symbol", "Filing Dates"], how="left")

# ✅ Display results
sp500.head(10)

```

    📡 Fetching stock data (Attempt 1/3)...
    YF.download() has changed argument auto_adjust default to True


    [*********************100%***********************]  501 of 501 completed
    
    16 Failed downloads:
    ['DISH', 'CDAY', 'PXD', 'RE', 'CTLT', 'PKI', 'WRK', 'SIVB', 'BRK.B', 'ABC', 'FLT', 'PEAK', 'FISV', 'MRO', 'ATVI']: YFTzMissingError('possibly delisted; no timezone found')
    ['BF.B']: YFPricesMissingError('possibly delisted; no price data found  (1d 2022-01-19 00:00:00 -> 2022-12-31 00:00:00)')





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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
      <th>5-Day Return (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
      <td>-4.894889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
      <td>-0.154269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
      <td>3.27939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
      <td>2.596676</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
      <td>5.594427</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>2022-02-25</td>
      <td>Ticker
ATVI   NaN
ATVI   NaN
dtype: float64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>2022-02-17</td>
      <td>2.935405</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>2022-01-21</td>
      <td>3.650651</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADP</td>
      <td>ADP</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Data Processing &amp; Outsourced Services</td>
      <td>Roseland, New Jersey</td>
      <td>1981-03-31</td>
      <td>8670</td>
      <td>1949</td>
      <td>0000008670-22-000038</td>
      <td>2022-08-03</td>
      <td>3.335776</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAP</td>
      <td>Advance Auto Parts</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Automotive Retail</td>
      <td>Raleigh, North Carolina</td>
      <td>2015-07-09</td>
      <td>1158449</td>
      <td>1932</td>
      <td>0001158449-22-000037</td>
      <td>2022-02-15</td>
      <td>-11.590314</td>
    </tr>
  </tbody>
</table>
</div>




```python

with open('inputs/ML_negative_unigram.txt', 'r') as file:
    BHR_negative = [line.strip().lower() for line in file]

BHR_negative.sort()
# BHR_negative

```


```python
with open('inputs/ML_positive_unigram.txt', 'r') as file:
    BHR_positive = [line.strip().lower() for line in file]

len(BHR_negative), len(BHR_positive)
BHR_positive.sort()
# BHR_positive # not exhaustive word forms
```


```python
file_path = "inputs/LM_MasterDictionary_1993-2021.csv"  # Update with actual path
df = pd.read_csv(file_path)
LM_positive = df[df['Positive'] > 0]['Word'].tolist()
LM_positive = [e.lower() for e in LM_positive] # to be consistent with our BHR input
df.describe() # there are negative numbers in the columns: years the word is removed!
len(LM_positive)
# LM_positive
```




    347




```python
LM_negative = df[df['Negative'] > 0]['Word'].tolist()
LM_negative = [e.lower() for e in LM_negative] # to be consistent with our BHR input
# LM_negative
```


```python
# ✅ Load LM Master Dictionary
lm_df = pd.read_csv("inputs/LM_MasterDictionary_1993-2021.csv")

# ✅ Extract LM Positive & Negative Word Lists
LM_positive = lm_df[lm_df["Positive"] > 0]["Word"].str.lower().tolist()
LM_negative = lm_df[lm_df["Negative"] > 0]["Word"].str.lower().tolist()

# ✅ Display Word Counts
print(f" BRH Positive Words: {len(BHR_positive)}")
print(f" BRH Negative Words: {len(BHR_negative)}")
print(f" LM Positive Words: {len(LM_positive)}")
print(f" LM Negative Words: {len(LM_negative)}")

# ✅ Display First 10 Words from Each List
print("\n BRH Positive Sample:", BHR_positive)
print(" BRH Negative Sample:", BHR_negative)
print(" LM Positive Sample:", LM_positive)
print(" LM Negative Sample:", LM_negative)

```

     BRH Positive Words: 75
     BRH Negative Words: 94
     LM Positive Words: 347
     LM Negative Words: 2345
    
     BRH Positive Sample: ['above', 'achieved', 'across', 'basis', 'benefit', 'benefited', 'benefiting', 'better', 'cash', 'congrats', 'congratulations', 'continue', 'continued', 'continues', 'curious', 'delivered', 'driving', 'drove', 'exceeded', 'exceeding', 'excellent', 'expansion', 'fantastic', 'favorable', 'flow', 'gains', 'generated', 'good', 'great', 'grew', 'growing', 'growth', 'helped', 'helping', 'impressive', 'improved', 'improvement', 'improvements', 'improving', 'income', 'increase', 'increased', 'increasing', 'job', 'leverage', 'lot', 'margin', 'momentum', 'nice', 'nicely', 'operating', 'outperformance', 'outstanding', 'over', 'performance', 'pleased', 'pretty', 'proud', 'raising', 'really', 'record', 'repurchase', 'results', 'share', 'solid', 'strength', 'strong', 'success', 'sustainable', 'terrific', 'think', 'up', 'upside', 'well', 'years']
     BRH Negative Sample: ['actions', 'address', 'affected', 'affecting', 'anticipated', 'associated', 'back', 'believe', 'below', 'caused', 'causing', 'certain', 'challenges', 'challenging', 'change', 'changed', 'changes', 'confident', 'costs', 'decision', 'decline', 'declined', 'declines', 'decrease', 'decreased', 'delay', 'delayed', 'delays', 'disappointed', 'disappointing', 'disappointment', 'down', 'due', 'dynamics', 'expectations', 'expected', 'experienced', 'factors', 'fell', 'goodwill', 'happened', 'headwinds', 'however', 'impact', 'impacted', 'impacting', 'impacts', 'inefficiencies', 'issue', 'issues', 'lack', 'longer', 'losing', 'loss', 'lost', 'lower', 'miss', 'necessary', 'need', 'negative', 'negatively', 'not', 'offset', 'pressure', 'pressures', 'pronounced', 'pushed', 'related', 'resolve', 'revised', 'short', 'shortfall', 'slipped', 'slowdown', 'slowed', 'slower', 'slowing', 'soft', 'softer', 'softness', 'steps', 'taking', 'temporary', 'term', 'timing', 'transition', 'trying', 'underperformance', 'understand', 'unexpected', 'unfortunately', 'weaker', 'weakness', 'worse']
     LM Positive Sample: ['able', 'abundance', 'abundant', 'acclaimed', 'accomplish', 'accomplished', 'accomplishes', 'accomplishing', 'accomplishment', 'accomplishments', 'achieve', 'achieved', 'achievement', 'achievements', 'achieves', 'achieving', 'adequately', 'advancement', 'advancements', 'advances', 'advancing', 'advantage', 'advantaged', 'advantageous', 'advantageously', 'advantages', 'alliance', 'alliances', 'assure', 'assured', 'assures', 'assuring', 'attain', 'attained', 'attaining', 'attainment', 'attainments', 'attains', 'attractive', 'attractiveness', 'beautiful', 'beautifully', 'beneficially', 'benefited', 'benefiting', 'benefitted', 'benefitting', 'best', 'better', 'bolstered', 'bolstering', 'bolsters', 'boom', 'booming', 'boost', 'boosted', 'breakthrough', 'breakthroughs', 'brilliant', 'charitable', 'collaborate', 'collaborated', 'collaborates', 'collaborating', 'collaboration', 'collaborations', 'collaborative', 'collaborator', 'collaborators', 'compliment', 'complimentary', 'complimented', 'complimenting', 'compliments', 'conclusive', 'conclusively', 'conducive', 'confident', 'constructive', 'constructively', 'courteous', 'creative', 'creatively', 'creativeness', 'creativity', 'delight', 'delighted', 'delightful', 'delightfully', 'delighting', 'delights', 'dependability', 'dependable', 'desirable', 'desired', 'despite', 'destined', 'diligent', 'diligently', 'distinction', 'distinctions', 'distinctive', 'distinctively', 'distinctiveness', 'dream', 'easier', 'easily', 'easy', 'efficiencies', 'efficiency', 'efficient', 'efficiently', 'empower', 'empowered', 'empowering', 'empowers', 'enable', 'enabled', 'enables', 'enabling', 'encouraged', 'encouragement', 'encourages', 'encouraging', 'enhance', 'enhanced', 'enhancement', 'enhancements', 'enhances', 'enhancing', 'enjoy', 'enjoyable', 'enjoyably', 'enjoyed', 'enjoying', 'enjoyment', 'enjoys', 'enthusiasm', 'enthusiastic', 'enthusiastically', 'excellence', 'excellent', 'excelling', 'excels', 'exceptional', 'exceptionally', 'excited', 'excitement', 'exciting', 'exclusive', 'exclusively', 'exclusiveness', 'exclusives', 'exclusivity', 'exemplary', 'fantastic', 'favorable', 'favorably', 'favored', 'favoring', 'favorite', 'favorites', 'friendly', 'gain', 'gained', 'gaining', 'gains', 'good', 'greatest', 'greatly', 'greatness', 'happiest', 'happily', 'happiness', 'happy', 'highest', 'honor', 'honored', 'honoring', 'honors', 'ideal', 'impress', 'impressed', 'impresses', 'impressing', 'impressive', 'impressively', 'improve', 'improved', 'improvement', 'improvements', 'improves', 'improving', 'incredible', 'incredibly', 'influential', 'informative', 'ingenuity', 'innovate', 'innovated', 'innovates', 'innovating', 'innovation', 'innovations', 'innovative', 'innovativeness', 'innovator', 'innovators', 'insightful', 'inspiration', 'inspirational', 'integrity', 'invent', 'invented', 'inventing', 'invention', 'inventions', 'inventive', 'inventiveness', 'inventor', 'inventors', 'leadership', 'leading', 'loyal', 'lucrative', 'meritorious', 'opportunities', 'opportunity', 'optimistic', 'outperform', 'outperformed', 'outperforming', 'outperforms', 'perfect', 'perfected', 'perfectly', 'perfects', 'pleasant', 'pleasantly', 'pleased', 'pleasure', 'plentiful', 'popular', 'popularity', 'positive', 'positively', 'preeminence', 'preeminent', 'premier', 'premiere', 'prestige', 'prestigious', 'proactive', 'proactively', 'proficiency', 'proficient', 'proficiently', 'profitability', 'profitable', 'profitably', 'progress', 'progressed', 'progresses', 'progressing', 'prospered', 'prospering', 'prosperity', 'prosperous', 'prospers', 'rebound', 'rebounded', 'rebounding', 'receptive', 'regain', 'regained', 'regaining', 'resolve', 'revolutionize', 'revolutionized', 'revolutionizes', 'revolutionizing', 'reward', 'rewarded', 'rewarding', 'satisfaction', 'satisfactorily', 'satisfactory', 'satisfied', 'satisfies', 'satisfy', 'satisfying', 'smooth', 'smoothing', 'smoothly', 'smooths', 'solves', 'solving', 'spectacular', 'spectacularly', 'stability', 'stabilization', 'stabilizations', 'stabilize', 'stabilized', 'stabilizes', 'stabilizing', 'stable', 'strength', 'strengthen', 'strengthened', 'strengthening', 'strengthens', 'strengths', 'strong', 'stronger', 'strongest', 'succeed', 'succeeded', 'succeeding', 'succeeds', 'success', 'successes', 'successful', 'successfully', 'superior', 'surpass', 'surpassed', 'surpasses', 'surpassing', 'transparency', 'tremendous', 'tremendously', 'unmatched', 'unparalleled', 'unsurpassed', 'upturn', 'upturns', 'valuable', 'versatile', 'versatility', 'vibrancy', 'vibrant', 'win', 'winner', 'winners', 'winning', 'worthy']
     LM Negative Sample: ['abandon', 'abandoned', 'abandoning', 'abandonment', 'abandonments', 'abandons', 'abdicated', 'abdicates', 'abdicating', 'abdication', 'abdications', 'aberrant', 'aberration', 'aberrational', 'aberrations', 'abetting', 'abnormal', 'abnormalities', 'abnormality', 'abnormally', 'abolish', 'abolished', 'abolishes', 'abolishing', 'abrogate', 'abrogated', 'abrogates', 'abrogating', 'abrogation', 'abrogations', 'abrupt', 'abruptly', 'abruptness', 'absence', 'absences', 'absenteeism', 'abuse', 'abused', 'abuses', 'abusing', 'abusive', 'abusively', 'abusiveness', 'accident', 'accidental', 'accidentally', 'accidents', 'accusation', 'accusations', 'accuse', 'accused', 'accuses', 'accusing', 'acquiesce', 'acquiesced', 'acquiesces', 'acquiescing', 'acquit', 'acquits', 'acquittal', 'acquittals', 'acquitted', 'acquitting', 'adulterate', 'adulterated', 'adulterating', 'adulteration', 'adulterations', 'adversarial', 'adversaries', 'adversary', 'adverse', 'adversely', 'adversities', 'adversity', 'aftermath', 'aftermaths', 'against', 'aggravate', 'aggravated', 'aggravates', 'aggravating', 'aggravation', 'aggravations', 'alerted', 'alerting', 'alienate', 'alienated', 'alienates', 'alienating', 'alienation', 'alienations', 'allegation', 'allegations', 'allege', 'alleged', 'allegedly', 'alleges', 'alleging', 'annoy', 'annoyance', 'annoyances', 'annoyed', 'annoying', 'annoys', 'annul', 'annulled', 'annulling', 'annulment', 'annulments', 'annuls', 'anomalies', 'anomalous', 'anomalously', 'anomaly', 'anticompetitive', 'antitrust', 'argue', 'argued', 'arguing', 'argument', 'argumentative', 'arguments', 'arrearage', 'arrearages', 'arrears', 'arrest', 'arrested', 'arrests', 'artificially', 'assault', 'assaulted', 'assaulting', 'assaults', 'assertions', 'attrition', 'aversely', 'backdating', 'bad', 'bail', 'bailout', 'balk', 'balked', 'bankrupt', 'bankruptcies', 'bankruptcy', 'bankrupted', 'bankrupting', 'bankrupts', 'bans', 'barred', 'barrier', 'barriers', 'bottleneck', 'bottlenecks', 'boycott', 'boycotted', 'boycotting', 'boycotts', 'breach', 'breached', 'breaches', 'breaching', 'break', 'breakage', 'breakages', 'breakdown', 'breakdowns', 'breaks', 'bribe', 'bribed', 'briberies', 'bribery', 'bribes', 'bribing', 'burden', 'burdened', 'burdening', 'burdens', 'burdensome', 'burned', 'calamities', 'calamitous', 'calamity', 'cancel', 'canceled', 'canceling', 'cancellation', 'cancellations', 'cancelled', 'cancelling', 'cancels', 'careless', 'carelessly', 'carelessness', 'catastrophe', 'catastrophes', 'catastrophic', 'catastrophically', 'caution', 'cautionary', 'cautioned', 'cautioning', 'cautions', 'cease', 'ceased', 'ceases', 'ceasing', 'censure', 'censured', 'censures', 'censuring', 'challenge', 'challenged', 'challenges', 'challenging', 'chargeoffs', 'circumvent', 'circumvented', 'circumventing', 'circumvention', 'circumventions', 'circumvents', 'claiming', 'claims', 'clawback', 'closeout', 'closeouts', 'closings', 'closure', 'closures', 'coerce', 'coerced', 'coerces', 'coercing', 'coercion', 'coercive', 'collapse', 'collapsed', 'collapses', 'collapsing', 'collision', 'collisions', 'collude', 'colluded', 'colludes', 'colluding', 'collusion', 'collusions', 'collusive', 'complain', 'complained', 'complaining', 'complains', 'complaint', 'complaints', 'complicate', 'complicated', 'complicates', 'complicating', 'complication', 'complications', 'compulsion', 'concealed', 'concealing', 'concede', 'conceded', 'concedes', 'conceding', 'concern', 'concerned', 'concerns', 'conciliating', 'conciliation', 'conciliations', 'condemn', 'condemnation', 'condemnations', 'condemned', 'condemning', 'condemns', 'condone', 'condoned', 'confess', 'confessed', 'confesses', 'confessing', 'confession', 'confine', 'confined', 'confinement', 'confinements', 'confines', 'confining', 'confiscate', 'confiscated', 'confiscates', 'confiscating', 'confiscation', 'confiscations', 'conflict', 'conflicted', 'conflicting', 'conflicts', 'confront', 'confrontation', 'confrontational', 'confrontations', 'confronted', 'confronting', 'confronts', 'confuse', 'confused', 'confuses', 'confusing', 'confusingly', 'confusion', 'conspiracies', 'conspiracy', 'conspirator', 'conspiratorial', 'conspirators', 'conspire', 'conspired', 'conspires', 'conspiring', 'contempt', 'contend', 'contended', 'contending', 'contends', 'contention', 'contentions', 'contentious', 'contentiously', 'contested', 'contesting', 'contraction', 'contractions', 'contradict', 'contradicted', 'contradicting', 'contradiction', 'contradictions', 'contradictory', 'contradicts', 'contrary', 'controversial', 'controversies', 'controversy', 'convict', 'convicted', 'convicting', 'conviction', 'convictions', 'corrected', 'correcting', 'correction', 'corrections', 'corrects', 'corrupt', 'corrupted', 'corrupting', 'corruption', 'corruptions', 'corruptly', 'corruptness', 'costly', 'counterclaim', 'counterclaimed', 'counterclaiming', 'counterclaims', 'counterfeit', 'counterfeited', 'counterfeiter', 'counterfeiters', 'counterfeiting', 'counterfeits', 'countermeasure', 'countermeasures', 'crime', 'crimes', 'criminal', 'criminally', 'criminals', 'crises', 'crisis', 'critically', 'criticism', 'criticisms', 'criticize', 'criticized', 'criticizes', 'criticizing', 'crucial', 'crucially', 'culpability', 'culpable', 'culpably', 'cumbersome', 'curtail', 'curtailed', 'curtailing', 'curtailment', 'curtailments', 'curtails', 'cut', 'cutback', 'cutbacks', 'cyberattack', 'cyberattacks', 'cyberbullying', 'cybercrime', 'cybercrimes', 'cybercriminal', 'cybercriminals', 'damage', 'damaged', 'damages', 'damaging', 'dampen', 'dampened', 'danger', 'dangerous', 'dangerously', 'dangers', 'deadlock', 'deadlocked', 'deadlocking', 'deadlocks', 'deadweight', 'deadweights', 'debarment', 'debarments', 'debarred', 'deceased', 'deceit', 'deceitful', 'deceitfulness', 'deceive', 'deceived', 'deceives', 'deceiving', 'deception', 'deceptions', 'deceptive', 'deceptively', 'decline', 'declined', 'declines', 'declining', 'deface', 'defaced', 'defacement', 'defamation', 'defamations', 'defamatory', 'defame', 'defamed', 'defames', 'defaming', 'default', 'defaulted', 'defaulting', 'defaults', 'defeat', 'defeated', 'defeating', 'defeats', 'defect', 'defective', 'defects', 'defend', 'defendant', 'defendants', 'defended', 'defending', 'defends', 'defensive', 'defer', 'deficiencies', 'deficiency', 'deficient', 'deficit', 'deficits', 'defraud', 'defrauded', 'defrauding', 'defrauds', 'defunct', 'degradation', 'degradations', 'degrade', 'degraded', 'degrades', 'degrading', 'delay', 'delayed', 'delaying', 'delays', 'deleterious', 'deliberate', 'deliberated', 'deliberately', 'delinquencies', 'delinquency', 'delinquent', 'delinquently', 'delinquents', 'delist', 'delisted', 'delisting', 'delists', 'demise', 'demised', 'demises', 'demising', 'demolish', 'demolished', 'demolishes', 'demolishing', 'demolition', 'demolitions', 'demote', 'demoted', 'demotes', 'demoting', 'demotion', 'demotions', 'denial', 'denials', 'denied', 'denies', 'denigrate', 'denigrated', 'denigrates', 'denigrating', 'denigration', 'deny', 'denying', 'deplete', 'depleted', 'depletes', 'depleting', 'depletion', 'depletions', 'deprecation', 'depress', 'depressed', 'depresses', 'depressing', 'deprivation', 'deprive', 'deprived', 'deprives', 'depriving', 'derelict', 'dereliction', 'derogatory', 'destabilization', 'destabilize', 'destabilized', 'destabilizing', 'destroy', 'destroyed', 'destroying', 'destroys', 'destruction', 'destructive', 'detain', 'detained', 'detention', 'detentions', 'deter', 'deteriorate', 'deteriorated', 'deteriorates', 'deteriorating', 'deterioration', 'deteriorations', 'deterred', 'deterrence', 'deterrences', 'deterrent', 'deterrents', 'deterring', 'deters', 'detract', 'detracted', 'detracting', 'detriment', 'detrimental', 'detrimentally', 'detriments', 'devalue', 'devalued', 'devalues', 'devaluing', 'devastate', 'devastated', 'devastating', 'devastation', 'deviate', 'deviated', 'deviates', 'deviating', 'deviation', 'deviations', 'devolve', 'devolved', 'devolves', 'devolving', 'difficult', 'difficulties', 'difficultly', 'difficulty', 'diminish', 'diminished', 'diminishes', 'diminishing', 'diminution', 'disadvantage', 'disadvantaged', 'disadvantageous', 'disadvantages', 'disaffiliation', 'disagree', 'disagreeable', 'disagreed', 'disagreeing', 'disagreement', 'disagreements', 'disagrees', 'disallow', 'disallowance', 'disallowances', 'disallowed', 'disallowing', 'disallows', 'disappear', 'disappearance', 'disappearances', 'disappeared', 'disappearing', 'disappears', 'disappoint', 'disappointed', 'disappointing', 'disappointingly', 'disappointment', 'disappointments', 'disappoints', 'disapproval', 'disapprovals', 'disapprove', 'disapproved', 'disapproves', 'disapproving', 'disassociates', 'disassociating', 'disassociation', 'disassociations', 'disaster', 'disasters', 'disastrous', 'disastrously', 'disavow', 'disavowal', 'disavowed', 'disavowing', 'disavows', 'disciplinary', 'disclaim', 'disclaimed', 'disclaimer', 'disclaimers', 'disclaiming', 'disclaims', 'disclose', 'disclosed', 'discloses', 'disclosing', 'discontinuance', 'discontinuances', 'discontinuation', 'discontinuations', 'discontinue', 'discontinued', 'discontinues', 'discontinuing', 'discourage', 'discouraged', 'discourages', 'discouraging', 'discredit', 'discredited', 'discrediting', 'discredits', 'discrepancies', 'discrepancy', 'disfavor', 'disfavored', 'disfavoring', 'disfavors', 'disgorge', 'disgorged', 'disgorgement', 'disgorgements', 'disgorges', 'disgorging', 'disgrace', 'disgraceful', 'disgracefully', 'dishonest', 'dishonestly', 'dishonesty', 'dishonor', 'dishonorable', 'dishonorably', 'dishonored', 'dishonoring', 'dishonors', 'disincentives', 'disinterested', 'disinterestedly', 'disinterestedness', 'disloyal', 'disloyally', 'disloyalty', 'dismal', 'dismally', 'dismiss', 'dismissal', 'dismissals', 'dismissed', 'dismisses', 'dismissing', 'disorderly', 'disparage', 'disparaged', 'disparagement', 'disparagements', 'disparages', 'disparaging', 'disparagingly', 'disparities', 'disparity', 'displace', 'displaced', 'displacement', 'displacements', 'displaces', 'displacing', 'dispose', 'dispossess', 'dispossessed', 'dispossesses', 'dispossessing', 'disproportion', 'disproportional', 'disproportionate', 'disproportionately', 'dispute', 'disputed', 'disputes', 'disputing', 'disqualification', 'disqualifications', 'disqualified', 'disqualifies', 'disqualify', 'disqualifying', 'disregard', 'disregarded', 'disregarding', 'disregards', 'disreputable', 'disrepute', 'disrupt', 'disrupted', 'disrupting', 'disruption', 'disruptions', 'disruptive', 'disrupts', 'dissatisfaction', 'dissatisfied', 'dissent', 'dissented', 'dissenter', 'dissenters', 'dissenting', 'dissents', 'dissident', 'dissidents', 'dissolution', 'dissolutions', 'distort', 'distorted', 'distorting', 'distortion', 'distortions', 'distorts', 'distract', 'distracted', 'distracting', 'distraction', 'distractions', 'distracts', 'distress', 'distressed', 'disturb', 'disturbance', 'disturbances', 'disturbed', 'disturbing', 'disturbs', 'diversion', 'divert', 'diverted', 'diverting', 'diverts', 'divest', 'divested', 'divesting', 'divestiture', 'divestitures', 'divestment', 'divestments', 'divests', 'divorce', 'divorced', 'divulge', 'divulged', 'divulges', 'divulging', 'doubt', 'doubted', 'doubtful', 'doubts', 'downgrade', 'downgraded', 'downgrades', 'downgrading', 'downsize', 'downsized', 'downsizes', 'downsizing', 'downsizings', 'downtime', 'downtimes', 'downturn', 'downturns', 'downward', 'downwards', 'drag', 'drastic', 'drastically', 'drawback', 'drawbacks', 'dropped', 'drought', 'droughts', 'duress', 'dysfunction', 'dysfunctional', 'dysfunctions', 'easing', 'egregious', 'egregiously', 'embargo', 'embargoed', 'embargoes', 'embargoing', 'embarrass', 'embarrassed', 'embarrasses', 'embarrassing', 'embarrassment', 'embarrassments', 'embezzle', 'embezzled', 'embezzlement', 'embezzlements', 'embezzler', 'embezzles', 'embezzling', 'encroach', 'encroached', 'encroaches', 'encroaching', 'encroachment', 'encroachments', 'encumber', 'encumbered', 'encumbering', 'encumbers', 'encumbrance', 'encumbrances', 'endanger', 'endangered', 'endangering', 'endangerment', 'endangers', 'enjoin', 'enjoined', 'enjoining', 'enjoins', 'erode', 'eroded', 'erodes', 'eroding', 'erosion', 'erratic', 'erratically', 'erred', 'erring', 'erroneous', 'erroneously', 'error', 'errors', 'errs', 'escalate', 'escalated', 'escalates', 'escalating', 'evade', 'evaded', 'evades', 'evading', 'evasion', 'evasions', 'evasive', 'evict', 'evicted', 'evicting', 'eviction', 'evictions', 'evicts', 'exacerbate', 'exacerbated', 'exacerbates', 'exacerbating', 'exacerbation', 'exacerbations', 'exaggerate', 'exaggerated', 'exaggerates', 'exaggerating', 'exaggeration', 'excessive', 'excessively', 'exculpate', 'exculpated', 'exculpates', 'exculpating', 'exculpation', 'exculpations', 'exculpatory', 'exonerate', 'exonerated', 'exonerates', 'exonerating', 'exoneration', 'exonerations', 'exploit', 'exploitation', 'exploitations', 'exploitative', 'exploited', 'exploiting', 'exploits', 'expose', 'exposed', 'exposes', 'exposing', 'expropriate', 'expropriated', 'expropriates', 'expropriating', 'expropriation', 'expropriations', 'expulsion', 'expulsions', 'extenuating', 'fail', 'failed', 'failing', 'failings', 'fails', 'failure', 'failures', 'fallout', 'false', 'falsely', 'falsification', 'falsifications', 'falsified', 'falsifies', 'falsify', 'falsifying', 'falsity', 'fatalities', 'fatality', 'fatally', 'fault', 'faulted', 'faults', 'faulty', 'fear', 'fears', 'felonies', 'felonious', 'felony', 'fictitious', 'fined', 'fines', 'fired', 'firing', 'flaw', 'flawed', 'flaws', 'forbid', 'forbidden', 'forbidding', 'forbids', 'forced', 'forcing', 'foreclose', 'foreclosed', 'forecloses', 'foreclosing', 'foreclosure', 'foreclosures', 'forego', 'foregoes', 'foregone', 'forestall', 'forestalled', 'forestalling', 'forestalls', 'forfeit', 'forfeited', 'forfeiting', 'forfeits', 'forfeiture', 'forfeitures', 'forgers', 'forgery', 'fraud', 'frauds', 'fraudulence', 'fraudulent', 'fraudulently', 'frivolous', 'frivolously', 'frustrate', 'frustrated', 'frustrates', 'frustrating', 'frustratingly', 'frustration', 'frustrations', 'fugitives', 'gratuitous', 'gratuitously', 'grievance', 'grievances', 'grossly', 'groundless', 'guilty', 'halt', 'halted', 'hamper', 'hampered', 'hampering', 'hampers', 'harass', 'harassed', 'harassing', 'harassment', 'hardship', 'hardships', 'harm', 'harmed', 'harmful', 'harmfully', 'harming', 'harms', 'harsh', 'harsher', 'harshest', 'harshly', 'harshness', 'hazard', 'hazardous', 'hazards', 'hinder', 'hindered', 'hindering', 'hinders', 'hindrance', 'hindrances', 'hostile', 'hostility', 'hurt', 'hurting', 'idle', 'idled', 'idling', 'ignore', 'ignored', 'ignores', 'ignoring', 'ill', 'illegal', 'illegalities', 'illegality', 'illegally', 'illegible', 'illicit', 'illicitly', 'illiquid', 'illiquidity', 'imbalance', 'imbalances', 'immature', 'immoral', 'impair', 'impaired', 'impairing', 'impairment', 'impairments', 'impairs', 'impasse', 'impasses', 'impede', 'impeded', 'impedes', 'impediment', 'impediments', 'impeding', 'impending', 'imperative', 'imperfection', 'imperfections', 'imperil', 'impermissible', 'implicate', 'implicated', 'implicates', 'implicating', 'impossibility', 'impossible', 'impound', 'impounded', 'impounding', 'impounds', 'impracticable', 'impractical', 'impracticalities', 'impracticality', 'imprisonment', 'improper', 'improperly', 'improprieties', 'impropriety', 'imprudent', 'imprudently', 'inability', 'inaccessible', 'inaccuracies', 'inaccuracy', 'inaccurate', 'inaccurately', 'inaction', 'inactions', 'inactivate', 'inactivated', 'inactivates', 'inactivating', 'inactivation', 'inactivations', 'inactivity', 'inadequacies', 'inadequacy', 'inadequate', 'inadequately', 'inadvertent', 'inadvertently', 'inadvisability', 'inadvisable', 'inappropriate', 'inappropriately', 'inattention', 'incapable', 'incapacitated', 'incapacity', 'incarcerate', 'incarcerated', 'incarcerates', 'incarcerating', 'incarceration', 'incarcerations', 'incidence', 'incidences', 'incident', 'incidents', 'incompatibilities', 'incompatibility', 'incompatible', 'incompetence', 'incompetency', 'incompetent', 'incompetently', 'incompetents', 'incomplete', 'incompletely', 'incompleteness', 'inconclusive', 'inconsistencies', 'inconsistency', 'inconsistent', 'inconsistently', 'inconvenience', 'inconveniences', 'inconvenient', 'incorrect', 'incorrectly', 'incorrectness', 'indecency', 'indecent', 'indefeasible', 'indefeasibly', 'indict', 'indictable', 'indicted', 'indicting', 'indictment', 'indictments', 'ineffective', 'ineffectively', 'ineffectiveness', 'inefficiencies', 'inefficiency', 'inefficient', 'inefficiently', 'ineligibility', 'ineligible', 'inequitable', 'inequitably', 'inequities', 'inequity', 'inevitable', 'inexperience', 'inexperienced', 'inferior', 'inflicted', 'infraction', 'infractions', 'infringe', 'infringed', 'infringement', 'infringements', 'infringes', 'infringing', 'inhibited', 'inimical', 'injunction', 'injunctions', 'injure', 'injured', 'injures', 'injuries', 'injuring', 'injurious', 'injury', 'inordinate', 'inordinately', 'inquiry', 'insecure', 'insensitive', 'insolvencies', 'insolvency', 'insolvent', 'instability', 'insubordination', 'insufficiency', 'insufficient', 'insufficiently', 'insurrection', 'insurrections', 'intentional', 'interfere', 'interfered', 'interference', 'interferences', 'interferes', 'interfering', 'intermittent', 'intermittently', 'interrupt', 'interrupted', 'interrupting', 'interruption', 'interruptions', 'interrupts', 'intimidation', 'intrusion', 'invalid', 'invalidate', 'invalidated', 'invalidates', 'invalidating', 'invalidation', 'invalidity', 'investigate', 'investigated', 'investigates', 'investigating', 'investigation', 'investigations', 'involuntarily', 'involuntary', 'irreconcilable', 'irreconcilably', 'irrecoverable', 'irrecoverably', 'irregular', 'irregularities', 'irregularity', 'irregularly', 'irreparable', 'irreparably', 'irreversible', 'jeopardize', 'jeopardized', 'justifiable', 'kickback', 'kickbacks', 'knowingly', 'lack', 'lacked', 'lacking', 'lackluster', 'lacks', 'lag', 'lagged', 'lagging', 'lags', 'lapse', 'lapsed', 'lapses', 'lapsing', 'laundering', 'layoff', 'layoffs', 'lie', 'limitation', 'limitations', 'lingering', 'liquidate', 'liquidated', 'liquidates', 'liquidating', 'liquidation', 'liquidations', 'liquidator', 'liquidators', 'litigant', 'litigants', 'litigate', 'litigated', 'litigates', 'litigating', 'litigation', 'litigations', 'lockout', 'lockouts', 'lose', 'loses', 'losing', 'loss', 'losses', 'lost', 'lying', 'malfeasance', 'malfunction', 'malfunctioned', 'malfunctioning', 'malfunctions', 'malice', 'malicious', 'maliciously', 'malpractice', 'manipulate', 'manipulated', 'manipulates', 'manipulating', 'manipulation', 'manipulations', 'manipulative', 'markdown', 'markdowns', 'misapplication', 'misapplications', 'misapplied', 'misapplies', 'misapply', 'misapplying', 'misappropriate', 'misappropriated', 'misappropriates', 'misappropriating', 'misappropriation', 'misappropriations', 'misbranded', 'miscalculate', 'miscalculated', 'miscalculates', 'miscalculating', 'miscalculation', 'miscalculations', 'mischaracterization', 'mischief', 'misclassification', 'misclassifications', 'misclassified', 'misclassify', 'miscommunication', 'misconduct', 'misdated', 'misdemeanor', 'misdemeanors', 'misdirected', 'mishandle', 'mishandled', 'mishandles', 'mishandling', 'misinform', 'misinformation', 'misinformed', 'misinforming', 'misinforms', 'misinterpret', 'misinterpretation', 'misinterpretations', 'misinterpreted', 'misinterpreting', 'misinterprets', 'misjudge', 'misjudged', 'misjudges', 'misjudging', 'misjudgment', 'misjudgments', 'mislabel', 'mislabeled', 'mislabeling', 'mislabelled', 'mislabels', 'mislead', 'misleading', 'misleadingly', 'misleads', 'misled', 'mismanage', 'mismanaged', 'mismanagement', 'mismanages', 'mismanaging', 'mismatch', 'mismatched', 'mismatches', 'mismatching', 'misplaced', 'misprice', 'mispricing', 'mispricings', 'misrepresent', 'misrepresentation', 'misrepresentations', 'misrepresented', 'misrepresenting', 'misrepresents', 'miss', 'missed', 'misses', 'misstate', 'misstated', 'misstatement', 'misstatements', 'misstates', 'misstating', 'misstep', 'missteps', 'mistake', 'mistaken', 'mistakenly', 'mistakes', 'mistaking', 'mistrial', 'mistrials', 'misunderstand', 'misunderstanding', 'misunderstandings', 'misunderstood', 'misuse', 'misused', 'misuses', 'misusing', 'monopolistic', 'monopolists', 'monopolization', 'monopolize', 'monopolized', 'monopolizes', 'monopolizing', 'monopoly', 'moratoria', 'moratorium', 'moratoriums', 'mothballed', 'mothballing', 'negative', 'negatively', 'negatives', 'neglect', 'neglected', 'neglectful', 'neglecting', 'neglects', 'negligence', 'negligences', 'negligent', 'negligently', 'nonattainment', 'noncompetitive', 'noncompliance', 'noncompliances', 'noncompliant', 'noncomplying', 'nonconforming', 'nonconformities', 'nonconformity', 'nondisclosure', 'nonfunctional', 'nonpayment', 'nonpayments', 'nonperformance', 'nonperformances', 'nonperforming', 'nonproducing', 'nonproductive', 'nonrecoverable', 'nonrenewal', 'nuisance', 'nuisances', 'nullification', 'nullifications', 'nullified', 'nullifies', 'nullify', 'nullifying', 'objected', 'objecting', 'objection', 'objectionable', 'objectionably', 'objections', 'obscene', 'obscenity', 'obsolescence', 'obsolete', 'obstacle', 'obstacles', 'obstruct', 'obstructed', 'obstructing', 'obstruction', 'obstructions', 'offence', 'offences', 'offend', 'offended', 'offender', 'offenders', 'offending', 'offends', 'omission', 'omissions', 'omit', 'omits', 'omitted', 'omitting', 'onerous', 'opportunistic', 'opportunistically', 'oppose', 'opposed', 'opposes', 'opposing', 'opposition', 'oppositions', 'outage', 'outages', 'outdated', 'outmoded', 'overage', 'overages', 'overbuild', 'overbuilding', 'overbuilds', 'overbuilt', 'overburden', 'overburdened', 'overburdening', 'overcapacities', 'overcapacity', 'overcharge', 'overcharged', 'overcharges', 'overcharging', 'overcome', 'overcomes', 'overcoming', 'overdue', 'overestimate', 'overestimated', 'overestimates', 'overestimating', 'overestimation', 'overestimations', 'overload', 'overloaded', 'overloading', 'overloads', 'overlook', 'overlooked', 'overlooking', 'overlooks', 'overpaid', 'overpayment', 'overpayments', 'overproduced', 'overproduces', 'overproducing', 'overproduction', 'overrun', 'overrunning', 'overruns', 'overshadow', 'overshadowed', 'overshadowing', 'overshadows', 'overstate', 'overstated', 'overstatement', 'overstatements', 'overstates', 'overstating', 'oversupplied', 'oversupplies', 'oversupply', 'oversupplying', 'overtly', 'overturn', 'overturned', 'overturning', 'overturns', 'overvalue', 'overvalued', 'overvaluing', 'panic', 'panics', 'penalize', 'penalized', 'penalizes', 'penalizing', 'penalties', 'penalty', 'peril', 'perils', 'perjury', 'perpetrate', 'perpetrated', 'perpetrates', 'perpetrating', 'perpetration', 'persist', 'persisted', 'persistence', 'persistent', 'persistently', 'persisting', 'persists', 'pervasive', 'pervasively', 'pervasiveness', 'petty', 'picket', 'picketed', 'picketing', 'plaintiff', 'plaintiffs', 'plea', 'plead', 'pleaded', 'pleading', 'pleadings', 'pleads', 'pleas', 'pled', 'poor', 'poorly', 'poses', 'posing', 'postpone', 'postponed', 'postponement', 'postponements', 'postpones', 'postponing', 'precipitated', 'precipitous', 'precipitously', 'preclude', 'precluded', 'precludes', 'precluding', 'predatory', 'prejudice', 'prejudiced', 'prejudices', 'prejudicial', 'prejudicing', 'premature', 'prematurely', 'pressing', 'pretrial', 'preventing', 'prevention', 'prevents', 'problem', 'problematic', 'problematical', 'problems', 'prolong', 'prolongation', 'prolongations', 'prolonged', 'prolonging', 'prolongs', 'prone', 'prosecute', 'prosecuted', 'prosecutes', 'prosecuting', 'prosecution', 'prosecutions', 'protest', 'protested', 'protester', 'protesters', 'protesting', 'protestor', 'protestors', 'protests', 'protracted', 'protraction', 'provoke', 'provoked', 'provokes', 'provoking', 'punished', 'punishes', 'punishing', 'punishment', 'punishments', 'punitive', 'purport', 'purported', 'purportedly', 'purporting', 'purports', 'question', 'questionable', 'questionably', 'questioned', 'questioning', 'questions', 'quit', 'quitting', 'racketeer', 'racketeering', 'rationalization', 'rationalizations', 'rationalize', 'rationalized', 'rationalizes', 'rationalizing', 'reassessment', 'reassessments', 'reassign', 'reassigned', 'reassigning', 'reassignment', 'reassignments', 'reassigns', 'recall', 'recalled', 'recalling', 'recalls', 'recession', 'recessionary', 'recessions', 'reckless', 'recklessly', 'recklessness', 'redact', 'redacted', 'redacting', 'redaction', 'redactions', 'redefault', 'redefaulted', 'redefaults', 'redress', 'redressed', 'redresses', 'redressing', 'refusal', 'refusals', 'refuse', 'refused', 'refuses', 'refusing', 'reject', 'rejected', 'rejecting', 'rejection', 'rejections', 'rejects', 'relinquish', 'relinquished', 'relinquishes', 'relinquishing', 'relinquishment', 'relinquishments', 'reluctance', 'reluctant', 'renegotiate', 'renegotiated', 'renegotiates', 'renegotiating', 'renegotiation', 'renegotiations', 'renounce', 'renounced', 'renouncement', 'renouncements', 'renounces', 'renouncing', 'reparation', 'reparations', 'repossessed', 'repossesses', 'repossessing', 'repossession', 'repossessions', 'repudiate', 'repudiated', 'repudiates', 'repudiating', 'repudiation', 'repudiations', 'resign', 'resignation', 'resignations', 'resigned', 'resigning', 'resigns', 'restate', 'restated', 'restatement', 'restatements', 'restates', 'restating', 'restructure', 'restructured', 'restructures', 'restructuring', 'restructurings', 'retaliate', 'retaliated', 'retaliates', 'retaliating', 'retaliation', 'retaliations', 'retaliatory', 'retribution', 'retributions', 'revocation', 'revocations', 'revoke', 'revoked', 'revokes', 'revoking', 'ridicule', 'ridiculed', 'ridicules', 'ridiculing', 'riskier', 'riskiest', 'risky', 'sabotage', 'sacrifice', 'sacrificed', 'sacrifices', 'sacrificial', 'sacrificing', 'scandalous', 'scandals', 'scrutinize', 'scrutinized', 'scrutinizes', 'scrutinizing', 'scrutiny', 'seize', 'seized', 'seizes', 'seizing', 'sentenced', 'sentencing', 'serious', 'seriously', 'seriousness', 'setback', 'setbacks', 'sever', 'severe', 'severed', 'severely', 'severities', 'severity', 'sharply', 'shocked', 'shortage', 'shortages', 'shortfall', 'shortfalls', 'shrinkage', 'shrinkages', 'shut', 'shutdown', 'shutdowns', 'shuts', 'shutting', 'slander', 'slandered', 'slanderous', 'slanders', 'slippage', 'slippages', 'slow', 'slowdown', 'slowdowns', 'slowed', 'slower', 'slowest', 'slowing', 'slowly', 'slowness', 'sluggish', 'sluggishly', 'sluggishness', 'solvencies', 'solvency', 'spam', 'spammers', 'spamming', 'staggering', 'stagnant', 'stagnate', 'stagnated', 'stagnates', 'stagnating', 'stagnation', 'standstill', 'standstills', 'stolen', 'stoppage', 'stoppages', 'stopped', 'stopping', 'stops', 'strain', 'strained', 'straining', 'strains', 'stress', 'stressed', 'stresses', 'stressful', 'stressing', 'stringent', 'subjected', 'subjecting', 'subjection', 'subpoena', 'subpoenaed', 'subpoenas', 'substandard', 'sue', 'sued', 'sues', 'suffer', 'suffered', 'suffering', 'suffers', 'suing', 'summoned', 'summoning', 'summons', 'summonses', 'susceptibility', 'susceptible', 'suspect', 'suspected', 'suspects', 'suspend', 'suspended', 'suspending', 'suspends', 'suspension', 'suspensions', 'suspicion', 'suspicions', 'suspicious', 'suspiciously', 'taint', 'tainted', 'tainting', 'taints', 'tampered', 'tense', 'terminate', 'terminated', 'terminates', 'terminating', 'termination', 'terminations', 'testify', 'testifying', 'threat', 'threaten', 'threatened', 'threatening', 'threatens', 'threats', 'tightening', 'tolerate', 'tolerated', 'tolerates', 'tolerating', 'toleration', 'tortuous', 'tortuously', 'tragedies', 'tragedy', 'tragic', 'tragically', 'traumatic', 'trouble', 'troubled', 'troubles', 'turbulence', 'turmoil', 'unable', 'unacceptable', 'unacceptably', 'unaccounted', 'unannounced', 'unanticipated', 'unapproved', 'unattractive', 'unauthorized', 'unavailability', 'unavailable', 'unavoidable', 'unavoidably', 'unaware', 'uncollectable', 'uncollected', 'uncollectibility', 'uncollectible', 'uncollectibles', 'uncompetitive', 'uncompleted', 'unconscionable', 'unconscionably', 'uncontrollable', 'uncontrollably', 'uncontrolled', 'uncorrected', 'uncover', 'uncovered', 'uncovering', 'uncovers', 'undeliverable', 'undelivered', 'undercapitalized', 'undercut', 'undercuts', 'undercutting', 'underestimate', 'underestimated', 'underestimates', 'underestimating', 'underestimation', 'underfunded', 'underinsured', 'undermine', 'undermined', 'undermines', 'undermining', 'underpaid', 'underpayment', 'underpayments', 'underpays', 'underperform', 'underperformance', 'underperformed', 'underperforming', 'underperforms', 'underproduced', 'underproduction', 'underreporting', 'understate', 'understated', 'understatement', 'understatements', 'understates', 'understating', 'underutilization', 'underutilized', 'undesirable', 'undesired', 'undetected', 'undetermined', 'undisclosed', 'undocumented', 'undue', 'unduly', 'uneconomic', 'uneconomical', 'uneconomically', 'unemployed', 'unemployment', 'unethical', 'unethically', 'unexcused', 'unexpected', 'unexpectedly', 'unfair', 'unfairly', 'unfavorability', 'unfavorable', 'unfavorably', 'unfavourable', 'unfeasible', 'unfit', 'unfitness', 'unforeseeable', 'unforeseen', 'unforseen', 'unfortunate', 'unfortunately', 'unfounded', 'unfriendly', 'unfulfilled', 'unfunded', 'uninsured', 'unintended', 'unintentional', 'unintentionally', 'unjust', 'unjustifiable', 'unjustifiably', 'unjustified', 'unjustly', 'unknowing', 'unknowingly', 'unlawful', 'unlawfully', 'unlicensed', 'unliquidated', 'unmarketable', 'unmerchantable', 'unmeritorious', 'unnecessarily', 'unnecessary', 'unneeded', 'unobtainable', 'unoccupied', 'unpaid', 'unperformed', 'unplanned', 'unpopular', 'unpredictability', 'unpredictable', 'unpredictably', 'unpredicted', 'unproductive', 'unprofitability', 'unprofitable', 'unqualified', 'unrealistic', 'unreasonable', 'unreasonableness', 'unreasonably', 'unreceptive', 'unrecoverable', 'unrecovered', 'unreimbursed', 'unreliable', 'unremedied', 'unreported', 'unresolved', 'unrest', 'unsafe', 'unsalable', 'unsaleable', 'unsatisfactory', 'unsatisfied', 'unsavory', 'unscheduled', 'unsellable', 'unsold', 'unsound', 'unstabilized', 'unstable', 'unsubstantiated', 'unsuccessful', 'unsuccessfully', 'unsuitability', 'unsuitable', 'unsuitably', 'unsuited', 'unsure', 'unsuspected', 'unsuspecting', 'unsustainable', 'untenable', 'untimely', 'untrusted', 'untruth', 'untruthful', 'untruthfully', 'untruthfulness', 'untruths', 'unusable', 'unwanted', 'unwarranted', 'unwelcome', 'unwilling', 'unwillingness', 'upset', 'urgency', 'urgent', 'usurious', 'usurp', 'usurped', 'usurping', 'usurps', 'usury', 'vandalism', 'verdict', 'verdicts', 'vetoed', 'victims', 'violate', 'violated', 'violates', 'violating', 'violation', 'violations', 'violative', 'violator', 'violators', 'violence', 'violent', 'violently', 'vitiate', 'vitiated', 'vitiates', 'vitiating', 'vitiation', 'voided', 'voiding', 'volatile', 'volatility', 'vulnerabilities', 'vulnerability', 'vulnerable', 'vulnerably', 'warn', 'warned', 'warning', 'warnings', 'warns', 'wasted', 'wasteful', 'wasting', 'weak', 'weaken', 'weakened', 'weakening', 'weakens', 'weaker', 'weakest', 'weakly', 'weakness', 'weaknesses', 'willfully', 'worries', 'worry', 'worrying', 'worse', 'worsen', 'worsened', 'worsening', 'worsens', 'worst', 'worthless', 'writedown', 'writedowns', 'writeoff', 'writeoffs', 'wrong', 'wrongdoing', 'wrongdoings', 'wrongful', 'wrongfully', 'wrongly']



```python
sp500
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
      <th>5-Day Return (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
      <td>-4.894889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
      <td>-0.154269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
      <td>3.27939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
      <td>2.596676</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
      <td>5.594427</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>0001041061-22-000009</td>
      <td>2022-02-23</td>
      <td>0.337203</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>0000877212-22-000026</td>
      <td>2022-02-10</td>
      <td>-9.24287</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>0001564590-22-007160</td>
      <td>2022-02-25</td>
      <td>-2.089038</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>0000109380-22-000072</td>
      <td>2022-02-25</td>
      <td>-7.888438</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>0001555280-22-000078</td>
      <td>2022-02-15</td>
      <td>-5.4106</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 12 columns</p>
</div>




```python
# Filter companies with valid filing dates
valid_companies = sp500.dropna(subset=["Filing Dates"])

lm_df = pd.read_csv("inputs/LM_MasterDictionary_1993-2021.csv")
LM_positive = set(lm_df[lm_df["Positive"] > 0]["Word"].str.lower())
LM_negative = set(lm_df[lm_df["Negative"] > 0]["Word"].str.lower())

# Initialize sentiment columns only if they don't already exist
sentiment_columns = ["BHR Positive Count", "BHR Negative Count", "LM Positive Count", "LM Negative Count"]

for col in sentiment_columns:
    if col not in valid_companies.columns:
        valid_companies[col] = 0  # Initialize column if missing

# Function to extract and clean 10-K text
def extract_10k_text(cik):
    firm_folder = f"10k_files/sec-edgar-filings/{str(cik).zfill(10)}/10-K/"
    html_files = glob.glob(firm_folder + "/*/primary-document.html")

    if not html_files:
        return None

    with open(html_files[0], "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml")

    # ✅ Remove hidden elements
    for div in soup.find_all("div", {"style": "display:none"}):
        div.decompose()

    return soup.get_text().lower()

# Count sentiment words for each company
for idx, row in tqdm(valid_companies.iterrows(), total=len(valid_companies), desc="Processing 10-Ks"):
    cik = row["CIK"]
    text = extract_10k_text(cik)

    if text:
        words = text.split()  # Tokenize text

        valid_companies.at[idx, "BHR Positive Count"] = sum(1 for word in words if word in BHR_positive)
        valid_companies.at[idx, "BHR Negative Count"] = sum(1 for word in words if word in BHR_negative)
        valid_companies.at[idx, "LM Positive Count"] = sum(1 for word in words if word in LM_positive)
        valid_companies.at[idx, "LM Negative Count"] = sum(1 for word in words if word in LM_negative)

# Merge sentiment analysis data with the dataset that contains "5-Day Return (%)"
sp500 = sp500.merge(valid_companies[["CIK"] + sentiment_columns], on="CIK", how="left", suffixes=("", "_new"))

# ✅ Prevent duplicate columns from being added
for col in sentiment_columns:
    if f"{col}_new" in sp500.columns:
        sp500[col] = sp500[f"{col}_new"]
        sp500.drop(columns=[f"{col}_new"], inplace=True)

# Save the updated dataset
sp500.to_csv(sp500_file, index=False)

# Display the first few rows
sp500.head(10)

```

    /var/folders/2s/rgjlyr356xq8fbht9p8_dttm0000gn/T/ipykernel_29792/4151790111.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      valid_companies[col] = 0  # Initialize column if missing
    /var/folders/2s/rgjlyr356xq8fbht9p8_dttm0000gn/T/ipykernel_29792/4151790111.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      valid_companies[col] = 0  # Initialize column if missing
    /var/folders/2s/rgjlyr356xq8fbht9p8_dttm0000gn/T/ipykernel_29792/4151790111.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      valid_companies[col] = 0  # Initialize column if missing
    /var/folders/2s/rgjlyr356xq8fbht9p8_dttm0000gn/T/ipykernel_29792/4151790111.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      valid_companies[col] = 0  # Initialize column if missing
    Processing 10-Ks:   0%|          | 0/501 [00:00<?, ?it/s]/var/folders/2s/rgjlyr356xq8fbht9p8_dttm0000gn/T/ipykernel_29792/4151790111.py:24: XMLParsedAsHTMLWarning: It looks like you're parsing an XML document using an HTML parser. If this really is an HTML document (maybe it's XHTML?), you can ignore or filter this warning. If it's XML, you should know that using an XML parser will be more reliable. To parse this document as XML, make sure you have the lxml package installed, and pass the keyword argument `features="xml"` into the BeautifulSoup constructor.
      soup = BeautifulSoup(file.read(), "lxml")
    Processing 10-Ks: 100%|██████████| 501/501 [02:58<00:00,  2.80it/s]





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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>Filing Dates</th>
      <th>5-Day Return (%)</th>
      <th>BHR Positive Count</th>
      <th>BHR Negative Count</th>
      <th>LM Positive Count</th>
      <th>LM Negative Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>2022-02-09</td>
      <td>-4.894889</td>
      <td>1611.0</td>
      <td>1834.0</td>
      <td>257.0</td>
      <td>1295.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>2022-02-11</td>
      <td>-0.154269</td>
      <td>704.0</td>
      <td>654.0</td>
      <td>100.0</td>
      <td>344.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>2022-02-18</td>
      <td>3.27939</td>
      <td>901.0</td>
      <td>1007.0</td>
      <td>157.0</td>
      <td>466.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>2022-02-18</td>
      <td>2.596676</td>
      <td>1045.0</td>
      <td>1124.0</td>
      <td>333.0</td>
      <td>751.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>2022-10-12</td>
      <td>5.594427</td>
      <td>1227.0</td>
      <td>1038.0</td>
      <td>376.0</td>
      <td>687.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>2022-02-25</td>
      <td>Ticker
ATVI   NaN
ATVI   NaN
dtype: float64</td>
      <td>1219.0</td>
      <td>1517.0</td>
      <td>302.0</td>
      <td>860.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>2022-02-17</td>
      <td>2.935405</td>
      <td>1124.0</td>
      <td>1064.0</td>
      <td>326.0</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>2022-01-21</td>
      <td>3.650651</td>
      <td>1269.0</td>
      <td>1143.0</td>
      <td>536.0</td>
      <td>710.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADP</td>
      <td>ADP</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Data Processing &amp; Outsourced Services</td>
      <td>Roseland, New Jersey</td>
      <td>1981-03-31</td>
      <td>8670</td>
      <td>1949</td>
      <td>0000008670-22-000038</td>
      <td>2022-08-03</td>
      <td>3.335776</td>
      <td>905.0</td>
      <td>787.0</td>
      <td>231.0</td>
      <td>468.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AAP</td>
      <td>Advance Auto Parts</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Automotive Retail</td>
      <td>Raleigh, North Carolina</td>
      <td>2015-07-09</td>
      <td>1158449</td>
      <td>1932</td>
      <td>0001158449-22-000037</td>
      <td>2022-02-15</td>
      <td>-11.590314</td>
      <td>678.0</td>
      <td>679.0</td>
      <td>140.0</td>
      <td>465.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Topics and their synonyms
# These were my second set of contex topics I chose when I intitially thought that the first topics 
# werent getting picked up from the code but that ended up not being the actual problem 
context_topics = {
    "revenue": ["revenue", "sales", "top line", "turnover"],
    "profit": ["profit", "net income", "earnings", "bottom line"],
    "debt": ["debt", "leverage", "borrowings", "interest payments", "repayment"]
}
```


```python
def get_clean_10k_text(cik):
    folder = f"10k_files/sec-edgar-filings/{str(cik).zfill(10)}/10-K/"
    files = glob.glob(folder + "*/primary-document.html")
    if not files:
        return ""
    with open(files[0], "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for div in soup.find_all("div", {"style": "display:none"}):
        div.decompose()
    return soup.get_text(separator=" ").lower()
```


```python
sp500
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>...</th>
      <th>supply_chain_pos_hits</th>
      <th>supply_chain_neg_hits</th>
      <th>litigation_pos_hits</th>
      <th>litigation_neg_hits</th>
      <th>revenue_pos_hits</th>
      <th>revenue_neg_hits</th>
      <th>profit_pos_hits</th>
      <th>profit_neg_hits</th>
      <th>debt_pos_hits</th>
      <th>debt_neg_hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>504</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>0001041061-22-000009</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>0000877212-22-000026</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>506</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>0001564590-22-007160</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>507</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>0000109380-22-000072</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>508</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>0001555280-22-000078</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>509 rows × 28 columns</p>
</div>



Testing for 10 - I finally got it to work 

Working test


```python
import pandas as pd
import signal
from tqdm import tqdm
from utils.near_regex import NEAR_finder  # Adjust if needed
tqdm.pandas(desc="🔍 Contextual Sentiment Analysis")

# Timeout setup - I got this from my 
class TimeoutException(Exception): pass
def timeout_handler(signum, frame):
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)

# Define your contextual topics 
# These were the the combination of the old context topics and new context topics 
# I figured  id keep them anyway 
context_topics = {
    "revenue": ["revenue", "sales", "top line"],
    "profit": ["profit", "earnings", "net income", "margin"],
    "debt": ["debt", "leverage", "interest payments", "credit facility"],
    "inflation": ["inflation", "price increases", "cost pressure", "rising prices"],
    "supply_chain": ["supply chain", "logistics", "distribution delays", "shipping"],
    "litigation": ["lawsuit", "litigation", "legal case", "settlement", "court", "class action"]
}

# Initialize sentiment columns
for topic in context_topics:
    sp500[f"{topic}_pos_hits"] = 0
    sp500[f"{topic}_neg_hits"] = 0

# Add clean text column
sp500["clean_10k_text"] = sp500["CIK"].apply(get_clean_10k_text)

# Define the compute_hits function with timeout

# I set this to a 90 second timeout before skipping to the next row after running this code so many times
# I figured if the row is running for almost a minute more than 90% of the time that row would get stuck
# Obv the bot helped me to write this but the coding logic is my own  
def compute_hits_with_timeout(row, timeout_seconds=90):
    signal.alarm(timeout_seconds)  # Set the timeout
    try:
        text = row["clean_10k_text"]
        if not text:
            return pd.Series({f"{topic}_pos_hits": 0 for topic in context_topics} | 
                             {f"{topic}_neg_hits": 0 for topic in context_topics})

        result = {}
        for topic, keywords in context_topics.items():
            pos_count, _ = NEAR_finder(LM_positive, keywords, text, max_words_between=10, greedy=False, partial=True)
            neg_count, _ = NEAR_finder(LM_negative, keywords, text, max_words_between=10, greedy=False, partial=True)
            result[f"{topic}_pos_hits"] = pos_count
            result[f"{topic}_neg_hits"] = neg_count
        return pd.Series(result)
    except TimeoutException:
        print(f"Timeout for CIK {row['CIK']}, skipping...")
        return pd.Series({f"{topic}_pos_hits": 0 for topic in context_topics} | 
                         {f"{topic}_neg_hits": 0 for topic in context_topics})
    finally:
        signal.alarm(0)  # Always cancel the alarm

# Apply sentiment extraction to full sp500
hit_results = sp500.progress_apply(compute_hits_with_timeout, axis=1)

# Append result columns to sp500
sp500.update(hit_results)

# Clean up
sp500.drop(columns=["clean_10k_text"], inplace=True)

# Display confirmation
print(sp500[[col for col in sp500.columns if "_hits" in col]].describe())

```


```python
valid_rows_8
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>...</th>
      <th>profit_neg_hits</th>
      <th>profit_pos_hits</th>
      <th>revenue_neg_hits</th>
      <th>revenue_pos_hits</th>
      <th>revenue_pos_hits</th>
      <th>revenue_neg_hits</th>
      <th>profit_pos_hits</th>
      <th>profit_neg_hits</th>
      <th>debt_pos_hits</th>
      <th>debt_neg_hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>37</td>
      <td>9</td>
      <td>27</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>19</td>
      <td>11</td>
      <td>20</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>41</td>
      <td>8</td>
      <td>31</td>
      <td>2</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>33</td>
      <td>11</td>
      <td>37</td>
      <td>8</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>36</td>
      <td>14</td>
      <td>25</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>reports</td>
      <td>Communication Services</td>
      <td>Interactive Home Entertainment</td>
      <td>Santa Monica, California</td>
      <td>2015-08-31</td>
      <td>718877</td>
      <td>2008</td>
      <td>0001628280-22-003992</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>41</td>
      <td>97</td>
      <td>2</td>
      <td>25</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADM</td>
      <td>ADM</td>
      <td>reports</td>
      <td>Consumer Staples</td>
      <td>Agricultural Products</td>
      <td>Chicago, Illinois</td>
      <td>1981-07-29</td>
      <td>7084</td>
      <td>1902</td>
      <td>0000007084-22-000008</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>18</td>
      <td>18</td>
      <td>26</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADBE</td>
      <td>Adobe Inc.</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Application Software</td>
      <td>San Jose, California</td>
      <td>1997-05-05</td>
      <td>796343</td>
      <td>1982</td>
      <td>0000796343-22-000032</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>41</td>
      <td>78</td>
      <td>2</td>
      <td>16</td>
      <td>4</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>




```python
sp500
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>...</th>
      <th>supply_chain_pos_hits</th>
      <th>supply_chain_neg_hits</th>
      <th>litigation_pos_hits</th>
      <th>litigation_neg_hits</th>
      <th>revenue_pos_hits</th>
      <th>revenue_neg_hits</th>
      <th>profit_pos_hits</th>
      <th>profit_neg_hits</th>
      <th>debt_pos_hits</th>
      <th>debt_neg_hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>504</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>0001041061-22-000009</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>0000877212-22-000026</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>506</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>0001564590-22-007160</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>507</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>0000109380-22-000072</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>508</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>0001555280-22-000078</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>509 rows × 28 columns</p>
</div>




```python
sp500.columns

```




    Index(['Symbol', 'Security', 'SEC filings', 'GICS Sector', 'GICS Sub-Industry',
           'Headquarters Location', 'Date first added', 'CIK', 'Founded',
           'Accession Number', 'Filing Dates', '5-Day Return (%)',
           'BHR Positive Count', 'BHR Negative Count', 'LM Positive Count',
           'LM Negative Count', 'inflation_pos_hits', 'inflation_neg_hits',
           'supply_chain_pos_hits', 'supply_chain_neg_hits', 'litigation_pos_hits',
           'litigation_neg_hits', 'revenue_pos_hits', 'revenue_neg_hits',
           'profit_pos_hits', 'profit_neg_hits', 'debt_pos_hits', 'debt_neg_hits'],
          dtype='object')




```python
#This took 160 minutes to run, thank god I had the Idea of saving this so that i didnt have to run it again if I ever came back to this
wsent = pd.read_csv('output/sp500_with_sentiment.csv')

wsent.head()
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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>Accession Number</th>
      <th>...</th>
      <th>profit_pos_hits</th>
      <th>profit_neg_hits</th>
      <th>debt_pos_hits</th>
      <th>debt_neg_hits</th>
      <th>inflation_pos_hits</th>
      <th>inflation_neg_hits</th>
      <th>supply_chain_pos_hits</th>
      <th>supply_chain_neg_hits</th>
      <th>litigation_pos_hits</th>
      <th>litigation_neg_hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0000066740-22-000010</td>
      <td>...</td>
      <td>13</td>
      <td>31</td>
      <td>9</td>
      <td>17</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>12</td>
      <td>19</td>
      <td>218</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0000091142-22-000028</td>
      <td>...</td>
      <td>11</td>
      <td>20</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0001104659-22-025141</td>
      <td>...</td>
      <td>12</td>
      <td>36</td>
      <td>2</td>
      <td>10</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0001551152-22-000007</td>
      <td>...</td>
      <td>12</td>
      <td>38</td>
      <td>10</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0001467373-22-000295</td>
      <td>...</td>
      <td>16</td>
      <td>28</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# Step 1: Filter columns that contain "hit" and are numeric
hit_cols = [col for col in wsent.columns if 'hit' in col and pd.api.types.is_numeric_dtype(wsent[col])]

# Step 2: Ensure '5-Day Return (%)' is numeric
wsent['5-Day Return (%)'] = pd.to_numeric(wsent['5-Day Return (%)'], errors='coerce')

# Step 3: Append it to the hit columns list if it's valid
if '5-Day Return (%)' in wsent.columns:
    hit_cols.append('5-Day Return (%)')

# Step 4: Drop rows with NaN in the selected columns
heatmap_data = wsent[hit_cols].dropna()

# Step 5: Plot the correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap: Hit Columns vs. 5-Day Return')
plt.show()
```


    
![png](output_30_0.png)
    



```python

```
