---
layout : post
title : "Web Scraping using BeautifulSoup"
date : 2019-08-12
permalink: /webscarping-using-beautifulsoup/
---
![webscraping-header](/images/web-header.png)



## Web Scraping (also termed Screen Scraping, Web Data Extraction, Web Harvesting etc.) is a technique employed to extract large amounts of data from websites whereby the data is extracted and saved to a local file in your computer or to a database in table (spreadsheet) format.


```python
# Loading libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
```

# Here is the link to Trump's lies artcile from NY-Times: [LINK](https://www.nytimes.com/interactive/2017/06/23/opinion/trumps-lies.html)

# Here is the structure of the first lie:
 <span class="short-desc"><strong>Jan. 21&nbsp;</strong>“I wasn't a fan of Iraq. I didn't want to go into Iraq.” <span class="short-truth"><a href="https://www.buzzfeed.com/andrewkaczynski/in-2002-donald-trump-said-he-supported-invading-iraq-on-the" target="_blank">(He was for an invasion before he was against it.)</a></span></span>&nbsp;&nbsp;

```python
link = "https://www.nytimes.com/interactive/2017/06/23/opinion/trumps-lies.html"
response = requests.get(link)
```

# Collecting all the records


```python
soup = BeautifulSoup(response.text, "html.parser")
```


```python
results = soup.find_all(name = "span", attrs = {"class" : "short-desc"})
```


```python
len(results)
```




    180




```python
results[0]
```




    <span class="short-desc"><strong>Jan. 21 </strong>“I wasn't a fan of Iraq. I didn't want to go into Iraq.” <span class="short-truth"><a href="https://www.buzzfeed.com/andrewkaczynski/in-2002-donald-trump-said-he-supported-invading-iraq-on-the" target="_blank">(He was for an invasion before he was against it.)</a></span></span>



# Parse the first lie into 4 structured columns:
* Date
* Lie
* Description
* Link

# The first result to see the pattern of the html


```python
r = results[0]
```


```python
r
```




    <span class="short-desc"><strong>Jan. 21 </strong>“I wasn't a fan of Iraq. I didn't want to go into Iraq.” <span class="short-truth"><a href="https://www.buzzfeed.com/andrewkaczynski/in-2002-donald-trump-said-he-supported-invading-iraq-on-the" target="_blank">(He was for an invasion before he was against it.)</a></span></span>



# Parsing date by looking at the strong tag


```python
date = r.find("strong").text[:-1] + ", 2017"
```


```python
date
```




    'Jan. 21, 2017'



# Parsing lie by parsing the content out


```python
lie = r.contents[1][1:-2]
```


```python
lie
```




    "I wasn't a fan of Iraq. I didn't want to go into Iraq."



# Parsing explanation by looking at the a tag


```python
explanation = r.find("a").text[1:-1]
```


```python
explanation
```




    'He was for an invasion before he was against it.'



# Parsing link out by looking at the href key of the a tag


```python
link = r.find("a")["href"]
```


```python
link
```




    'https://www.buzzfeed.com/andrewkaczynski/in-2002-donald-trump-said-he-supported-invading-iraq-on-the'



# Now, iterating over the whole records and put it into a dataframe


```python
rows = []
for r in results:
    date = r.find("strong").text[:-1] + ", 2017"
    lie = r.contents[1][1:-2]
    explanation = r.find("a").text[1:-1]
    link = r.find("a")["href"]
    rows.append( (date, lie, explanation, link) )
```


```python
df = pd.DataFrame(data = rows, columns=["Date", "Lie", "Explanation", "Link"])
df["Date"] = pd.to_datetime(df["Date"])
```


```python
df.head(20)
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
      <th>Date</th>
      <th>Lie</th>
      <th>Explanation</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-01-21</td>
      <td>I wasn't a fan of Iraq. I didn't want to go in...</td>
      <td>He was for an invasion before he was against it.</td>
      <td>https://www.buzzfeed.com/andrewkaczynski/in-20...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-01-21</td>
      <td>A reporter for Time magazine — and I have been...</td>
      <td>Trump was on the cover 11 times and Nixon appe...</td>
      <td>http://nation.time.com/2013/11/06/10-things-yo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-23</td>
      <td>Between 3 million and 5 million illegal votes ...</td>
      <td>There's no evidence of illegal voting.</td>
      <td>https://www.nytimes.com/2017/01/23/us/politics...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-01-25</td>
      <td>Now, the audience was the biggest ever. But th...</td>
      <td>Official aerial photos show Obama's 2009 inaug...</td>
      <td>https://www.nytimes.com/2017/01/21/us/politics...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-01-25</td>
      <td>Take a look at the Pew reports (which show vot...</td>
      <td>The report never mentioned voter fraud.</td>
      <td>https://www.nytimes.com/2017/01/24/us/politics...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-01-25</td>
      <td>You had millions of people that now aren't ins...</td>
      <td>The real number is less than 1 million, accord...</td>
      <td>https://www.nytimes.com/2017/03/13/us/politics...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-01-25</td>
      <td>So, look, when President Obama was there two w...</td>
      <td>There were no gun homicide victims in Chicago ...</td>
      <td>https://www.dnainfo.com/chicago/2017-chicago-m...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-01-26</td>
      <td>We've taken in tens of thousands of people. We...</td>
      <td>Vetting lasts up to two years.</td>
      <td>https://www.nytimes.com/interactive/2017/01/29...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-01-26</td>
      <td>I cut off hundreds of millions of dollars off ...</td>
      <td>Most of the cuts were already planned.</td>
      <td>https://www.washingtonpost.com/news/fact-check...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-01-28</td>
      <td>The coverage about me in the @nytimes and the ...</td>
      <td>It never apologized.</td>
      <td>https://www.nytimes.com/2016/11/13/us/election...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-01-29</td>
      <td>The Cuban-Americans, I got 84 percent of that ...</td>
      <td>There is no support for this.</td>
      <td>http://www.pewresearch.org/fact-tank/2016/11/1...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-01-30</td>
      <td>Only 109 people out of 325,000 were detained a...</td>
      <td>At least 746 people were detained and processe...</td>
      <td>http://markets.on.nytimes.com/research/stocks/...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-02-03</td>
      <td>Professional anarchists, thugs and paid protes...</td>
      <td>There is no evidence of paid protesters.</td>
      <td>https://www.nytimes.com/2017/01/28/nyregion/jf...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-02-04</td>
      <td>After being forced to apologize for its bad an...</td>
      <td>It never apologized.</td>
      <td>https://www.nytimes.com/2016/11/13/us/election...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2017-02-05</td>
      <td>We had 109 people out of hundreds of thousands...</td>
      <td>About 60,000 people were affected.</td>
      <td>http://www.politifact.com/truth-o-meter/statem...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2017-02-06</td>
      <td>I have already saved more than $700 million wh...</td>
      <td>Much of the price drop was projected before Tr...</td>
      <td>https://www.washingtonpost.com/news/fact-check...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2017-02-06</td>
      <td>It's gotten to a point where it is not even be...</td>
      <td>Terrorism has been reported on, often in detail.</td>
      <td>https://www.nytimes.com/2017/02/07/us/politics...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2017-02-06</td>
      <td>The failing @nytimes was forced to apologize t...</td>
      <td>It didn't apologize.</td>
      <td>https://www.nytimes.com/2016/11/13/us/election...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2017-02-06</td>
      <td>And the previous administration allowed it to ...</td>
      <td>The group’s origins date to 2004.</td>
      <td>https://www.nytimes.com/2015/11/19/world/middl...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2017-02-07</td>
      <td>And yet the murder rate in our country is the ...</td>
      <td>It was higher in the 1980s and '90s.</td>
      <td>http://www.politifact.com/truth-o-meter/statem...</td>
    </tr>
  </tbody>
</table>
</div>


