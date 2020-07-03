
## Analyze A/B Test Results

You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**

This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!

## Table of Contents
- [Introduction](#intro)
- [Part I - Probability](#probability)
- [Part II - A/B Test](#ab_test)
- [Part III - Regression](#regression)


<a id='intro'></a>
### Introduction

A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 

For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

**As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).

<a id='probability'></a>
#### Part I - Probability

To get started, let's import our libraries.


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)
```

`1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**

a. Read in the dataset and take a look at the top few rows here:


```python
df = pd.read_csv('ab_data.csv')
df.head()
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
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>851104</td>
      <td>2017-01-21 22:11:48.556739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>804228</td>
      <td>2017-01-12 08:01:45.159739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864975</td>
      <td>2017-01-21 01:52:26.210827</td>
      <td>control</td>
      <td>old_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



b. Use the cell below to find the number of rows in the dataset.


```python
df.shape
```




    (294478, 5)



c. The number of unique users in the dataset.


```python
df.nunique()
```




    user_id         290584
    timestamp       294478
    group                2
    landing_page         2
    converted            2
    dtype: int64




```python
df.groupby('group').nunique()
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
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>control</th>
      <td>146195</td>
      <td>147202</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>146284</td>
      <td>147276</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['user_id'].nunique()
```




    290584



d. The proportion of users converted.


```python
df['converted'].sum()/df['user_id'].nunique()
```




    0.12126269856564711



e. The number of times the `new_page` and `treatment` don't match.


```python
treat = df.query("group == 'treatment' and landing_page == 'old_page'").shape[0]
control= df.query("group == 'control' and landing_page == 'new_page'").shape[0]

treat + control
```




    3893



f. Do any of the rows have missing values?


```python
df.isnull().sum()
```




    user_id         0
    timestamp       0
    group           0
    landing_page    0
    converted       0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 294478 entries, 0 to 294477
    Data columns (total 5 columns):
    user_id         294478 non-null int64
    timestamp       294478 non-null object
    group           294478 non-null object
    landing_page    294478 non-null object
    converted       294478 non-null int64
    dtypes: int64(2), object(3)
    memory usage: 11.2+ MB


`2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  

a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.


```python
df2= df.query("group == 'control' and landing_page == 'old_page'")
df2= df2.append(df.query("group == 'treatment' and landing_page == 'new_page'"))
```


```python
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
```




    0



`3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

a. How many unique **user_id**s are in **df2**?


```python
df2['user_id'].nunique()
```




    290584




```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 290585 entries, 0 to 294477
    Data columns (total 5 columns):
    user_id         290585 non-null int64
    timestamp       290585 non-null object
    group           290585 non-null object
    landing_page    290585 non-null object
    converted       290585 non-null int64
    dtypes: int64(2), object(3)
    memory usage: 13.3+ MB


b. There is one **user_id** repeated in **df2**.  What is it?


```python
sum(df2['user_id'].duplicated())
```




    1



c. What is the row information for the repeat **user_id**? 


```python
df2[df2['user_id'].duplicated()]
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
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2893</th>
      <td>773192</td>
      <td>2017-01-14 02:55:59.590927</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.


```python
df2['user_id'].drop_duplicates(inplace=True)
```

`4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.

a. What is the probability of an individual converting regardless of the page they receive?


```python
df2['converted'].mean()
```




    0.11959667567149027



b. Given that an individual was in the `control` group, what is the probability they converted?


```python
df2.query("group == 'control'")['converted'].mean()
```




    0.1203863045004612



c. Given that an individual was in the `treatment` group, what is the probability they converted?


```python
df2.query("group == 'treatment'")['converted'].mean()
```




    0.11880724790277405



d. What is the probability that an individual received the new page?


```python
df2.query("landing_page == 'new_page'").shape[0]/df2.shape[0]
```




    0.5000636646764286



e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.


```python
obs_diff = df2['converted'][df2['group'] == 'treatment'].mean() - df2['converted'][df2['group'] == 'control'].mean()
```


```python
obs_diff
```




    -0.0015790565976871451



**No, because the new page has a lower conversion rate than the old page, but the difference is not significant - 0.15 percent**

**Probability of converting regardless of page -0.1196**

**Given that an individual was in the control group, the probability of converting - 0.1204**

**Given that an individual was in the treatment group, the probability of converting -0.1188**

**The probability of receiving the new page - 0.5001**

<a id='ab_test'></a>
### Part II - A/B Test

Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  

However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  

These questions are the difficult parts associated with A/B tests in general.  


`1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

$H_{0}$:$p_{old}$=$p_{new}$

$H_{1}$:$p_{new}$>$p_{old}$

or...

$H_{0}$:$p_{old}$−$p_{new}$=0 

$H_{1}$:$p_{new}$−$p_{old}$>0

`2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>

Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>

Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>

Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

a. What is the **conversion rate** for $p_{new}$ under the null? 


```python
pnull=df2['converted'].mean()
pnull
```




    0.11959667567149027



b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>


```python
pnull
```




    0.11959667567149027



c. What is $n_{new}$, the number of individuals in the treatment group?


```python
nnew=df2.query("group == 'treatment'").shape[0]
nnew
```




    145311



d. What is $n_{old}$, the number of individuals in the control group?


```python
nold=df2.query("group == 'control'").shape[0]
nold
```




    145274



e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.


```python
new_page_converted = np.random.choice(2, nnew, replace = True, p=[(1-pnull), pnull])
new_page_converted
```




    array([0, 0, 0, ..., 0, 0, 0])



f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.


```python
old_page_converted =np.random.choice(2, nold, replace = True, p=[(1-pnull), pnull])
old_page_converted
```




    array([0, 0, 0, ..., 0, 0, 1])



g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).


```python
simvalue= new_page_converted.mean() - old_page_converted.mean()
simvalue
```




    -7.1847865984478454e-05



h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.


```python
p_diffs=[]
for _ in range (1000):
    new_page_converted = np.random.choice(2, nnew, replace = True, p=[(1-pnull), pnull])
    old_page_converted =np.random.choice(2, nold, replace = True, p=[(1-pnull), pnull])
    p_diffs.append(new_page_converted.mean() - old_page_converted.mean())
```

i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.


```python
plt.hist(p_diffs)
```




    (array([  16.,   35.,  100.,  164.,  233.,  208.,  146.,   68.,   28.,    2.]),
     array([-0.00341678, -0.00270785, -0.00199892, -0.00128998, -0.00058105,
             0.00012788,  0.00083682,  0.00154575,  0.00225468,  0.00296362,
             0.00367255]),
     <a list of 10 Patch objects>)




![png](output_62_1.png)


j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?


```python
p_diffs = np.array(p_diffs)
null_value = np.random.normal (0,p_diffs.std(),p_diffs.size)
plt.hist(null_value)
plt.axvline(x=obs_diff, color ='r')
```




    <matplotlib.lines.Line2D at 0x7f6996363a58>




![png](output_64_1.png)



```python
(p_diffs > ( obs_diff ) ).mean()
```




    0.90200000000000002



k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

**In this section, we examined the basic AB test step:**

**1) Find observed difference - 0.15 percent**

**2) Find Sample distribution** 

**3) Distribution under the null**

**4) And finally p-value - 0.113**

**The results show that there is not much difference in the pages. Old pages work a little better than new ones.**
    

l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.


```python
import statsmodels.api as sm
convert_old = df2.query('group == "control" & converted == 1').shape[0]
convert_new = df2.query('group == "treatment" & converted == 1').shape[0]
n_old = df2.shape[0] - df2.query('group == "control"').shape[0]
n_new = df2.query('group == "treatment"').shape[0]
```

    /opt/conda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.


```python
z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old])
z_score, p_value
```




    (-1.2862985368041895, 0.19833889295811258)



n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

**Calculations of the z-score and p-values from test statistics suggest that the conversion rates for old and new pages do not differ from each other**



<a id='regression'></a>
### Part III - A regression approach

`1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 

a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

**Logistic regression**

b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.


```python
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
```

c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 


```python
df2['intercept']=1
lm = sm.OLS(df2['converted'],df2[['intercept','ab_page',]])
results=lm.fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>converted</td>    <th>  R-squared:         </th> <td>   0.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.720</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jul 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.190</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:56:48</td>     <th>  Log-Likelihood:    </th> <td> -85267.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>290585</td>      <th>  AIC:               </th> <td>1.705e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>290583</td>      <th>  BIC:               </th> <td>1.706e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.1204</td> <td>    0.001</td> <td>  141.407</td> <td> 0.000</td> <td>    0.119</td> <td>    0.122</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0016</td> <td>    0.001</td> <td>   -1.312</td> <td> 0.190</td> <td>   -0.004</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>125554.337</td> <th>  Durbin-Watson:     </th>  <td>   2.000</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>414318.532</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 2.345</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>        <td> 6.497</td>   <th>  Cond. No.          </th>  <td>    2.62</td> 
</tr>
</table>



d. Provide the summary of your model below, and use it as necessary to answer the following questions.


```python
np.exp(results.params)
```




    intercept    1.127932
    ab_page      0.998422
    dtype: float64




```python
1/_
```




    0.001001001001001001




```python
1/np.exp(results.params)
```




    intercept    0.886578
    ab_page      1.001580
    dtype: float64



e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

**p-value associated with ab_page -  0.190 is different from the value that I found in the second part, because they support different hypotheses**

$H_{0}$:$p_{old}$=$p_{new}$

$H_{1}$:$p_{new}$!=$p_{old}$

f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

**Yes, for a complete and accurate analysis, it would be great if there was more information about the conversion rate. The difficulty will be the addition of additional words that can affect the regression model and the complexity of filling multicollinearity.**

g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 

Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.


```python
country = pd.read_csv('./countries.csv')
dfn = country.set_index('user_id').join(df2.set_index('user_id'))
```


```python
dfn.head()
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
      <th>country</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>ab_page</th>
      <th>intercept</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630000</th>
      <td>US</td>
      <td>2017-01-19 06:26:06.548941</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>630001</th>
      <td>US</td>
      <td>2017-01-16 03:16:42.560309</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>630002</th>
      <td>US</td>
      <td>2017-01-19 19:20:56.438330</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>630003</th>
      <td>US</td>
      <td>2017-01-12 10:09:31.510471</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>630004</th>
      <td>US</td>
      <td>2017-01-18 20:23:58.824994</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfn[['CA', 'US', 'UK']] = pd.get_dummies(dfn['country'])
```

h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  

Provide the summary results, and your conclusions based on the results.


```python
dfn['intercept']=1
lm = sm.OLS(dfn['converted'],dfn[['intercept','ab_page','CA', 'UK']])
results=lm.fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>converted</td>    <th>  R-squared:         </th> <td>   0.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.640</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jul 2020</td> <th>  Prob (F-statistic):</th>  <td> 0.178</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:57:01</td>     <th>  Log-Likelihood:    </th> <td> -85266.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>290585</td>      <th>  AIC:               </th> <td>1.705e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>290581</td>      <th>  BIC:               </th> <td>1.706e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.1214</td> <td>    0.001</td> <td>   90.150</td> <td> 0.000</td> <td>    0.119</td> <td>    0.124</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0016</td> <td>    0.001</td> <td>   -1.308</td> <td> 0.191</td> <td>   -0.004</td> <td>    0.001</td>
</tr>
<tr>
  <th>CA</th>        <td>   -0.0053</td> <td>    0.003</td> <td>   -1.784</td> <td> 0.074</td> <td>   -0.011</td> <td>    0.001</td>
</tr>
<tr>
  <th>UK</th>        <td>   -0.0010</td> <td>    0.001</td> <td>   -0.744</td> <td> 0.457</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>125552.050</td> <th>  Durbin-Watson:     </th>  <td>   2.000</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>414302.956</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 2.345</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>        <td> 6.497</td>   <th>  Cond. No.          </th>  <td>    6.92</td> 
</tr>
</table>




```python
np.exp(results.params)
```




    intercept    1.129053
    ab_page      0.998427
    CA           0.994746
    UK           0.998956
    dtype: float64




```python
1/np.exp(results.params)
```




    intercept    0.885698
    ab_page      1.001575
    CA           1.005282
    UK           1.001045
    dtype: float64




```python
log= sm.Logit(dfn['converted'], dfn[['intercept', 'ab_page', 'CA', 'UK']])
results = log.fit()
results.summary2()
```

    Optimization terminated successfully.
             Current function value: 0.366112
             Iterations 6





<table class="simpletable">
<tr>
        <td>Model:</td>              <td>Logit</td>       <td>No. Iterations:</td>    <td>6.0000</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>     <td>converted</td>    <td>Pseudo R-squared:</td>    <td>0.000</td>   
</tr>
<tr>
         <td>Date:</td>        <td>2020-07-03 10:57</td>       <td>AIC:</td>        <td>212781.3782</td>
</tr>
<tr>
   <td>No. Observations:</td>       <td>290585</td>            <td>BIC:</td>        <td>212823.6968</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>         <td>Log-Likelihood:</td>  <td>-1.0639e+05</td>
</tr>
<tr>
     <td>Df Residuals:</td>         <td>290581</td>          <td>LL-Null:</td>      <td>-1.0639e+05</td>
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>           <td>Scale:</td>         <td>1.0000</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>       <th>Coef.</th>  <th>Std.Err.</th>     <th>z</th>      <th>P>|z|</th> <th>[0.025</th>  <th>0.975]</th> 
</tr>
<tr>
  <th>intercept</th> <td>-1.9794</td>  <td>0.0127</td>  <td>-155.4143</td> <td>0.0000</td> <td>-2.0043</td> <td>-1.9544</td>
</tr>
<tr>
  <th>ab_page</th>   <td>-0.0150</td>  <td>0.0114</td>   <td>-1.3076</td>  <td>0.1910</td> <td>-0.0374</td> <td>0.0075</td> 
</tr>
<tr>
  <th>CA</th>        <td>-0.0506</td>  <td>0.0284</td>   <td>-1.7835</td>  <td>0.0745</td> <td>-0.1063</td> <td>0.0050</td> 
</tr>
<tr>
  <th>UK</th>        <td>-0.0099</td>  <td>0.0133</td>   <td>-0.7437</td>  <td>0.4570</td> <td>-0.0359</td> <td>0.0162</td> 
</tr>
</table>




```python
dfn['UK_ab_page'] = dfn['UK']*dfn['ab_page'] 
dfn['CA_ab_page'] = dfn['CA']*dfn['ab_page'] 

logit3 = sm.Logit(dfn['converted'], dfn[['intercept', 'ab_page', 'UK', 'CA', 'UK_ab_page', 'CA_ab_page']]) 
results = logit3.fit() 
results.summary2()
```

    Optimization terminated successfully.
             Current function value: 0.366108
             Iterations 6





<table class="simpletable">
<tr>
        <td>Model:</td>              <td>Logit</td>       <td>No. Iterations:</td>    <td>6.0000</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>     <td>converted</td>    <td>Pseudo R-squared:</td>    <td>0.000</td>   
</tr>
<tr>
         <td>Date:</td>        <td>2020-07-03 10:58</td>       <td>AIC:</td>        <td>212782.9124</td>
</tr>
<tr>
   <td>No. Observations:</td>       <td>290585</td>            <td>BIC:</td>        <td>212846.3903</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>         <td>Log-Likelihood:</td>  <td>-1.0639e+05</td>
</tr>
<tr>
     <td>Df Residuals:</td>         <td>290579</td>          <td>LL-Null:</td>      <td>-1.0639e+05</td>
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>           <td>Scale:</td>         <td>1.0000</td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>       <th>Coef.</th>  <th>Std.Err.</th>     <th>z</th>      <th>P>|z|</th> <th>[0.025</th>  <th>0.975]</th> 
</tr>
<tr>
  <th>intercept</th>  <td>-1.9922</td>  <td>0.0161</td>  <td>-123.4571</td> <td>0.0000</td> <td>-2.0238</td> <td>-1.9606</td>
</tr>
<tr>
  <th>ab_page</th>    <td>0.0108</td>   <td>0.0228</td>   <td>0.4749</td>   <td>0.6349</td> <td>-0.0339</td> <td>0.0555</td> 
</tr>
<tr>
  <th>UK</th>         <td>0.0057</td>   <td>0.0188</td>   <td>0.3057</td>   <td>0.7598</td> <td>-0.0311</td> <td>0.0426</td> 
</tr>
<tr>
  <th>CA</th>         <td>-0.0118</td>  <td>0.0398</td>   <td>-0.2957</td>  <td>0.7674</td> <td>-0.0899</td> <td>0.0663</td> 
</tr>
<tr>
  <th>UK_ab_page</th> <td>-0.0314</td>  <td>0.0266</td>   <td>-1.1811</td>  <td>0.2375</td> <td>-0.0835</td> <td>0.0207</td> 
</tr>
<tr>
  <th>CA_ab_page</th> <td>-0.0783</td>  <td>0.0568</td>   <td>-1.3783</td>  <td>0.1681</td> <td>-0.1896</td> <td>0.0330</td> 
</tr>
</table>




```python
np.exp(results.params)

```




    intercept     0.136392
    ab_page       1.010893
    UK            1.005761
    CA            0.988285
    UK_ab_page    0.969079
    CA_ab_page    0.924703
    dtype: float64




```python
1/np.exp(results.params)
```




    intercept     7.331806
    ab_page       0.989224
    UK            0.994272
    CA            1.011854
    UK_ab_page    1.031907
    CA_ab_page    1.081428
    dtype: float64



**This regression summary shows that the country of origin and page type based on their p values do not provide a statistical basis for rejecting the null hypothesis**

<a id='conclusions'></a>
## Conclusion

> From this analysis, we can see that we do not need to switch to a new page, since the changes on the page are practically insignificant.Ultimately, we do not have enough evidence to reject the null hypothesis based on any of our A / B tests. As a result, there is no reason to switch to a new page when the old works just as well.


## Directions to Submit

> Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).

> Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.

> Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])
```
