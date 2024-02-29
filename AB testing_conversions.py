#!/usr/bin/env python
# coding: utf-8

# # AB Testing

# #### In this experiment, the design team is considering implementing a new version of a product page. But before they do that, they want to be sure that the conversion rate of 13% achieved by the old version will be increased by 2% when the new version is implemented. The new version is only viable if it can achieve a conversion rate of 15%. 
# 
# #### Given we don’t know if the new design will perform better or worse (or the same) as our current design, we’ll choose a two-tailed test:
# 
# Hₒ: p = pₒ
# 
# Hₐ: p ≠ pₒ
# 
# Where p is the conversion rate of the new version and pₒ is the conversion rate of the old version.
# 
# #### The power of the test (1 - β), the probability of the test rejecting the null hypothesis when it is false, agreed upon is 0.8
# 
# #### The level of significance (α) agreed upon is 0.05
# 
# #### First we import the python libraries we are going to use then we determine the size of our sample.


# Packages imports
import numpy as np 
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import pandas as pd


# Calculating effect size based on our expected rates

effect_size = sm.proportion_effectsize(0.13, 0.15)    

required_n = sm.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  
# Calculating sample size needed
required_n = ceil(required_n)

# Rounding up to next whole number  

print(f'Sample size: {required_n}')

# #### Load dataset using Pandas


df = pd.read_csv('/Users/andrew/Downloads/ab_data.csv')

# Inspect dataset

df.head(3)

df.info()

df.isnull().sum()

# Summarise control and treatment group values in dataset
pd.crosstab(df['group'], df['landing_page'])


# #### There are 294478 rows in the DataFrame, each representing a user session, as well as 5 columns :
# 
# #### user_id - The user ID of each session
# #### timestamp - Timestamp for the session
# #### group - Which group the user was assigned to for that session {control, treatment}
# #### landing_page - Which design each user saw on that session {old_page, new_page}
# #### converted - Whether the session ended in a conversion or not (binary, 0=not converted, 1=converted)
# 
# #### We shall only be using the 'group' and 'converted' columns for our analysis.

# #### We will also determine how many users have visited the website more than once to avoid sampling a user more than once hence avoiding duplicated records

session_counts = df['user_id'].value_counts()

session_counts[session_counts > 1].count()

print(f'The number of duplicated records is',session_counts[session_counts > 1].count())


# #### Since this number is small relative to the size of our dataset, we will drop these rows

users_to_drop = session_counts[session_counts > 1]

df = df[~df['user_id'].isin(users_to_drop)]

print(f'The updated dataset now has {df.shape[0]} entries')

# #### Next, we proceed and create a sample of n = 4720 records using the DataFrame.sample() Pandas method

control_sample = df[df['group'] == 'control'].sample(n=required_n,random_state=101)

treatment_sample = df[df['group'] == 'treatment'].sample(n=required_n,random_state=101)

# Merge control and treatment sample
ab_test_data = pd.concat([control_sample,treatment_sample]).reset_index()

# Validate that new dataset has the required number of values per group
ab_test_data['group'].value_counts().reset_index()

# #### We then calculate some basic statistics and visualise our results to get an idea of what our sample looks like

# Conversion rates per group
conversion_rates = ab_test_data.groupby('group')['converted']

# Std. deviation of the proportion
std_p = lambda x: np.std(x, ddof=0) 

# Std. error of the proportion (std / sqrt(n))
se_p = lambda x: stats.sem(x, ddof=0)            

# Display conversion rate mean, standard dev, and standard error by group
conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']

conversion_rates.style.format('{:.3f}')

# Visualise conversion rate by group
plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'])

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);


# #### The last step of our analysis is testing our hypothesis. Since we have a very large sample, we can use the normal approximation for calculating our p-value (i.e. z-test).

from statsmodels.stats.proportion import proportions_ztest,proportion_confint

control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

# Number of conversions by group in sample
control_results.sum(),treatment_results.sum()

# Count of values by group in sample
control_results.count(),treatment_results.count()

n_con = control_results.count() # Control group count
n_treat = treatment_results.count() # Treatment group count
successes = [control_results.sum(), treatment_results.sum()] # Conversions by group
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs) # Z statistic and p-value
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05) # Confidence intervals

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# #### Since our p-value=0.732 is way above our α=0.05 threshold, we cannot reject the Null hypothesis Hₒ, which means that our new design did not perform significantly different (let alone better) than our old one.
# 
# #### Additionally, if we look at the confidence interval for the treatment group ([0.116, 0.135], or 11.6-13.5%) we notice that:
# 
# #### It includes our baseline value of 13% conversion rate. It does not include our target value of 15% (the 2% increase we were aiming for)
# #### What this means is that it is more likely that the true conversion rate of the new design is similar to our baseline, rather than the 15% target we had hoped for. This is further proof that our new design is not likely to be an improvement on our old design, and that unfortunately we are back to the drawing board!






