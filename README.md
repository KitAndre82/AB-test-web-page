# AB-test-web-page
## AB Testing Project README

### Introduction

In this project, we conducted an A/B test to evaluate the effectiveness of a new version of a product page compared to the old version. The goal was to determine if the new design could increase the conversion rate by 2% from the baseline rate of 13% to meet a target conversion rate of 15%.

### Hypotheses

We formulated our hypotheses as follows:

Null Hypothesis (Hₒ): The conversion rate of the new version is equal to the conversion rate of the old version (p = pₒ).

Alternative Hypothesis (Hₐ): The conversion rate of the new version is not equal to the conversion rate of the old version (p ≠ pₒ).

### Experiment Design

We chose a two-tailed test as we were interested in whether the new design would perform better or worse than the old design.
The power of the test (1 - β) was set to 0.8, and the significance level (α) was set to 0.05.
We calculated the required sample size using the Normal Distribution Z-test based on the expected effect size and desired power.
The dataset used for the experiment contained user information such as group assignment (control or treatment) and conversion status.

### Data Preprocessing

We removed duplicated user sessions to avoid sampling the same user multiple times.
After preprocessing, the dataset contained 286,690 entries.

### Sampling

We created samples of size 4720 for both the control and treatment groups.
Basic statistics were calculated, and visualizations were generated to understand the sample distributions.

### Hypothesis Testing

We used the Z-test for proportions to compare the conversion rates of the control and treatment groups.
The p-value obtained was 0.732, much higher than the significance level of 0.05.
Therefore, we failed to reject the null hypothesis, indicating that the new design did not perform significantly different from the old design.
Additionally, the confidence interval for the treatment group did not include the target conversion rate of 15%.

### Conclusion

Based on the results, the new design is unlikely to improve the conversion rate compared to the old design.
Further iterations or alternative approaches may be necessary to achieve the desired increase in conversion rate.

### Repository Structure

Code: File containing the code for the A/B test and analysis.

README.md: Markdown file providing an overview of the project, instructions for running the code, and interpretation of the results.

### Dependencies

Python 3.x
Libraries: pandas, numpy, scipy, statsmodels, matplotlib, seaborn

I welcome any feedback or suggestions for improvement. Thank you for your interest in our project!
