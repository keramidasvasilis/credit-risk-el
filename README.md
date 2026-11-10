# Credit Risk: Expected Loss = PD x LGD x EAD

Personal, non-commercial learning project using the Freddie Mac Single-Family Loan-Level Dataset.
The data is not included in this repository. Download it from Freddie Mac (registration required) and place it in `data/`.

## Data

Freddie Mac Single-Family Loan-Level Dataset, sample files for the vintages 2005–2009 (50,000 loans per vintage, 250,000 in total). The performance history is available until March 2026. After removing the loans closed with codes `16` and `96`, which do not describe a clear credit outcome, 244,967 loans remain.

**Default.** `default = 1` if the loan closed with zero balance code `02`, `03`, `09` or `15` (third-party sale, short sale or charge-off, REO disposition, whole loan sale), and `default = 0` otherwise. Loans that have not closed yet (7,001) are counted as non-defaults.

The default rate is 5.7% (14,084 defaults). By vintage it rises from 5.8% (2005) to 9.3% (2007) and falls to 4.4% (2008) and 1.1% (2009), consistent with the 2007–2009 housing crisis.

## PD model

The PD model estimates the probability of default $\text{PD} = P(\text{default} = 1)$ of a loan from its characteristics at origination (notebook `02_pd_model.ipynb`).

**Split.** The loans are split randomly into a training set (195,973 loans, 80%) and a test set (48,994 loans, 20%), stratified by `default`. Everything learned from the data (bins, weights of evidence, coefficients) uses the training set only.

**Inputs.** For each category $i$ of an input, the *weight of evidence* is
$$\text{WoE}(i) = \ln \frac{n_{\text{good}}(i) / N_{\text{good}}}{n_{\text{bad}}(i) / N_{\text{bad}}},$$
and the *information value* of the input is
$$\text{IV} = \sum_i \left( \frac{n_{\text{good}}(i)}{N_{\text{good}}} - \frac{n_{\text{bad}}(i)}{N_{\text{bad}}} \right) \text{WoE}(i),$$
where good and bad mean non-default and default. Continuous inputs are first split into fine bins. Neighbouring categories are then merged (coarse classing) when they are too small or have similar WoE. Of the 22 candidate inputs, 11 are kept. Eight are dropped because $\text{IV} < 0.02$ (merging categories never increases the IV, so no grouping would change this), and three are excluded without computing their IV.

**Model.** The categories of the 11 inputs give 54 dummy variables. In each input we omit the category with the lowest WoE, the *reference category*, so the model has 43 inputs. The model is the logistic regression
$$\text{PD}(x) = \frac{1}{1 + e^{-(\beta_0 + \beta^\top x)}},$$
fitted by maximum likelihood with `statsmodels`. For each coefficient we test $H_0 : \beta_j = 0$ with the Wald statistic $z = \hat\beta_j / \text{SE}(\hat\beta_j)$, where the standard errors come from the Fisher information matrix. We apply the 5% level per original input: an input is kept if at least one of its dummies is significant. All 11 inputs passed this criterion.

### Results

On the test set the model has AUC 0.838, Gini coefficient 0.676 and KS statistic 0.521. The AUC on the training set is 0.840, so there is no sign of overfitting.

The model is also well calibrated: when the test loans are sorted by predicted PD and split into ten groups of equal size, the mean predicted PD is close to the observed default rate in every group (from 0.10% vs 0.08% in the lowest group to 26.1% vs 25.0% in the highest).

**Scorecard.** The model is converted into a scorecard with scores between 300 and 850 (a higher score means a lower probability of default). The score of a loan is the sum of the points of its categories plus the points of the intercept. The points of every category are in `outputs/scorecard.csv`. On the test set the scores range from 308 to 841, with mean 536.