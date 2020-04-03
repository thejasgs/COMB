# COMB: A Hybrid Method for Cross-validated Feature Selection
**Feature Selex** is a Python 3 module that contains the official implementation of the COMB feature selection method for classification.

The methods in this module take two dataframes *X* and *Y*. *X* is contains column of features which may only contain *int64* datatypes. *Y* contains one column of labels. 

If you are feeling adventurous, feel free to take the package and tweak the parameters!

## Dependancies
- numpy
- pandas
- scikitlearn

You may install them using:

```python3 -m pip install --user numpy sklearn pandas```

## Usage
```python
import featureSelex
X = featureSelex.COMB(X,Y)
```

#### OR

```python 
from featureSelex import COMB
X = COMB(X,Y)
```
*Voila! **X** now only contains the columns of the selected features.*

## Permissions
If you using this for research, please cite our research paper.
```ACM REFERENCE FORMAT
Thejas G. S., Daniel Jimenez, S.S. Iyengar, Jerry Miller, N.R. Sunitha, and Prajwal Badrinath. 2020. COMB: A Hybrid Method for Cross-validated Feature Selection. In 2020 ACM Southeast Conference (ACMSE 2020), April 2–4, 2020, Tampa, FL, USA. ACM, New York, NY, USA, 8 pages. https: //doi.org/10.1145/3374135.3385285
```
If you are using this code for any other project, please contact Thejas Gubbi Sadashiva <tgs001@fiu.edu> or Daniel Jimenez <djime072@fiu.edu>.
