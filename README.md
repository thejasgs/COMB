# Feature Selex Module
**Feature Selex** is a Python 3 module that contains 5 feature selection methods for supervised classification.

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
X = featureSelex.nameOfMethod(X,Y)
```

*Voila! **X** now only contains the columns of the selected features.*
#### OR

```python 
from featureSelex import nameOfMethod
X = nameOfMethod(X,Y)
```

You may use these methods.

- `from featureSelex import COMB`
- `from featureSelex import ARMB`
- `from featureSelex import VMMB`
- `from featureSelex import FIMB`
- `from featureSelex import BF`

## Permissions
If you using a for research, please cite our research paper.
If you are using this code for any other project, please contact Thejas Gubbi Sadashiva <tgs001@fiu.edu> or Daniel Jimenez <djime072@fiu.edu>.