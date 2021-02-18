---
layout : post
title : "Updates on Regression Error Characteristics Curves"
date : 2021-01-10
permalink: /updated-rec/
---
![updated-rec-header](/images/updated-rec-header.png)


# Updates on Regression Error Characteristics Curve

## Introduction

###  In machine learning, a Receiver Operating Characteristic (ROC) curve visualizes the performance of a classifier applied on a binary class problem across all possible trade-offs between the false positive rates and the true positive rates. A graph consists of multiple ROC curves of different models characterizes the performance of the models on a binary problem and makes the comparison process of the models easier by visualization. Additionally, the area under the ROC curve (AUC) represents the expected performance of the classification model as a single scalar value. 

### Although ROC curves are limited to classification problems, Regression Error Characteristic (REC) curves can be used to visualize the performance of the regressor models. REC illustrates the absolute deviation tolerance versus the fraction of the exemplars predicted correctly within the tolerance interval. The resulting curve estimates the cumulative distribution function of the error. The area over the REC curve (AOC), which can be calculated via the area under the REC curve (AOC = 1 - AUC) is a biased estimate of the expected error.

### Furthermore, the coefficient of determination $R^2$ can be also calculated with respect to the AOC. Likewise the ROC curve, the shape of the REC curve can also be used as a guidance for the users to reveal additional information about the data modeling. The REC curve was implemented in Python.

### Previously, I have implemented a simple version of REC [LINK](https://amirhessam88.github.io/regression-error-characteristics-curve/). Here, is an overview of the objected oriented implementation of REC and its updates.

### You can clone the project from my GitHub Repo [LINK](https://github.com/amirhessam88/Regression-Error-Characteristic-Curve) or run the following cell.


```python
# cloning the project
!git clone https://github.com/amirhessam88/Regression-Error-Characteristic-Curve.git

# change directory to the project
%cd Regression-Error-Characteristic-Curve

# installing requirements
!pip install -r requirements.txt
```

### Or you can follow the class below which is the content of `src/rec.py`


```python
# Loading Packages
import numpy as np
import scipy as scp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
```


```python
# class definition
class RegressionErrorCharacteristic:
    """Regression Error Characteristics (REC).
    This is wrapper to implement the REC algorithm.
    REC is implemented based on the following paper:
    Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
    In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
    https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf
    Parameters
    ----------
    y_true: numpy.array[int] or list[float]
        List of ground truth target (response) values
    y_pred: numpy.array[float] or list[float]
        List of predicted target values list[float]
    Attributes
    ----------
    auc_rec: float value between 0. and 1
        Area under REC curve.
    deviation:  numpy.array[float] or list[float]
        List of deviations to plot REC curve.
    accuracy:  numpy.array[float] or list[float]
        Calculated accuracy at each deviation to plot REC curve.
    plotting_dict: dict()
        Plotting object as a dictionary consists of all
        calculated metrics which was used to plot curves
    plot_rec(): Func
        Function to plot the REC curve.
    """

    def __init__(self, y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        else:
            self.y_true = y_true
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = y_pred
        self.deviation, self.accuracy, self.auc_rec = self._rec_curve()

    def _rec_curve(self):
        """
        Function to calculate the rec curve elements: deviation, accuracy, auc.
        Simpson method is used as the integral method to calculate the area under
        regression error characteristics (REC).
        REC is implemented based on the following paper:
        Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
        In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
        https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf
        """
        begin = 0.0
        end = 1.0
        interval = 0.01
        accuracy = []
        deviation = np.arange(begin, end, interval)

        # main loop to calculate norm and compare with each deviation
        for i in range(len(deviation)):
            count = 0.0
            for j in range(len(self.y_true)):
                calc_norm = np.linalg.norm(self.y_true[j] - self.y_pred[j]) / np.sqrt(
                    np.linalg.norm(self.y_true[j]) ** 2
                    + np.linalg.norm(self.y_pred[j]) ** 2
                )
                if calc_norm < deviation[i]:
                    count += 1
            accuracy.append(count / len(self.y_true))

        auc_rec = scp.integrate.simps(accuracy, deviation) / end

        return deviation, accuracy, auc_rec

    def plot_rec(self, figsize=None, color=None, linestyle=None, fontsize=None):
        """Function to plot REC curve.
        Parameters
        ----------
        figsize: tuple, optional, (default=(8, 5))
            Figure size
        color: str, optional, (default="navy")
            Color of the curve.
        linestyle: str, optional, (default="--")
        fontsize: int or float, optional, (default=15)
            Fontsize for xlabel and ylabel, and ticks parameters
        """
        sns.set_style("ticks")
        mpl.rcParams["axes.linewidth"] = 3
        mpl.rcParams["lines.linewidth"] = 3

        # initializing figsize
        if figsize is None:
            figsize = (8, 5)
        elif isinstance(figsize, list) or isinstance(figsize, tuple):
            figsize = figsize
        else:
            raise TypeError("Only tuple and list types are allowed for figsize.")

        # initializing color
        if color is None:
            color = "navy"
        elif isinstance(color, str):
            color = color
        else:
            raise TypeError("Only str type is allowed for color.")

        # initializing linestyle
        if linestyle is None:
            linestyle = "--"
        elif isinstance(linestyle, str):
            linestyle = linestyle
        else:
            raise TypeError("Only str type is allowed for linestyle.")

        # initializing fontsize
        if fontsize is None:
            fontsize = 15
        elif isinstance(fontsize, float) or isinstance(fontsize, int):
            fontsize = fontsize
        else:
            raise TypeError("Only int and float types are allowed for fontsize.")

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            self.deviation,
            self.accuracy,
            color=color,
            linestyle=linestyle,
            label=f"AUC = {self.auc_rec:.3f}",
        )

        ax.set(
            xlim=[-0.01, 1.01],
            ylim=[-0.01, 1.01],
        )
        ax.set_xlabel("Deviation", fontsize=fontsize)
        ax.set_ylabel("Accuracy", fontsize=fontsize)
        ax.set_title("REC Curve", fontsize=fontsize)

        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(prop={"size": fontsize}, loc=4, framealpha=0.0)

        plt.show()
```

### Load Data and Train a Regression Model


```python
# load data, train model
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_predict

# loading a sample regression dataset
boston = datasets.load_boston()
X = boston.data
y_true = boston.target

# defining a simple linear regression model
model = linear_model.LinearRegression()

# predicting using 4-folds cross-validation
y_pred = cross_val_predict(model, X, y_true, cv=4)
```

### Let's try to plot REC curve


```python
# initiate the RegressionErrorCharacteristic with y_pred and y_true
myREC = RegressionErrorCharacteristic(y_true, y_pred)

# now we can use the class function plot_rec()
myREC.plot_rec()
```


![png](/notebooks/updated-regression-error-characteristics-curve_files/output_11_0.png)


### Currently, REC is also implemented in my Machine Learning Python Library, [SlickML](https://github.com/slickml/slick-ml) which can be easily installed via pip.


```python
!pip install slickml
```


```python
# plot regression metrics
from slickml.metrics import RegressionMetrics
reg_metrics = RegressionMetrics(y_true, y_pred)
reg_metrics.plot()
```


<style>.container { width:95% !important; }</style>



<style  type="text/css" >
    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87aca th {
          font-size: 12px;
          text-align: left;
          font-weight: bold;
    }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87aca td {
          font-size: 12px;
          text-align: center;
    }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col0 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col1 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col2 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col3 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col5 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col6 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col7 {
            background-color:  #e5e5ff;
            color:  #000000;
        }    #T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col8 {
            background-color:  #e5e5ff;
            color:  #000000;
        }</style><table id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87aca" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >R2 Score</th>        <th class="col_heading level0 col1" >Explained Variance Score</th>        <th class="col_heading level0 col2" >Mean Absolute Error</th>        <th class="col_heading level0 col3" >Mean Squared Error</th>        <th class="col_heading level0 col4" >Mean Squared Log Error</th>        <th class="col_heading level0 col5" >Mean Absolute Percentage Error</th>        <th class="col_heading level0 col6" >REC AUC</th>        <th class="col_heading level0 col7" >Coeff. of Variation</th>        <th class="col_heading level0 col8" >Mean of Variation</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acalevel0_row0" class="row_heading level0 row0" >Metrics</th>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col0" class="data row0 col0" >0.499000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col1" class="data row0 col1" >0.509000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col2" class="data row0 col2" >4.646000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col3" class="data row0 col3" >42.299000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col4" class="data row0 col4" >None</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col5" class="data row0 col5" >0.254000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col6" class="data row0 col6" >0.835000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col7" class="data row0 col7" >0.342000</td>
                        <td id="T_dbbde7d4_71aa_11eb_af6d_ff828eb87acarow0_col8" class="data row0 col8" >1.110000</td>
            </tr>
    </tbody></table>



![png](/notebooks/updated-regression-error-characteristics-curve_files/output_14_2.png)


### Extra notes about the REC class


```python
# list of attributes of the REC object
dir(myREC)
```




    ['__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_rec_curve',
     'accuracy',
     'auc_rec',
     'deviation',
     'plot_rec',
     'y_pred',
     'y_true']




```python
# dict of attributes values of the REC object
myREC.__dict__
```




    {'y_true': array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15. ,
            18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6,
            15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21. , 12.7, 14.5, 13.2,
            13.1, 13.5, 18.9, 20. , 21. , 24.7, 30.8, 34.9, 26.6, 25.3, 24.7,
            21.2, 19.3, 20. , 16.6, 14.4, 19.4, 19.7, 20.5, 25. , 23.4, 18.9,
            35.4, 24.7, 31.6, 23.3, 19.6, 18.7, 16. , 22.2, 25. , 33. , 23.5,
            19.4, 22. , 17.4, 20.9, 24.2, 21.7, 22.8, 23.4, 24.1, 21.4, 20. ,
            20.8, 21.2, 20.3, 28. , 23.9, 24.8, 22.9, 23.9, 26.6, 22.5, 22.2,
            23.6, 28.7, 22.6, 22. , 22.9, 25. , 20.6, 28.4, 21.4, 38.7, 43.8,
            33.2, 27.5, 26.5, 18.6, 19.3, 20.1, 19.5, 19.5, 20.4, 19.8, 19.4,
            21.7, 22.8, 18.8, 18.7, 18.5, 18.3, 21.2, 19.2, 20.4, 19.3, 22. ,
            20.3, 20.5, 17.3, 18.8, 21.4, 15.7, 16.2, 18. , 14.3, 19.2, 19.6,
            23. , 18.4, 15.6, 18.1, 17.4, 17.1, 13.3, 17.8, 14. , 14.4, 13.4,
            15.6, 11.8, 13.8, 15.6, 14.6, 17.8, 15.4, 21.5, 19.6, 15.3, 19.4,
            17. , 15.6, 13.1, 41.3, 24.3, 23.3, 27. , 50. , 50. , 50. , 22.7,
            25. , 50. , 23.8, 23.8, 22.3, 17.4, 19.1, 23.1, 23.6, 22.6, 29.4,
            23.2, 24.6, 29.9, 37.2, 39.8, 36.2, 37.9, 32.5, 26.4, 29.6, 50. ,
            32. , 29.8, 34.9, 37. , 30.5, 36.4, 31.1, 29.1, 50. , 33.3, 30.3,
            34.6, 34.9, 32.9, 24.1, 42.3, 48.5, 50. , 22.6, 24.4, 22.5, 24.4,
            20. , 21.7, 19.3, 22.4, 28.1, 23.7, 25. , 23.3, 28.7, 21.5, 23. ,
            26.7, 21.7, 27.5, 30.1, 44.8, 50. , 37.6, 31.6, 46.7, 31.5, 24.3,
            31.7, 41.7, 48.3, 29. , 24. , 25.1, 31.5, 23.7, 23.3, 22. , 20.1,
            22.2, 23.7, 17.6, 18.5, 24.3, 20.5, 24.5, 26.2, 24.4, 24.8, 29.6,
            42.8, 21.9, 20.9, 44. , 50. , 36. , 30.1, 33.8, 43.1, 48.8, 31. ,
            36.5, 22.8, 30.7, 50. , 43.5, 20.7, 21.1, 25.2, 24.4, 35.2, 32.4,
            32. , 33.2, 33.1, 29.1, 35.1, 45.4, 35.4, 46. , 50. , 32.2, 22. ,
            20.1, 23.2, 22.3, 24.8, 28.5, 37.3, 27.9, 23.9, 21.7, 28.6, 27.1,
            20.3, 22.5, 29. , 24.8, 22. , 26.4, 33.1, 36.1, 28.4, 33.4, 28.2,
            22.8, 20.3, 16.1, 22.1, 19.4, 21.6, 23.8, 16.2, 17.8, 19.8, 23.1,
            21. , 23.8, 23.1, 20.4, 18.5, 25. , 24.6, 23. , 22.2, 19.3, 22.6,
            19.8, 17.1, 19.4, 22.2, 20.7, 21.1, 19.5, 18.5, 20.6, 19. , 18.7,
            32.7, 16.5, 23.9, 31.2, 17.5, 17.2, 23.1, 24.5, 26.6, 22.9, 24.1,
            18.6, 30.1, 18.2, 20.6, 17.8, 21.7, 22.7, 22.6, 25. , 19.9, 20.8,
            16.8, 21.9, 27.5, 21.9, 23.1, 50. , 50. , 50. , 50. , 50. , 13.8,
            13.8, 15. , 13.9, 13.3, 13.1, 10.2, 10.4, 10.9, 11.3, 12.3,  8.8,
             7.2, 10.5,  7.4, 10.2, 11.5, 15.1, 23.2,  9.7, 13.8, 12.7, 13.1,
            12.5,  8.5,  5. ,  6.3,  5.6,  7.2, 12.1,  8.3,  8.5,  5. , 11.9,
            27.9, 17.2, 27.5, 15. , 17.2, 17.9, 16.3,  7. ,  7.2,  7.5, 10.4,
             8.8,  8.4, 16.7, 14.2, 20.8, 13.4, 11.7,  8.3, 10.2, 10.9, 11. ,
             9.5, 14.5, 14.1, 16.1, 14.3, 11.7, 13.4,  9.6,  8.7,  8.4, 12.8,
            10.5, 17.1, 18.4, 15.4, 10.8, 11.8, 14.9, 12.6, 14.1, 13. , 13.4,
            15.2, 16.1, 17.8, 14.9, 14.1, 12.7, 13.5, 14.9, 20. , 16.4, 17.7,
            19.5, 20.2, 21.4, 19.9, 19. , 19.1, 19.1, 20.1, 19.9, 19.6, 23.2,
            29.8, 13.8, 13.3, 16.7, 12. , 14.6, 21.4, 23. , 23.7, 25. , 21.8,
            20.6, 21.2, 19.1, 20.6, 15.2,  7. ,  8.1, 13.6, 20.1, 21.8, 24.5,
            23.1, 19.7, 18.3, 21.2, 17.5, 16.8, 22.4, 20.6, 23.9, 22. , 11.9]),
     'y_pred': array([30.5267484 , 25.0863876 , 30.50798059, 28.18447014, 27.27659833,
            25.0206824 , 22.97649705, 18.83349203, 10.41056667, 18.22035867,
            17.95015824, 21.25976346, 20.80047342, 19.60016143, 19.196703  ,
            19.47833086, 20.78071079, 16.6822844 , 16.65544328, 18.72117678,
            12.29985737, 17.59449166, 15.35105412, 13.40048216, 15.32683646,
            13.20887423, 15.17435384, 14.21411649, 19.05918486, 20.41568379,
            10.89254337, 17.90439239,  7.89185461, 14.15801366, 13.24116244,
            24.47046953, 22.94182224, 23.67632915, 23.36460532, 32.39402215,
            35.14808049, 27.98190352, 25.45873917, 24.73521528, 23.00170805,
            22.57211674, 20.58753997, 17.52214917,  8.13490741, 16.97684727,
            21.06815411, 23.91508564, 27.64691619, 24.14485615, 15.56029526,
            30.79882223, 24.80264078, 33.4623118 , 21.30659952, 20.90728705,
            17.44043906, 17.9999192 , 23.53676251, 21.48419631, 21.78225019,
            31.51422368, 26.62239223, 21.4259965 , 17.5785947 , 21.04346836,
            25.64100972, 22.2476999 , 25.28041972, 24.53308237, 26.43625821,
            24.64444619, 23.58527409, 24.18580195, 21.5659332 , 23.34277551,
            28.76031545, 27.27313482, 26.55259368, 25.59918469, 24.89705822,
            28.07627352, 22.42719907, 26.60952109, 31.01227403, 31.12757221,
            27.72734566, 28.06284015, 30.0598584 , 30.47238708, 28.12557027,
            29.29080253, 25.36109703, 35.73204957, 35.27318261, 32.47046941,
            24.93842677, 26.00700668, 20.31952914, 20.78433732, 22.06589527,
            19.2069564 , 17.68898542, 21.40525709, 23.09291626, 20.11633401,
            21.10974578, 27.00546521, 21.45765484, 21.16538012, 26.0590909 ,
            21.10224572, 24.0431348 , 24.54699762, 21.06883648, 21.61550914,
            22.7970042 , 23.29465489, 21.21031219, 16.65846995, 21.31241829,
            23.35272473, 15.00484047, 13.98216192, 16.31224834, 12.82892742,
            17.67100485, 17.35169438, 18.06590466, 14.54231082, 12.43429351,
            14.70629958, 14.13498891, 16.76691574, 11.86766115, 14.10364074,
            10.48750529,  2.60155375, 17.25421879, 10.95127839,  8.45772726,
             9.92604258, 16.03249641,  8.13668368,  8.93992704, 13.84872187,
            19.46012801, 18.81839351, 25.07151207, 17.06680828, 24.76523955,
            22.84078468, 14.8100259 , 32.08569446, 29.75556827, 24.11794019,
            36.71250211, 34.44106735, 41.40717071, 41.14172393, 25.14455214,
            25.88779219, 33.83456259, 23.53734156, 26.16880192, 26.08817423,
            22.99156864, 24.85059583, 24.84904356, 29.49513872, 27.7346177 ,
            29.86929462, 25.88565372, 29.83332418, 30.83380806, 31.60060406,
            32.04371601, 27.97667106, 33.12994507, 31.59105041, 24.50879627,
            24.84504636, 32.59723405, 32.6452962 , 32.13317928, 32.94471956,
            29.48976084, 29.76312024, 31.45418695, 30.53869113, 30.8009377 ,
            38.04946061, 34.94816639, 31.63807099, 33.42340016, 28.38581989,
            28.58556215, 29.92318278, 34.22091216, 38.91580814, 39.72884181,
            21.62558428, 22.49795701, 17.75037721, 26.49671976, 21.82975475,
            26.21074138, 21.32617731, 25.95677617, 23.52170299,  8.98474282,
            23.49843966, 29.84060044, 27.35266917, 28.57173841, 33.16259161,
            36.11656764, 27.36112851, 34.95044465, 29.70227353, 34.88411689,
            35.37154272, 34.9656523 , 31.30723951, 31.67098863, 30.16041348,
            25.22307605, 31.68482563, 34.49607873, 33.59011098, 34.88609777,
            25.65542526, 33.56085549, 31.12769825, 27.71172655, 27.9700297 ,
            26.07987213, 24.53909622, 24.22949914, 26.79548935, 18.36775975,
            14.56236196, 20.0211073 , 20.77494205, 20.94318096, 22.45082716,
            23.00302431, 24.13854883, 22.8773354 , 24.94523676, 23.43062427,
            21.9220964 , 37.75348485, 46.0387449 , 36.29956397, 33.42227994,
            35.63121347, 37.73550614, 43.15871658, 35.54582198, 35.78581939,
            26.2145508 , 33.04359663, 44.15984014, 40.10445186, 23.67097031,
            22.83333215, 27.75872967, 28.30030813, 35.97207441, 33.5821157 ,
            33.00259849, 34.53104554, 33.03762186, 30.33610549, 35.79537384,
            40.14368337, 35.13832636, 39.72628001, 44.66430225, 33.53476984,
            28.39893868, 21.44108838, 26.46426956, 26.45036422, 28.19254823,
            32.28714563, 34.00450932, 30.73048779, 27.58136754, 25.50416679,
            30.41717428, 28.5537616 , 21.26499536, 28.56613136, 33.32854451,
            30.70616356, 29.65558785, 29.99829605, 33.95640231, 34.03666941,
            29.92228892, 35.26268183, 31.4684171 , 27.27185199, 22.45919981,
            17.35290013, 25.78745656, 22.18307636, 24.27488685, 25.22185983,
            19.3730908 , 18.75693281, 18.96546422, 24.70680454, 22.25319816,
            24.90064011, 24.58899763, 22.27170985, 18.29406617, 25.31675015,
            25.74825622, 24.24006005, 20.74247194, 22.2724639 , 25.99697212,
            22.95074899, 21.08849735, 24.02165592, 22.27198713, 21.97337322,
            20.83605635, 19.88580193, 18.8456278 , 22.22297757, 21.17604193,
            20.66020263, 32.18750342, 23.77228515, 27.10826844, 28.88233483,
            17.79339712, 16.23249967, 25.02233521, 27.40697378, 24.01081973,
            20.9085694 , 20.94202826, 17.03969642, 25.0247316 , 13.01837226,
            15.28516465, 16.05874373, 18.43463798, 17.16282305, 16.50646587,
            18.9234259 , 16.53289546, 12.3301473 , 14.91260929, 36.53029657,
             3.44934894,  9.86726338,  2.15755822, 14.51490398, 25.64604461,
            28.14171927, 20.55676673, 18.99700387,  6.29338716,  0.36723264,
            25.21231157, 19.00722091, 20.84373786, 16.83178745, 16.52858787,
             6.94732752, 23.14935994, 18.00555339, 17.87839155,  5.58726281,
            11.43561647,  7.11854835,  7.71483181,  9.85102228, 18.58218928,
            22.06088402, 23.23262238, 13.50474107, 25.30921032, 21.85118929,
            25.93778574, 25.4333105 , 21.25055767,  6.49120345, 16.70422006,
            14.62619407, 22.47437656, 23.96833655, 14.36796361,  7.2210479 ,
             3.00634116,  8.63434   , 23.73687785, 19.24233528, 26.55582952,
            14.69932053, 24.6247661 ,  6.53985825, 13.3906666 , -4.21497838,
            17.08348653, 22.44850424, 10.47410586,  2.99281815, 23.29672629,
            25.22113115, 23.89095978, 22.77551145, 21.75192822, 21.64393502,
            16.85421874, 22.9314145 , 17.00701415, 22.25307386, 21.39011639,
            25.78774037, 26.91686181, 29.16977078, 25.4209194 , 22.48929316,
            21.52325279, 22.11480045, 16.51672591, 12.5907749 , 17.84338686,
            15.60712236, 23.04596371, 24.61591619, 23.81288272, 17.27815149,
            20.49211538, 24.31173475, 23.4049838 , 22.79344004, 23.91122798,
            25.99821758, 26.340751  , 24.80431406, 29.81887588, 24.31283927,
            25.00306443, 21.45560985, 20.94064609, 23.81165456, 23.93028539,
            26.86069153, 26.55911716, 25.52133615, 28.37067106, 25.58182393,
            23.86249106, 22.9542643 , 23.23084127, 20.85348155, 22.14460316,
            25.95941808, 28.56570542, 28.66950016, 32.40424752, 20.80900555,
            22.63860862, 26.82479237, 15.10173037, 24.19086321, 25.83217571,
            28.72040665, 32.90390596, 34.77066018, 26.30591365, 25.34381775,
            28.23758999, 25.17944017, 26.4705689 , 11.10717431,  8.17635637,
             3.73550246, 13.8301605 , 15.69551789, 20.29222007, 20.71648521,
            17.02842287, 13.84795979, 19.07576995, 21.45547232, 18.09944532,
            20.72111835, 23.33408351, 21.35832828, 27.58215819, 25.92833448,
            21.13637498]),
     'deviation': array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
            0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
            0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
            0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
            0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
            0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
            0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
            0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
            0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
            0.99]),
     'accuracy': [0.0,
      0.04940711462450593,
      0.10869565217391304,
      0.17391304347826086,
      0.22924901185770752,
      0.274703557312253,
      0.3201581027667984,
      0.3616600790513834,
      0.40711462450592883,
      0.4407114624505929,
      0.4881422924901186,
      0.5276679841897233,
      0.5612648221343873,
      0.5849802371541502,
      0.616600790513834,
      0.6442687747035574,
      0.6600790513833992,
      0.6818181818181818,
      0.7055335968379447,
      0.717391304347826,
      0.7351778656126482,
      0.7509881422924901,
      0.7648221343873518,
      0.7766798418972332,
      0.7865612648221344,
      0.7924901185770751,
      0.808300395256917,
      0.8181818181818182,
      0.8260869565217391,
      0.8320158102766798,
      0.8399209486166008,
      0.8458498023715415,
      0.8557312252964426,
      0.8675889328063241,
      0.8774703557312253,
      0.8893280632411067,
      0.8952569169960475,
      0.9051383399209486,
      0.9090909090909091,
      0.9189723320158103,
      0.9229249011857708,
      0.9288537549407114,
      0.9308300395256917,
      0.9347826086956522,
      0.9387351778656127,
      0.9446640316205533,
      0.950592885375494,
      0.950592885375494,
      0.9525691699604744,
      0.9545454545454546,
      0.9565217391304348,
      0.9624505928853755,
      0.9644268774703557,
      0.9644268774703557,
      0.9664031620553359,
      0.9683794466403162,
      0.9703557312252964,
      0.9703557312252964,
      0.974308300395257,
      0.9782608695652174,
      0.9802371541501976,
      0.9822134387351779,
      0.9822134387351779,
      0.9841897233201581,
      0.9861660079051383,
      0.9881422924901185,
      0.9881422924901185,
      0.9881422924901185,
      0.9881422924901185,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9901185770750988,
      0.9920948616600791,
      0.9920948616600791,
      0.9920948616600791,
      0.9920948616600791,
      0.9920948616600791,
      0.9920948616600791,
      0.9940711462450593,
      0.9940711462450593,
      0.9940711462450593,
      0.9940711462450593,
      0.9960474308300395,
      0.9960474308300395,
      0.9960474308300395,
      0.9960474308300395,
      0.9960474308300395,
      0.9960474308300395,
      0.9960474308300395,
      0.9980237154150198,
      0.9980237154150198],
     'auc_rec': 0.8348534255599472}


