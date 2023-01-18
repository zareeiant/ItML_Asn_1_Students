#identify missisng values or zeroe, are there str in numrical col, 
# describe for numrical, different plots

import pandas as pd
import numpy as np
import math
import statistics as stats
import sklearn.datasets
import ipywidgets as widgets
import thinkplot
import thinkstats2
##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def describe(self):
        for col in self.num:
            print(col)
            print(self.data[col].describe())
            print('\n \n')

    def value_count(self):
        for col in self.cat:
            print(col)
            print(self.data[col].value_counts())
            print('\n \n')

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList
    def missing_values(self): 
        for col in self.data:
            print(col)
            print('Before handling nan values:',self.data[col].count())
            self.data[col]=self.data[col][~np.isnan(self.data[col])]
            print('After handling nan values:',self.data[col].count())
            

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def displot(self,splitTarg=False, show=True):
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.displot(data=self.data, x=col,col=self.target,kind="kde")
            if splitTarg == True:
                sns.displot(data=self.data, x=col, hue=self.target, col=self.target,kind="kde")



    def correlation(self, show=True):
        data=self.data.apply(pd.to_numeric, errors='coerce')
        data=data.drop(columns=self.cat)
        data=data.corr()
        mask=np.triu(np.ones_like(data, dtype=bool))
        sns.heatmap(data, center=0, linewidths=0.5, annot=True, cmap="YlGnBu", yticklabels=True, mask=mask)
        if show==True:
            plt.show()
        return plt 




    def find_outliers_IQR(self):

        for col in self.num:

            q1=self.data[col].quantile(0.25)
            q3=self.data[col].quantile(0.75)
            IQR=q3-q1
            outliers = self.data[col][((self.data[col]<(q1-1.5*IQR)) | (self.data[col]>(q3+1.5*IQR)))]
            print(col)
            print('number of outliers: '+ str(len(outliers)))
            print('max outlier value: '+ str(outliers.max()))
            print('min outlier value: '+ str(outliers.min())+'\n')

    #def remove_outliers(self):
     #   for col in self.num:


    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4,out5, out6, out7, out8, out9])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        tab.set_title(3, 'Outlier')
        tab.set_title(4, 'Correlation')
        tab.set_title(5, 'Displot')
        tab.set_title(6, 'Describe')
        tab.set_title(7, 'Value_count')
        tab.set_title(8, 'Missing value')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)
        with out4:
            self.find_outliers_IQR()
        with out5:
            fig4=self.correlation(show=False)
            plt.show(fig4)
        with out6:
            fig5=self.displot(show=False)
            plt.show(fig5)
        with out7:
            self.describe()
        with out8:
            self.value_count()
        with out9:
            self.missing_values()
