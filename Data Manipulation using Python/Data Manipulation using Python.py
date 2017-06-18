import numpy as np
import pandas as pd


train = pd.read_csv('train.csv')

###############################################################################
##                      Imputing Missing File                                ##
'''
‘fillna()’ does it in one go.
 It is used for updating missing values with the overall mean/mode/median of the column.
 Let’s impute the ‘Gender’, ‘Married’ and ‘Self_Employed’ columns with their respective modes.
'''
from scipy.stats import mode

# to find missing values
train.apply(lambda x: sum(x.isnull()),axis=0)  
# axis = 0 applied on each columns


(train['Gender'].mode())[0] 
#  [0] postion function gives mode and another position gives count

train['Gender'].fillna((train['Gender'].mode())[0],inplace= True)
train['Self_Employed'].fillna((train['Self_Employed'].mode())[0],inplace= True)
train['Married'].fillna((train['Self_Employed'].mode())[0],inplace= True )


###############################################################################
##                             Pivote Table                                  ##
'''
Pandas can be used to create MS Excel style pivot tables. 
For instance, in this case, a key column is “LoanAmount” which has missing values. 
We can impute it using mean amount of each ‘Gender’, ‘Married’ and ‘Self_Employed’ group
'''
# Determine the pivote table
table = train.pivote_table(values=train['LoanAmount'],
                           index=['Gender', 'Married', 'Self_Employed'],
                           aggfunc=np.mean)


##############################################################################

