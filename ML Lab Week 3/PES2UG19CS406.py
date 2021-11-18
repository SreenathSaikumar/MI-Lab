'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO
    entropy=0
    temp=df[[df.columns[-1]]].values
    vals,counts=np.unique(temp,return_counts=True)
    tot_ctr=np.sum(counts)
    for i in counts:
        temp2=i/tot_ctr
        if temp2!=0:
            entropy=entropy-(temp2*(np.log2(temp2)))
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    avginfo=0
    attr=df[attribute].values
    attruni=np.unique(attr)
    for i in attruni:
        tempdf=df[df[attribute]==i]
        res=tempdf[[tempdf.columns[-1]]].values
        vals,counts=np.unique(res,return_counts=True)
        tot_ctr=np.sum(counts)
        entropy=0
        for j in counts:
            temp=j/tot_ctr
            if temp!=0:
                entropy=entropy-temp*np.log2(temp)
        avginfo+=entropy*(np.sum(counts)/df.shape[0])
    return abs(avginfo)


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    information_gain=get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
    return information_gain




#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    selected_column=''
    infogains={}
    maxval = -10000000
    for i in df.columns[:-1]:
        infogainattr = get_information_gain(df,i)
        if infogainattr > maxval:
            selected_column = i
            maxval = infogainattr
        infogains[i] = infogainattr
    ans=(infogains,selected_column)
    return ans
