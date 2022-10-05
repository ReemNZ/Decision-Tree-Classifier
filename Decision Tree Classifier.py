#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# - 1. [Libraries and EDA](#1.-Libraries-and-EDA)
# - 2. [Train test split function](#2.-Train-test-split-function)
# - 3. [Helper function](#3.-Helper-function)
# - 4. [Model modules](#4.-Model-modules) 
#     - 4.1 [Pure data](#4.1-Pure-data)
#     - 4.2 [Classify](#4.2-Classify)
#     - 4.3 [Feature type](#4.3-Feature-type)
#     - 4.4 [Potential splits](#4.4-Potential-splits)
#     - 4.5 [Split data](#4.5-Split-data)
#     - 4.6 [Lowest entropy calculation](#4.6-Lowest-entropy-calculation)
#     - 4.7 [Best split](#4.7-Best-split)
# - 5. [Decision tree algorithm](#5.-Decision-tree-algorithm)
#     - 5.1 [Tree algorithm](#5.1-Tree-algorithm)
#     - 5.2 [Example classification](#5.2-Example-classification)
# - 6. [Accuracy checking](#6.-Accuracy-checking)
# - 7. [Categorical example (Titanic)](#7.-Categorical-example-(Titanic))

# ![te.PNG](attachment:te.PNG)

# # 1. Libraries and EDA

# In[413]:


import pandas as pd
import numpy as np
from sklearn import datasets
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import pprint 


# In[414]:


flower = datasets.load_iris()


# In[415]:


dir(flower)


# In[416]:


print(flower.data[0], flower.feature_names, flower.target_names)


# In[417]:


df = pd.DataFrame(flower.data, columns=flower.feature_names)
df['type'] = flower.target
df.head(2)


# In[418]:


df['type'].replace({0 : flower.target_names[0], 
                    1 : flower.target_names[1],
                    2 : flower.target_names[2]}, inplace=True)
df.rename(columns={"sepal length (cm)" : "sepal_length", "sepal width (cm)" : "sepal_width", 
                  "petal length (cm)" : "petal_length", "petal width (cm)" : "petal_width"}, inplace=True)
df.sample(3)


# # 2. Train test split function

# In[419]:


def split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    
    indicies = df.index.tolist() #must be converted to a list.
    test_species = random.sample(population=indicies, k=test_size)
    test_df = df.iloc[test_species]
    train_df = df.drop(test_species)
    return train_df, test_df


# In[420]:


random.seed(0) #To always get the same split.
train_df, test_df = split(df,0.4)


# In[421]:


test_df.head(3)


# In[422]:


train_df.head(3)


# Since numpy is faster than Pandas, so this classifier will be built using numpy.

# # 3. Helper function

# In[423]:


df0 = train_df[train_df['type'] == "setosa"]
df1 = train_df[train_df['type'] == "versicolor"]
df2 = train_df[train_df['type'] == "virginica"]


# In[425]:


iris1 = plt.scatter(df0['petal_width'], df0['petal_length']);
iris2 = plt.scatter(df1['petal_width'], df1['petal_length']);
iris3 = plt.scatter(df2['petal_width'], df2['petal_length']);
plt.xlabel('petal width (cm)')
plt.ylabel('petal length (cm)');
plt.legend([iris1, iris2, iris3],["0 : setosa","1 : versicolor","2 : virginica"]);


# In[426]:


data = train_df.values
data[:5]


# # 4. Model modules

# ## 4.1 Pure data

# Check using boelians if number of unique values for the "Classification" is equal to 1.

# In[427]:


def data_purity(data):
    type_column = data[:,-1] #access all rows then access the last column 
    unique_species = np.unique(type_column)
    if len(unique_species) == 1:
        return True
    else:
        return False


# In[429]:


train_df[train_df['sepal_width'] <= 2.0]


# In[430]:


data_purity((train_df[train_df['sepal_width'] <= 2.0]).values) #it is inserted as a dataframe, then it is converted to an array by "values".


# ## 4.2 Classify

# In[431]:


def classify_data(data):
    type_column = data[:,-1]
    unique_species, species_counts = np.unique(type_column, return_counts= True) # array([1,2]), array([27,2])
    max_value = np.max(species_counts) #27
    index_max = np.where(species_counts == max_value)[0] # index 0
    classification = unique_species[index_max][0] # 1.0
    #classification = flower.target_names[int(iris_classify)]
    return classification


# In[432]:


classify_data((train_df[train_df['petal_width'] <= 1]).values)


# In[433]:


classify_data(train_df[ (train_df['petal_width'] <= 1.5) & (train_df['petal_width'] >= 1) ].values)


# In[434]:


classify_data((train_df[train_df['petal_width'] >= 2]).values)


# In[435]:


classify_data(train_df[ (train_df['petal_width'] <= 1.7) & (train_df['petal_width'] >= 1.3) ].values)


# ### Draft

# In[436]:


type_column = train_df[ (train_df['petal_width'] <= 1.5) & (train_df['petal_width'] >= 1) ].values[:,-1]
unique_species, species_counts = np.unique(type_column, return_counts= True)
max_value = np.max(species_counts)
index = np.where(species_counts == max_value)[0]
index2 = unique_species[index][0]
unique_species, species_counts, type_column, max_value, index, index2


# In[437]:


type_column =(train_df[train_df['petal_width'] >= 2]).values[:,-1]
unique_species, species_counts = np.unique(type_column, return_counts= True) # array([1,2]), array([27,2])
max_value = np.max(species_counts)  #27
index = np.where(species_counts == max_value)[0]
index2 = unique_species[index][0]
type_column, unique_species, species_counts, max_value, index, index2


# ## 4.3 Feature type

# In[438]:


def determine_type_feature(df):
    feature_type = []
    threshold = 15
    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]
        if (isinstance(example_value, str)) or (len(unique_values) <= threshold):
            feature_type.append("categorical")
        else:
            feature_type.append("continous")
    return feature_type


# In[439]:


FEATURE_TYPE = determine_type_feature(df)
FEATURE_TYPE


# ## 4.4 Potential splits

# Based on feature type whether continous or categorical, potential splits are calculated as the midpoint between each 2 successive points.

# In[440]:


def get_potential_split(data):
    potential_splits = {}
    column_index = data.shape
    for ind in range((column_index[1])-1):
        extracted_row = data[:, ind]
        unique_row = np.unique(extracted_row)
        
        type_of_feature = FEATURE_TYPE[ind]
        if type_of_feature == "continous":
            potential_splits[ind] = []
            for i in range(len(unique_row)):
                if i != 0:
                    present = unique_row[i]
                    previous = unique_row[i-1]
                    split = (present + previous) / 2
                    potential_splits[ind].append(split)
        
        else:
            potential_splits[ind] = unique_row
    
    
    return potential_splits


# In[442]:


splits = get_potential_split(train_df.values)
splits


# In[444]:


sn.lmplot(data = train_df, x='petal_width', y='petal_length', fit_reg= False, hue='type', size=6, aspect=1.5);
plt.vlines(x= splits[3], ymin=0, ymax=7, colors='k');
plt.hlines(y= splits[2], xmin=0, xmax=2.5, colors='y');


# ## 4.5 Split data

# Spliting data into below and above based on their feature type (continous) or (categorical).

# In[445]:


def split_data(data, column_index, split_value):
    split_column_value = data[:, column_index]
    type_of_feature = FEATURE_TYPE[column_index]
    
    if type_of_feature == "continous":
        data_below = data[split_column_value <= split_value]
        data_above = data[split_column_value > split_value]
    else:
        data_below = data[split_column_value == split_value]
        data_above = data[split_column_value != split_value]
    return data_below, data_above


# In[446]:


column_index = 3
split_value = 0.8

split_data(data, column_index, split_value)


# In[447]:


data_below, data_above = split_data(data, column_index, split_value)


# In[448]:


sn.lmplot(data= train_df, x='petal_width', y='petal_length', fit_reg=False, aspect=1.5);
plt.vlines(x=split_value, ymin= 0, ymax= 7, colors='k');


# ## 4.6 Lowest entropy calculation

# In[449]:


def calculate_entropy(data):
    type_column = data[:,-1]
    _, species_counts = np.unique(type_column, return_counts=True)
    probability = species_counts / species_counts.sum()
    entropy = sum(probability * -np.log2(probability))
    return entropy


# In[450]:


calculate_entropy(data_below)


# In[451]:


calculate_entropy(data_above)


# In[452]:


def overall_entropy(data_below, data_above):
    p_data_below = len(data_below) / (len(data_below) + len(data_above))
    p_data_above = len(data_above) / (len(data_below) + len(data_above))
    overall_entropy = (p_data_below * calculate_entropy(data_below)) + (p_data_above * calculate_entropy(data_above))
    return overall_entropy


# In[453]:


overall_entropy(data_below, data_above)


# ## 4.7 Best split

# In[454]:


potential_split = get_potential_split(data)


# In[455]:


def determine_best_split(data, potential_split):
    current_entropy = 99
    for column_index in potential_split:
        for val in potential_split[column_index]:
            data_below, data_above = split_data(data, column_index = column_index, split_value= val) 
            current_overall_entropy = overall_entropy(data_below, data_above)
            if current_overall_entropy < current_entropy:
                current_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = val
    return best_split_column, best_split_value


# In[456]:


determine_best_split(data, potential_split)


# In[457]:


def determine_best_split(data, potential_split):
    current_entropy = 99
    for column_index in potential_split:
        for val in potential_split[column_index]:
            data_below, data_above = split_data(data, column_index = column_index, split_value= val) 
            current_overall_entropy = overall_entropy(data_below, data_above)
            if current_overall_entropy <= current_entropy:   ### <=
                current_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = val
    return best_split_column, best_split_value


# In[458]:


determine_best_split(data, potential_split)


# # 5. Decision tree algorithm

# In[459]:


example_tree = {"petal_width <= 0.8" : ["Setosa" , 
                                        {"petal_width <= 1.65" : [{"petal_width <= 4.9" : ["versicolor","virginica"]},
                                        "virginica"]}]}


# ## 5.1 Tree algorithm

# In[460]:


def decision_tree_algorithm(df, counter=0, min_sample=2, max_depth=5):
    
    # data_preparation
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPE #To make the order excuted every loop not only under certain loops.
        COLUMN_HEADERS = df.columns
        FEATURE_TYPE = determine_type_feature(df)
        data = df.values
    else:
        data = df #after counter 1, data is delivered from data below/above as np arrays
    
    # base case
    if (data_purity(data)) or (len(data) < min_sample) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    
    else:
        counter += 1
        potential_split = get_potential_split(data)
        best_split_column, best_split_value = determine_best_split(data, potential_split)
        data_below, data_above = split_data(data, best_split_column, best_split_value)
        
        #intiate sub-tree 1
        type_of_feature = FEATURE_TYPE[best_split_column]
        if type_of_feature == "continous":
            question = "{} <= {}".format(COLUMN_HEADERS[best_split_column], best_split_value)
        else:
            question = "{} = {}".format(COLUMN_HEADERS[best_split_column], best_split_value)
            
        yes_answer = decision_tree_algorithm(data_below, counter, min_sample, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_sample, max_depth)
        sub_tree = {question : [yes_answer, no_answer]}
        
        return sub_tree


# In[461]:


tree = decision_tree_algorithm(train_df[train_df['type'] != 2])
tree


# In[462]:


tree = decision_tree_algorithm(train_df)
tree


# In[463]:


pprint.pp(tree)


# ![te.PNG](attachment:te.PNG)

# In[464]:


tree1 = decision_tree_algorithm(train_df, min_sample=50)
tree1 # if data is few, take the classification of the majority.


# In[465]:


pprint.pp(tree1)


# In[466]:


tree2 = decision_tree_algorithm(train_df, max_depth=3)
pprint.pprint(tree2)


# ## 5.2 Example classification

# In[467]:


example = test_df.iloc[0]
example


# In[468]:


print(tree.keys(), list(tree.keys())[0])


# In[483]:


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    
    if comparison_operator == "<=":
        if example[feature_name] <= float(value): #whether answer is class or dict
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    else:
        if str(example[feature_name]) == value: #whether answer is class or dict
            answer = tree[question][0]
        else:
            answer = tree[question][1]
        
    if not isinstance(answer, dict): # if answer is a class not dictionary
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
        


# In[484]:


classify_example(example, tree)


# # 6. Accuracy checking

# In[471]:


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["correct_classification"] = (df["classification"] == df["type"])
    accuracy = df["correct_classification"].mean()
    return accuracy


# In[472]:


calculate_accuracy(test_df, tree)


# In[473]:


test_df.loc[129]


# In[474]:


pprint.pp(tree)


# petal width is below 1.65 by only 0.05 which made the classification incorrect. Since the difference is small, so the error is acceptable.

# In[475]:


train1_df, test1_df = split(df, test_size=0.05)
tree = decision_tree_algorithm(train_df, max_depth=3)
accuracy = calculate_accuracy(test_df, tree)

pprint.pp(tree)
print(accuracy)


# # 7. Categorical example (Titanic)

# In[476]:


tit = pd.read_csv("ml_tut9.csv")
tit.sample(2)


# In[477]:


tit['type'] = tit['Survived']
tit = tit.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
tit.sample(2)


# In[478]:


tit.info()


# In[479]:


median_age = tit['Age'].median()
mode_embarked = tit['Embarked'].mode()[0]
tit = tit.fillna({"Age" : median_age, "Embarked" : mode_embarked})
tit.info()


# In[480]:


FEATURE_TYPE = determine_type_feature(tit)
FEATURE_TYPE #You must calculate it before running the decision tree classifier.


# In[481]:


get_potential_split(tit.values)


# In[482]:


tit_tree = decision_tree_algorithm(tit, max_depth=3)
pprint.pp(tit_tree, width=50) #width adjust it not to be on the same line.


# In[485]:


example_tit = tit.iloc[3]
example_tit


# In[487]:


classify_example(example_tit, tit_tree) 


# In[492]:


train2_df, test2_df = split(tit, test_size=0.2)
tree = decision_tree_algorithm(train2_df, max_depth=3)
accuracy = calculate_accuracy(test2_df, tree)

pprint.pp(tree, width=50)
print(accuracy)


# In[ ]:




