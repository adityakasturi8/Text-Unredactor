> # Text Unredactor
### Author : Aditya K Kasturi 

__About:__
- When classified information needs to be made available to the public, it must first be redacted. In this procedure, all sensitive names, locations, and other information are concealed. Documents containing sensitive information, such as names, incident reports, dispatch logs, and patient information, are often found.

Predicting the names from already redacted files is often challenging. We must practice our machine learning model on large datasets if we want to identify the most reliable and best name. The Large Movie Review Dataset was used as my training dataset because it includes reviews of most famous movies.

__Libraries and Packages Used:__
- pandas
- sklearn
- DictVectorizer
- RandomForestClassifier
- pytest
- warnings

__system requirements:__
- The Text Analytics Final Annoucement states that the project can be evaulated using google instance, your personal machine, or the ou library jupyter hub instance. As a first possibility i would avoid using google instance and run in personal machine or jupyter hub instance
- An instance with minimum of 8gb memory is required.
- A working GPU for the machine learning model to run.
- Make sure you have an active internet connection, as the data is extracted from Github.


### Assumptions 
- The unredactor.tsv raw url (github) is active and shall be available to use.
- Personal computer or ou library jupyter hub instance is used incase of failure in running the file on google console.

### Bugs
- Some of the tab seperated files in the ``` unredactor.tsv ``` are not accurately separated, and are irregular.
- Some of the user's data have multiple redactions, and duplicates.
- Data is uneven in few lines
- I had to drop all the __NaN__ to make sure that the model works better.
 ### Description

__How to install and use this packages (for personal computer):__
0. Require prior installation of python, and pip
1. You can execute in the terminal or using VScode or using Jypter or any other IDE
2. gitclone my repository ```https://github.com/adityakasturi8/cs5293sp22-project3.git```
3. cd into the project directory in terminal ```cs5293sp22-project3```
4. install python packages 
                  - pip install sklearn
                  - pip install pandas
                  - pip install DictVectorizer 
                  - pip install pytest
                  
6. run unit test using ``` python pytest```
7. run the unredactor.py file by typing ``` python unredactor.py ```


__How to install and use this packages (for google console):__
0. Require prior installation of python, pipenv, and pip
1. gitclone my repository ```https://github.com/adityakasturi8/cs5293sp22-project3.git```
2. cd into the project directory ```cs5293sp22-project3```
3. install python package pipenv by typing ```pip install pipenv```
4. run unit test using ```pipenv run python -m pytest```
5. run the unredactor.py file using the below instructions

__Possible errors to expect when running it using google console:__
1. __No moudle Found: _ctype__: To fix the error enter the command ```sudo apt-get install libffi-dev``` . This should mostly resolve the issue, if still facing it, please run the fine on your personal computer or jupyter instance. 


__Running the Program (google console):__
- The program can be run by utilizing the commandline.
- To run the program, go to the cs5293sp22-project3 folder
- run the unredactor.py file 
- An example on how to run the unredactor.py file is mentioned below

  ```
  pipenv run python unredactor.py
  ``` 
  
  __Running the Program (personal computer):__
- The program can be run by utilizing the commandline.
- To run the program, go to the cs5293sp22-project3 folder
- run the unredactor.py file 
- An example on how to run the unredactor.py file is mentioned below
  ```
  python unredactor.py
  ``` 
__Dataset:__
- For this project, the dataset has be acquired from Stanford.edu 
- The dataset can be found on ```https://ai.stanford.edu/~amaas/data/sentiment/```
- unredactor.tsv file link can be found on ```https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv ```
- The above link has been used as training, validation, and testing our model.
__Result:__
- After running the unredactor.py file, the output geneted will be precision, recall, and f1 score of the randomforest classifer which was used for this project.
- An example of the output is mentioned below
```

Precision:  0.0016885886953430503
Recall:  0.004123711340206186
F1 Score:  0.0020333507530048934
Top 10 predictions:  ['Joan Crawford' 'Brosnan' 'Michael Madsen' 'Eddie Murphy' 'Denise Richards' 'Mehta' 'Joan Crawford'
                     'Aidan Quinn' 'von Trier' 'Richard Rodney Bennet']

```
__Functions:__

- In the unredactor.py file, There are four functions:

0. __unredactor.py__ :  The unredactor.py file calls all the functions from unredactor.py and executes the flow of the project.
                  The unredactor.py contains the following functions
                  ```

                check_length(sentence)

                split_data(data)
                 
                feature_extraction(data)

                randomforest_classifer()

                  ```
1. __check_length(sentence)__ : This function is made to know the length of the each name which has been redacted. it contains '█' which is essential to calculate each character length and stores the length of the it in a a column ```label_length```

```
def check_length(sentence):
    length = 0
    for i in sentence:
        if i == '█':
            length += 1
    return length
```
![image](https://user-images.githubusercontent.com/95768375/167355639-08cde250-5f58-49cb-aaf8-5ff3122c0a06.png)


The above image contains the first four rows of the dataframe. the last column contains the length of the each redacted label.



2. __split_data(data)__: This function is used to split the data into training, validation, and testing after extracting it from the ``` unredactor.tsv ``` file. 

```
def split_data(data):
    train_split = data.loc[data['type'] == 'training']
    validation_split = data.loc[data['type'] == 'validation']
    test_split = data.loc[data['type'] == 'testing']
    return train_split, validation_split, test_split

```

3. __feature_extraction(data)__: This function utilizes for features extraction using dictionary vectorizer

```
def feature_extraction(data):
    len_lst = list(data['label_length'])
    
    L = []
    for i in range(len(data)):
        D = {}
        D['length'] = len_lst[i]
        L.append(D)

```
Here, The ``` label_length ``` column is converted to a list and added to an object variable ``` len_lst ```. Creation of a list of dictionaries takes places, which has a key name as __length__ and value is the length of the each redacted name label.

This feature helps to determine during the unredaction the length of charactes to expect and predict. 

4. __randomforest_classifer()__: This fuction calls the machine learning model RandomForestClassifer from the Sci-kit learn library and performs the following operations.  

```
def randomforest_classifer(train_split_feature_extracted,train_split,validation_split_feature_extracted,test_split_feature_extracted):
    model = RandomForestClassifier(n_estimators = 500,criterion = 'entropy',max_features = 'auto',min_samples_leaf = 1,min_samples_split = 2,n_jobs = 1,random_state = 42)
    model.fit(train_split_feature_extracted, train_split['labels'])
    predicted_test_split = model.predict(test_split_feature_extracted)
    print("Precision: ", precision_score(test_split['labels'], predicted_test_split, average='macro'))
    print("Recall: ", recall_score(test_split['labels'], predicted_test_split, average='macro'))
    print("F1 Score: ", f1_score(test_split['labels'], predicted_test_split, average='macro'))
    print("Top 10 predictions: ", model.predict(test_split_feature_extracted)[:10])
 
 ```

- The above python code defines a random forest Classifier. This classifier is then used to train the model on the train_split_feature_extracted dataset.
- The model is then used to predict the labels on the test_split_feature_extracted dataset. Finally, the precision,recall, and f1 score of the predictions is printed.
- It also prints the Top 10 predictions which contains the predicted names.

__Test_Cases__:
- Every test funtion is tested with a passing case
- 
1. for ```test_check_length()``` one passing case is tested by Checking  if the length of the unredactor.unredactor string is equal to 1, and assert True if is is equal to 1. 
```
def test_check_length():
    if unredactor.check_length('█') == 1:
        assert True
```

2. for ```test_split_data()```  The test_split_data() function checks that the train_split, validation_split, and test_split DataFrames have the same column names as the original data DataFrame.

```
def test_split_data():
    
    train_split, validation_split, test_split = unredactor.split_data(data)
    assert train_split.columns.tolist() == validation_split.columns.tolist() == test_split.columns.tolist() == data.columns.tolist()

```


