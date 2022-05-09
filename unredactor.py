import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score,recall_score,f1_score
import warnings

warnings.filterwarnings('ignore') 

data = pd.read_table("https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv",on_bad_lines='skip', sep='\t', names = ['GitID','type','labels','redacted_data'])

def check_length(sentence):
    length = 0
    for i in sentence:
        if i == '█':
            length += 1
    return length


def split_data(data):
    train_split = data.loc[data['type'] == 'training']
    validation_split = data.loc[data['type'] == 'validation']
    test_split = data.loc[data['type'] == 'testing']
    return train_split, validation_split, test_split

def feature_extraction(data):
    len_lst = list(data['label_length'])
    
    L = []
    for i in range(len(data)):
        D = {}
        D['length'] = len_lst[i]
        L.append(D)
    
    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(L)  


def randomforest_classifer(train_split_feature_extracted,train_split,validation_split_feature_extracted,test_split_feature_extracted):
    model = RandomForestClassifier(n_estimators = 500,criterion = 'entropy',max_features = 'auto',min_samples_leaf = 1,min_samples_split = 2,n_jobs = 1,random_state = 42)
    model.fit(train_split_feature_extracted, train_split['labels'])
    predicted_test_split = model.predict(test_split_feature_extracted)
    print("Precision: ", precision_score(test_split['labels'], predicted_test_split, average='macro'))
    print("Recall: ", recall_score(test_split['labels'], predicted_test_split, average='macro'))
    print("F1 Score: ", f1_score(test_split['labels'], predicted_test_split, average='macro'))
    print("Top 10 predictions: ", model.predict(test_split_feature_extracted)[:10])

if __name__ == "__main__":
        data['label_length'] = data['redacted_data'].apply(check_length)
        train_split, validation_split, test_split = split_data(data)
        train_split_data = train_split['redacted_data']
        validation_split_data = validation_split['redacted_data']
        test_split_data = test_split['redacted_data']
        train_split_feature_extracted = feature_extraction(train_split)
        validation_split_feature_extracted = feature_extraction(validation_split)
        test_split_feature_extracted = feature_extraction(test_split)
        randomforest_classifer(train_split_feature_extracted,train_split,validation_split_feature_extracted,test_split_feature_extracted)
        

