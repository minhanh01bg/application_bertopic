from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# print(len(docs))
data = pd.read_csv('./data_test/baomoi12092023.csv')


data['body'] = data['body'].astype(str)



docs = data['body'].tolist()



print(len(docs))
topic_model = BERTopic(language="vietnamese", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info())
print(topic_model.get_topic_freq().head())