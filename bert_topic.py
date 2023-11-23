import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import pickle
import matplotlib.pyplot as plt
def save_pickle(list_data, path):
    with open(path, 'wb') as f:
        pickle.dump(list_data, f)
        
def load_pickle_list(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def process_bert(path):
    papers = pd.read_csv(path)
    # convert type of column
    papers.rename(columns={'doc_ner': 'paper_text'}, inplace=True)
    papers['paper_text'].astype(str)
    
    # -------------------------------- PROCESS --------------------------------
    # ------------------ DROP_DUPLICATED ------------------
    import numpy as np
    # duplicated
    print(papers.duplicated().sum())
    papers.drop_duplicates(inplace=True)

    # drop nan
    papers.dropna(axis=0, inplace=True, how="any")
    # drop inf
    papers.replace([np.inf, -np.inf], np.nan, inplace=True)
    papers.replace('', np.nan, inplace=True)
    papers.dropna(inplace=True,how='any',axis=0)
    # Remove the columns
    try:
        papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
    except:
        pass

    # Print out the first rows of papers
    print(papers.head())
    print(papers.info())
    papers.reset_index(inplace=True, drop=True)
    
    # ------------------ timestamps ------------------
    # get timestamps
    timestamps = papers.date.to_list()
    import datetime
    import re
    timestamps = [re.sub(r'\s+', ' ', timestamp).strip() for timestamp in timestamps]
    timestamps1 = [re.sub(r',','',time.split()[2]) for time in timestamps]
    timestamp_ = [datetime.datetime.strptime(timestamp, "%d/%m/%Y").date() for timestamp in timestamps1]
    timestamp_ = [str(time.year)+'-'+str(time.month)+'-'+str(time.day) for time in timestamp_]
    print('timestamps: ',timestamp_)
    
    # ------------------ Classes ------------------
    classes = [i for i in papers["title"]]
    
    # ------------------ PAPERS ------------------
    # Load the regular expression library
    import re

    # Remove punctuation
    papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # replace \n
    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub('\n', ' ', x))
    # Convert the titles to lowercase
    papers['paper_text_processed'] = \
    papers['paper_text_processed'].map(lambda x: x.lower())

    # Print out the first rows of papers
    papers['paper_text_processed'][0]
    
    # ------------------ process ------------------
    
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from unidecode import unidecode
    import string
    import re
    import pyvi
    from pyvi import ViTokenizer, ViPosTagger
    from underthesea import word_tokenize
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words = set(open("./data_test/vietnamese-stopwords-dash.txt", "r", encoding="utf-8").read().splitlines())

    def remove_whitespace(text):
        return re.sub(r'\s+', ' ', text).strip()

    def remove_number(text):
        return re.sub(r'\d+', '', text).strip()

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_punctuation_not_underscore(text): # remove dấu câu để lại dấu _
        # Tạo một bản sao của string.punctuation và loại bỏ dấu _
        punctuation = string.punctuation.replace("_", "")
        # punctuation = punctuation.replace("-", "")
        return text.translate(str.maketrans('', '', punctuation))

    def remove_(text):
        text = remove_punctuation_not_underscore(text)
        text = remove_number(text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = text.replace("_  _", "_")
        return re.sub(r'“|”', '', text).strip()
        # return ''.join(char for char in text if char not in ['“', '”'])

    def remove_punctuation_underscore(text): # remove dấu _
        punctuation = "_"
        return text.translate(str.maketrans('', '', punctuation))
        
    def word_to_phrase(text): # chuyển từ thành cụm từ
        # return ViTokenizer.tokenize(text)
        return word_tokenize(text, format="text")

    # def sent_to_words(sentences):
    #     for sentence in sentences:
    #         # deacc=True removes punctuations
    #         # loại bỏ dấu câu and split
    #         yield([remove_(str(word)) for word in re.findall(r'\w+|\S+', word_to_phrase(str(sentence))) if remove_(str(word)) != ''])

    def remove_stopwords(texts):
        words = [word for word in texts.split() if word not in stop_words]
        words_to_sentence = ' '.join(words)
        return words_to_sentence.strip()

    def remove_emoj(text):
        emoj = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642" 
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                                        "]+", re.UNICODE)
        return re.sub(emoj, '', text)

    # drop url
    def remove_url(text):
        text = re.sub(r'http\S+', '', text).strip()
        text = re.sub(r'www\S+', '', text).strip()
        text = re.sub(r'pic\S+', '', text).strip()
        text = re.sub(r'bit.ly\S+', '', text).strip()
        text = re.sub(r'fb\S+', '', text).strip()
        text = re.sub(r'com\S+', '', text).strip()
        text = re.sub(r'https\S+', '', text).strip()
        return text
    # drop email
    def remove_email(text):
        # \S kí tự không phải khoảng trắng
        # \S* kí tự không phải khoảng trắng xuất hiện 0 hoặc nhiều lần
        # \s? có thể có hoặc không có khoảng trắng cuối chuỗi
        return re.sub(r'\S*@\S*\s?', '', text).strip()

    data = papers['paper_text_processed']
    data = data.apply(lambda x: remove_whitespace(x))
    data = data.apply(lambda x: remove_emoj(x))
    data = data.apply(lambda x: remove_email(x))
    data = data.apply(lambda x: remove_url(x)) 
    data = data.apply(lambda x: remove_number(x))
    # data = data.apply(lambda x: remove_punctuation_underscore(x))
    data = data.apply(lambda x: word_to_phrase(x))
    data = data.apply(lambda x: remove_(x))
    data = data.apply(lambda x: remove_stopwords(x))
    text = data[0]
    print(text)
    # print(remove_stopwords(text))
    docs = data.tolist()
    
    topics1 = []
    # ------------------ train ------------------
    topic_model = BERTopic.load(f'./models/model-2023-11-22 16:15:42.469136.pickle')
    topics1.extend(topic_model.topics_)
    
    topic_model = topic_model.partial_fit(docs)
    
    # ------------------ save topics ------------------
    topics1.extend(topic_model.topics_)
    topic_model.topics_ = topics1
    
    # ------------------ save topics ------------------
    docs1 = load_pickle_list('./models/data/docs.pkl')
    timestamp_1 = load_pickle_list('./models/data/timestamp.pkl')
    classes_1 = load_pickle_list('./models/data/classes.pkl')
    docs1.extend(docs)
    timestamp_1.extend(timestamp_)
    classes_1.extend(classes)
    
    # ------------------ Topic over time ------------------
    
    topics_over_time = topic_model.topics_over_time(docs1, timestamp_1, datetime_format="%Y-%m-%d")
    print(topics_over_time)
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10,title="Các chủ đề theo thời gian")
    fig.write_html("./media/results/bert_topic_visualize_topics_over_time.html")
    
    # ------------------ Topic per class ------------------
    topics_per_class = topic_model.topics_per_class(docs1, classes=classes_1)
    fig = topic_model.visualize_topics_per_class(topics_per_class,title="Các bài báo theo chủ đề")
    fig.write_html("./media/results/bert_topic_visualize_topics_per_class.html")
    # count
    color_green = '#0AF712'
    color_red='#FA4D43'
    color_blue ='#3D50FA'
    x = np.arange(topic_model.nr_topics)
    width = 0.35 # the width of the bars
    count_values = topics_per_class['Name'].value_counts().sort_index()
    print(count_values)
    fig, ax = plt.subplots(figsize=(20, 7))
    rect_docs = ax.bar(x, count_values.values,width, color=color_blue,label='Số lượng bài báo')
    ax.set_xticks(x)
    ax.set_xticklabels(count_values.index, rotation=45)# , rotation=45
    ax.set_xlabel("Chủ đề", fontsize=14)
    ax.set_ylabel("Số lượng",fontsize=14)
    ax.set_title("Thống kê số lượng theo chủ đề",fontsize=16)

    ax.legend()
    def autolabel(rects,ax):
        for rect in rects:
            height = rect.get_height() # get height of bar
            # ax.annotate -> add text for bar
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height), # x,y location of text
                        xytext=(0, 3),  # 3 points vertical offset # text location <-> height
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rect_docs,ax)
    plt.savefig('./media/results/bert_topic_visualize_topics_per_class.png',bbox_inches='tight',pad_inches=0,dpi=300)
    # ------------------ save data ------------------
    # save data
    # save_pickle(docs1,'./models/data/docs.pkl')
    # save_pickle(timestamp_1,'./models/data/timestamp.pkl')
    # save_pickle(classes_1,'./models/data/classes.pkl')
    
    # save model
    # topic_model.save(f"./models/model-2023-11-22 16:15:42.469136.pickle",serialization="pickle")
    
    # ------------------ save topic per doc ------------------
    df = pd.DataFrame()
    topic_per_doc = topic_model.topics_
    df['topic'] = topic_per_doc
    df['timestamp'] = timestamp_1
    df['title'] = classes_1
    df['docs'] = docs1
    df.to_csv('./media/results/data_topics_per_doc.csv',index=False)
    
process_bert('./data_test/vnexpress-thoisu-ner1.csv')