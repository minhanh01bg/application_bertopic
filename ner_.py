import pandas as pd

# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# print(len(docs))
def ner_(path):
    name_file = path.split('/')[-1].split('.')[0]
    papers = pd.read_csv(path)
    papers['text'].astype(str)
    # Print head
    print(papers.head())
    print(papers.shape)
    # ---------------------------------
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
    # --------------------------------- process ner ---------------------------------
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline
    from underthesea import word_tokenize, sent_tokenize
    from underthesea import ner
    import torch

    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer)

    def ner_results(text) -> dict[str, list[str]]: # dict[str, list[str]]: key là tên thực thể, value là list các từ thuộc thực thể đó
        sentences = sent_tokenize(text)
        # print(sentences)
        sentences = [s for s in sentences if len(s) > 0]

        list_ner = []
        results_sentences = []

        for s in sentences:
            if len(s) > 0:
                tensor_input = torch.tensor([tokenizer.encode(s)])
                # print(tensor_input.shape)
                if(tensor_input.shape[1] < 512):
                    # print(preprocess(s))
                    # print(nlp(s))
                    if len(nlp(s)) > 0:
                        word = ""
                        original_word = ""
                        # s = preprocess(s)
                        for i in nlp(s):
                            if 'B-' in i['entity']:
                                word = i['word']
                                original_word = i['word']
                            elif 'I-' in i['entity']:
                                original_word += " " + i['word']
                                if i['word'] != "-":
                                    word += " " + i['word']
                            elif 'E-' in i['entity']:
                                word += " " + i['word']
                                original_word += " " + i['word']
                                list_ner.append({'entity': i['entity'].split('-')[1], 'word': word})
                                word = ""
                                original_word = ""
                            elif 'S-' in i['entity']:
                                list_ner.append({'entity': i['entity'].split('-')[1], 'word': i['word']})
                                word = ""
                                original_word = ""
                        if word != "":
                            list_ner.append({'entity': i['entity'].split('-')[1], 'word': word})
                            # print(word)
                            s = s.replace(original_word, "_".join(word.split(" ")).lower())
                            # print(s)
                            results_sentences.append(s)
                            word = ""
                            original_word = ""
                    #
                    list_ner += nlp(s)
                    # print(nlp(preprocess(s)))
                    # print("=========================================")
                else:
                    print("sentence too long")

        # print(list_ner)
        dict_ner = {}
        for i in list_ner:
            if i['entity'] in dict_ner:
                dict_ner[i['entity']].append(i['word'])
            else:
                dict_ner[i['entity']] = [i['word']]

        results = {}
        for key, value in dict_ner.items():
            if '-' not in key:
                # print(key + ": " + str(value))
                results[key] = value
        
        print(results_sentences)
        return results, " ".join(results_sentences)

    ner_results_location_ = []
    ner_results_person_ = []
    ner_results_organization_ = []
    doc = []
    for i in range(len(papers['text'])):
        results, doc_ = ner_results(papers['text'][i])
        dict_ner = results
        doc.append(doc_)
        print(i)
        if 'LOCATION' not in dict_ner:
            dict_ner['LOCATION'] = []
        if 'PERSON' not in dict_ner:
            dict_ner['PERSON'] = []
        if 'ORGANIZATION' not in dict_ner:
            dict_ner['ORGANIZATION'] = []    
        ner_results_location_.append(dict_ner['LOCATION'])
        ner_results_person_.append(dict_ner['PERSON'])
        ner_results_organization_.append(dict_ner['ORGANIZATION'])
        # print(dict_ner)
        print(doc_)
        
    papers['location'] = ner_results_location_
    papers['person'] = ner_results_person_
    papers['organization'] = ner_results_organization_
    papers['doc_ner'] = doc
    papers.to_csv(path, index=False)
