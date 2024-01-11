import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import nltk
import re
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW, BertConfig
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from torch.utils.data import TensorDataset
from nltk.tokenize.toktok import ToktokTokenizer
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import f1_score
import warnings 
warnings.filterwarnings("ignore")

def build_datasets(df):
    df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

    tokenizer=ToktokTokenizer()
    nltk.download('stopwords')
    stopword_list=nltk.corpus.stopwords.words('english')


    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    def denoise_text(text):
        text = remove_between_square_brackets(text)
        return text
    df['text']=df['text'].apply(denoise_text)

    def remove_special_characters(text, remove_digits=True):
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text
    df['text']=df['text'].apply(remove_special_characters)

    df['text']=df['text'].str.lower()
    df_sexist = df[['text','label_sexist']]
    le_sexist = preprocessing.LabelEncoder()
    le_sexist.fit(df_sexist['label_sexist'])
    df_sexist['label_sexist'] = le_sexist.transform(df_sexist['label_sexist'])
    df_sexist.head()

    df_category = df.loc[df['label_sexist']=='sexist'][['text','label_category']]
    le_category = preprocessing.LabelEncoder()
    le_category.fit(df_category['label_category'])
    df_category['label_category'] = le_category.transform(df_category['label_category'])
    df_category.head()

    df_vector = []
    for i in range(1,len(df_category['label_category'].unique())+1):
        df_vector.append(df.loc[df['label_sexist']=='sexist'][
                                        df['label_category'].str.startswith(str(i))][['text','label_vector']])
        # print(df_vector[-1]['label_vector'].unique())
        le_vector = preprocessing.LabelEncoder()
        le_vector.fit(df_vector[-1]['label_vector'])
        df_vector[-1]['label_vector'] = le_vector.transform(df_vector[-1]['label_vector'])
        # print(df_vector[-1]['label_vector'].unique())
    
    return df_sexist, df_category, df_vector

def get_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('GPU: ', end='')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU')

test_df = pd.read_csv('./Group_1/test.csv')
test_df.head()
test_sexist, test_category, test_vector = build_datasets(test_df)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

test_sentences = []
test_sentences.append(test_sexist['text'].values)
test_sentences.append(test_category['text'].values)
for i in range(len(test_vector)):
    test_sentences.append(test_vector[i]['text'].values)

num_labels = [2,4,2,3,4,2]
model_class = RobertaForSequenceClassification.from_pretrained(
"roberta-base", 
num_labels = num_labels[i], 
output_attentions = False,
output_hidden_states = False,
)
model_class.cuda()
tokenizer_class = RobertaTokenizer.from_pretrained('roberta-base')
loaded_tokenizer = list()
loaded_model = list()
for i in range(len(test_sentences)):
    loaded_model.append(model_class.from_pretrained('./model_saves_'+str(i)))
    loaded_tokenizer.append(tokenizer_class.from_pretrained('./model_saves_'+str(i)))
    loaded_model[i].cuda()

test_ids = []
test_attention_masks = []
test_label = []

for i in range(len(test_sentences)):
    test_ids.append([])
    test_attention_masks.append([])

    for sent in test_sentences[i]:
        encoded_dict = loaded_tokenizer[i].encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 68,          # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation = True
                        )

        # Add the encoded sentence to the list.    
        test_ids[i].append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        test_attention_masks[i].append(encoded_dict['attention_mask'])

    test_ids[i] = torch.cat(test_ids[i], dim=0)
    test_attention_masks[i] = torch.cat(test_attention_masks[i], dim=0)
    if i==0:
        test_label.append(torch.tensor(test_sexist['label_sexist'].values))
    elif i==1:
        test_label.append(torch.tensor(test_category['label_category'].values))
    else:
        test_label.append(torch.tensor(test_vector[i-2]['label_vector'].values))

test_dataset, test_size = [],[]
for i in range(len(test_sentences)):
    test_dataset.append(TensorDataset(test_ids[i], test_attention_masks[i], test_label[i]))
    test_size.append(len(test_dataset[i]))

batch_size = 32
test_dataloader = []
for i in range(len(test_sentences)):
    test_dataloader.append(DataLoader(
                test_dataset[i],
                sampler = SequentialSampler(test_dataset[i]),
                batch_size = batch_size 
            ))

print('----------------------------------------------------------------------------------------------------------')
print('Independent Task Training Results on Test Dataset. For the final model only task A classifier was used.')
print('----------------------------------------------------------------------------------------------------------')

for i in range(len(test_sentences)):
    loaded_model[i].eval()
    total_eval_accuracy = 0
    nb_eval_steps = 0
    for batch in test_dataloader[i]:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            loss, logits = loaded_model[i](b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)[0:2]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += get_f1(logits, label_ids)
    avg_test_accuracy = total_eval_accuracy / len(test_dataloader[i])
    if i == 0:
        print('Task A')
    elif i==1:
        print('Task B')
    else:
        print(f'Task C {i-1}')
    print(" Macro F1_Score: {0:.5f}".format(avg_test_accuracy))

print('----------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------')

predictions = list()
for batch in test_dataloader[0]:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            loss, logits = loaded_model[0](b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)[0:2]
        logits = logits.detach().cpu().numpy()
        predictions.extend(logits)
predictions = np.argmax(predictions, axis=1).flatten()

new_test_df = pd.DataFrame()
for i in range(len(predictions)):
    if predictions[i] == 1:
        new_test_df = new_test_df.append(test_df.iloc[[i]],ignore_index =True)
new_test_df

import pickle
pkl_file = open('/content/encoder_dict/sexist_encoder.pkl', 'rb')
le_sexist = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('/content/encoder_dict/category_encoder.pkl', 'rb')
le_category = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('/content/encoder_dict/vector_encoder.pkl', 'rb')
le_vector = pickle.load(pkl_file)
pkl_file.close()
new_test_df['label_sexist'] = le_sexist.transform(new_test_df['label_sexist'])
new_test_df['label_category'] = le_category.transform(new_test_df['label_category'])
new_test_df['label_vector'] = le_vector.transform(new_test_df['label_vector'])
pkl_file = open('/content/encoder_dict/inv_map.pkl', 'rb')
mapping_inv = pickle.load(pkl_file)
pkl_file.close()

tt = new_test_df[new_test_df.label_sexist == mapping_inv['sexist']]
new_test_data_cat = pd.DataFrame({
    'text': tt['text'].replace(r'\n', ' ', regex=True),
    'labels':tt['label_vector']
})

model_args = ClassificationArgs(num_train_epochs = 20)
model11 = ClassificationModel('roberta', './model_saves_6/roberta')

def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='macro')
result, model_outputs, wrong_predictions = model11.eval_model(new_test_data_cat,f1=f1_multiclass)

print('----------------------------------------------------------------------------------------------------------')
print('Task C Macro F1 Score: ', end="")
print(result['f1'])
print('----------------------------------------------------------------------------------------------------------')

from scipy.special import softmax
from sklearn.metrics import f1_score,accuracy_score
def truelabels(outputs):
    pred = []
    for y in outputs:
        if(y==0 or y==1):
            pred.append(0)
        elif(y==2 or y==3 or y==4):
            pred.append(1)
        elif(y==5 or y==6 or y==7 or y==8):
            pred.append(2)
        elif(y==9 or y==10):
            pred.append(3)
    return np.array(pred)
def backprop(outputs):
    pred = []
    for y1 in outputs:
        y = np.argmax(y1)
        if(y==0 or y==1):
            pred.append(0)
        elif(y==2 or y==3 or y==4):
            pred.append(1)
        elif(y==5 or y==6 or y==7 or y==8):
            pred.append(2)
        elif(y==9 or y==10):
            pred.append(3)
    return np.array(pred)

            
pred = backprop(model_outputs)
true = truelabels(new_test_data_cat['labels'])

print('----------------------------------------------------------------------------------------------------------')
print('Hierarchical Task B F1 Score: ', end='')
print(f1_score(true, pred, average='macro'))
print('----------------------------------------------------------------------------------------------------------')