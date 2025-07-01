import torch
from torch import nn,optim
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset,DataLoader
import chardet
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import argparse

class Model(nn.Module):
    def __init__(self,dataset):
        super(Model,self).__init__()
        self.lstm_size=128
        self.embedding_dim=128
        self.num_layers=3
        self.dropout=0.2

        n_vocab=len(dataset.uniq_words)
        self.embedding=nn.Embedding(num_embedding=n_vocab,embedding_dim=self.embedding_dim)
        self.lstm=nn.LSTM(input_size=self.embedding_dim,
                          hidden_size=self.lstm_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        self.fc=nn.Linear(self.lstm_size,n_vocab)

        def forward(self,x,prev_state):
            embed=self.embedding(x)
            output,state=self.lstm(embed.prev_state)
            logits=self.fc(output)
            return logits,state#递归返回
        
        def init_state(self,sequence_length):
            return (torch.zeros(self.num_layers,sequence_length,self.lstm_size),
            torch.zeros(self.num_layers,sequence_length,self.lstm_size))
class Dataset(Dataset):
    def __init__(self,args,path):
        self.args=args
        self.path=path
        self.worldlist=self.loadword(path)
        self.uniq_words=self.get_uniq_words()
        self.index_to_words={index:word for index,word in enumerate(self.uniq_words)}
        self.words_to_index={word:index for index,word in enumerate(self.uniq_words)}
        self.words_index=[self.word_to_index[word]for word in self.wordlist]
    def loadword(self,path):
        with open(path,'rb')as f:
            raw_data=f.read()
            detected_encoding=chardet.detect(raw_data)["encoding"] 
        data=pd.read_csv(path,encoding=detected_encoding)
        data=data[:200] 
        text=data["text"].str.cat(sep="")
        text-re.sub(r"[^a-zA-Z]"," ",text)
        text=re.sub(r"[' ']"," ",text)

        text=text.split(" ")
        wordlist=text
        print(wordlist[:10])
        return wordlist
    
    def get_uniq_words(self):
        word_counts=Counter(self.wordlist)
        return sorted(word_counts,key=word_counts.get,reverse=True)
    
    def __len__(self):
        return len(self.words_index)-self.args.sequence_length
    
    def __getitem(self,index):
        return (
            torch.tensor(self.words_index[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexs[index+1:index+1+self.args.sequence_length])
               )
class Train:
    def __init__(self,dataset,model,args):
        model.trian()
        dataloader=DataLoader(dataset,batch_size=args.batch_size)
        criterion=nn.CrossEntropyLoss()
        optim=optim.SGD(model.parameters(),lr=0.001)
        for epoch in range(args.max_epochs):
            state_h,state_c=model.init_state(args.sequence_length)
            for batch,(x,y)in enumerate(dataloader):
                optim.zero_grad()
                y_pred,(state_h,state_c)=model(x,(state_h,state_c))
                loss=criterion(y_pred.transpose(1,2),y)
                state_h=state_h.detach()
                state_c=state_c.detach()
                loss.backward()     #进行反向传播
                optim.step()        #使用优化器更新模型参数,最小化损失
                print({"epoch":epoch,"batch":batch,"loss":loss.item()})     
                

def predect(dataset,model,text,next_words=100):
    model.eval()

    words=text.split("")
    state_h,state_c=model.init_state(len(words))
    
    for i in range(next_words):     #next_words表示要根据已有text连续生成的单词的数量
        x=torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred,(state_h,state_c)=model(x,(state_h,state_c))
        last_word_logits=y_pred[0][-1]      #y_pred是一个三维张量,形状为(batch_size,sequence_length,vocab_size)     由于只处理一个样本,batch_size为1;sequence_length表示输入的text的长度        因此last_word_logits的形状为(vocab_size,)
        p=torch.nn.functional.softmax(last_word_logits,dim=0).detach().numpy()
        word_index=np.random.choice(len(last_word_logits),p=p)
        words.append(dataset.index_to_words[word_index])
    
    return words

parser=argparse.ArgumentParser()
parser.add_argument("--max-epochs",type=int,default=10)
parser.add_argument("--batch-size",type=int,default=256)
parser.add_argument("--sequence-length",type=int,default=2)
args=parser.parse_args()


dataset=Dataset(args,"F:\点春季学习\文本相似度\imdb_tr (1).csv")
model=Model(dataset)
train=Train(dataset,model,args)
print(predect(dataset,model,"think movie"))