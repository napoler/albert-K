from albertk import *
import numpy as np
import torch
import tkitFile
from tqdm import tqdm
import tkitFile
from harvesttext import HarvestText
import numpy as np
 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F     # 激励函数都在这

fname='data/train.txt'
fname='data/train_mini.txt'
 
tt=tkitText.Text()
# limit='null'


def run_text():
    """
    进行文章级别的聚类 
    """
    limit=3000
    # pre_json=tkitFile.Json("data/pre_train.json")
    # presentence_embedding=[]
    text_list=[]
    labels=[]
    for it in tqdm(DB.content_pet.find({}).limit(limit)):
        # print(it)
        text_list.append(it['content'][:300])
    # print(text_list)
    presentence_embedding,text_list,labels=get_embedding(text_list,[],tokenizer,model)

    num_clusters=100

    cluster_ids_x, cluster_centers = kmeans(
        X=presentence_embedding, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu'),tol=1e-8
    )
    # print('cluster_ids_x',cluster_ids_x)
    # print("cluster_centers",cluster_centers)

    output_dir='./model'
    torch.save(cluster_centers, os.path.join(output_dir, 'Kmean_text.bin'))
    

    klist={}

    for i,c in enumerate (cluster_ids_x.tolist()):
        # print(i,c,text_list)
        if klist.get(c):
            klist[c].append(text_list[i])
        else:
            klist[c]=[text_list[i]]
    pprint.pprint(klist)
    #绘图

    x=presentence_embedding
    # plot
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
    # plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.axis([-1, 1, -1, 1])
    plt.tight_layout()
    plt.show()
# run_text()



# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# print(x)
# print(y)
# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()





def get_data(f='data/data.json'):
    data_json=tkitFile.Json(f)
    for it in data_json.auto_load():
        # print(it['num'])
        if type(it['num'])==int:
            pass
        elif  it['num'].endswith("万"):
            it['num']=it['num'].replace("万",'')
            it['num']=float(it['num'])*10000
        it['num']=int(it['num'])/10000
        # print(it['title'])
        # print(it['num'])
        yield it

def using_typed_words():
    from harvesttext.resources import get_qh_typed_words,get_baidu_stopwords
    ht0 = HarvestText()
    typed_words, stopwords = get_qh_typed_words(), get_baidu_stopwords()
    ht0.add_typed_words(typed_words)
    # sentence = "THUOCL是自然语言处理的一套中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。"
    # print(sentence)
    # print(ht0.posseg(sentence,stopwords=stopwords))
    return ht0,stopwords
# using_typed_words()

# tol={}
# ht = HarvestText()
# ht,stopwords=using_typed_words()
# for it in get_data():
#     # print(it)
#     # print(ht.seg(it['title']))    # return_sent=False时，则返回词语列表
#     # for word in list(set(ht.seg(it['title']))):
#     for word,flag in list(set(ht.posseg(it['title'],standard_name=True,stopwords=stopwords))):
 
#         if tol.get(word) ==None:
#             tol[word]={"num":int(it['num']),'occ':1,'flag':flag}
#         else:
#             pre=tol.get(word)
#             tol[word]={"num":int(it['num'])+int(pre['num']),'occ':1+int(pre['occ']),'flag':flag}



def load_data():
    data=[]
    text_list=[]
    y=[]
    i=0
    for it in get_data():
        # print(it)
        data.append(it)
        text_list.append(it['title'][:30])
        y.append(it['num'])
        
        # if i>30000:
        #     break
        i=i+1
    # print(data[:2])

    x,text_list,labels=get_embedding(text_list,y,tokenizer,model)
    y=[[u] for u in labels ]
    y= torch.tensor(y,dtype=torch.float64) 


    # print(x,text_list,labels)
    y=y.float()
    x=x.float()
    return x,y

# x,y=load_data()
# torch.save(x,"x.model")
# torch.save(y,"y.model")

x=torch.load('x.model')
y=torch.load('y.model')








# plt.ion()   # 画图
# plt.show()


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        # self.hidden1 = torch.nn.Linear(n_hidden, 200)   # 输出层线性输出
        # self.predict = torch.nn.Linear(200, n_output)   # 输出层线性输出
        n_hidden_1=1024
        n_hidden_2=4096
        n_hidden_3=1024
        n_hidden_4=4096
        #  torch.nn.Dropout(0.5),  # drop 50% of the neuron
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), torch.nn.Dropout(0.1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),  torch.nn.Dropout(0.1), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),  torch.nn.Dropout(0.1), nn.ReLU(True))
        # self.layer5 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),  torch.nn.Dropout(0.01), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        # print("x",x.long)
        # a=self.hidden(x)
        # print(a)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer5(x)
        x = self.layer4(x)
        return x



net = Net(in_dim=312,out_dim=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# optimizer 是训练的工具
# optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
optimizer = torch.optim.Adam(net.parameters(), lr=2e-6)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)


# x=x.float()
# print(x.size())
# print(y)
# print(y.size())
try:
    net=torch.load("net.model")
except:
    print("new")

try:
    optimizer=torch.load("optimizer.model")
except:
    print("new")

net.train()

print(optimizer)
print(x[:1])
print(y[:1])
for t in range(10000000):
    # print(optimizer)
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    print(prediction[0])
    print(y[0])

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    print(loss.item())
    # 接着上面来
    # if t % 5 == 0:
    #     # plot and show learning process
    #     plt.cla()
    #     plt.scatter(x.numpy(), y.numpy())
    #     plt.plot(x.numpy(), prediction.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.numpy(), fontdict={'size': 20, 'color':  'red'})
    #     plt.pause(0.1)
    # prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    if t%5==0:
        torch.save(net,"net.model")
        torch.save(optimizer,"optimizer.model")
# net.eval()