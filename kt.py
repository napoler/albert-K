import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans,kmeans_predict
import os

from transformers import AlbertModel, BertTokenizer,AlbertConfig
import torch
import torch.nn as nn
import tkitText
from albert_kmeans import *
device="cpu"
tokenizer = BertTokenizer.from_pretrained('data/albert_tiny/vocab.txt')
# print(tokenizer)
config = AlbertConfig.from_pretrained('data/albert_tiny')
model = AlbertModel.from_pretrained('data/albert_tiny',config=config)
# text_list=["你好吗",'我很不错',"哈哈",'我喜欢吃肉','我喜欢猪肉',"哈哈",'我喜欢吃肉']
text="""

威尔士柯基犬（welsh corgi pembroke）是一种小型犬，它们的胆子很大，也相当机警，能高度警惕地守护家园，是最受欢迎的小型护卫犬之一。 [1] 
TA说

腿短，我是认真的——柯基的故事2018-09-05 15:15
有这样一种宠物狗，当看第一眼的时候，一定有三个字从你脑海跳出——“小短腿”。时下它已变得非常出名，甚至可以说已经成为一种网红品种。它就鼎鼎大名的“电动肥臀”——柯基犬。...详情
内容来自
中文学名威尔士柯基犬别    称卡迪根威尔士柯基犬、彭布罗克威尔士柯基犬界动物界门脊索动物门 Chordata亚    门脊椎动物亚门纲哺乳纲Mammalia亚    纲真兽亚纲目食肉目亚    目裂脚亚目科犬科亚    科犬亚科属犬属亚    属狗亚属种威尔士柯基犬亚    种家犬英文名welsh corgi pembroke犬 种牧牛犬寿 命12-15年身    高25cm－30cm体    重10kg－12kg市场参考价1000-10000元


"""
# li=torch.tensor([])  # 现有list时尽量用这种方式


def get_embedding(text_list,tokenizer,model):
    """
    获取文本特征
    """
    # text_list=["你好吗",'我很不错']
    # li=torch.tensor([])  # 现有list时尽量用这种方式

    for i,text in enumerate( text_list):
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        x=outputs[0].double()
        # print('x',x)
        sentence_embedding = torch.mean(x, 1)
        # print(sentence_embedding)
        if i!=0:
            # presentence_embedding=torch.cat((sentence_embedding, sentence_embedding), 0)	# 在 0 维(纵向)进行拼接

            # presentence_embedding=presentence_embedding+sentence_embedding.detach().numpy()
            presentence_embedding=np.concatenate((presentence_embedding,sentence_embedding.detach().numpy()),axis=0)
            
        else:
            presentence_embedding=sentence_embedding.detach().numpy()

    presentence_embedding = torch.from_numpy(presentence_embedding)   #为numpy类型
    # print( "presentence_embedding",presentence_embedding.size())
    return presentence_embedding


# 训练

tt=tkitText.Text()

text_list=tt.sentence_segmentation_v1(text)
presentence_embedding=get_embedding(text_list,tokenizer,model)
num_clusters=10
# print('x',x )
# # # kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=presentence_embedding, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu'),tol=1e-8
)
print('cluster_ids_x',cluster_ids_x)
print("cluster_centers",cluster_centers)


 

output_dir='./'
# torch.save(cluster_centers, os.path.join(output_dir, 'Kmeanpytroch_model.bin'))
 

cluster_centers=torch.load(os.path.join(output_dir, 'Kmeanpytroch_model.bin'))






text="""

英国女王伊丽莎白二世对柯基犬情有独钟， 72年间亲自喂养了30多只柯基犬。伊丽莎白女王打算出资30万英镑，为35只去世的爱犬建立一个纪念堂，为每只狗树碑立传。



伊丽莎白二世与柯基犬在一起

话说，我们都知道，

柯基一直是英国皇室的标志之一......

女王爱柯基是出了名的，

不少重要场合，

她都会把柯基带在身边。
"""

tt=tkitText.Text()
text_list=tt.sentence_segmentation_v1(text)
test_embedding=get_embedding(text_list,tokenizer,model)
# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    test_embedding, cluster_centers, 'euclidean', device=device
)

print("cluster_ids_y",cluster_ids_y)

for i,c in enumerate (cluster_ids_y):
    print(i,c,text_list[i])
    







# #         p=torch.cat(inputs=(p, x), dimension=1)
#     else:
#         p=x
# print(p)
# print("outputs",outputs)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states)
# dropout_ratio=0.1
# rnn_layers=10
# hidden_dim=10
# embedding_dim=10
# lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)

# # set device
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')

# tagset_size=2
# albert_path='data/albert_tiny'
# albert_embedding=312
# rnn_hidden=200
# dropout_ratio=0.1
# use_cuda = False
# dropout1 = 0.2
# rnn_layer=2
# model = ALBERT_KMEAMS(albert_path, tagset_size, albert_embedding, rnn_hidden, rnn_layer, dropout_ratio=dropout_ratio, dropout1=dropout1, use_cuda=use_cuda)
# # print(model)
# a=model(input_ids)
# print(a)
# print(a.double()[0] )
# # # data
# data_size, dims, num_clusters = 10000, 30, 100
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)
# print('x',x )
# x=outputs[0].double()[0] 
# num_clusters=30
# print('x',x )
# # # # kmeans
# cluster_ids_x, cluster_centers = kmeans(
#     X=li, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu')
# )

# print('cluster_ids_x',cluster_ids_x)
# # print("cluster_centers",cluster_centers)

# output_dir='./'
# torch.save(cluster_centers, os.path.join(output_dir, 'pytroch_model.bin'))


# # y = np.random.randn(5, dims) / 6
# # y = torch.from_numpy(y)



# # # predict cluster ids for y
# # cluster_ids_y = kmeans_predict(
# #     y, cluster_centers, 'euclidean', device=device
# # )

# # print("cluster_ids_y",cluster_ids_y)














# plot
# plt.figure(figsize=(4, 3), dpi=160)
# plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
# plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
# plt.scatter(
#     cluster_centers[:, 0], cluster_centers[:, 1],
#     c='white',
#     alpha=0.6,
#     edgecolors='black',
#     linewidths=2
# )
# plt.axis([-1, 1, -1, 1])
# plt.tight_layout()
# plt.show()