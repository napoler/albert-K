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
import pprint

device="cpu"
tokenizer = BertTokenizer.from_pretrained('data/albert_tiny/vocab.txt')
# print(tokenizer)
config = AlbertConfig.from_pretrained('data/albert_tiny')
model = AlbertModel.from_pretrained('data/albert_tiny',config=config)
# text_list=["你好吗",'我很不错',"哈哈",'我喜欢吃肉','我喜欢猪肉',"哈哈",'我喜欢吃肉']
 

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

 

output_dir='./model'
# torch.save(cluster_centers, os.path.join(output_dir, 'Kmeanpytroch_model.bin'))


cluster_centers=torch.load(os.path.join(output_dir, 'Kmean.bin'))






text="""

英国女王伊丽莎白二世对柯基犬情有独钟， 72年间亲自喂养了30多只柯基犬。伊丽莎白女王打算出资30万英镑，为35只去世的爱犬建立一个纪念堂，为每只狗树碑立传。



伊丽莎白二世与柯基犬在一起

话说，我们都知道，

柯基一直是英国皇室的标志之一......

女王爱柯基是出了名的，

不少重要场合，

她都会把柯基带在身边。
布偶猫体态舒展，被毛丰厚，一双蓝色的大眼睛不知道夺去了多少爱猫人士的心，是当之无愧的猫界“颜值担当”。然而，虽然有一颗养猫的心在蠢蠢欲动，但却不是每个人都适合养布偶猫的，下面我们就来看看哪些人不适合养布偶猫。


第一种：工作或学习繁忙，空闲时间不多的人。布偶猫性格温顺，对人友好。与别的猫咪的不同的是布偶猫异常黏人，她们喜欢和主人待在一起，像小狗一样讨主人欢心。她们没有足够的安全感，需要时时刻刻待在主人身边，如果你长时间不在，布偶猫会抑郁生病。所以没有时间陪伴猫咪的上班族建议不要选布偶猫。


第二种：粗心大意、怕麻烦的“大老粗”。布偶猫体质不好，肠胃敏感，幼猫对食物和水非常挑剔。不注意饮食会经常拉稀，铲屎官们做好为布偶猫擦屁股的打算了吗？你如果很怕麻烦建议不要选布偶猫。补充一点，水最好选择纯净水。


第三种：喜欢室外散养宠物的人。布偶猫胆小喜静，忍耐性强。就算受伤也不会表现出来，她们只会默默地忍受。如果放到室外与别的动物打架或意外受伤，一般主人不会察觉。喜欢室外散养的朋友们建议不要选布偶猫。

小编也跟风刷一波，觉得部分确实能够吸引人观看，例如不时地插入黄磊做饭的情节，

真的让人有忍不住流口水的冲动，其次看黄磊做饭一个直观感受就是“油腻”，黄磊在这期节目中做了两次面条，第一个是女嘉宾们刚来不久，早上都没怎么吃，都在吵闹着比较饿，这个时候黄磊给大家做了碗猪油拌面，锅里放了特别多的猪油，后来拌面的时候也浇上一层，这么浓厚的油水，能不香吗？


主人想知道，在有太阳的天气里，家里的一群猫是怎么过的，于是架起摄像机，拍下了下边的画面。

一群猫来来去去，都蹲在有太阳的地方。


随着太阳的移动，猫也跟着移动。


当只剩下最后一点微弱的阳光后，一群猫挤在一起，一边相互取暖，一边争取晒到最后一点太阳。


看上去就是猫猫聚众晒太阳的场景，可当细心的网友看完后，发现了其中一晃而过的亮点。

这是完整视频，你们来找找。

没找到吗？没找到来看看截出来的画面。

右下边的两只猫猫，不知道在做什么奇怪的游戏。


当然游戏不游戏什么的不重要，对猫来说最重要的事还是晒太阳。

一只超喜欢晒太阳的狸花猫

ins博主@taruchoro家里有一只狸花猫，对晒太阳十分痴迷。

只要太阳一出来，

它就会跑到门口来晒。


一开始趴着晒，

让阳光充足的晒到背部。



还有节目最后的“大碗宽面”，这个时候应景给大家做了碗宽面，因为吴亦凡有首歌就是叫做“大碗宽面”，所以直接就让吴亦凡上手给大家演示了一遍把面下锅，最后出锅的时候，黄磊又是浇上了一层烧开的油，浇上去还“滋滋”作响。

"""

tt=tkitText.Text()
text_list=tt.sentence_segmentation_v1(text)
test_embedding=get_embedding(text_list,tokenizer,model)
# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    test_embedding, cluster_centers, 'euclidean', device=device
)

print("cluster_ids_y",cluster_ids_y)


klist={}

for i,c in enumerate (cluster_ids_y.tolist()):
    # print(i,c,text_list[i])
    if klist.get(c):
        klist[c].append(text_list[i])
    else:
        klist[c]=[text_list[i]]
pprint.pprint(klist)






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