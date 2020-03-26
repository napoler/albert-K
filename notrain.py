from albertk import *
import numpy as np
import torch
import tkitFile
from tqdm import tqdm
# fname='data/train.txt'
# fname='data/train_mini.txt'
 
tt=tkitText.Text()
# limit='null'


def run_sent():
    limit=30000
    # pre_json=tkitFile.Json("data/pre_train.json")
    presentence_embedding=[]
    text_list=[]
    labels=[]
    for it in tqdm(KDB.pre_train.find({}).limit(limit)):
        text_list.append(it['sent'])
        presentence_embedding.append(it['embedding'])
    
    print("总共数据：",len(presentence_embedding))
    presentence_embedding=torch.tensor(presentence_embedding, dtype=torch.long)
    # print(text_list,labels)
    num_clusters=100
    # # print('x',x )
    # # # # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=presentence_embedding, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu'),tol=1e-8
    )
    # print('cluster_ids_x',cluster_ids_x)
    # print("cluster_centers",cluster_centers)

    # output_dir='./model'
    # torch.save(cluster_centers, os.path.join(output_dir, 'Kmean.bin'))
    

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










# keyword=input("输入关键词：")
keyword="边境牧羊犬智商"
run_search_sent(keyword)





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

    # output_dir='./model'
    # torch.save(cluster_centers, os.path.join(output_dir, 'Kmean_text.bin'))
    

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