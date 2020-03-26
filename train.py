from albertk import *



 

 
fname='data/train.txt'
fname='data/train_mini.txt'
 
tt=tkitText.Text()
text_list=[]
# with open(fname) as file:
#     for line in file:
#         # print(line)
#         if len(line)>0:
#             text_list=text_list+tt.sentence_segmentation_v1(line)

kw = input("输入关键词:")
text=''
for it in search_content(kw):
    text=text+it.title+"。"+it.content
text_list=text_list+tt.sentence_segmentation_v1(text)
# 训练


presentence_embedding=get_embedding(text_list,tokenizer,model)

find_k(presentence_embedding,10)


num_clusters=input("输入K：")
num_clusters=10
# # print('x',x )
# # # # kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=presentence_embedding, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu'),tol=1e-8
)
# print('cluster_ids_x',cluster_ids_x)
# print("cluster_centers",cluster_centers)

output_dir='./model'
torch.save(cluster_centers, os.path.join(output_dir, 'Kmean.bin'))
 

klist={}

for i,c in enumerate (cluster_ids_x.tolist()):
    # print(i,c,text_list[i])
    if klist.get(c):
        klist[c].append(text_list[i])
    else:
        klist[c]=[text_list[i]]
pprint.pprint(klist)


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