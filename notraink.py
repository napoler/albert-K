from albertk import *
from tqdm import tqdm
fname='data/train.txt'
fname='data/train_mini.txt'
 
tt=tkitText.Text()
text_list=[]


# text="""

# 英国女王伊丽莎白二世对柯基犬情有独钟， 72年间亲自喂养了30多只柯基犬。伊丽莎白女王打算出资30万英镑，为35只去世的爱犬建立一个纪念堂，为每只狗树碑立传。



# 伊丽莎白二世与柯基犬在一起

# 话说，我们都知道，
# 眼鳉 [SEP]  [CLS] 杰瑞拟四眼鳉杰瑞拟四眼鳉，为辐鳍鱼纲鲤齿目𫚥鳉亚目溪鳉科的其中一种，为热带淡水鱼，分布于南美洲巴西北部Jari河下游流域，体长可达3.1公分，栖息在底中层水域，生活习性不明。 [SEP] [PT] 杰瑞拟四眼鳉 [SEP] 

#  [TT] 日本树蛙 [SEP]  [CLS] 日本树蛙日本树蛙（学名：'）又称日本溪树蛙，日本河鹿树蛙，温泉蛙，是树蛙科溪树蛙属的一个不会爬树的树蛙物种，体型较其他树蛙小，而雌性的日本树蛙比雄性树蛙稍大，成年的雄性长2.5至3公分，雌性长3至4公分，而其头部的宽度约相等于全身的宽度。牠们身体背面有一些小颗粒突起，表面极其粗糙，牠们其中一个明显的特征是其背部的中央近肩胛的地方有一对短棒状突起。日本树蛙的身体颜色会随著环境而变化，一般有三种不同色系的变色方式，即铅灰色、淡褐色或黄褐色三种色系。日本树蛙并非日本的特有种，牠们也能在日本周边地区见到。最初，牠们的命名标本是采集自日本，因此牠们被命名为「日本树蛙」，但其实牠们仅分布于台湾及日本琉球群岛。日本树蛙的蝌蚪一般呈浅灰色，身体呈椭圆形，尾巴幼而长，其长度达其身躯的两倍以上，也有一些较深色的横蚊。在蝌蚪期之后，日本树蛙踏入成年期，这时，一只成年的雄性日本树蛙可长达3公分，而雌性一般比雄性稍大一点，可长达4公分。牠们的吻端钝而圆，头部的长度约相等于头部的宽度，上下唇均有呈黑色的横带。牠们的背部可随环境而变为铅灰色、淡褐色或黄褐色。牠们的双眼
# 之间有一道深色的横带，背部的花纹可呈X型或H型，其背部的中央近肩胛的地方有一对短棒状突起。牠们身体背面有一些小颗粒突起，表面极其粗糙。牠们的体侧呈灰黑色或深棕色，而腹部则呈白色或淡黄色，其表面光滑，但也跟背部一样有一些小颗粒突起。牠们的前腿及后腿均带有深色的横带，而前腿无蹼，后腿有蹼。雄性的日本树蛙有单一咽下外鸣囊。雌性日本树蛙在交配后会把卵产进温泉高达摄氏40至50度的水中。这些卵会一直沉下水底，并黏在水底的植物上，直到卵中的蝌蚪在适当的时机破卵而出。蝌蚪在孵化后就会开始在水底进行觅食，牠们会大量进食水中的藻类和碎屑。直到发育中期，牠们会长出后腿，而牠们会继续觅食，直到发育末期时，牠们会长出前腿，并一直把之藏在胸前的透明袋中，直到牠们接近变态的时候才会真正伸出来。在前腿伸出来后，这双前腿就会挡住了牠们的鳃，令牠们再也不能使用鳃来呼吸，而是改用肺部和皮肤来呼吸。牠们的骨骼和身体中的消化系统亦在此时开始转变，牠们的身体结构逐渐相似于一只成年的日本树蛙。但在这些变化完全发育完成之前，牠们是无法摄食的。在这情况下，牠们的尾部会随著发育的时间而萎缩，萎缩时分解出来的养份就可以提供发育中的日本树

# 还有节目最后的“大碗宽面”，这个时候应景给大家做了碗宽面，因为吴亦凡有首歌就是叫做“大碗宽面”，所以直接就让吴亦凡上手给大家演示了一遍把面下锅，最后出锅的时候，黄磊又是浇上了一层烧开的油，浇上去还“滋滋”作响。

# """



# text_list_no=text_list+tt.sentence_segmentation_v1(text)

# text_list= ["20世纪初，美国阿拉斯加开始引入了西伯利亚雪撬犬。","像所有的狗一样，西伯利亚雪撬犬的鼻子通常都是凉且潮湿的。","品种标准",'鼻子','尾巴','1938年，美国哈士奇犬俱乐部成立。']
# labels=[1,2,3,3,3,1]+[-1]*len(text_list_no)
# # 训练
# labels=np.array(labels)

# print(len(text_list),text_list)

# text_list=text_list+text_list_no



# tt=tkitText.Text()

# kw = input("输入关键词:")
# text=''
# i=0
# for it in search_content(kw):
#     i=i+1
#     # print(i)
#     text=text+it.title+"。"+it.content
# text_list=tt.sentence_segmentation_v1(text)










# text=''
# i=0
# import pymongo
# client = pymongo.MongoClient("localhost", 27017)
# DB = client.gpt2Write
# print(DB.name)
# text_list=[]
# for it in DB.content_pet.find({}):
#     # print(it)
#     # text=text+it['title']+"。"+it['content']

#     tx='[esp]'.join(tt.sentence_segmentation_v1(it['content'][:300]))
#     text_list.append(tx)
#     i=i+1
#     if i==10000:
#         break

# # text=''
# # i=0
# # import pymongo
# # client = pymongo.MongoClient("localhost", 27017)
# # DB = client.gpt2Write
# # print(DB.name)

# # for it in DB.content_pet.find({}):
# #     # print(it)
# #     text=text+it['title']+"。"+it['content']
# #     i=i+1
# #     if i==100:
# #         break
# #     # its.append(it)   
# # # exit()
# # text_list=tt.sentence_segmentation_v1(text)
# # # print(text_list)




# presentence_embedding,text_list,labels=get_embedding_np(text_list,[],tokenizer,model)



# def find_k(presentence_embedding,max=10):
#     """
#     选择拐点
#     https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
    
#     """
#     from sklearn.metrics import silhouette_score
#     X=presentence_embedding
#     p_y=[]
#     p_x=[]
#     for n_cluster in range(2, max):
#         kmeans = KMeans(n_clusters=n_cluster).fit(X)
#         label = kmeans.labels_
#         sil_coeff = silhouette_score(X, label, metric='euclidean')
#         print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
#         p_y.append(sil_coeff)
#         p_x.append(n_cluster)


#     plt.figure()
#     plt.plot(p_x, p_y)
#     plt.xlabel("k ")
#     plt.ylabel("SSE")
#     plt.show()
# # find_k(presentence_embedding,len(presentence_embedding)-2)





limit=30000
# pre_json=tkitFile.Json("data/pre_train.json")
presentence_embedding=[]
text_list=[]
labels=[]
for it in tqdm(KDB.pre_train.find({}).limit(limit)):
    text_list.append(it['sent'])
    presentence_embedding.append(it['embedding'])






find_k(presentence_embedding,20)










# def bulid_pre_dict(text_list,output_labels):
#     """
#     将分类结果包装成词典
#     """
#     klist={}
#     for i,c in enumerate (output_labels):
#         # print(i,c,text_list[i])
#         if klist.get(c):
#             klist[c].append(text_list[i])
#         else:
#             klist[c]=[text_list[i]]
#     return klist
# klist=bulid_pre_dict(text_list,output_labels)
# pprint.pprint(klist)





# label_spread.predict( X)

# #绘图

# x=presentence_embedding
# # plot
# plt.figure(figsize=(4, 3), dpi=160)
# plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
# # plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
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