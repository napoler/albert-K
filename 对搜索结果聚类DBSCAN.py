from albertk import *
import pprint
import json
# fname='data/train.txt'
# fname='data/train_mini.txt'
 
tt=tkitText.Text()
text_list=[]


# text="""
# 20世纪初，美国阿拉斯加开始引入了西伯利亚雪撬犬。
# 1909年，西伯利亚雪橇犬第一次在阿拉斯加的犬赛中亮相。
# 1925年1月，阿拉斯加偏僻小镇白喉流行，由于最近的存有血清的城市远在955英里以外，为快速运回治疗白喉的血清，人们决定用哈士奇雪橇队代替运送，657英里的路程按正常的运送速度来算需要25天，由于病症快速蔓延，雪橇队决定以接力运送的方式来运送，雪橇队最后仅用了5天半时间就完成了任务，挽救了无数生命 [1]  。
# 1930年，西伯利亚雪橇犬俱乐部得到了美国养犬俱乐部的正式承认，并制订了其犬种标准。
# 1938年，美国哈士奇犬俱乐部成立。从此，哈士奇犬从极地环境走进都市生活，它不但是优秀的雪橇犬，而且是出色的伴侣犬 [5]  。
# 鼻子
# 像所有的狗一样，西伯利亚雪撬犬的鼻子通常都是凉且潮湿的。 在某些情况下，西伯利亚雪撬犬（哈士奇）表现出所谓'雪鼻'或'冬鼻'的现象 [1]  。这种现象学术上被称作"hypopigmentation"，由于冬季里缺少阳光，这导致了鼻（或其一部分）褪色成棕色或粉红色，在夏季到来时便能恢复正常。雪鼻现象在其它的短毛种类里也有出现；老年犬只的这种颜色变化可能是永久的，尽管它并不与疾病相联系。
# 尾巴
# 呈三角形，毛发浓密，大小中等，一般直立尾部像毛刷一样，有着类似
# 耳朵和尾巴参考图
# 耳朵和尾巴参考图
# 于狐狸尾巴的外形，恰好位于背线之下，犬立正时尾巴通常以优美的镰刀形曲线背在背上。尾巴举起时不卷在身体的任何一侧，也不平放在背上。正常情况下，应答时犬会摇动尾巴。尾巴上的毛中等长度，上面、侧面和下面的毛长度基本一致，因此看起来很像一个圆的狐狸尾巴 [4]  。
# 被毛
# 西伯利亚雪橇犬的被毛为双层，中等长度，看上去毛很浓密。下层毛柔软，浓密，长度足以支撑外层被毛。外层毛的粗毛平直，光滑伏贴，不粗糙，不能直立。应该指出的是，换毛期没有下层被毛是正常的 [4]  。
# 品种标准
# 哈士奇最早的作用就是拉小车，仍十分擅长此项工作，它的身体比例和体形反映了力量、速度、耐力的最基本的平衡状况。雄性肌肉发达，但轮廓不粗糙；雌性充满阴柔美，但不孱弱。在正常条件下，一只肌肉结实、发育良好的哈士奇犬也不能拖曳过重的东西 [4]  。
# """

# kw = input("输入关键词:")
# text=''
# i=0
# for it in search_content(kw):
#     i=i+1
#     # print(i)
#     text=text+it.title+"。"+it.content
# text_list=tt.sentence_segmentation_v1(text)

i=0
text=''
import pymongo
client = pymongo.MongoClient("localhost", 27017)
DB = client.gpt2Write
print(DB.name)

for it in DB.content_pet.find({}):
    # print(it)
    text=text+it['title']+"。"+it['content']
    i=i+1
    if i==10000:
        break
    # its.append(it)   
# exit()
text_list=tt.sentence_segmentation_v1(text)

model,tokenizer=load_albert("data/albert_tiny")
keyword="柯基犬"
# pre=Pre_KMeans(text_list,tokenizer,model,100)
# klist=bulid_pre_dict(text_list,pre.tolist())


# klist=run_search_content_DBSCAN(keyword,tokenizer,model)




pre=auto_train_DBSCAN(text_list,tokenizer,model)
klist=bulid_pre_dict(text_list,pre.tolist())

submit = './data/klist.json'
# save_list(klist)
with open(submit, 'w') as f:
    json.dump(klist, f)



# pprint.pprint(klist)


# pre=run_search_content_sk(keyword,tokenizer,model,num_clusters=20)
# pprint.pprint(pre)
# # # for i in  pre:
# # #     print(i)
 