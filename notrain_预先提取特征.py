from albertk import *
import numpy as np
import tkitFile
from tqdm import tqdm
fname='data/train.txt'
fname='data/train_mini.txt'
 
tt=tkitText.Text()

# kw = input("输入关键词:")
text=''
i=0
# for it in search_content(kw):
#     i=i+1
#     # print(i)
#     text=text+it.title+"。"+it.content


text=''
i=0

# print(DB.name)
# pre_json=tkitFile.Json("data/pre_train.json")
text_list=[]
for it in tqdm(DB.content_pet.find({})):
    # print(it)
    text=text+it['title']+"。"+it['content']

    # tx='[esp]'.join(tt.sentence_segmentation_v1(it['content'][:300]))
    # text_list.append(tx)
    i=i+1
    if i%10==0:
        text_list=tt.sentence_segmentation_v1(text)
        presentence_embedding,text_list,labels=get_embedding(text_list,[],tokenizer,model)
        data=[]
        for it,t in zip(presentence_embedding.tolist(),text_list) :
            key=tt.md5(t)
            if KDB.pre_train.find_one({"_id":key}):
                pass
            else:
                one={"_id":key,'sent':t,"embedding":it}
                # data.append(one)
                try:
                    KDB.pre_train.insert_one(one)
                    # print("ojk")
                except:
                    pass
            
        # pre_json.save(data)
        text=''
