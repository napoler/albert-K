 

from albertk import *







 


def run():

    tt=tkitText.Text()
    text_list=[]

    text=''
    kw = input("输入关键词:")
    for it in search_sent(kw):

        # text=text+"。"+it.content
        if len(it.content)>5:
            text_list.append(it.content)

    # for it in search_content(kw):
        # text=text+it.title+"。"+it.content
    # text_list=text_list+tt.sentence_segmentation_v1(text)
    mjson=tkitFile.Json("data/marked.json")
    # print(text_list)

    c_list=read_labels()
    data=[]

    for it in text_list:
        print("##"*29)
        pprint.pprint(c_list)
        print("句子：",it)

        new_text_list=[it]

        # print(new_text_list,marked_text,marked_label)
        # exit()
        try:
            marked_text,marked_label=bulid_t_data(mjson)
        except:
            pass
        # print("cd",len(c_list))
        # print(marked_label[:2])
        # print(marked_text[-2:])
        # print(len(marked_text),len(marked_label))
        # pre=auto_train(new_text_list,marked_text,marked_label,n_neighbors=len(c_list))
        try:


            pre=auto_train(new_text_list,marked_text,marked_label,n_neighbors=len(c_list))

            # print("预测：",pre)
            for i,p in enumerate( pre):
                # print(type(p))
                # print("句子：",new_text_list[i])
                
                try:
                    print("预测结果:",p,c_list[str(p)])
                    # print("预测结果:",p,c_list[int(p)])
                except:
                    pass
        except:
            pass
        c = input("输入对应标签(新建输入n):")
        if c=="n":
                n= input("输入新建标签:")
                c_list[len(c_list)]=n
                save_labels(c_list)
                one={"label":len(c_list)-1,'sentence':it}
                print(one)
                mjson.save([one]) 
        else:
            try:
                # c=int(c)
                # print(c)
                if c_list.get(str(c)):
                    one={"label":int(c),'sentence':it}
                    print(one)
                    mjson.save([one])
                # elif int(c)>=0:
                #     c_l= input("11输入新建标签:")
                #     if len(c_l)>0:
                #         c_list[len(c_list)]=c_l
                #         save_labels(c_list)
                #         one={"label":int(c),'sentence':it}
                #         mjson.save([one])
                # data.append
            except:
                pass




while True:
    print("###"*20)
    run()