from albertk import *
import pprint
model,tokenizer=load_albert("data/albert_tiny")


keyword=input("输入关键词：")
# keyword="边境牧羊犬智商"
klist=run_search_sent(keyword,tokenizer,model,20)
pprint.pprint(klist)