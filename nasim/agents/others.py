import pandas as pd
# excel文件的写入
def save_data(datas,name):
    name_attribute=['reward']
    test=pd.DataFrame(columns=name_attribute,data=datas)#数据有三列，列名分别为one,two,three
    #print(test)
    test.to_csv(name,encoding='utf-8')

    