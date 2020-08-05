import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
'''bath_list=['6_2','6_3','6_4','6_5','6_6','6_7','6_8','6_9']
for bath in bath_list:
    df1 = pd.DataFrame(pd.read_excel('test/true_%s.xlsx'%bath))

    df2 = pd.DataFrame(pd.read_excel('test/flase_%s.xlsx'%bath))


    id_true=list(set(df1['order_display_id'].values))
    id_flase=list(set(df2['order_display_id'].values))
    print(len(id_true),len(id_flase),len(id_true)+len(id_flase))
    df1['msg_content']=df1['msg_content'].astype("str")
    df1 = df1[True^df1['msg_content'].str.contains(r'.*?im.*')]
    df1 = df1['msg_content'].groupby(df1['order_display_id']).agg("/temp/".join).tolist()
    df2['im']=df2['im'].astype("str")
    df2 = df2[True^df2['im'].str.contains(r'.*?im.*')]
    df2 = df2['im'].groupby(df2['order_display_id']).agg("/temp/".join).tolist()

    print('zhen',len(df1),'jia',len(df2))
    x=df1+df2
    print(len(x))

    y=[0 for i in range(len(df1))]+[1 for j in range(len(df2))]

    #x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)
    #print(len(x_train),len(x_test),y_test)
    train = pd.DataFrame({'id':[i for i in range(len(x))],'im':x,'label':y})
    #test = pd.DataFrame({'id':[i for i in range(len(x_test))],'im':x_test,'label':y_test})

    train.to_csv(bath+'.csv', encoding='utf-8',index=False)
    #test.to_csv('test.csv', encoding='utf-8',index=False)'''
def undeee(a):
    b = str(a)
    b = b.replace('[', '')
    b = b.replace(']', '')
    a = list(eval(b))
    return a

img_pth=np.array([])
ocr_text=np.array([])
dfid = pd.DataFrame(pd.read_excel('OCR_id_8.3.xlsx'))
df_c = pd.DataFrame(pd.read_excel('OCR_TEST2.xlsx'))
order_display_id=dfid['order_display_id'].values
id=[]
im=dfid['msg_content'].values
img_pth_c=df_c['im_pth'].values
predict_c=df_c['predict'].values
predict_c_dl=df_c['predict_dl'].values
predict=[]
predict_dl=[]
for bath in [1,2,3,4,5,6,7,8]:
    df1 = pd.DataFrame(pd.read_excel('chineseocr_lite-OCRThread-%s.xlsx'%bath))
    img_pth=np.append(img_pth,df1['img_name'].values)
    ocr_text=np.append(ocr_text, df1['ocr内容'].values)
for i in img_pth:
    print(i[1:])
    try:
        index=np.argwhere(im==i[1:])[0][0]
        id.append(order_display_id[index])
        index_2=np.argwhere(img_pth_c==i)[0][0]
        predict.append(predict_c[index_2])
        predict_dl.append(predict_c_dl[index_2])
        print(i)
    except:
        print('none')


print(img_pth)
print(ocr_text)
df=pd.DataFrame({'img_pth':img_pth,'ocr_text':ocr_text,'predict':predict,'predict_dl':predict_dl,'order_display_id':id,'dt':["2020-08-02" for i in range(len(img_pth))]})
df.to_excel('result_OCzr——8.3.xlsx', index=False)
