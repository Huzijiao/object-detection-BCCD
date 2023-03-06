import os
'''查看文件夹下有多少个文件'''
path =  "data/crop_img/Platelets/"
#Platelets
count = 0
for root, dirs, files in os.walk(path):#用walk()函数遍历目录下所有的文件
    for name in files:
        count = count+1
print(count)
#调用方法
