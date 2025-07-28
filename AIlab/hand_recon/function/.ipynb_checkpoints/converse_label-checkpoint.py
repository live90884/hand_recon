import csv
import numpy as np

def get_label(name):
    point = []
    label = []
    with open(name, newline='') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.DictReader(csvfile)
        k = 0
        category = 0
        for row in rows:          
            try:
                label.append(label[point.index(row['label'])])##如果原先就有資料則label = 其原先label
                k += 1
            except:
                label.append(len(point)-k)
                if len(point) - k > category:
                    category = len(point) - k
            point.append(row['label'])
    csvfile.close()
    return label, category+1
        
def getPointByLabel(label_index, name):
    point = []
    label = []
    with open(name, newline='') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.DictReader(csvfile)
        k = 0
        for row in rows:
            string = row['label']
            string = string.replace('-', ',')
            for i in range(len(string)): #to make ans index group order equal with groupby result order
                characters = "()"
                for x in range(len(characters)):
                    string = string.replace(characters[x], '')
            a = string.split(',')
            try:
                label.append(label[point.index(a)])##如果原先就有資料則label = 其原先label, 如果原先沒資料這行會出錯, 進到except
                k += 1
            except:
                label.append(len(point)-k)
            point.append(a)
    csvfile.close()
    return ((int(point[label_index][0]),int(point[label_index][1])),
            (int(point[label_index][2]),int(point[label_index][3])),
            (int(point[label_index][4]),int(point[label_index][5])))