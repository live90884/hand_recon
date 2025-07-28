import numpy as np
import math
import cv2
infinity = 100000
def getCenter(x, y, img):
    shift = (__right__(x, y, img) - __left__(x, y, img))/2#需要平移的距離
    return (round(x + shift), y)
def getValue(x, y, m, img):
    width = __right__(x, y, img) + __left__(x, y, img) - 1
    if m == infinity:#直接計算
        value = width
    else:
        value = math.sin(math.atan(m)) * width
    return abs(round(value))
def filterPass(a, value):# filter mistake situation
    min1 = min2 = min3 = infinity
    start1 = start2 = start3 = 0
    end1 = end2 = end3 = 0
    for i in range(len(a)):
        if end1 == 0:
            if a[i] != 0:
                if start1 == 0:
                    start1 = i
                if a[i] <= min1:
                    min1 = a[i]
        if start1 != 0 and end1 == 0:
            if a[i] == 0:
                end1 = i - 1
        if end1 != 0 and end2 == 0:
            if a[i] != 0:
                if start2 == 0:
                    start2 = i
                if a[i] <= min2:
                    min2 = a[i]
        if start2 != 0 and end2 == 0:
            if a[i] == 0:
                end2 = i - 1
        if end2 != 0 and end3 == 0:
            if a[i] != 0:
                if start3 == 0:
                    start3 = i
                if a[i] <= min3:
                    min3 = a[i]
        if start3 != 0 and end3 == 0:
            if a[i] == 0:
                end3 = i - 1
                break
    for i in range(len(a)):
        if i >= start1 and i <= end1:
            if a[i] > min1 + value:
                a[i] = 0
        if i >= start2 and i <= end2:
            if a[i] > min2 + value:
                a[i] = 0
        if i >= start3 and i <= end3:
            if a[i] > min3 + value:
                a[i] = 0
    return a
def getPoint(x, y, m, value, direct, whether_y,img):#輸入想位移的值以及斜率即可返還點座標
    tmpx = ((value * value) / (m * m + 1)) ** 0.5
    tmpy = m * tmpx
    if direct == "Left": #右手
        if whether_y == True and m > 0:
            return (round(x - tmpx), round(y + tmpy))
        return (round(x + tmpx), round(y - tmpy))
    else: #左手
        if whether_y == True and m < 0:
            return (round(x + tmpx), round(y - tmpy))
        return (round(x - tmpx), round(y + tmpy))
def whetherPassFromYaxis(x, y, m, direct, value, img):#return(dis, first_x, first_y), dis等於0表示沒有
    ori = img[y][x]
    if m <= 1 and m >= -1:
        unit = abs(round(1/m))
        first_shift = abs(round((1/m)/2))
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    
    count = 0
    first_touch_distance = 0
    first_touch_x = 0
    first_touch_y = 0
    if direct == "Left": #Left是右手
        k = 0
        leave_edge = 1#二值化邊線可能粗度超過1像素
        for i in range(1, img.shape[1]): #right
            if i - 1 == first_shift:
                if m < 0:
                    k += 1
                else:
                    k -= 1
            if (i - 1 - first_shift) % unit == 0:
                if m < 0:
                    k += 1
                else:
                    k -= 1
            if leave_edge == 1:
                if (ori != img[y + k][x + i]).any():#記錄
                    leave_edge = 0
                    count += 1
                    cv2.circle(img, (x + i, y + k), 4, (0, 255, 0), -1)#print
                    if count == 2:
                        return (first_touch_distance, first_touch_x, first_touch_y)#紀錄該點根據x軸方向延伸後第一個碰到的邊界距離, 以及它的x, y
                    first_touch_distance = (i * i + k * k) ** 0.5 
                    first_touch_x = x + i
                    first_touch_y = y + k
            if (ori == img[y + k][x + i]).all():
                leave_edge = 1
            if (i * i + k * k) ** 0.5 >= value:#尋找完畢後count還沒等於2
                return (0, 0, 0) #表示沒找到
    else:
        k = 0
        leave_edge = 1#二值化邊線可能粗度超過1像素
        for i in range(1, img.shape[1]): #left
            if i - 1 == first_shift:
                if m < 0:
                    k -= 1
                else:
                    k += 1
            if (i - 1 - first_shift) % unit == 0:
                if m < 0:
                    k -= 1
                else:
                    k += 1
            if leave_edge == 1:
                if (ori != img[y + k][x - i]).any():#記錄
                    leave_edge = 0
                    count += 1
                    cv2.circle(img, (x - i, y + k), 4, (0, 255, 0), -1)#print
                    if count == 2:
                        return (first_touch_distance, first_touch_x, first_touch_y) #紀錄該點根據x軸方向延伸後第一個碰到的邊界距離
                    first_touch_distance = (i * i + k * k) ** 0.5
                    first_touch_x = x - i
                    first_touch_y = y + k
            if (ori == img[y + k][x - i]).all():
                leave_edge = 1
            if (i * i + k * k) ** 0.5 >= value:
                return (0, 0, 0)
    return (0, 0, 0)

def whetherPassFromXaxis(x, y, m, img):
    ori = img[y][x]
    if m <= 1 and m >= -1:
        unit = abs(round(1/m))
        first_shift = abs(round((1/m)/2))
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    k = 0
    first_touch_distance = 0
    for i in range(1, img.shape[0]): #down
        if i - 1 == first_shift:
            if m < 0:
                k += 1
            else:
                k -= 1
        if (i - 1 - first_shift) % unit == 0:
            if m < 0:
                k += 1
            else:
                k -= 1
        if (ori != img[y + i][x + k]).any():
            first_touch_distance = (i * i + k * k) ** 0.5
            return (first_touch_distance, x + i, y + k)
        if (i * i + k * k) ** 0.5 >= value:
                return (0, 0, 0)
def getCollidePoint(x, y, m, direct, img):#斜率1/10的話, 表示x移動10格y往上一格, 實際上則需要往右五格
    ori = img[y][x]
    alter = 0
    if m <= 1 and m >= -1:
        alter = 1
        unit = abs(round(1/m))
        first_shift = abs(round((1/m)/2))
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    print("m ", m)
    print("first ",first_shift)
    print("unit ", unit)
    if direct == "Left": #Left是右手
        if alter == 1:
            k = 0
            for i in range(1, img.shape[1]): #right
                print(i)
                if i - 1 == first_shift:
                    if m < 0:
                        k += 1
                    else:
                        k -= 1
                if (i - 1 - first_shift) % unit == 0:
                    if m < 0:
                        k += 1
                    else:
                        k -= 1
                if (ori != img[y + k][x + i]).any():
                    return (x + i, y + k)
    else:
        if alter == 1:
            k = 0
            for i in range(1, img.shape[1]): #left
                if i - 1 == first_shift:
                    if m < 0:
                        k -= 1
                    else:
                        k += 1
                if (i - 1 - first_shift) % unit == 0:
                    if m < 0:
                        k -= 1
                    else:
                        k += 1
                if (ori != img[y + k][x - i]).any():
                    return (x - i, y + k)
    return (0,0)
        
def __left__(x, y, img):
    ori = img[y][x]
    for i in range(1, img.shape[1]): #left
        if (ori != img[y][x-i]).any():
            return i
    return 0

def __right__(x, y, img):
    ori = img[y][x]
    for i in range(1, img.shape[1]): #right
        if (ori != img[y][x+i]).any():
            return i
    return 0