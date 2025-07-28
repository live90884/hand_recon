import cv2, os, csv
import mediapipe as mp
from function.label_fuc import getValue, getPoint, whetherPassFromYaxis, whetherPassFromXaxis, find_y_points, find_first_x, find_other_x
import matplotlib.pyplot as plt
#define
infinity = 10000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
K = 3 #Y軸是X軸的三倍
whether_point = 0

def exe(id, const_rate, grid_number, width_rate, alter_orix, alter_oriy, low_bound, upper_bound):
    xaxis_grid = grid_number
    yaxis_grid = K * xaxis_grid
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.5, max_num_hands = 1)
    path = 'data/' + id

    for file in os.listdir(path):
        if file == "train" or file[0] == '.':
            continue
        img = cv2.imread(os.path.join(path, file))
    
    #img = cv2.imread('test/photo/6.PNG')
    result = hands.process(img)
    ori_img = img
    #二值化描邊
    img = cv2.Canny(img, low_bound, upper_bound)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ####

    if result.multi_hand_landmarks:
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        try:
            originx, originy, xEnd_x, xEnd_y = zuobiao(ori_img, result.multi_handedness[0].classification[0].label)
        except:
            print("please try again")
            return False

    print("腕橫紋",originx, originy, "腕旁點", xEnd_x, xEnd_y)
    xdir_m = -1 * (xEnd_y - originy) / (xEnd_x - originx)
    if  xdir_m == 0:
        ydir_m = infinity
    else:
        ydir_m = -1 / xdir_m
    xdirValue = ((xEnd_x - originx) ** 2 + (xEnd_y - originy) ** 2)**0.5
    
    ydirValue = round(K * xdirValue)
    xdirValue = round(xdirValue)
    xdirEnd_x, xdirEnd_y = getPoint(originx, originy, xdir_m,  xdirValue, result.multi_handedness[0].classification[0].label, False)
    ydirEnd_x, ydirEnd_y = getPoint(originx, originy, ydir_m,  ydirValue, result.multi_handedness[0].classification[0].label, True)
    xdir_point = [[0 for _ in range(2)] for _ in range(xdirValue)]#每個點有x和y
    ydir_point = [[0 for _ in range(2)] for _ in range(ydirValue)]#每個點有x和y
    for i in range(xdirValue):
        xdir_point[i][0], xdir_point[i][1] = getPoint(originx, originy, xdir_m, i, result.multi_handedness[0].classification[0].label, False)
    for i in range(ydirValue):
        ydir_point[i][0], ydir_point[i][1] = getPoint(originx, originy, ydir_m, i, result.multi_handedness[0].classification[0].label, True)
    whethery = [0] * ydirValue #紀錄每個點所碰到的最近距離, 0表示沒有成功通過兩個點以上
    ycollid_point = [[0 for _ in range(2)] for _ in range(ydirValue)]#每個點有x和y
    for i in range(ydirValue):
        whethery[i], ycollid_point[i][0],  ycollid_point[i][1] = whetherPassFromYaxis(ydir_point[i][0], ydir_point[i][1], xdir_m, result.multi_handedness[0].classification[0].label, xdirValue, img)
        #print(xdirValue,whethery[i])
    ylabels = [0] * 3
    ylabels[0], ylabels[1], ylabels[2] = find_y_points(whethery, xdirValue/10)#8是基於人的手臂粗度的一半不可能貼超過8張
    xcollid_point = [[0 for _ in range(2)] for _ in range(xdirValue)]#每個點有x和y
    whetherx = [0] * xdirValue #紀錄每個點所碰到的最近距離, 0表示沒有成功通過兩個點以上
    for i in range(xdirValue):
        whetherx[i], xcollid_point[i][0], xcollid_point[i][1] = whetherPassFromXaxis(xdir_point[i][0], xdir_point[i][1], ydir_m, ydirValue, img)
        #print(i, whetherx[i])
    xlabels = [0] * 3
    xlabels[0] = find_first_x(whetherx)
    shift_x = xcollid_point[xlabels[0]][0] - ycollid_point[ylabels[0]][0]
    shift_y = xcollid_point[xlabels[0]][1] - ycollid_point[ylabels[0]][1]
    xlabels[1] = find_other_x(ydir_m, xdir_point, ycollid_point[ylabels[1]][0] + shift_x, ycollid_point[ylabels[1]][1] + shift_y, img)
    xlabels[2] = find_other_x(ydir_m, xdir_point, ycollid_point[ylabels[2]][0] + shift_x, ycollid_point[ylabels[2]][1] + shift_y, img)
    ####################################print################################################
    cv2.circle(img, (originx, originy), 10, color_red, -1)   
    cv2.circle(img, (xdirEnd_x, xdirEnd_y), 10, color_red, -1)
    cv2.line(img, (originx, originy), (xdirEnd_x, xdirEnd_y), color_red, 5)
    cv2.circle(img, (ydirEnd_x, ydirEnd_y), 10, color_red, -1)
    cv2.line(img, (originx, originy), (ydirEnd_x, ydirEnd_y), color_red, 5)
    
    if result.multi_handedness[0].classification[0].label == "Left":# hand type
        print("hand type: right")
    else: 
        print("hand type: left")
    for i in range(xdirValue):
        cv2.circle(img, (xdir_point[i][0], xdir_point[i][1]), 1, color_blue, -1)
    for i in range(ydirValue):
        cv2.circle(img, (ydir_point[i][0], ydir_point[i][1]), 1, color_blue, -1)
    #只是為了print出網格
    xgrid_point = [[0 for _ in range(2)] for _ in range(xaxis_grid)]#每個點有x和y
    ygrid_point = [[0 for _ in range(2)] for _ in range(yaxis_grid)]#每個點有x和y
    for i in range(xaxis_grid):
        xgrid_point[i][0], xgrid_point[i][1] = getPoint(originx, originy, xdir_m, (i/xaxis_grid  * xdirValue), result.multi_handedness[0].classification[0].label, False)
    for i in range(yaxis_grid):
        ygrid_point[i][0], ygrid_point[i][1] = getPoint(originx, originy, ydir_m, (i/yaxis_grid * ydirValue), result.multi_handedness[0].classification[0].label, True)
    another_xdir_point = [[0 for _ in range(2)] for _ in range(xaxis_grid)]#每個點有x和y
    another_ydir_point = [[0 for _ in range(2)] for _ in range(yaxis_grid)]#每個點有x和y
    for i in range(xaxis_grid):
        another_xdir_point[i][0], another_xdir_point[i][1] = getPoint(xgrid_point[i][0], xgrid_point[i][1], ydir_m, ydirValue, result.multi_handedness[0].classification[0].label, True)
    for i in range(yaxis_grid):
        another_ydir_point[i][0], another_ydir_point[i][1] = getPoint(ygrid_point[i][0], ygrid_point[i][1], xdir_m, xdirValue, result.multi_handedness[0].classification[0].label, False)
    for i in range(xaxis_grid):
        cv2.line(img, (xgrid_point[i][0], xgrid_point[i][1]), (another_xdir_point[i][0], another_xdir_point[i][1]), color_red, 1)
    for i in range(yaxis_grid):
        cv2.line(img, (ygrid_point[i][0], ygrid_point[i][1]), (another_ydir_point[i][0], another_ydir_point[i][1]), color_red, 1)
    #只是為了print出網格end

    for i in range(3):
        if xlabels[i] != 0:
            xlabels[i] = int(xlabels[i] / (xdirValue / xaxis_grid)) + 1
        if ylabels[i] != 0:
            ylabels[i] = int(ylabels[i] / (ydirValue / yaxis_grid)) + 1
    
    img = cv2.resize(img, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img', img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return '((' + str(xlabels[0]) + '-' + str(ylabels[0]) + ')-(' + str(xlabels[1]) + '-' + str(ylabels[1]) + ')-(' + str(xlabels[2]) + '-' + str(ylabels[2]) + '))'

def zuobiao(img, hand):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//2, h//2))  #對影像進行縮小便于后續選擇坐標點
    a = []  # 用于存放橫坐標
    b = []  # 用于存放縱坐
    print('請依序點擊"碗關節"及垂直於手臂"任一另一點"及劃出的線之延伸"腕旁點"坐標：')

    # 定義點擊事件
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 如果存在滑鼠點擊事件
            xy = "%d,%d" % (x, y)  # 得到坐標x,y
            a.append(x*2)  # 將每次的坐標存放在a陣列里面 由于原先圖片尺寸縮小一半，這里坐標點需要變回原圖片的尺寸位置
            b.append(y*2)  # 將每次的坐標存放在b陣列里面
            cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)  # 點擊的地方小紅圓點顯示
            if len(a) == 2:
                ydir_m = -1 * (b[0]//2 - b[1]//2) / (a[0]//2 - a[1]//2)
                if  ydir_m == infinity:
                    xdir_m = 0
                else:
                    xdir_m = -1 / ydir_m
                xdirEnd_x, xdirEnd_y = getPoint(a[0]//2, b[0]//2, xdir_m,  100, hand, False)
                cv2.line(img, (a[0]//2, b[0]//2), (a[1]//2, b[1]//2), color_blue, 1)
                cv2.line(img, (a[0]//2, b[0]//2), (xdirEnd_x, xdirEnd_y), color_blue, 1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,  # 點擊的地方顯示坐標數字 引數1圖片，引數2添加的文字，引數3左上角坐標，引數4字體，引數5字體粗細
                        1.0, (0, 0, 0), thickness=1)
            
            cv2.imshow("image", img)  # 顯示圖片
    cv2.namedWindow("image")  # 定義圖片視窗
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)  # 回呼函式，引數1視窗的名字，引數2滑鼠回應函式
    
    cv2.imshow("image", img)  # 顯示圖片
    
    cv2.waitKey(0)
    try:
        return a[0], b[0], a[2], b[2]
    except:
        return 0