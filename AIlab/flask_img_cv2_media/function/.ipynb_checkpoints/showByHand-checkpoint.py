import cv2, os, csv
import mediapipe as mp
from function.label_fuc import getPoint
import matplotlib.pyplot as plt
#define
infinity = 10000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
K = 3 #Y軸是X軸的三倍

def show_pointByhand(img, point1, point2, point3, grid_number, whether_grid):
    xaxis_grid = grid_number
    yaxis_grid = K * xaxis_grid
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.05, min_tracking_confidence=0.2, max_num_hands = 1)
    result = hands.process(img)
    
    if result.multi_hand_landmarks:
        print("")
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if result.multi_handedness[0].classification[0].label == "Left":# hand type
            print("hand type: right hand")
        else: 
            print("hand type: left hand")
        try:
            originx, originy, xEnd_x, xEnd_y = zuobiao(img, result.multi_handedness[0].classification[0].label)
        except:
            print("error")
            return 0
    print("腕橫紋", (originx, originy), "腕旁點", (xEnd_x, xEnd_y))
    xdir_m = -1 * (xEnd_y - originy) / (xEnd_x - originx)
    if  xdir_m == 0:
        ydir_m = infinity
    else:
        ydir_m = -1 / xdir_m
    xdirValue = ((xEnd_x - originx) ** 2 + (xEnd_y - originy) ** 2)**0.5
    
    ydirValue = round(K * xdirValue)
    xdirValue = round(xdirValue)
        
    point1_xvalue = ((2 * point1[0]) - 1)/(2 * grid_number) * xdirValue
    point1_yvalue = ((2 * point1[1]) - 1)/(6 * grid_number) * ydirValue
    point2_xvalue = ((2 * point2[0]) - 1)/(2 * grid_number) * xdirValue
    point2_yvalue = ((2 * point2[1]) - 1)/(6 * grid_number) * ydirValue
    point3_xvalue = ((2 * point3[0]) - 1)/(2 * grid_number) * xdirValue
    point3_yvalue = ((2 * point3[1]) - 1)/(6 * grid_number) * ydirValue
    point1Xdir_x, point1Xdir_y = getPoint(originx, originy, xdir_m,   point1_xvalue, result.multi_handedness[0].classification[0].label, False)
    point1_x, point1_y = getPoint(point1Xdir_x, point1Xdir_y, ydir_m,  point1_yvalue, result.multi_handedness[0].classification[0].label, True)
    point2Xdir_x, point2Xdir_y = getPoint(originx, originy, xdir_m,   point2_xvalue, result.multi_handedness[0].classification[0].label, False)
    point2_x, point2_y = getPoint(point2Xdir_x, point2Xdir_y, ydir_m,  point2_yvalue, result.multi_handedness[0].classification[0].label, True)
    point3Xdir_x, point3Xdir_y = getPoint(originx, originy, xdir_m,   point3_xvalue, result.multi_handedness[0].classification[0].label, False)
    point3_x, point3_y = getPoint(point3Xdir_x, point3Xdir_y, ydir_m,  point3_yvalue, result.multi_handedness[0].classification[0].label, True)
    ####################################print################################################
    cv2.circle(img, (point1_x, point1_y), 5, color_red, -1)   
    cv2.circle(img, (point2_x, point2_y), 5, color_red, -1)
    cv2.circle(img, (point3_x, point3_y), 5, color_red, -1)         
    if whether_grid == True:
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
    img = cv2.resize(img, None, fx = 0.6, fy = 0.6, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img', img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
                ydir_m = -1 * (b[0]//2 - b[1]//2) / (a[0]//2 - a[1]//2)#劃出垂直於手臂邊緣的線
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
    cv2.moveWindow("image", 100, 10)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.putText(img, 'please click your wrist, ', (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), thickness=1)
    cv2.putText(img, 'then click the point that make ', (10, 45), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), thickness=1)
    cv2.putText(img, 'the line betweens two point parallel, ', (10, 70), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), thickness=1)
    cv2.putText(img, 'and click the point that intersect ', (10, 95), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), thickness=1)
    cv2.putText(img, 'with your arm edge.', (10, 120), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), thickness=1)
    cv2.imshow("image", img)  # 顯示圖片
    cv2.waitKey(0)
    try:
        return a[0], b[0], a[2], b[2]
    except:
        return 0