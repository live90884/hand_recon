import cv2, os, csv
import mediapipe as mp
from function.label_fuc import getCenter, getValue, getPoint
import matplotlib.pyplot as plt
#define
infinity = 10000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
K = 3 #Y軸是X軸的三倍
whether_point = 0

def show_point(img, point1, point2, point3, grid_number, const_rate, width_rate, show_ori, whether_grid, whether_show, lower_bound, upper_bound):
    xaxis_grid = grid_number
    yaxis_grid = K * xaxis_grid
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.01, min_tracking_confidence=0.5, max_num_hands = 1)
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color = color_blue, thickness=3)
    handConStyle = mpDraw.DrawingSpec(color = color_green, thickness=5)
    result = hands.process(img)

    #二值化描邊
    ori_img = img
    img = cv2.Canny(img, lower_bound, upper_bound)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ####

    if result.multi_hand_landmarks:
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        originx = 0 #腕關節
        originy = 0

        for handLms in result.multi_hand_landmarks: 
            for i, lm in enumerate(handLms.landmark):
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                if i == 0:
                    originx = xPos
                    originy = yPos

    m_x, m_y = getCenter(originx, round(originy + (imgHeight / const_rate)), img) #找斜率用的x和y
    if m_x == originx:
        print("infinity")
        ydir_m = infinity
    else:
        ydir_m = -1 * (originy - m_y) / (originx - m_x)
    if  ydir_m == infinity:
        xdir_m = 0
    else:
        xdir_m = -1 / ydir_m

    xdirValue = getValue(m_x, m_y, ydir_m, img)/width_rate #如果手臂形狀很特別***********************************************************************
    #視少數奇怪的人之情況自行更動值
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
    
    
    
    if result.multi_handedness[0].classification[0].label == "Left":# hand type
        print("hand type: right")
    else: 
        print("hand type: left")
    
    if show_ori == True:
        cv2.circle(ori_img, (point1_x, point1_y), 5, color_red, -1)   
        cv2.circle(ori_img, (point2_x, point2_y), 5, color_red, -1)
        cv2.circle(ori_img, (point3_x, point3_y), 5, color_red, -1)
        if whether_show == True:
            cv2.circle(ori_img, (m_x, m_y), 10, color_blue, -1)          
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
                cv2.line(ori_img, (xgrid_point[i][0], xgrid_point[i][1]), (another_xdir_point[i][0], another_xdir_point[i][1]), color_red, 1)
            for i in range(yaxis_grid):
                cv2.line(ori_img, (ygrid_point[i][0], ygrid_point[i][1]), (another_ydir_point[i][0], another_ydir_point[i][1]), color_red, 1)
        mpDraw.draw_landmarks(ori_img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
        ori_img = cv2.resize(ori_img, None, fx = 0.6, fy = 0.6, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('ori_img', ori_img)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.circle(img, (point1_x, point1_y), 5, color_red, -1)   
        cv2.circle(img, (point2_x, point2_y), 5, color_red, -1)
        cv2.circle(img, (point3_x, point3_y), 5, color_red, -1)
        if whether_show == True:
            cv2.circle(img, (m_x, m_y), 10, color_blue, -1)          #用這條來顯示當前選擇的手臂粗度位置
        
        #只是為了print出網格
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
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
        img = cv2.resize(img, None, fx = 0.6, fy = 0.6, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('img', img)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()