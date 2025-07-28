import cv2, os, csv
import mediapipe as mp
import matplotlib.pyplot as plt
#define
infinity = 10000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
K = 3 #Y軸是X軸的三倍
whether_point = 0

def check_label(id, const_rate, whether_show, grid_number, width_rate, alter_ori, low_bound, upper_bound):
    xaxis_grid = grid_number
    yaxis_grid = K * xaxis_grid
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color = color_blue, thickness=3)
    handConStyle = mpDraw.DrawingSpec(color = color_green, thickness=5)

    path = 'data/' + id

    for file in os.listdir(path):
        if file == "train" or file[0] == '.':
            continue
        img = cv2.imread(os.path.join(path, file))

    result = hands.process(img)

    #二值化描邊
    img = cv2.Canny(img, low_bound, upper_bound)
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

    
    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
    #cv2.circle(img, (m_x, m_y), 10, color_blue, -1)          #用這條來顯示當前選擇的手臂粗度位置***********************
    img = cv2.resize(img, None, fx = 0.6, fy = 0.6, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img', img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()