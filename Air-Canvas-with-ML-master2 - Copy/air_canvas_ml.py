# # # # All the imports go here
# # # import cv2
# # # import numpy as np
# # # import mediapipe as mp
# # # from collections import deque

# # # # Giving different arrays to handle colour points of different colour
# # # bpoints = [deque(maxlen=1024)]
# # # gpoints = [deque(maxlen=1024)]
# # # rpoints = [deque(maxlen=1024)]
# # # ypoints = [deque(maxlen=1024)]

# # # # These indexes will be used to mark the points in particular arrays of specific colour
# # # blue_index = 0
# # # green_index = 0
# # # red_index = 0
# # # yellow_index = 0

# # # # Kernel for dilation
# # # kernel = np.ones((5, 5), np.uint8)

# # # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# # # colorIndex = 0

# # # # Canvas setup
# # # paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
# # # paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# # # cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # # # Initialize mediapipe
# # # mpHands = mp.solutions.hands
# # # hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# # # mpDraw = mp.solutions.drawing_utils

# # # # Initialize the webcam
# # # cap = cv2.VideoCapture(0)
# # # ret = True
# # # while ret:
# # #     # Read each frame from the webcam
# # #     ret, frame = cap.read()

# # #     x, y, c = frame.shape

# # #     # Flip the frame vertically
# # #     frame = cv2.flip(frame, 1)
# # #     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # #     # Draw UI buttons
# # #     frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
# # #     frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
# # #     frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
# # #     frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
# # #     frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
# # #     cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# # #     # Get hand landmark prediction
# # #     result = hands.process(framergb)

# # #     # Post process the result
# # #     if result.multi_hand_landmarks:
# # #         landmarks = []
# # #         for handslms in result.multi_hand_landmarks:
# # #             for lm in handslms.landmark:
# # #                 lmx = int(lm.x * 640)
# # #                 lmy = int(lm.y * 480)
# # #                 landmarks.append([lmx, lmy])

# # #             # Drawing landmarks on frames
# # #             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# # #         fore_finger = (landmarks[8][0], landmarks[8][1])
# # #         center = fore_finger
# # #         thumb = (landmarks[4][0], landmarks[4][1])
# # #         cv2.circle(frame, center, 3, (0, 255, 0), -1)

# # #         # Clear the canvas if the thumb is close to the forefinger
# # #         if (thumb[1] - center[1] < 30):
# # #             bpoints.append(deque(maxlen=512))
# # #             blue_index += 1
# # #             gpoints.append(deque(maxlen=512))
# # #             green_index += 1
# # #             rpoints.append(deque(maxlen=512))
# # #             red_index += 1
# # #             ypoints.append(deque(maxlen=512))
# # #             yellow_index += 1

# # #         elif center[1] <= 65:
# # #             if 40 <= center[0] <= 140:  # Clear Button
# # #                 bpoints = [deque(maxlen=512)]
# # #                 gpoints = [deque(maxlen=512)]
# # #                 rpoints = [deque(maxlen=512)]
# # #                 ypoints = [deque(maxlen=512)]
# # #                 blue_index = 0
# # #                 green_index = 0
# # #                 red_index = 0
# # #                 yellow_index = 0
# # #                 paintWindow[67:, :, :] = 255
# # #             elif 160 <= center[0] <= 255:
# # #                 colorIndex = 0  # Blue
# # #             elif 275 <= center[0] <= 370:
# # #                 colorIndex = 1  # Green
# # #             elif 390 <= center[0] <= 485:
# # #                 colorIndex = 2  # Red
# # #             elif 505 <= center[0] <= 600:
# # #                 colorIndex = 3  # Yellow
# # #         else:
# # #             if colorIndex == 0:
# # #                 bpoints[blue_index].appendleft(center)
# # #             elif colorIndex == 1:
# # #                 gpoints[green_index].appendleft(center)
# # #             elif colorIndex == 2:
# # #                 rpoints[red_index].appendleft(center)
# # #             elif colorIndex == 3:
# # #                 ypoints[yellow_index].appendleft(center)

# # #     # Draw lines of all the colors on the canvas and frame
# # #     points = [bpoints, gpoints, rpoints, ypoints]
# # #     for i in range(len(points)):
# # #         for j in range(len(points[i])):
# # #             for k in range(1, len(points[i][j])):
# # #                 if points[i][j][k - 1] is None or points[i][j][k] is None:
# # #                     continue
# # #                 cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
# # #                 cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

# # #     # Shape detection and classification
# # #     gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
# # #     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# # #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # #     for cnt in contours:
# # #         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
# # #         cv2.drawContours(frame, [approx], 0, (0), 5)

# # #         x = approx.ravel()[0]
# # #         y = approx.ravel()[1] - 5

# # #         if len(approx) == 3:
# # #             cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# # #         elif len(approx) == 4:
# # #             x, y, w, h = cv2.boundingRect(approx)
# # #             aspectRatio = float(w) / h
# # #             if aspectRatio >= 0.95 and aspectRatio <= 1.05:
# # #                 cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# # #             else:
# # #                 cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# # #         elif len(approx) > 6:
# # #             cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# # #     # Show all the windows
# # #     cv2.imshow("Paint", frame)
# # #     cv2.imshow("Canvas", paintWindow)

# # #     if cv2.waitKey(1) & 0xFF == ord("q"):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()








# # # All the imports go here*****************code2.0*********
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # from collections import deque

# # # Giving different arrays to handle colour points of different colour
# # bpoints = [deque(maxlen=1024)]
# # gpoints = [deque(maxlen=1024)]
# # rpoints = [deque(maxlen=1024)]
# # ypoints = [deque(maxlen=1024)]

# # # These indexes will be used to mark the points in particular arrays of specific colour
# # blue_index = 0
# # green_index = 0
# # red_index = 0
# # yellow_index = 0

# # # Kernel for dilation
# # kernel = np.ones((5, 5), np.uint8)

# # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# # colorIndex = 0

# # # Canvas setup
# # paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
# # paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# # paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# # cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # # Initialize mediapipe
# # mpHands = mp.solutions.hands
# # hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# # mpDraw = mp.solutions.drawing_utils

# # # Initialize the webcam
# # cap = cv2.VideoCapture(0)
# # ret = True
# # while ret:
# #     # Read each frame from the webcam
# #     ret, frame = cap.read()

# #     x, y, c = frame.shape

# #     # Flip the frame vertically
# #     frame = cv2.flip(frame, 1)
# #     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     # Draw UI buttons
# #     frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
# #     frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
# #     frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
# #     frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
# #     frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
# #     cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# #     # Get hand landmark prediction
# #     result = hands.process(framergb)

# #     # Post process the result
# #     if result.multi_hand_landmarks:
# #         landmarks = []
# #         for handslms in result.multi_hand_landmarks:
# #             for lm in handslms.landmark:
# #                 lmx = int(lm.x * 640)
# #                 lmy = int(lm.y * 480)
# #                 landmarks.append([lmx, lmy])

# #             # Drawing landmarks on frames
# #             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# #         fore_finger = (landmarks[8][0], landmarks[8][1])
# #         center = fore_finger
# #         thumb = (landmarks[4][0], landmarks[4][1])
# #         cv2.circle(frame, center, 3, (0, 255, 0), -1)

# #         # Clear the canvas if the thumb is close to the forefinger
# #         if (thumb[1] - center[1] < 30):
# #             bpoints.append(deque(maxlen=512))
# #             blue_index += 1
# #             gpoints.append(deque(maxlen=512))
# #             green_index += 1
# #             rpoints.append(deque(maxlen=512))
# #             red_index += 1
# #             ypoints.append(deque(maxlen=512))
# #             yellow_index += 1

# #         elif center[1] <= 65:
# #             if 40 <= center[0] <= 140:  # Clear Button
# #                 bpoints = [deque(maxlen=512)]
# #                 gpoints = [deque(maxlen=512)]
# #                 rpoints = [deque(maxlen=512)]
# #                 ypoints = [deque(maxlen=512)]
# #                 blue_index = 0
# #                 green_index = 0
# #                 red_index = 0
# #                 yellow_index = 0
# #                 paintWindow[67:, :, :] = 255
# #             elif 160 <= center[0] <= 255:
# #                 colorIndex = 0  # Blue
# #             elif 275 <= center[0] <= 370:
# #                 colorIndex = 1  # Green
# #             elif 390 <= center[0] <= 485:
# #                 colorIndex = 2  # Red
# #             elif 505 <= center[0] <= 600:
# #                 colorIndex = 3  # Yellow
# #         else:
# #             if colorIndex == 0:
# #                 bpoints[blue_index].appendleft(center)
# #             elif colorIndex == 1:
# #                 gpoints[green_index].appendleft(center)
# #             elif colorIndex == 2:
# #                 rpoints[red_index].appendleft(center)
# #             elif colorIndex == 3:
# #                 ypoints[yellow_index].appendleft(center)

# #     # Draw lines of all the colors on the canvas and frame
# #     points = [bpoints, gpoints, rpoints, ypoints]
# #     for i in range(len(points)):
# #         for j in range(len(points[i])):
# #             for k in range(1, len(points[i][j])):
# #                 if points[i][j][k - 1] is None or points[i][j][k] is None:
# #                     continue
# #                 cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
# #                 cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

# #     # Shape detection and classification
# #     gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
# #     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     for contour in contours:
# #         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
# #         x, y, w, h = cv2.boundingRect(approx)

# #         if len(approx) == 3:
# #             cv2.putText(frame, "Triangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #         elif len(approx) == 4:
# #             aspect_ratio = float(w) / h
# #             if 0.95 <= aspect_ratio <= 1.05:
# #                 cv2.putText(frame, "Square", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #             else:
# #                 cv2.putText(frame, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #         elif len(approx) > 4:
# #             cv2.putText(frame, "Circle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# #     # Show the frame and canvas
# #     cv2.imshow("Output", frame)
# #     cv2.imshow("Paint", paintWindow)

# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()




# {
#  "cells": [
#   {
#    "cell_type": "markdown",
#    "id": "30b08bb9",
#    "metadata": {},
#    "source": [
#     "### HANDWRITTEN EQUATION SOLVER USING CNN by TITHI DEB\n",
#     "\n",
#     "##### Explanation\n",
#     "* Import the libraries.\n",
#     "* Load the dataset.\n",
#     "* Preprocessing the Data.\n",
#     "* Then Building the CNN model.\n",
#     "* Then we trainthe model.\n",
#     "* Lastly we test the model."
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "id": "435a2778",
#    "metadata": {},
#    "source": [
#     "##### IMPORTING LIBRARIES "
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 85,
#    "id": "52a56941",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import numpy as np\n",
#     "import cv2\n",
#     "from PIL import Image\n",
#     "from matplotlib import pyplot as plt\n",
#     "%matplotlib inline\n",
#     "import os\n",
#     "from os import listdir\n",
#     "from os.path import isfile, join\n",
#     "import pandas as pd"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "id": "1586a081",
#    "metadata": {},
#    "source": [
#     "##### LOADING THE DATASET"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 86,
#    "id": "60e800ac",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def load_images_from_folder(folder):\n",
#     "    train_data=[]\n",
#     "    for filename in os.listdir(folder):\n",
#     "        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)\n",
#     "        img=~img\n",
#     "        if img is not None:\n",
#     "            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)\n",
#     "\n",
#     "            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)\n",
#     "            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])\n",
#     "            w=int(28)\n",
#     "            h=int(28)\n",
#     "            maxi=0\n",
#     "            for c in cnt:\n",
#     "                x,y,w,h=cv2.boundingRect(c)\n",
#     "                maxi=max(w*h,maxi)\n",
#     "                if maxi==w*h:\n",
#     "                    x_max=x\n",
#     "                    y_max=y\n",
#     "                    w_max=w\n",
#     "                    h_max=h\n",
#     "            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]\n",
#     "            im_resize = cv2.resize(im_crop,(28,28))\n",
#     "            im_resize=np.reshape(im_resize,(784,1))\n",
#     "            train_data.append(im_resize)\n",
#     "    return train_data"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 87,
#    "id": "09b190b3",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "data=[]"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 89,
#    "id": "9b8db932",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "1300\n"
#      ]
#     }
#    ],
#    "source": [
#     "# Assign '-' = 10\n",
#     "data=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\!\\\\')\n",
#     "len(data)\n",
#     "for i in range(0,len(data)):\n",
#     "    data[i]=np.append(data[i],['10'])\n",
#     "    \n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 90,
#    "id": "dc4bde24",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "26412\n"
#      ]
#     }
#    ],
#    "source": [
#     "# Assign + = 11\n",
#     "data11=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\+\\\\')\n",
#     "\n",
#     "for i in range(0,len(data11)):\n",
#     "    data11[i]=np.append(data11[i],['11'])\n",
#     "data=np.concatenate((data,data11))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 91,
#    "id": "dba74258",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "33326\n"
#      ]
#     }
#    ],
#    "source": [
#     "data0=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\0\\\\')\n",
#     "\n",
#     "for i in range(0,len(data0)):\n",
#     "    data0[i]=np.append(data0[i],['0'])\n",
#     "data=np.concatenate((data,data0))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 92,
#    "id": "b452fa94",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "59846\n"
#      ]
#     }
#    ],
#    "source": [
#     "data1=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\1\\\\')\n",
#     "\n",
#     "for i in range(0,len(data1)):\n",
#     "    data1[i]=np.append(data1[i],['1'])\n",
#     "data=np.concatenate((data,data1))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 93,
#    "id": "6f8dd715",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "85987\n"
#      ]
#     }
#    ],
#    "source": [
#     "data2=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\2\\\\')\n",
#     "\n",
#     "for i in range(0,len(data2)):\n",
#     "    data2[i]=np.append(data2[i],['2'])\n",
#     "data=np.concatenate((data,data2))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 94,
#    "id": "e718a73d",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "96896\n"
#      ]
#     }
#    ],
#    "source": [
#     "data3=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\3\\\\')\n",
#     "\n",
#     "for i in range(0,len(data3)):\n",
#     "    data3[i]=np.append(data3[i],['3'])\n",
#     "data=np.concatenate((data,data3))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 95,
#    "id": "8533e5fc",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "104292\n"
#      ]
#     }
#    ],
#    "source": [
#     "data4=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\4\\\\')\n",
#     "\n",
#     "for i in range(0,len(data4)):\n",
#     "    data4[i]=np.append(data4[i],['4'])\n",
#     "data=np.concatenate((data,data4))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 96,
#    "id": "7258dde0",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "107837\n"
#      ]
#     }
#    ],
#    "source": [
#     "data5=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\5\\\\')\n",
#     "\n",
#     "for i in range(0,len(data5)):\n",
#     "    data5[i]=np.append(data5[i],['5'])\n",
#     "data=np.concatenate((data,data5))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 97,
#    "id": "caa5bf07",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "110955\n"
#      ]
#     }
#    ],
#    "source": [
#     "data6=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\6\\\\')\n",
#     "\n",
#     "for i in range(0,len(data6)):\n",
#     "    data6[i]=np.append(data6[i],['6'])\n",
#     "data=np.concatenate((data,data6))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 98,
#    "id": "7dca697b",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "113864\n"
#      ]
#     }
#    ],
#    "source": [
#     "data7=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\7\\\\')\n",
#     "\n",
#     "for i in range(0,len(data7)):\n",
#     "    data7[i]=np.append(data7[i],['7'])\n",
#     "data=np.concatenate((data,data7))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 99,
#    "id": "ef32568c",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "116932\n"
#      ]
#     }
#    ],
#    "source": [
#     "data8=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\8\\\\')\n",
#     "\n",
#     "for i in range(0,len(data8)):\n",
#     "    data8[i]=np.append(data8[i],['8'])\n",
#     "data=np.concatenate((data,data8))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 100,
#    "id": "a7a2b42c",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "120669\n"
#      ]
#     }
#    ],
#    "source": [
#     "data9=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\9\\\\')\n",
#     "\n",
#     "for i in range(0,len(data9)):\n",
#     "    data9[i]=np.append(data9[i],['9'])\n",
#     "data=np.concatenate((data,data9))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 101,
#    "id": "dfac1059",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "123920\n"
#      ]
#     }
#    ],
#    "source": [
#     "data12=load_images_from_folder('Desktop\\ML PROJECTS\\Handwritten Equation Slover USing CNN\\extracted\\\\times\\\\')\n",
#     "\n",
#     "for i in range(0,len(data12)):\n",
#     "    data12[i]=np.append(data12[i],['12'])\n",
#     "data=np.concatenate((data,data12))\n",
#     "print(len(data))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 102,
#    "id": "7efc8d8c",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df=pd.DataFrame(data,index=None)\n",
#     "df.to_csv('train_final.csv',index=False)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 103,
#    "id": "dc782fae",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# Importing Libraries\n",
#     "import pandas as pd\n",
#     "import numpy as np\n",
#     "import pickle"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 104,
#    "id": "fdd0f025",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df_train=pd.read_csv('train_final.csv',index_col=False)\n",
#     "labels=df_train[['784']]"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 105,
#    "id": "7d5c48d6",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/html": [
#        "<div>\n",
#        "<style scoped>\n",
#        "    .dataframe tbody tr th:only-of-type {\n",
#        "        vertical-align: middle;\n",
#        "    }\n",
#        "\n",
#        "    .dataframe tbody tr th {\n",
#        "        vertical-align: top;\n",
#        "    }\n",
#        "\n",
#        "    .dataframe thead th {\n",
#        "        text-align: right;\n",
#        "    }\n",
#        "</style>\n",
#        "<table border=\"1\" class=\"dataframe\">\n",
#        "  <thead>\n",
#        "    <tr style=\"text-align: right;\">\n",
#        "      <th></th>\n",
#        "      <th>0</th>\n",
#        "      <th>1</th>\n",
#        "      <th>2</th>\n",
#        "      <th>3</th>\n",
#        "      <th>4</th>\n",
#        "      <th>5</th>\n",
#        "      <th>6</th>\n",
#        "      <th>7</th>\n",
#        "      <th>8</th>\n",
#        "      <th>9</th>\n",
#        "      <th>...</th>\n",
#        "      <th>774</th>\n",
#        "      <th>775</th>\n",
#        "      <th>776</th>\n",
#        "      <th>777</th>\n",
#        "      <th>778</th>\n",
#        "      <th>779</th>\n",
#        "      <th>780</th>\n",
#        "      <th>781</th>\n",
#        "      <th>782</th>\n",
#        "      <th>783</th>\n",
#        "    </tr>\n",
#        "  </thead>\n",
#        "  <tbody>\n",
#        "    <tr>\n",
#        "      <th>0</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>1</th>\n",
#        "      <td>0</td>\n",
#        "      <td>36</td>\n",
#        "      <td>146</td>\n",
#        "      <td>255</td>\n",
#        "      <td>146</td>\n",
#        "      <td>36</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>2</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>3</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>32</td>\n",
#        "      <td>150</td>\n",
#        "      <td>241</td>\n",
#        "      <td>123</td>\n",
#        "      <td>4</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>4</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>64</td>\n",
#        "      <td>191</td>\n",
#        "      <td>191</td>\n",
#        "      <td>64</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>5</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>132</td>\n",
#        "      <td>241</td>\n",
#        "      <td>105</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>6</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>7</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>8</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>9</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "  </tbody>\n",
#        "</table>\n",
#        "<p>10 rows × 784 columns</p>\n",
#        "</div>"
#       ],
#       "text/plain": [
#        "     0    1    2    3    4    5    6    7    8    9  ...  774  775  776  777  \\\n",
#        "0  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "1    0   36  146  255  146   36    0    0    0    0  ...    0    0    0    0   \n",
#        "2  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "3    0    0    0   32  150  241  123    4    0    0  ...    0    0    0    0   \n",
#        "4    0    0    0    0    0   64  191  191   64    0  ...    0    0    0    0   \n",
#        "5    0    0    0    0    0    0    0  132  241  105  ...    0    0    0    0   \n",
#        "6  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "7  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "8  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "9  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "\n",
#        "   778  779  780  781  782  783  \n",
#        "0    0    0    0    0    0    0  \n",
#        "1    0    0    0    0    0    0  \n",
#        "2    0    0    0    0    0    0  \n",
#        "3    0    0    0    0    0    0  \n",
#        "4    0    0    0    0    0    0  \n",
#        "5    0    0    0    0    0    0  \n",
#        "6    0    0    0    0    0    0  \n",
#        "7    0    0    0    0    0    0  \n",
#        "8    0    0    0    0    0    0  \n",
#        "9    0    0    0    0    0    0  \n",
#        "\n",
#        "[10 rows x 784 columns]"
#       ]
#      },
#      "execution_count": 105,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "df_train.drop(df_train.columns[[784]],axis=1,inplace=True)\n",
#     "df_train.head(10)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 106,
#    "id": "aabe2c12",
#    "metadata": {
#     "scrolled": True
#    },
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "'channels_last'"
#       ]
#      },
#      "execution_count": 106,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "np.random.seed(1212)\n",
#     "import keras\n",
#     "from keras.models import Model\n",
#     "from keras.layers import *\n",
#     "from keras import optimizers\n",
#     "from keras.layers import Input, Dense\n",
#     "from keras.models import Sequential\n",
#     "from keras.layers import Dense\n",
#     "from keras.layers import Dropout\n",
#     "from keras.layers import Flatten\n",
#     "from keras.layers.convolutional import Conv2D\n",
#     "from keras.layers.convolutional import MaxPooling2D\n",
#     "from keras.utils import np_utils\n",
#     "from keras import backend as K\n",
#     "K.image_data_format()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 107,
#    "id": "fa394857",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "labels=np.array(labels)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 108,
#    "id": "0e520efd",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "from keras.utils.np_utils import to_categorical\n",
#     "cat=to_categorical(labels,num_classes=13)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 109,
#    "id": "3f97c3ff",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n"
#      ]
#     }
#    ],
#    "source": [
#     "print(cat[0])"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 111,
#    "id": "c241d392",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/html": [
#        "<div>\n",
#        "<style scoped>\n",
#        "    .dataframe tbody tr th:only-of-type {\n",
#        "        vertical-align: middle;\n",
#        "    }\n",
#        "\n",
#        "    .dataframe tbody tr th {\n",
#        "        vertical-align: top;\n",
#        "    }\n",
#        "\n",
#        "    .dataframe thead th {\n",
#        "        text-align: right;\n",
#        "    }\n",
#        "</style>\n",
#        "<table border=\"1\" class=\"dataframe\">\n",
#        "  <thead>\n",
#        "    <tr style=\"text-align: right;\">\n",
#        "      <th></th>\n",
#        "      <th>0</th>\n",
#        "      <th>1</th>\n",
#        "      <th>2</th>\n",
#        "      <th>3</th>\n",
#        "      <th>4</th>\n",
#        "      <th>5</th>\n",
#        "      <th>6</th>\n",
#        "      <th>7</th>\n",
#        "      <th>8</th>\n",
#        "      <th>9</th>\n",
#        "      <th>...</th>\n",
#        "      <th>774</th>\n",
#        "      <th>775</th>\n",
#        "      <th>776</th>\n",
#        "      <th>777</th>\n",
#        "      <th>778</th>\n",
#        "      <th>779</th>\n",
#        "      <th>780</th>\n",
#        "      <th>781</th>\n",
#        "      <th>782</th>\n",
#        "      <th>783</th>\n",
#        "    </tr>\n",
#        "  </thead>\n",
#        "  <tbody>\n",
#        "    <tr>\n",
#        "      <th>0</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>1</th>\n",
#        "      <td>0</td>\n",
#        "      <td>36</td>\n",
#        "      <td>146</td>\n",
#        "      <td>255</td>\n",
#        "      <td>146</td>\n",
#        "      <td>36</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>2</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>3</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>32</td>\n",
#        "      <td>150</td>\n",
#        "      <td>241</td>\n",
#        "      <td>123</td>\n",
#        "      <td>4</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>4</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>64</td>\n",
#        "      <td>191</td>\n",
#        "      <td>191</td>\n",
#        "      <td>64</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>5</th>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>132</td>\n",
#        "      <td>241</td>\n",
#        "      <td>105</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>6</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>7</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>8</th>\n",
#        "      <td>255</td>\n",
#        "      <td>218</td>\n",
#        "      <td>109</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "    <tr>\n",
#        "      <th>9</th>\n",
#        "      <td>255</td>\n",
#        "      <td>232</td>\n",
#        "      <td>132</td>\n",
#        "      <td>32</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>...</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "      <td>0</td>\n",
#        "    </tr>\n",
#        "  </tbody>\n",
#        "</table>\n",
#        "<p>10 rows × 784 columns</p>\n",
#        "</div>"
#       ],
#       "text/plain": [
#        "     0    1    2    3    4    5    6    7    8    9  ...  774  775  776  777  \\\n",
#        "0  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "1    0   36  146  255  146   36    0    0    0    0  ...    0    0    0    0   \n",
#        "2  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "3    0    0    0   32  150  241  123    4    0    0  ...    0    0    0    0   \n",
#        "4    0    0    0    0    0   64  191  191   64    0  ...    0    0    0    0   \n",
#        "5    0    0    0    0    0    0    0  132  241  105  ...    0    0    0    0   \n",
#        "6  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "7  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "8  255  218  109    0    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "9  255  232  132   32    0    0    0    0    0    0  ...    0    0    0    0   \n",
#        "\n",
#        "   778  779  780  781  782  783  \n",
#        "0    0    0    0    0    0    0  \n",
#        "1    0    0    0    0    0    0  \n",
#        "2    0    0    0    0    0    0  \n",
#        "3    0    0    0    0    0    0  \n",
#        "4    0    0    0    0    0    0  \n",
#        "5    0    0    0    0    0    0  \n",
#        "6    0    0    0    0    0    0  \n",
#        "7    0    0    0    0    0    0  \n",
#        "8    0    0    0    0    0    0  \n",
#        "9    0    0    0    0    0    0  \n",
#        "\n",
#        "[10 rows x 784 columns]"
#       ]
#      },
#      "execution_count": 111,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "df_train.head(10)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 112,
#    "id": "bfdb67ce",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "(123920, 784)"
#       ]
#      },
#      "execution_count": 112,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "df_train.shape"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 113,
#    "id": "d90ce99e",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "temp=df_train.to_numpy()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 114,
#    "id": "477ad747",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "X_train = temp.reshape(temp.shape[0], 28, 28, 1)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 115,
#    "id": "3a48bb88",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "123920"
#       ]
#      },
#      "execution_count": 115,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "temp.shape[0]"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 116,
#    "id": "71c39d22",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "(123920, 28, 28, 1)"
#       ]
#      },
#      "execution_count": 116,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "X_train.shape"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 117,
#    "id": "2f157d22",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "l=[]\n",
#     "for i in range(50621):\n",
#     "    l.append(np.array(df_train[i:i+1]).reshape(1,28,28))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 118,
#    "id": "338272f4",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "np.random.seed(7)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 119,
#    "id": "c0f024a2",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "(123920, 28, 28, 1)"
#       ]
#      },
#      "execution_count": 119,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "X_train.shape"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "id": "46132f27",
#    "metadata": {},
#    "source": [
#     "### Building the model"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 120,
#    "id": "4d6e852d",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "model = Sequential()\n",
#     "model.add(Conv2D(32, (3,3), input_shape=(28, 28,1), activation='relu',padding='same'))\n",
#     "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
#     "model.add(Conv2D(15, (3, 3), activation='relu'))\n",
#     "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
#     "model.add(Dropout(0.2))\n",
#     "model.add(Flatten())\n",
#     "model.add(Dense(128, activation='relu'))\n",
#     "model.add(Dense(50, activation='relu'))\n",
#     "model.add(Dense(13, activation='softmax'))"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 121,
#    "id": "25e36f98",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# Compile model\n",
#     "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 122,
#    "id": "94ff54e2",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "from keras.models import model_from_json"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 123,
#    "id": "c0829508",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "Epoch 1/10\n",
#       "620/620 [==============================] - 18s 28ms/step - loss: 0.5222 - accuracy: 0.8705\n",
#       "Epoch 2/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0958 - accuracy: 0.9722\n",
#       "Epoch 3/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0583 - accuracy: 0.9826\n",
#       "Epoch 4/10\n",
#       "620/620 [==============================] - 16s 27ms/step - loss: 0.0409 - accuracy: 0.9875\n",
#       "Epoch 5/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0331 - accuracy: 0.9900\n",
#       "Epoch 6/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0265 - accuracy: 0.9916\n",
#       "Epoch 7/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0244 - accuracy: 0.9922\n",
#       "Epoch 8/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0203 - accuracy: 0.9935\n",
#       "Epoch 9/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0200 - accuracy: 0.9936\n",
#       "Epoch 10/10\n",
#       "620/620 [==============================] - 17s 27ms/step - loss: 0.0163 - accuracy: 0.9948\n"
#      ]
#     },
#     {
#      "data": {
#       "text/plain": [
#        "<keras.callbacks.History at 0x2807204bac0>"
#       ]
#      },
#      "execution_count": 123,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "model.fit(X_train, cat, epochs=10, batch_size=200,shuffle=True,verbose=1)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 124,
#    "id": "8b3d7607",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "model_json = model.to_json()\n",
#     "with open(\"model_final.json\", \"w\") as json_file:\n",
#     "    json_file.write(model_json)\n",
#     "# serialize weights to HDF5\n",
#     "model.save_weights(\"model_final.h5\")"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 125,
#    "id": "6961530a",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import cv2\n",
#     "import numpy\n",
#     "from keras.datasets import mnist\n",
#     "from keras.models import Sequential\n",
#     "from keras.layers import Dense\n",
#     "from keras.layers import Dropout\n",
#     "from keras.layers import Flatten\n",
#     "from keras.layers.convolutional import Conv2D\n",
#     "from keras.layers.convolutional import MaxPooling2D\n",
#     "from keras.utils import np_utils\n",
#     "from keras import backend as K\n",
#     "# K.set_image_dim_ordering('th')\n",
#     "from keras.models import model_from_json"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 139,
#    "id": "cd79bd92",
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "json_file = open('model_final.json', 'r')\n",
#     "loaded_model_json = json_file.read()\n",
#     "json_file.close()\n",
#     "loaded_model = model_from_json(loaded_model_json)\n",
#     "# load weights into new model\n",
#     "loaded_model.load_weights(\"model_final.h5\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "id": "6b16e2f9",
#    "metadata": {},
#    "source": [
#     "##### TESTING THE MODEL"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 240,
#    "id": "5b4752ac",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "array([[198, 198, 199, ..., 198, 198, 198],\n",
#        "       [196, 197, 197, ..., 198, 198, 198],\n",
#        "       [195, 195, 196, ..., 198, 198, 198],\n",
#        "       ...,\n",
#        "       [197, 196, 196, ..., 198, 198, 198],\n",
#        "       [196, 196, 195, ..., 198, 198, 198],\n",
#        "       [196, 195, 195, ..., 199, 198, 198]], dtype=uint8)"
#       ]
#      },
#      "execution_count": 240,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "import cv2\n",
#     "import numpy as np\n",
#     "img = cv2.imread('Desktop\\\\ML PROJECTS\\\\Handwritten Equation Slover USing CNN\\\\d.jpg',cv2.IMREAD_GRAYSCALE)\n",
#     "img"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 241,
#    "id": "44138cb6",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "image/jpeg": "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\n",
#       "text/plain": [
#        "<IPython.core.display.Image object>"
#       ]
#      },
#      "execution_count": 241,
#      "metadata": {
#       "image/jpeg": {
#        "width": 360
#       }
#      },
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "from IPython import display\n",
#     "display.Image('Desktop\\\\ML PROJECTS\\\\Handwritten Equation Slover USing CNN\\\\d.jpg',width=360)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 242,
#    "id": "55c8bc00",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "10\n",
#       "[[0, 17, 17, 16], [24, 4, 213, 479], [397, 130, 211, 230], [757, 44, 197, 415], [856, 353, 4, 10], [859, 351, 3, 3], [860, 247, 4, 3], [867, 183, 3, 3], [870, 121, 3, 4], [929, 427, 6, 6]]\n",
#       "[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]\n",
#       "8\n",
#       "[[24, 4, 213, 479], [397, 130, 211, 230], [757, 44, 197, 415]]\n"
#      ]
#     }
#    ],
#    "source": [
#     "if img is not None:\n",
#     "    #images.append(img)\n",
#     "    img=~img\n",
#     "    ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)\n",
#     "    ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)\n",
#     "    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])\n",
#     "    w=int(28)\n",
#     "    h=int(28)\n",
#     "    train_data=[]\n",
#     "    print(len(cnt))\n",
#     "    rects=[]\n",
#     "    for c in cnt :\n",
#     "        x,y,w,h= cv2.boundingRect(c)\n",
#     "        rect=[x,y,w,h]\n",
#     "        rects.append(rect)\n",
#     "    print(rects)\n",
#     "    bool_rect=[]\n",
#     "    for r in rects:\n",
#     "        l=[]\n",
#     "        for rec in rects:\n",
#     "            flag=0\n",
#     "            if rec!=r:\n",
#     "                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):\n",
#     "                    flag=1\n",
#     "                l.append(flag)\n",
#     "            if rec==r:\n",
#     "                l.append(0)\n",
#     "        bool_rect.append(l)\n",
#     "    print(bool_rect)\n",
#     "    dump_rect=[]\n",
#     "    for i in range(0,len(cnt)):\n",
#     "        for j in range(0,len(cnt)):\n",
#     "            if bool_rect[i][j]==1:\n",
#     "                area1=rects[i][2]*rects[i][3]\n",
#     "                area2=rects[j][2]*rects[j][3]\n",
#     "                if(area1==min(area1,area2)):\n",
#     "                    dump_rect.append(rects[i])\n",
#     "    print(len(dump_rect)) \n",
#     "    final_rect=[i for i in rects if i not in dump_rect]\n",
#     "    print(final_rect)\n",
#     "    for r in final_rect:\n",
#     "        x=r[0]\n",
#     "        y=r[1]\n",
#     "        w=r[2]\n",
#     "        h=r[3]\n",
#     "        im_crop =thresh[y:y+h+10,x:x+w+10]\n",
#     "        \n",
#     "\n",
#     "        im_resize = cv2.resize(im_crop,(28,28))\n",
#     "\n",
#     "\n",
#     "        im_resize=np.reshape(im_resize,(28,28,1))\n",
#     "        train_data.append(im_resize)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 243,
#    "id": "007510e6",
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "1/1 [==============================] - 0s 14ms/step\n",
#       "1/1 [==============================] - 0s 16ms/step\n",
#       "1/1 [==============================] - 0s 17ms/step\n",
#       "7+1\n"
#      ]
#     }
#    ],
#    "source": [
#     "s=''\n",
#     "for i in range(len(train_data)):\n",
#     "    train_data[i]=np.array(train_data[i])\n",
#     "    train_data[i]=train_data[i].reshape(1,28,28,1)\n",
#     "    result=np.argmax(loaded_model.predict(train_data[i]), axis=-1)\n",
#     "    if(result[0]==10):\n",
#     "        s=s+'-'\n",
#     "    elif(result[0]==11):\n",
#     "        s=s+'+'\n",
#     "    elif(result[0]==12):\n",
#     "        s=s+'*'\n",
#     "    elif(result[0]==0):\n",
#     "        s=s+'0'\n",
#     "    elif(result[0]==1):\n",
#     "        s=s+'1'\n",
#     "    elif(result[0]==2):\n",
#     "        s=s+'2'\n",
#     "    elif(result[0]==3):\n",
#     "        s=s+'3'\n",
#     "    elif(result[0]==4):\n",
#     "        s=s+'4'\n",
#     "    elif(result[0]==5):\n",
#     "        s=s+'5'\n",
#     "    elif(result[0]==6):\n",
#     "        s=s+'6'\n",
#     "    elif(result[0]==7):\n",
#     "        s=s+'7'\n",
#     "    elif(result[0]==8):\n",
#     "        s=s+'8'\n",
#     "    elif(result[0]==9):\n",
#     "        s=s+'9'\n",
#     "        \n",
#     "print(s)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 244,
#    "id": "75bdcf48",
#    "metadata": {},
#    "outputs": [
#     {
#      "data": {
#       "text/plain": [
#        "8"
#       ]
#      },
#      "execution_count": 244,
#      "metadata": {},
#      "output_type": "execute_result"
#     }
#    ],
#    "source": [
#     "eval(s)"
#    ]
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3 (ipykernel)",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.10.9"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load pre-trained MNIST model (you can train or use any suitable model)
model = load_model('mnist_model.h5')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize canvas and settings
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
brush_thickness = 8
color = (255, 255, 255)
points = []

# Function to predict number
def predict_number(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28, 28))
    img_resized = img_resized.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict([img_resized])
    return np.argmax(prediction)

# Main Loop
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame from the webcam
    success, frame = cap.read()
    if not success:
        break
    
    # Flip the frame (optional)
    frame = cv2.flip(frame, 1)
    
    # Detect hand landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark coordinates for the index fingertip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Store points for drawing
            points.append((cx, cy))
            
            # Draw on the canvas
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(canvas, points[i-1], points[i], color, brush_thickness)
    
    # Show the canvas on a separate window
    cv2.imshow("Canvas", canvas)
    
    # Combine canvas and frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Canvas", combined)
    
    # Check for user inputs
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas[:] = 0
        points = []
    elif key == ord('p'):  # Press 'p' to predict the drawn number
        # Predict based on the drawing
        prediction = predict_number(canvas)
        print(f"Predicted Number: {prediction}")
    
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
