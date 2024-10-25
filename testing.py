from detecto import core, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
import time
import os
from tkinter import *
from tkinter import filedialog
import skimage.exposure
from skimage import measure
from math import atan2,degrees
from scipy import ndimage
col_img=None
identifiers = ["tank"]
# identifiers = ['FORK ASSY L FR 51500-ABG-2100','FORK ASSY L FRONT 51500-AAB-A020','FORK ASSY L FRONT 51500-AAK-1000','FORK ASSY L FRONT 51500-AAW-3000','FORK ASSY R FR 51400-ABG-2100','FORK ASSY R FRONT 51400-AAB-A020','FORK ASSY R FRONT 51400-AAK-1000','FORK ASSY R FRONT 51400-AAW-3000','SET ILLUST FORK ASSY L FR 51560-ABW-100R','SET ILLUST FORK ASSY L FR 51560-ABW-200R','SET ILLUST FORK ASSY L FR 51560-ABW-300R','SET ILLUST FORK ASSY R FR 51460-ABW-100R','SET ILLUST FORK ASSY R FR 51460-ABW-200R','SET ILLUST FORK ASSY R FR 51460-ABW-300R']
# # identifiers = ['AIR CLEANER ASSY 17200-ABW-2000','AIR/C ASSY 17200-AAW-3000','AIR/C SUB ASSY 1720A-AAB-A000','AIR/C SUB ASSY 1720A-AAK-2000','AIRC SUB ASSY 1720A-ABA-1000','AIR/C SUB ASSY 1720A-ABS-0000','AIR/C SUB ASSY 1720A-ABT-0000','AIR/C SUB ASSY 1720A-ABW-0000']
# # identifiers = ['LIGHT ASSY HEAD 33100-ABW-0010','LIGHT ASSY HEAD 33100-ABW-2010']
# # identifiers = ['WHEEL 1','WHEEL 2','WHEEL ASSY FR WITHOUT TYRE (ABWD) 4460P-ABW-3000','WHEEL 3','WHEEL 4','WHEEL 5','WHEEL ASSY FRONT DISC CEAT 44600-ABG-2100C','WHEEL 6','WHEEL 8','WHEEL 9','WHEEL 10','WHEEL 11','WHEEL ASSY RR WITHOUT TYRE ABWD 4260P-ABW-3000']
# # new_wheel_pth =  core.Model.load(r"wheelpth.pth", new_wheel)
# model = core.Model.load(r"C:\Users\R2\Downloads\drive-download-20230423T173351Z-001\All_LightAssyHead_April_18_model.pth", identifiers)
# identifiers = ['AIR CLEANER ASSY 17200-ABW-2000','AIR-C ASSY 17200-AAW-3000','AIR-C SUB ASSY 1720A-AAB-A000','AIR-C SUB ASSY- 1720A-AAB-A000','AIR-C SUB ASSY 1720A-AAK-2000','BASH PLATE 64300-ABW 3000','BASH PLATE 64300-ABW-1000','COVER CENTER 80151-AAW-0000','COVER HANDLE RR SUB ASSY- 5320A-ABG-3000','COVER HANDLE RR SUB ASSY- 5320A-ABH-3000','COVER INNER SUB ASSY 8113A-AAW-3000','COVER TOOLBOX 80102-AAB-A000','COWL FR. CENTER SUB ASSY 6420A-AAK-1000','FRONT	COWL SUB ASSY 6130A- AAB-A000','GRIP L REAR-50450-AAB-A000','GRIP R REAR-50400-AAB-A000','GRIP REAR SUB ASSY 5040A-ABW-0000','GRIP REAR(MAT AXIS GREY) 5035-KN-9700','HEAD LIGHT BRAKET SUB AS HEADLIGHT SUPPORT-BKT- 6131A-ABW-1000','HEAD LIGHT BRAKET SUB AS HEADLIGHT SUPPORT-BKT- 6131B-ABW-0000','HEAD LIGHT BRAKET SUB AS HEADLIGHT SUPPORT-BKT- 6131B-ABW-1000','HEAD LIGHT BRAKET SUB AS HEADLIGHT SUPPORT-BKT- 6131C-ABW-0000','LIGHT ASSY HEAD -33100-AAK-2010','LIGHT ASSY HEAD -33100-AAW-1020','METER ASSY COMB 3710A-ABW-1010','METER ASSY COMB 37100-AAB-A210','METER ASSY COMB 37100-AAB-C010','METER ASSY COMB 37100-AAK-6010','METER ASSY COMB 37100-ABA-0010','MIRROR ASSY L BACK - 88120-AAB-A000','MIRROR ASSY L BACK - 88120-AAW-0000','MIRROR ASSY L BACK - 88120-ABA-300T','MIRROR ASSY L BACK - 88120-ABG-2000','MIRROR ASSY L BACK PBM-IN - 88120-AAK-H01R','MIRROR ASSY LH(BLACK) -88120-AAB-100R','MIRROR ASSY R BACK - 88110-AAB-A000','MIRROR ASSY R BACK - 88110-ABA-300T','MIRROR ASSY R BACK - 88110-ABG-2000','MIRROR ASSY R BACK -88110-AAW-0000','MIRROR ASSY R BACK PBM- R - 88110-AAK-H01R','MIRROR ASSY RH(BLACK) -88110-AAB-100R','RR FENDER COVER 88150-KVN-9000','RR FENDER SUB ASSY 8010A-AAW-0000']
model = core.Model.load(r"C:\Users\REKHA\Downloads\defult1.pth", identifiers)
global part

def readPattern_new():
    global openFile
    global isCapture
    global folder_chr
    global part
    if openFile == True:
        cv2_image = cv2.imread(filepath, 1)
        image = utils.read_image(filepath)
        img = image.copy()
        img1 = image.copy()
        
        # openFile = False
    if isCapture == True:
        print("isCapture")
        cv2_image = cv2.imread("Saved.jpg", 1)
        image = utils.read_image("Saved.jpg")
        isCapture = False
    
    if folder_chr == True:
        image = cv2.imread(os.path.join(filenames,filename))
        cv2_image = cv2.imread(os.path.join(filenames,filename))
        img = image.copy()
        img1 = image.copy()
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        folder_chr = False
   

    tic=time.time()

 
    predictions = model.predict(image)
    toc=time.time()
    # print('Done in {:.4f} seconds'.format(toc - tic))
    labels, boxes, scores = predictions
    # print(scores)
    thresh = 0.7
    filtered_indices = np.where(scores > thresh)
    filtered_scores = scores[filtered_indices]
    print(filtered_scores)
    filtered_boxes = boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    cv2_image = image
    i = 0   

    while i < len(filtered_boxes):
        if(filtered_scores[0] > 0.7):
            img = image.copy()
            startPoint = (int(filtered_boxes[i][0]), int(filtered_boxes[i][1]))
            endPoint = (int(filtered_boxes[i][2]), int(filtered_boxes[i][3]))
            
            text = filtered_labels[0]
            position = (10, 50)  # Coordinates of the text (x, y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            color = (0, 255, 0)  # Text color in BGR format
            thickness = 2  # Thickness of the text

            # Draw the text on the image
            cv2.putText(image,str(text) , position, font, scale, color, thickness)
   
            cv2.rectangle(img, startPoint, endPoint, (0, 0, 255), 5)
            
            cv2.namedWindow("final",cv2.WINDOW_FREERATIO)
            cv2.imshow("final",img)
            # cv2.waitKey(0)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # If 'q' is pressed, break the loop
                break
            # startPoint2 = (int(filtered_boxes[i][0])-50, int(filtered_boxes[i][1] )-40)
            # endPoint2 = (int(filtered_boxes[i][2])-50 ,int(filtered_boxes[i][3]) - 40)
            # #cv2.rectangle(cv2_image, startPoint, endPoint, (0, 0, 255), 5)
            # print(filtered_labels[0])
          
            # input = img.copy()
            # input1 = img.copy()
            # # print((input.shape))
     
            # print ('pass')
        
            # lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
            # A = lab[:,:,1]
            # cv2.imwrite("A.bmp", A)
           
            # resultimage = np.zeros((800, 800))
          
            # out = cv2.normalize(A,None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # out = out * 255
         
            # cv2.imwrite("out.bmp", out)
            # norm_mg = cv2.imread("out.bmp", 0)

            # kernel1 = np.ones((8,8), np.float32)/35                                                         
          
            # img = cv2.filter2D(src=norm_mg, ddepth=-1, kernel=kernel1)
           
            # blur = cv2.GaussianBlur(img, (0,0), sigmaX=4, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    
            # blur =~ blur
         
            # blur_out = cv2.normalize(blur,None,0, 0.5, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
           
            # blur_outw = blur_out * 255
         
            # cv2.imwrite("blur_outw.bmp", blur_outw)
            # blur_out1 = cv2.imread("blur_outw.bmp", 0)
         
            # ret, threshold = cv2.threshold(blur_out1, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
          
            # threshold =~ threshold
            
            # threshold = cv2.Canny(threshold, 50, 100)
            # kernel = np.ones([3,3], "uint8")

            # threshold = cv2.dilate(threshold , kernel, 2)
            # cv2.imwrite("threshold.bmp", threshold)
            # # cv2.namedWindow('threshold', cv2.WINDOW_FREERATIO)
            # # cv2.imshow('threshold',threshold)
            # image1 = threshold.copy()
            # rect_img = threshold.copy()
            # rect_img = cv2.cvtColor(rect_img, cv2.COLOR_GRAY2BGR)
            # #global col_img
            # col_img = rect_img.copy()
            # all_pixelsX = []
            # all_pixelsY = []

            # dataset=None
            # listarea = []
            # contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
            # for pic, contour in enumerate(contours):
            #     # print(str(contour))
            #     area = cv2.contourArea(contour)
            #     # print(len(contours))
            #     #print("area = "+str(area))
              
            #     x, y, w, h = cv2.boundingRect(contour)
            #     if( x>0 and area > 100000 and area < 20000000 ):
            #         #c += 1
            #         # cv2.drawContours(col_img, contours, pic, (255, 255, 255), -1)
            #         bx, by, bw, bh = cv2.boundingRect(contour)
            #         rect = cv2.minAreaRect(contour)
            #         box = cv2.boxPoints(rect)
            #         box = np.int0(box)
            #         def euclidian_distance(a,b):
            #             return np.linalg.norm(a-b)
            #         #global dataset
            #         dataset=[box[0],box[1],box[2],box[3]]
            #         dataset.sort(key=lambda x:box[0][0])
            #         dataset = np.array(dataset)
            #         dataset = sorted(dataset, key = lambda point: euclidian_distance(point,dataset[0]))

                    
            #         # draw minimum area rectangle (rotated rectangle)
            #         col_img = cv2.drawContours(col_img,[box],0,(0,255,255),2)

            #         cv2.circle(col_img,(dataset[0]), 3, (0,255,0), -1)
            #         cv2.circle(col_img,(dataset[1]), 3, (255,0,0), -1)
            #         run = True

            #         #print(threshold.size().height)
            #         # print((threshold.shape[0]))
            #         # print((threshold.shape[1]))

            #         #Img = np.zeros((512, 512, 3), dtype='uint8')     
            #         blank_image = np.zeros((threshold.shape[0],threshold.shape[1]), dtype='uint8')#hxw
            #         blank_image = cv2.line(blank_image, dataset[0], dataset[1], (255,255,255), 3)
            #         blank_image = cv2.line(blank_image, dataset[2], dataset[3], (255,255,255), 3) 

            #         bitwiseAnd = cv2.bitwise_and(threshold, blank_image)
            #         bitwiseAnd = cv2.dilate(bitwiseAnd , kernel, 2)
            #         cv2.imwrite("bitwiseAnd.bmp", bitwiseAnd)
            #         # cv2.namedWindow('AND', cv2.WINDOW_NORMAL)
            #         # cv2.imshow("AND", bitwiseAnd)
            #         contours1, hierarchy1 = cv2.findContours(bitwiseAnd, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
            #         cv2.drawContours(col_img, contours1, -1, (0, 255, 0), 3) 
            #         def AngleBtw2Points(pointA, pointB):
            #             changeInX = pointB[0] - pointA[0]
            #             changeInY = pointB[1] - pointA[1]
            #             return degrees(atan2(changeInY,changeInX))
            #         pts_list=[]
            #         for pic1, contour1 in enumerate(contours1):
            #             rect = cv2.minAreaRect(contour1)
            #             # print(rect[0])
            #             pts_list.append((int(rect[0][0]),int(rect[0][1])))

            #             cv2.circle(col_img,(int(rect[0][0]),int(rect[0][1])), 3, (0,0,255), -1)
                        
            #         angle = AngleBtw2Points(pts_list[0],pts_list[1])

            #         #def rotate(image, angle, center = None):
            #         #   (h, w) = image.shape[:2]

            #         #   if center is None:
            #         #      center = (w / 2, h / 2)

            #             # Perform the rotation
            #         # M = cv2.getRotationMatrix2D(center, angle, 1.0)
            #         # rotated = cv2.warpAffine(image, M, (w, h))

            #             #return rotated
            #         rotated1 = ndimage.rotate(threshold, angle)
            #         # cv2.namedWindow('rotated_image1', cv2.WINDOW_NORMAL)
            #         # cv2.imshow('rotated_image1', rotated1)

            #         org_rotated = ndimage.rotate(img1, angle)
            #         # cv2.namedWindow('org_rotated_image1', cv2.WINDOW_NORMAL)
            #         # cv2.imshow('org_rotated_image1', org_rotated)
            #         #rotated_image=rotate(threshold,angle,None)
            #         #cv2.namedWindow('rotated_image', cv2.WINDOW_NORMAL)
            #         #cv2.imshow('rotated_image', rotated_image)
                    
            #         cv2.imwrite("rotated.bmp",rotated1)
            #         cv2.imwrite("org_rotated.bmp",org_rotated)

            

            #         part=True
            #         rotated_image_process()
            #         # middle_body()
            #         #cv2.waitKey(0)
            #         cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #         cv2.imshow('image', col_img)
            # if(run == True):
                # break
        else:
            print("fail")
           # break


def color_detection():
    global part
    image = cv2.imread("org_ROI.bmp")
    img = cv2.imread("org_ROI.bmp")
    crop=image[17:188,327:431]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([0,0,0])
    upper_hue = np.array([80,80,100])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask',mask) 
    white_pixel_count = np.sum(mask == 255) 
    black=None
    print(white_pixel_count)
    image_rubber = cv2.imread("org_ROI.bmp")
    image_rubber_crop=image_rubber[27:190,332:425]
    cv2.namedWindow('image_rubber_crop', cv2.WINDOW_NORMAL)
    cv2.imshow('image_rubber_crop', image_rubber_crop)
    cv2.imwrite("image_rubber_crop.bmp", image_rubber_crop)



    ####################  rubber width##########################
    image_rubber_Filder2D=cv2.cvtColor(image_rubber_crop, cv2.COLOR_BGR2GRAY)
    kernel2 = np.array([-3, -3, 0,12,0,-3,-3]
                    ) 
  
    # Applying the filter2D() function 
    img = cv2.filter2D(src=image_rubber_Filder2D, ddepth=-2, kernel=kernel2) 
    cv2.imwrite("image_rubber_crop_filterimg.bmp", img)

    # ret, img = cv2.threshold(image_rubber_crop, 128, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("image_rubber_crop_THRE_128.bmp", img)
    ret, img = cv2.threshold(img, 68, 255, cv2.THRESH_BINARY)
    cv2.imwrite("image_rubber_crop_THRE.bmp", img)

    contours_, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    image_rubber_Draw=image_rubber_crop.copy()
    # print(len(contours_))
    rectangle_array = []
    for pic, contour in enumerate(contours_):
        # print(str(contour))
        area = cv2.contourArea(contour)        
        # 148,154
        # print("area = "+str(area)) 
        x, y, w, h = cv2.boundingRect(contour)  
        # print("height = "+str((y+h)-y))       
        if( area > 100 and ((y+h)-y)>15):
            # print("area = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_rubber_Draw, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            new_rectangle = {'x': x, 'y': y, 'width': x+w, 'height': y+h}
            rectangle_array.append(new_rectangle) 
    
    # WIN_20230223_11_11_46_Pro.jpg
    max_y = 0
    min_height = float('inf')  
    for rect in rectangle_array:
        # print(rect)      
        max_y = max(max_y, rect['y'])
        min_height = min(min_height, rect['height'])

    final_height = max_y-min_height

    cv2.imwrite("image_rubber_crop_contours.bmp", image_rubber_Draw)
    cv2.namedWindow('image_rubber_crop_contours', cv2.WINDOW_NORMAL)
    cv2.imshow('image_rubber_crop_contours', image_rubber_Draw)
    cv2.namedWindow('image_rubber_crop_THRE', cv2.WINDOW_NORMAL)
    cv2.imshow('image_rubber_crop_THRE', img)

    ###################  rubber width ##########################


    image_rubber = cv2.imread("org_ROI.bmp")
    image_rubber_crop=image_rubber[27:190,312:425]
    image_rubber_threshold=cv2.cvtColor(image_rubber_crop, cv2.COLOR_BGR2GRAY)
    ret, img_width = cv2.threshold(image_rubber_threshold, 126, 255, cv2.THRESH_BINARY)
    cv2.imwrite("image_rubber_crop_THRE_width.bmp", img_width)
    # img_width=~img_width
    contours_, hierarchy = cv2.findContours(img_width, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    image_rubber_Draw_width=image_rubber_crop.copy()
    # print(len(contours_))
    rectangle_array_width = []
    rectangle_array_width_final = []

    for pic, contour in enumerate(contours_):
        # print(str(contour))
        area = cv2.contourArea(contour)        
        # 148,154
        x, y, w, h = cv2.boundingRect(contour)          
        if( area > 300 ):
            # print("area width = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_rubber_Draw_width, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            new_rectangle = {'x': x, 'y': y, 'width': x+w, 'height': y+h}
            rectangle_array_width.append(new_rectangle) 
    

   

    sorted_rectangles_width = sorted(rectangle_array_width, key=lambda rect: rect['x'])  

    for rect in sorted_rectangles_width:
        if(rect['y']!=0 and rect['height']!=image_rubber_Draw_width.shape[0] ):  #and rect['width']!=image_rubber_Draw_width.shape[1]
            rectangle_array_width_final.append(rect)
            cv2.rectangle(image_rubber_Draw_width, (rect['x'],rect['y']), (rect['width'], rect['height']), (0,0,255), 2, 8, 0 )
            # print(rect)
    cv2.namedWindow('image_rubber_Draw_width', cv2.WINDOW_NORMAL)
    cv2.imshow('image_rubber_Draw_width', image_rubber_Draw_width)
   
    rubber_min_x = float('inf')
    rubber_max_width = 0
    for rect in rectangle_array_width_final:
        # print(rect)      
        rubber_min_x = min(rubber_min_x, rect['x'])
        rubber_max_width = max(rubber_max_width, rect['width'])

    final_width = rubber_max_width-rubber_min_x
    
    print(f"Rubber Final Width: {final_width}")
    print(f"Rubber Final Height: {final_height}")

    # image_center = cv2.imread("org_ROI.bmp")    
    image_center_crop=image[39:177,398:1180]
    cv2.namedWindow('image_center_crop', cv2.WINDOW_NORMAL)
    cv2.imshow('image_center_crop', image_center_crop)
    cv2.imwrite("image_center_crop.bmp", image_center_crop)
    
   #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
    image_center_width=image_center_crop.copy()
    


    hsv = cv2.cvtColor(image_center_width, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([0,0,0])
    upper_hue = np.array([115,110,170])
    image_mask_width = cv2.inRange(hsv, lower_hue, upper_hue)
    cv2.namedWindow('image_mask_width', cv2.WINDOW_NORMAL)
    cv2.imshow('image_mask_width', image_mask_width)
    cv2.imwrite("image_mask_width.bmp", image_mask_width)

    image_mask_width=~image_mask_width
    contours_, hierarchy = cv2.findContours(image_mask_width, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    image_center_Draw=image_center_crop.copy()
    # print(len(contours_))
    rectangle_array = []
    for pic, contour in enumerate(contours_):
        # print(str(contour))
        area = cv2.contourArea(contour)
        
        # 148,154
        x, y, w, h = cv2.boundingRect(contour)    
      
        if( area > 10000 ):
            # print("area = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_center_Draw, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )

            new_rectangle = {'x': x, 'y': y, 'width': x+w, 'height': y+h}           
            rectangle_array.append(new_rectangle) 
    center_min_x = float('inf')
    center_max_width = 0
    for rect in rectangle_array:
        # print(rect)      
        center_min_x = min(center_min_x, rect['x'])
        center_max_width = max(center_max_width, rect['width'])
    
    center_final_width=center_max_width-center_min_x
    cv2.namedWindow('image_center_Draw', cv2.WINDOW_NORMAL)
    cv2.imshow('image_center_Draw', image_center_Draw)
    cv2.imwrite("image_center_Draw.bmp", image_center_Draw)


    ##height
    image_center_height=image_center_crop.copy()
    hsv_height = cv2.cvtColor(image_center_height, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([0,0,0])
    upper_hue = np.array([115,160,210])
    image_mask_height = cv2.inRange(hsv_height, lower_hue, upper_hue)
    cv2.namedWindow('image_mask_height', cv2.WINDOW_NORMAL)
    cv2.imshow('image_mask_height', image_mask_height)
    cv2.imwrite("image_mask_height.bmp", image_mask_height)

    image_mask_height=~image_mask_height
    contours_, hierarchy = cv2.findContours(image_mask_height, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    image_center_Draw=image_center_crop.copy()
    # print(len(contours_))
    rectangle_array = []
    for pic, contour in enumerate(contours_):
        # print(str(contour))
        area = cv2.contourArea(contour)
        
        # 148,154
        x, y, w, h = cv2.boundingRect(contour)    
      
        if( area < 1500 and area >200):
            # print("area = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_center_Draw, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )

            new_rectangle = {'x': x, 'y': y, 'width': x+w, 'height': y+h}
            # print(new_rectangle)
            rectangle_array.append(new_rectangle) 

    sorted_rectangles = sorted(rectangle_array, key=lambda rect: rect['x'])
    first_three_rectangles = sorted_rectangles[:3]
    # print(first_three_rectangles)

    
    center_max_y =0
    center_min_height = float('inf')    
    for rect in first_three_rectangles:
        # print(rect)      
        center_max_y = max(center_max_y, rect['y'])
        center_min_height = min(center_min_height, rect['height'])
    
    center_final_height=center_max_y-center_min_height
    print("..........................................")
    print(f"Center Final Width: {center_final_width}")
    print(f"Center Final Height: {center_final_height}")
    print("..........................................")


    cv2.namedWindow('image_mask_height_Draw', cv2.WINDOW_NORMAL)
    cv2.imshow('image_mask_height_Draw', image_center_Draw)
    cv2.imwrite("image_mask_height_Draw.bmp", image_center_Draw)
    if (center_final_width > 600 and center_final_width< 700 and center_final_height > 70 and center_final_height < 120 and white_pixel_count>3000 and final_width > 80 and final_width < 130 ):#and final_height > 50 and final_height < 100
        part = part and True
    else:
        part = part and False

   
    return part

def middle_body():
    global col_img
    global part
    mid_img = cv2.imread("org_ROI.bmp",0)
    mid_img1 = cv2.imread("org_ROI.bmp")
    
    crop_left=mid_img[29:217,3:390]
    crop_left1=mid_img1[29:217,3:390]
    cv2.namedWindow('crop_img', cv2.WINDOW_NORMAL)
    cv2.imshow('crop_img', crop_left)
    
    # matchLoc,res=template_match(crop_left,left_part_template)

    # col_img = cv2.rectangle(col_img, (3,29), (390,217), (0,255,0), 2)
    # cv2.namedWindow('col_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('col_img', col_img)
    # if res>0.2:
    #     part=part and True
        # cv2.rectangle(col_img, (matchLoc[0]+3,matchLoc[1]+25), (matchLoc[0] + left_part_template.shape[0]+290, matchLoc[1] + left_part_template.shape[1]-240), (255,0,0), 2, 8, 0 )
    crop_img = cv2.equalizeHist(crop_left) 
    ret, threshold_img = cv2.threshold(crop_img, 140, 255, cv2.THRESH_BINARY)#140
    ker = np.ones((3, 3), np.uint8)
    # kernel2 = np.array([[-1, -1, -1], 
    #                 [-1, 8, -1], 
    #                 [-1, -1, -1]]) 
    # crop_img = cv2.filter2D(src=crop_img, ddepth=-1, kernel=kernel2) 
    threshold_img = cv2.erode(threshold_img, ker)
    cv2.namedWindow('crop_img1', cv2.WINDOW_NORMAL)
    cv2.imshow('crop_img1', threshold_img)
    con_count = 0
    contours_, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    # print(len(contours_))
    for pic, contour in enumerate(contours_):
        # print(str(contour))
        area = cv2.contourArea(contour)
        #print("area = "+str(area)) 
        x, y, w, h = cv2.boundingRect(contour)     
        if( x > 0 and y > 0 and area > 3000 and w < 350 and w> 300  ):
            x, y, w, h = cv2.boundingRect(contour)
            print("width"+str(w))
            cv2.rectangle(crop_left1, (x,y), (x+w, y+h), (255,200,0), 2, 8, 0 )

            # ROI = image[y-5:y-5+h+10, x-5:x-5+w+10]
            # cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
            # cv2.imshow('ROI', ROI)
            # org_ROI=org_img[y-5:y-5+h+10, x-5:x-5+w+10]
            cv2.namedWindow('crop_left1', cv2.WINDOW_NORMAL)
            cv2.imshow('crop_left1', crop_left1)
            # cv2.imwrite("org_ROI.bmp",org_ROI)
            # cv2.imwrite("ROI_temp.bmp",ROI)
            con_count += 1

    print(con_count) 
    if(con_count == 1):
        part=part and True
    else:
        part = part and False
    
    return part
# global part
# col_img = 
        
def body_part_checking():
    global part
    image = cv2.imread("org_ROI.bmp",0)
    col_img = cv2.imread("org_ROI.bmp")
    
    kernel2 = np.array([[-1, -1, -1], 
                    [-1, 8, -1], 
                    [-1, -1, -1]])
  
    # Applying the filter2D() function 
    img = cv2.filter2D(src=image, ddepth=-2, kernel=kernel2) 
    ret, threshold_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    diaker = np.ones((1, 1), np.uint8)
    threshold_img =cv2.morphologyEx(threshold_img, cv2.MORPH_BLACKHAT, diaker)
    cv2.imwrite("filterimg.bmp", img)
    cv2.imwrite("threshold_img.bmp", threshold_img)
    cv2.namedWindow('threshold_img', cv2.WINDOW_NORMAL)
    cv2.imshow('threshold_img', threshold_img)
    # # template saving
    # left_upper_part=img[5:69,448:535]
    # right_upper_part=img[4:72,565:649]
    # bottom_left_part=img[134:221,550:643]
    # bottom_right_part=img[131:206,676:1012]
    # middle_square=img[76:138,726:812]

    # cv2.imwrite("left_upper_part.bmp",left_upper_part)
    # cv2.imwrite("right_upper_part.bmp",right_upper_part)
    # cv2.imwrite("bottom_left_part.bmp",bottom_left_part)
    # cv2.imwrite("bottom_right_part.bmp",bottom_right_part)
    # cv2.imwrite("middle_square.bmp",middle_square)

    #ROI creation
    left_upper_part=img[4:80,440:535]
    right_upper_part=img[3:80,555:665]
    bottom_left_part=img[132:228,540:655]
    bottom_right_part=img[128:220,670:1022]
    middle_square=img[70:148,720:825]
    left_part=image[29:217,3:390]
    right_mid_part=image[1:227,423:1065]

    col_img = cv2.rectangle(col_img, (440,4), (535,80), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (555,3), (665,80), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (540,132), (655,228), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (670,128), (1022,220), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (720,70), (825,148), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (3,29), (390,217), (0,255,0), 2)
    col_img = cv2.rectangle(col_img, (423,1), (1065,227), (0,255,0), 2)

    left_upper_part_template = cv2.imread("left_upper_part.bmp", 0)
    right_upper_part_template = cv2.imread("right_upper_part.bmp", 0)
    bottom_left_part_template = cv2.imread("bottom_left_part.bmp", 0)
    bottom_right_part_template = cv2.imread("bottom_right_part.bmp", 0)
    middle_square_template = cv2.imread("middle_square.bmp", 0)
    left_part_template = cv2.imread("start_part1.bmp", 0)
    right_mid_part_template = cv2.imread("mid_part.bmp", 0)
    
    
    
    #upper left part
    matchLoc,res=template_match(left_upper_part,left_upper_part_template)
    print(matchLoc)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+440,matchLoc[1]), (matchLoc[0] + left_upper_part_template.shape[0]+470, matchLoc[1] + left_upper_part_template.shape[1]-20), (255,0,0), 2, 8, 0 )
    else:
        feature_ext(left_upper_part,left_upper_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False
    
    #upper right part
    matchLoc,res=template_match(right_upper_part,right_upper_part_template)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+555,matchLoc[1]+3), (matchLoc[0] + right_upper_part_template.shape[0]+575, matchLoc[1] + right_upper_part_template.shape[1]-20), (255,0,0), 2, 8, 0 )
    else:
        feature_ext(right_upper_part,right_upper_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False

    #bottom left part
    matchLoc,res=template_match(bottom_left_part,bottom_left_part_template)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+540,matchLoc[1]+130), (matchLoc[0] + bottom_left_part_template.shape[0]+540, matchLoc[1] + bottom_left_part_template.shape[1]+130), (255,0,0), 2, 8, 0 )
    else:
        feature_ext(bottom_left_part,bottom_left_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False

    #bottomrightpart
    matchLoc,res=template_match(bottom_right_part,bottom_right_part_template)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+660,matchLoc[1]+130), (matchLoc[0] + bottom_right_part_template.shape[0]+900, matchLoc[1] + bottom_right_part_template.shape[1]-90), (255,0,0), 2, 8, 0 )
        # cv2.rectangle(col_img, (matchLoc[0]+670,matchLoc[1]+125), (matchLoc[0] + bottom_right_part_template.shape[0]+670, matchLoc[1] + bottom_right_part_template.shape[1]+125), (255,0,0), 2, 8, 0 )
    else:
        feature_ext(bottom_right_part,bottom_right_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False

    #middlepart
    matchLoc,res=template_match(middle_square,middle_square_template)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+720,matchLoc[1]+70), (matchLoc[0] + middle_square_template.shape[0]+760, matchLoc[1] + middle_square_template.shape[1]+50), (255,0,0), 2, 8, 0 )
    else:
        feature_ext(middle_square,middle_square_template)
        if(part == True):
            part =part and True
        else:
            part=part and False
    matchLoc,res=template_match(left_part,left_part_template)

    # col_img = cv2.rectangle(col_img, (3,29), (390,217), (0,255,0), 2)
    # cv2.namedWindow('col_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('col_img', col_img)
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+3,matchLoc[1]+25), (matchLoc[0] + left_part_template.shape[0]+290, matchLoc[1] + left_part_template.shape[1]-240), (255,0,0), 2, 8, 0 )
    else :
        feature_ext(left_part,left_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False

    matchLoc,res=template_match(right_mid_part,right_mid_part_template)
    col_img = cv2.rectangle(col_img, (423,1), (1065,227), (0,255,0), 2)  
    if res>0.5:
        part=part and True
        cv2.rectangle(col_img, (matchLoc[0]+400,matchLoc[1]+1), (matchLoc[0] + right_mid_part_template.shape[0]+500, matchLoc[1] + right_mid_part_template.shape[1]+200), (255,0,0), 2, 8, 0 )
    else :
        feature_ext(left_part,left_part_template)
        if(part == True):
            part =part and True
        else:
            part=part and False
    if(part==True):
        print("childs parts are present")
    else:
        print("child parts are missing")
    
    # Shoeing the original and output image
    cv2.namedWindow('filter2d', cv2.WINDOW_NORMAL) 
    cv2.imshow('filter2d', img) 
    cv2.namedWindow('col_img', cv2.WINDOW_NORMAL) 
    cv2.imshow('col_img', col_img) 
    return part

def feature_ext(master, temp):
    global part
    mainImg = master.copy()
    FDTemplate = temp.copy()
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(FDTemplate, None)
    kp2, des2 = sift.detectAndCompute(mainImg, None)
    FLANN_INDEX_KDTREE=0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=30)  #50
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # print(kp1)
    # print(des1)
    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance: 
            good.append(m)
    # flip_img = cam1_img.copy()
    # flip_img = cv2.rotate(flip_img, cv2.ROTATE_180)
    # flip_img = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
    # # cv2.putText(flip_img, text="WHEEL ASSY FRONT DISC CEAT 44600-ABG-2100C", startPoint2, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)#(0, 0, 255)
    print("Len(good) = " + str(len(good)))

    if len(good) > 20 :
        part = part and True
    else:
        part = part and False

def template_match(master,template):
    # print(master.shape)
    # print(template.shape)
    res = cv2.matchTemplate(master, template, cv2.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
    # print(max_val)
    return max_loc,max_val

def rotated_image_process():
    global part
    # part = False
    image = cv2.imread("rotated.bmp", 0)
    org_img=cv2.imread("org_rotated.bmp")
    ret, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    template = cv2.imread("ROI.bmp")
    loc,res=template_match(org_img,template)
    ROI=None
    if res>0.7:
        contours_, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
        # print(len(contours_))
        for pic, contour in enumerate(contours_):
            # print(str(contour))
            area = cv2.contourArea(contour)
            #print("area = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)     
            if( x>0 and area > 100000 and area < 20000000 and w > 900 and w < 1150):
                x, y, w, h = cv2.boundingRect(contour)
                
                ROI = image[y-5:y-5+h+10, x-5:x-5+w+10]
                cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
                cv2.imshow('ROI', ROI)
                org_ROI=org_img[y-5:y-5+h+10, x-5:x-5+w+10]
                # cv2.namedWindow('org_ROI', cv2.WINDOW_NORMAL)
                # cv2.imshow('org_ROI', org_ROI)
                cv2.imwrite("org_ROI.bmp",org_ROI)
                # cv2.imwrite("ROI_temp.bmp",ROI)
                part=part and True
                # print(part)
    #         else:
    # #             if len(good) > 30 :
    # #     
    # # else:
    #             part = part and False

    else:
        rotated_ROI = ndimage.rotate(image, 180)
        rotated_org = ndimage.rotate(org_img, 180)
        cv2.namedWindow('ROI_rotate', cv2.WINDOW_NORMAL)
        cv2.imshow('ROI_rotate', rotated_ROI)
        contours_, hierarchy = cv2.findContours(rotated_ROI, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
        # pri/nt(len(contours_))
        for pic, contour in enumerate(contours_):
            # print(str(contour))
            area = cv2.contourArea(contour)
            #print("area = "+str(area)) 
            x, y, w, h = cv2.boundingRect(contour)     
            if( x>0 and area > 100000 and area < 20000000 and w > 900 and w < 1150):
                x, y, w, h = cv2.boundingRect(contour)
                
                ROI = image[y:y+h, x:x+w]
                cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
                cv2.imshow('ROI', ROI)
                org_ROI=rotated_org[y-5:y-5+h+10, x-5:x-5+w+10]
                # cv2.namedWindow('org_ROI', cv2.WINDOW_NORMAL)
                # cv2.imshow('org_ROI', org_ROI)
                cv2.imwrite("org_ROI.bmp",org_ROI)
                #cv2.imwrite("ROI.bmp",ROI)
                part=part and True
                # print(part)
    #         else:
    # #             if len(good) > 30 :
    # #     
    # # else:
    #             part = part and False
    if( part ==  True):
        color_check=color_detection()
        if(color_check == True):
            bodycheck = body_part_checking()
            if(bodycheck == True ):
                middlecheck = middle_body() 
                if(middlecheck == True):
                    print("ok")
                else:
                    print("fail")
            else:
                print("fail")
        else:
            print("fail")
    else: 
        print("fail")


def open_file():
    global filepath
    global isCapture
    global openFile
    global folder_chr
    filepath = askopenfilename(filetypes=[("Image Files", "*.jpg"), ("All Files", "*.*")])
    if not filepath:
        return
    with open(filepath, "r") as input_file:
        openFile = True
        isCapture = False
        folder_chr = False
        # test()
        # conturs1()
        # new_check()
        # threshold()
        # identifyCharacter()
        # template_macth()
        readPattern_new()
        # conturs1()
        # backgrounextract()
        # conturs()
        # greenback()


def openCam():
    global cap  
    i = 0
    cap=cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # print(camera)

def captureImage():
    global isCapture
    global openFile
    global folder_chr
    global i  

    if cap.isOpened():
        while True:
            global count
            # while True:
            ret, imgInput=cap.read()
            isCapture = True
            openFile = False
            folder_chr = False
           
            count = count + 1
            cv2.imwrite("Saved.jpg", imgInput)
            # cv2.imwrite(r'C:\Users\R2\Desktop\Images' + "\\" + str(count) + ".jpg", imgInput)
            # readPattern()
            if cv2.waitKey(1) & 0xFF==ord('a'):
                break
        cap.release()
        cv2.destroyAllWindows()


def browse_folder():    
    global filename
    global filenames

    filenames=filedialog.askdirectory()
    # print(filenames)
    # global browse_img,annotation_img,original_img
    global isCapture
    global openFile
    global folder_chr 
    global folderclick
    global nxtbtnclick
    global prvsbtnclick
    global originalImgTop
    global cv_img
    global originalImgSide
    folderclick=0
    nxtbtnclick=0
    prvsbtnclick=0
    outlist = []
    namefile = []
    i=0
    for filename in os.listdir(filenames):
        cv_img = cv2.imread(os.path.join(filenames,filename))
        if cv_img is not None:
            # images.append(img)
            # processSegment(img)
            print(cv_img.shape[1])
            print(cv_img.shape[0])
            folder_chr = True
            openFile = False
            isCapture = False
            i += 1
            # cv_img=cv2.resize(cv_img,(512,512))                
            # if len(cv_img.shape) == 3 :
            #     cv_img =  cv2.cvtColor(cv_img,  cv2.COLOR_BGR2GRAY)    
            # conturs1()         
            # prediction=new_model.predict([prepare(os.path.join(filenames,filename))])
            # threshold()
            # identifyCharacter()
            # new_check()
            # template_macth()
            # test()
            # greenback()
            # conturs()
            # readPattern()
            readPattern_new()
            cv2.imwrite(r'D:\HERO MOTOCORP\fork' + "\\" + str(i) + ".jpg", col_img)
            
            





window = tk.Tk()
window.geometry("40x150")
window.title("Hasbro Inspection")

fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=1)
fr_buttons.place(x=5, y=5)

btn_open = tk.Button(fr_buttons, text="BROWSE", command=open_file)
btn_open.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

btn_capture = tk.Button(fr_buttons, text = "FOLDER", command=browse_folder)
btn_capture.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

btn_openCam = tk.Button(fr_buttons, text="Open Cam", command=openCam)
btn_openCam.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

btn_capture = tk.Button(fr_buttons, text = "Capture", command=captureImage)
btn_capture.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

window.mainloop()