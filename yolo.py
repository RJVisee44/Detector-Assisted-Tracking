# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:10:24 2019

@author: ryan4
"""

import numpy as np
import cv2
from intersection_over_union import bbox_inter_over_union
from utils import process_yolo_pred_bbox, fix_bounds
from darknet import performDetect

def set_with_yolo(img_list,index,result_file,cls,path_to_images,con_ious):
    #This function 1) Evaluates predictions compared to ground truth 
    #This function 2) Finds good IoU to (re)initialize tracker
    #Returns the results dataframe, current frame, and index for gt
    #Also returns bbox of final frame in (xmin,ymin,w,h) form
    con_iou = 0 
    first_find = initialize_tracker = True

    #Yolo Detection Files
    model_cfg = "yolov2_DAT.cfg"
    data_cfg = "yolov2_DAT.data"
    weights = "yolov2_DAT.weights"
        
    while initialize_tracker == True:        
        
        if index < len(img_list)-1:

            #Write filename to result file            
            result_file.write("%s\n" % img_list[index])                    
                
            #Get frame
            path = path_to_images + '%s.jpg' % img_list[index] 

            #Start timer
            timer = cv2.getTickCount()
                        
            #Get prediction, Image loaded in detect()
            detections = performDetect(imagePath=path, thresh= 0.30, configPath = model_cfg, weightPath = weights, metaPath= data_cfg, showImage= True, makeImageOnly = True, initOnly= False)
            
            #Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            result_file.write("Processing time: %.4f seconds\n" % (1/fps))
            
            #Process predictions
            boxes = []
            for detection in detections["detections"]:
                if detection[0] == cls:
                    boxes.append(detection)       
            pred_bb,pred_cls,pred_conf = process_yolo_pred_bbox(boxes)
                
            frame = cv2.imread(path)
                  
            #If more than one prediction, want the one with the highest confidence
            if len(pred_conf) > 0:
                pred_bb = pred_bb[pred_conf.index(max(pred_conf))] 
                pred_bb = fix_bounds(pred_bb[0],pred_bb[1],pred_bb[2],pred_bb[3])
                result_file.write("YOLOv2 det: %s %s\n" % (cls,np.array(pred_bb)))
                #Display
                cv2.rectangle(frame, (int(pred_bb[0]),int(pred_bb[1])), (int(pred_bb[2]),int(pred_bb[3])), (0,255,0), 5, 1)
            else:
                pred_bb = []
                result_file.write("YOLOv2 det: %s [0 0 0 0]\n" % cls)                
            
            # Display detection
            cv2.putText(frame, "Detecting using YOLO", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                            
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                            
            # Display result
            cv2.imshow("Detecting", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            
            #video_writer.write(frame)
            
            if first_find == True:
                if len(pred_bb) > 0:
                    con_iou = 1
                    con_tn = 0
                else:
                    con_tn = 1
                    con_iou = 0
                prev_pred = pred_bb
                first_find = False
            
            else:
                #Consecutive negatives
                if len(pred_bb) == 0 and len(prev_pred) == 0:
                    con_tn += 1
                    con_iou = 0
                #Switch to positive, but could be a FP
                elif len(pred_bb) > 0 and len(prev_pred) == 0:
                    con_iou += 1
                #Switch to negative, but could be a FN
                elif len(pred_bb) == 0 and len(prev_pred) > 0:
                    con_tn += 1
                #Consecutive positives
                elif len(pred_bb) > 0 and len(prev_pred) > 0:
                    iou = bbox_inter_over_union(prev_pred,pred_bb)
                    if iou > 0.1:
                        con_iou += 1
                        con_tn = 0
                
                prev_pred = pred_bb
                
                if con_iou >= con_ious:
                    print("Initialized")
                    initialize_tracker = False
                    initial = (prev_pred[0],prev_pred[1],prev_pred[2]-prev_pred[0],prev_pred[3]-prev_pred[1])
                    return initial, index
                
                if con_tn >= con_ious:
                    print("Hand lost for many frames")
                    initialize_tracker = False
                    initial = (0,0,0,0)
                    return initial, index
            
            index += 1
            
        else:
            initial = None #If at the max, don't need initial
            return initial, index
        
def check_yolo(path,result_file,cls):
    #This function quickly checks if hand has returned to frame
    #Yes or No return, along with updated results
    #Write all results to the same file
    
    #Yolo Detection Files
    model_cfg = "yolov2_DAT.cfg"
    data_cfg = "yolov2_DAT.data"
    weights = "yolov2_DAT.weights"
    
    #Start timer
    timer = cv2.getTickCount()
                    
    #Get prediction, Image loaded in detect()
    detections = performDetect(imagePath=path, thresh= 0.30, configPath = model_cfg, weightPath = weights, metaPath= data_cfg, showImage= True, makeImageOnly = True, initOnly= False)
        
    #Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    result_file.write("Processing time: %.4f seconds\n" % (1/fps))

    #Process predictions
    boxes = []
    for detection in detections["detections"]:
        if detection[0] == cls:
            boxes.append(detection)       
    pred_bb,pred_cls,pred_conf = process_yolo_pred_bbox(boxes)
                           
    #If more than one prediction, want the one with the highest confidence
    if len(pred_conf) > 0:
        pred_bb = pred_bb[pred_conf.index(max(pred_conf))] 
        
    #Predictions
    if pred_bb != []:
        print("Found something!")
        pred_bb = fix_bounds(pred_bb[0],pred_bb[1],pred_bb[2],pred_bb[3])
        result_file.write("YOLOv2 det: %s %s\n" % (cls,np.array(pred_bb)))
        initial = (pred_bb[0],pred_bb[1],pred_bb[2]-pred_bb[0],pred_bb[3]-pred_bb[1])
        return initial

    #No predictions
    else:
        print("Found Nothing :(")
        result_file.write("YOLOv2 det: %s [0 0 0 0]\n" % cls)
        initial = (0,0,0,0)
        return initial
