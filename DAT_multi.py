# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:53:01 2019

@author: ryan4
"""

import cv2
import argparse
import numpy as np
from utils import create_tracker, fix_bounds, initialize_tracker
from yolo_multi import set_with_yolo_multi,check_yolo_multi

def yolo_tracker(ri,coi,chi,tracker_type,img_list,path_to_images):

    result_file = open("result_LR.txt",'w')
   
    with open(img_list) as f: 
        img_list = f.read().splitlines()
        
    ###############################
    #   Predict bounding boxes 
    ###############################
    print("Initializing tracker using YOLOv2")   
    i = 0
    initial, i = set_with_yolo_multi(img_list, i, result_file,path_to_images, con_ious=coi)
    
    path = path_to_images + '%s.jpg' % img_list[i]
    
    #Initialize both trackers
    if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):                
        tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path)   
    
    reset_iter = check_iter = check_iter_r = check_iter_l = 1    
    
    while i < len(img_list):
        result_file.write("%s\n" % img_list[i])   
            
        reset_iter += 1
        
        #Both hands aren't initialized
        if (initial[0] == (0,0,0,0)) and (initial[1] == (0,0,0,0)):
            check_iter += 1
            
            #Get frame
            path = path_to_images + '%s.jpg' % img_list[i]
                        
            #Start timer, processing time is only time required to load image
            timer = cv2.getTickCount()
            
            cv2.imread(path) 
    
            #Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
            #Check iterations, look for hands
            if check_iter%chi == 0:
                reset_iter = check_iter = check_iter_r = check_iter_l = 1

                initial = check_yolo_multi(path,result_file)
                
                if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):
                    tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path)
                
            else:
                result_file.write("Processing time: %.4f seconds\n" % (1/fps))
                result_file.write("Neither: L [0 0 0 0]\n")
                result_file.write("Neither: R [0 0 0 0]\n")
               
        
        elif reset_iter%ri != 0:
            
            #Left hand not initialized, check but go with reset iters
            if initial[0] == (0,0,0,0):    
                
                check_iter_l += 1
                path = path_to_images + '%s.jpg' % img_list[i]
                
                if check_iter_l%chi == 0:
                    check_iter = check_iter_r = check_iter_l = 1
                    
                    initial = check_yolo_multi(path,result_file)
                    
                    #Left hand found
                    if (initial[0] != (0,0,0,0)):
                        frame = cv2.imread(path_to_images + '%s.jpg' % img_list[i])
                        tracker_l = create_tracker(tracker_type,both=False)
                        okl = tracker_l.init(frame, initial[0]) 
                        okr,bbox = tracker_r.update(frame) #Need to update tracker
                
                        if (okl == False):
                            print("Unable to initialize tracker")
  
                else:                                      
                    #Start timer 
                    timer = cv2.getTickCount()                         
                    
                    #Update right and read next frame
                    frame = cv2.imread(path)                                          
                                            
                    #Get estimate  
                    okr,bbox = tracker_r.update(frame)
                        
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                    #Successful right track
                    if okr:
                        #If tracked, write to file
                        result_file.write("Processing time: %.4f seconds\n" % (1/fps))                        
                    
                        pred_bbox = int(bbox[0]),int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                        pred_bbox = fix_bounds(pred_bbox[0],pred_bbox[1],pred_bbox[2],pred_bbox[3])
                        result_file.write("Neither: L [0 0 0 0]\n")
                        result_file.write("%s track: R %s\n" % (tracker_type, np.array(pred_bbox)))
                        
                        # Tracking success for tracker 1
                        p1,p2 = (int(pred_bbox[0]),int(pred_bbox[1])), (int(pred_bbox[2]),int(pred_bbox[3]))
                        cv2.rectangle(frame, p1, p2, (0,255,0), 5, 1)
                        
                        # Display tracker type on frame
                        cv2.putText(frame, "%s Tracker" % tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                    
                        # Display FPS on frame
                        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                    
                        # Display result
                        cv2.imshow("Tracking", frame)

                    else:
                        #Tracking failure, use detector right away, reset both
                        cv2.putText(frame, "Tracking Failure Detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),4)
                        print("Tracking Failure Detected")
                        initial, i = set_with_yolo_multi(img_list, i, result_file,path_to_images, con_ious=coi)
                        path = path_to_images + '%s.jpg' % img_list[i]
                                                    
                        if initial == None:
                            break
                        
                        if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):
                            tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path) 
                        
                        check_iter = check_iter_r = check_iter_l = 1
                        reset_iter = 1

            #Right hand not initialized, check but go with reset iters
            elif initial[1] == (0,0,0,0):    
                
                check_iter_r += 1
                path = path_to_images + '%s.jpg' % img_list[i]
                
                if check_iter_r%chi == 0:
                    check_iter = check_iter_r = check_iter_l = 1
                    
                    initial = check_yolo_multi(path,result_file)
                    
                    if (initial[1] != (0,0,0,0)): #Found right
                        frame = cv2.imread(path_to_images + '%s.jpg' % img_list[i])
                        tracker_r = create_tracker(tracker_type,both=False)
                        okr = tracker_r.init(frame, initial[1]) 
                        okl,bbox = tracker_l.update(frame) #Need to update tracker
                
                        if (okr == False):
                            print("Unable to initialize tracker")
  
                else:    
                    #Start timer 
                    timer = cv2.getTickCount()                         
                    
                    #Update right and read next frame
                    frame = cv2.imread(path)                                          
                                            
                    #Get estimate  
                    okl,bbox = tracker_l.update(frame)
                        
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                    #Successful left track
                    if okl:
                        #If tracked, write to file
                        result_file.write("Processing time: %.4f seconds\n" % (1/fps))                        
                    
                        pred_bbox = int(bbox[0]),int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                        pred_bbox = fix_bounds(pred_bbox[0],pred_bbox[1],pred_bbox[2],pred_bbox[3])
                        result_file.write("%s track: L %s\n" % (tracker_type, np.array(pred_bbox)))
                        result_file.write("Neither: R [0 0 0 0]\n")
                        
                        # Tracking success for tracker 1
                        p1,p2 = (int(pred_bbox[0]),int(pred_bbox[1])), (int(pred_bbox[2]),int(pred_bbox[3]))
                        cv2.rectangle(frame, p1, p2, (0,255,0), 5, 1)
                        
                        # Display tracker type on frame
                        cv2.putText(frame, "%s Tracker" % tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                    
                        # Display FPS on frame
                        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                    
                        # Display result
                        cv2.imshow("Tracking", frame)

                    else:
                        #Tracking failure, use detector right away, reset both
                        cv2.putText(frame, "Tracking Failure Detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),4)
                        initial, i = set_with_yolo_multi(img_list, i, result_file,path_to_images, con_ious=coi)
                        path = path_to_images + '%s.jpg' % img_list[i]
                                                    
                        if initial == None:
                            break
                        
                        if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):
                            tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path)
                        
                        check_iter = check_iter_r = check_iter_l = 1
                        reset_iter = 1 
        
            #Both hands are initialized
            else:
                path = path_to_images + '%s.jpg' % img_list[i]
                
                #Start timer 
                timer = cv2.getTickCount()                         
                
                #Update right and read next frame
                frame = cv2.imread(path)                                          
                                        
                #Get estimate  
                okl,bbox_l = tracker_l.update(frame)
                okr,bbox_r = tracker_r.update(frame)
                    
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)                    
                
                #Successful tracks 
                if okr and okl:
                    #If tracked, write to file
                    result_file.write("Processing time: %.4f seconds\n" % (1/fps))                        
                
                    pred_bbox = int(bbox_l[0]),int(bbox_l[1]), int(bbox_l[0] + bbox_l[2]), int(bbox_l[1] + bbox_l[3])
                    pred_bbox = fix_bounds(pred_bbox[0],pred_bbox[1],pred_bbox[2],pred_bbox[3])
                    result_file.write("%s track: L %s\n" % (tracker_type, np.array(pred_bbox)))
                    p1,p2 = (int(pred_bbox[0]),int(pred_bbox[1])), (int(pred_bbox[2]),int(pred_bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0,255,0), 5, 1)
                        
                    pred_bbox = int(bbox_r[0]),int(bbox_r[1]), int(bbox_r[0] + bbox_r[2]), int(bbox_r[1] + bbox_r[3])
                    pred_bbox = fix_bounds(pred_bbox[0],pred_bbox[1],pred_bbox[2],pred_bbox[3])
                    result_file.write("%s track: R %s\n" % (tracker_type, np.array(pred_bbox)))
                    p1,p2 = (int(pred_bbox[0]),int(pred_bbox[1])), (int(pred_bbox[2]),int(pred_bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0,255,0), 5, 1)    
                    
                    # Display tracker type on frame
                    cv2.putText(frame, "%s Tracker" % tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                
                    # Display FPS on frame
                    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                
                    # Display result
                    cv2.imshow("Tracking", frame)

                else:
                    #Tracking failure, use detector right away, reset both
                    cv2.putText(frame, "Tracking Failure Detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),4)
                    initial, i = set_with_yolo_multi(img_list, i, result_file,path_to_images, con_ious=coi)
                    path = path_to_images + '%s.jpg' % img_list[i]
                                                
                    if initial == None:
                        break
                    
                    if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):
                        tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path) 
                    
                    check_iter = check_iter_r = check_iter_l = 1
                    reset_iter = 1 
        
        # Reset both upon reset iteration
        elif reset_iter%ri == 0:
            #print("{}th frame: {}".format(ri,reset_iter))
            reset_iter = check_iter = check_iter_r = check_iter_l = 1
            initial, i = set_with_yolo_multi(img_list, i, result_file,path_to_images, con_ious=coi)
            path = path_to_images + '%s.jpg' % img_list[i]
                                                
            if initial == None:
                break
            
            if (initial[0] != (0,0,0,0)) and (initial[1] != (0,0,0,0)):
                tracker_l,tracker_r = initialize_tracker(initial,tracker_type,path)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    
        i += 1
    
    cv2.destroyAllWindows()
    result_file.close()
