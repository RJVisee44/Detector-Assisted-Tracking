#!python3

import cv2
import argparse
import numpy as np
from utils import create_tracker, fix_bounds
from yolo import set_with_yolo,check_yolo

def yolo_tracker(ri,coi,chi,cls,tracker_type,img_list,path_to_images):
    
    #In first frame, use YOLO to predict location of hand 
    #Repeat this for 'coi' frames to ensure that it is actually a hand by using IOU
    #If IOU > 0.1 (With previous pred bbox), then correct
    #Use tracker to track object until failure
    #After failure, use YOLO in same manner as initialization to reinitialize tracker  
    
    #Write all results to the same file
    result_file = "result_%s.txt" % cls
    result_file = open(result_file,'w')
        
    with open(img_list) as f: 
        img_list = f.read().splitlines()
    
    result_file.write("%s\n" % img_list[0])
    
   
    ###############################
    #   Predict bounding boxes 
    ###############################
    
    print("Initializing tracker using YOLO")
    i = 1
    #Check to see if this frame contains a hand, otherwise load the next frame
    initial, i = set_with_yolo(img_list, i, result_file, cls,path_to_images, con_ious=coi)
    
    if initial != (0,0,0,0):
        frame = cv2.imread(path_to_images + '%s.jpg' % img_list[i])
        tracker = create_tracker(tracker_type)
        ok = tracker.init(frame, initial)        

        if ok == False:
            print("Unable to initialize tracker")
 
    check_iter = reset_iter = 1
    
    while i < len(img_list):
        result_file.write("%s\n" % img_list[i])   
        
        reset_iter += 1
        
        if initial == (0,0,0,0):
            check_iter += 1
            
            #Start timer
            timer = cv2.getTickCount()

            #Get frame
            path = path_to_images + '%s.jpg' % img_list[i]
            cv2.imread(path) 
    
            #Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
            if check_iter%chi == 0:
                check_iter = 1
                #print("CHECK INDEX: ",index)
                initial = check_yolo("%s/%s.jpg" % (path_to_images,img_list[i]),result_file,cls)
                
                if initial != (0,0,0,0):
                    i += 1
                    initial, i = set_with_yolo(img_list,i,result_file,cls,path_to_images,con_ious=coi-1)
                    
                    if initial != (0,0,0,0):
                        frame = cv2.imread(path_to_images + '%s.jpg' % img_list[i])
                        tracker = create_tracker(tracker_type)
                        ok = tracker.init(frame, initial) 
            else:
                result_file.write("Processing time: %.4f seconds\n" % (1/fps))
                result_file.write("Neither: %s [0 0 0 0]\n" % cls)
               
        elif reset_iter%ri != 0:
            
            #Start timer 
            timer = cv2.getTickCount()                         
            
            #Update and read next frame
            frame = cv2.imread(path_to_images + '%s.jpg' % img_list[i])                                          
                                    
            #Get estimate  
            ok,bbox = tracker.update(frame)
                
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            #Show bounding box
            if ok:
                #If tracked, write to file
                result_file.write("Processing time: %.4f seconds\n" % (1/fps))                        
            
                pred_bbox = int(bbox[0]),int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                pred_bbox = fix_bounds(pred_bbox[0],pred_bbox[1],pred_bbox[2],pred_bbox[3])
                result_file.write("%s track: %s %s\n" % (tracker_type, cls, np.array(pred_bbox)))
                result_file.close()
                
                # Tracking success for tracker 1
                p1,p2 = (int(pred_bbox[0]),int(pred_bbox[1])), (int(pred_bbox[2]),int(pred_bbox[3]))
                cv2.rectangle(frame, p1, p2, (0,255,0), 5, 1)
                
                # Display tracker type on frame
                cv2.putText(frame, " %s: %s Tracker" % (cls,tracker_type), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            
                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            
                # Display result
                cv2.imshow("Tracking", frame)
                
            else:
                #Tracking failure, use detector right away
                cv2.putText(frame, "Tracking Failure Detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),4)
                print("Tracking Failure Detected")
                initial, i = set_with_yolo(img_list, i, result_file, cls,path_to_images, con_ious=coi)
                
                if initial == None:
                    break
                
                if initial != (0,0,0,0):
                    tracker = create_tracker(tracker_type)
                    frame = cv2.imread(path_to_images + "%s.jpg" % img_list[i])            
                    ok = tracker.init(frame,initial)
                
        elif reset_iter%ri == 0:
            print("{}th frame: {}".format(ri,reset_iter))
            reset_iter = 1
            initial, i = set_with_yolo(img_list, i, result_file, cls,path_to_images, con_ious=coi)
            
            if initial == None:
                break
            
            if initial != (0,0,0,0):
                tracker = create_tracker(tracker_type)
                frame = cv2.imread(path_to_images + "%s.jpg" % img_list[i])            
                ok = tracker.init(frame,initial)

        i += 1
        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break    
    cv2.destroyAllWindows()
            
def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset_iters', dest='ri', type=int, default=0, help='Reset Iterations')
    parser.add_argument('--con_ious', dest='coi',type=int, default=0, help='Consecutive IoUs before Initialization')
    parser.add_argument('--check_iters', dest='chi', type=int, default=0, help='Check Iterations')
    parser.add_argument('--cls',dest='cls',type=str,default=None,help='Object for tracking')
    parser.add_argument('--tracker_type',dest='tracker_type',type=str,default='KCF',help='Tracker to combine with YOLO. Try: "MF","KCF","MIL","OLB"')
    parser.add_argument('--img_list', dest='img_list', type=str, default=0, help='Path to list of filename test images')
    parser.add_argument('--image_path',dest="path_to_images",type=str,default=None,help='Absolute path to images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print('DAT using %s for %s_%s_%s' % (args.tracker_type, args.ri, args.coi, args.chi))
    yolo_tracker(args.ri,args.coi,args.chi,args.cls,args.tracker_type,args.img_list,args.path_to_images)
