# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:08:45 2019

@author: ryan4
"""

import cv2

def process_yolo_pred_bbox(boxes):
    # process predict box
    pred_bb = []
    pred_cls = []
    pred_conf = []
    
    for box in boxes:
        xmin = box[2][0] - (box[2][2]/2)
        ymin = box[2][1] - (box[2][3]/2)
        xmax = xmin + box[2][2]
        ymax = ymin + box[2][3]
        pred_bb.append([round(xmin), round(ymin), round(xmax), round(ymax)])
        pred_cls.append(box[0])
        pred_conf.append(box[1])
        
    return pred_bb, pred_cls, pred_conf

def fix_bounds(xmin,ymin,xmax,ymax):
    if xmax > 720:
        xmax = 719
    if xmin < 0:
        xmin = 0
    if ymax > 405:
        ymax = 404
    if ymin < 0:
        ymin = 0
    return int(xmin),int(ymin),int(xmax),int(ymax)

def create_tracker(tracker_type):
    #This function modulates the call to create a tracker
    if tracker_type == 'OLB':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MF':
        tracker = cv2.TrackerMedianFlow_create()
    else:
        tracker = cv2.TrackerGOTURN_create()
        
    return tracker

def bbox_inter_over_union(bbox_gt, bbox_pred):
    
    """
    Function returns the intersection over union (IoU) between a ground truth
    bounding box (bbox) and predicted bbox
    """
    
    #Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(int(bbox_gt[0]), int(bbox_pred[0]))
    yA = max(int(bbox_gt[1]), int(bbox_pred[1]))
    xB = min(int(bbox_gt[2]), int(bbox_pred[2]))
    yB = min(int(bbox_gt[3]), int(bbox_pred[3]))
        
    #Compute the area of intersection rectangle
    width = xB - xA + 1
    height = yB - yA + 1
    #If no overlap, either width or height negative
    if width < 0 or height < 0:
        inter_area = 0
    else:
        inter_area = width * height
    
    #Compute the area of both the prediction and ground-truth rectangles
    bboxA_area = (int(bbox_gt[2]) - int(bbox_gt[0]) + 1) * (int(bbox_gt[3]) - int(bbox_gt[1]) + 1)
    bboxB_area = (int(bbox_pred[2]) - int(bbox_pred[0]) + 1) * (int(bbox_pred[3]) - int(bbox_pred[1]) + 1)
 
    #Compute the intersection over union by taking the intersection
    #Area and dividing it by the sum of prediction + ground-truth
    #Areas - the intersection area
    iou = inter_area / float(bboxA_area + bboxB_area - inter_area)
 
    #Return the intersection over union value
    return iou