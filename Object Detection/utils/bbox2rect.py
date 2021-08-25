import matplotlib.pyplot as plt
import cv2
def bbox2rect(bbox, color = 'red', label = ''):
    return plt.Rectangle(xy = (bbox[0], bbox[1]), width = bbox[2],
                         height = bbox[3], fill = False, 
                         edgecolor = color, linewidth = 1.5)

def plot_bbox_label(img, label, bbox, color = 'red'):
    #xmin,ymin,w,h
    #画边框
    c1 = (int(bbox[0]),int(bbox[1]))
    c2 = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
    cv2.rectangle(img, c1, c2, color = color, thickness = 2)
    if label != '':
        #画label
        t_size = cv2.getTextSize(label, 0, 1.1, 2)[0]
        c1 = (int(bbox[0]),int(bbox[1] - t_size[1] - 14))
        c2 = (int(bbox[0] + t_size[0]),int(bbox[1]))

        cv2.rectangle(img, c1, c2, color, -1,cv2.LINE_AA)
        c1 = (int(bbox[0]),int(bbox[1]-12))
        cv2.putText(img, label, c1,0,1,[255,255,255],2,cv2.LINE_AA)
    return img

def plot_anchor_box(img, label, bbox, color = 'red'):
    #xc,yc,w,h
    #画边框
    c1 = (int(bbox[0]-bbox[2]/2),int(bbox[1] - bbox[3]/2))
    c2 = (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2))
    cv2.rectangle(img, c1, c2, color = color, thickness = 2)
    if label != '':
        #画label
        t_size = cv2.getTextSize(label, 0, 1.1, 2)[0]
        c1 = (int(bbox[0]-bbox[2]/2),int(bbox[1] - bbox[3]/2 - t_size[1] - 14))
        c2 = (int(bbox[0]-bbox[2]/2 + t_size[0]),int(bbox[1]-bbox[3]/2))

        cv2.rectangle(img, c1, c2, color, -1,cv2.LINE_AA)
        c1 = (int(bbox[0]-bbox[2]/2),int(bbox[1] - bbox[3]/2-12))
        cv2.putText(img, label, c1,0,1,[255,255,255],2,cv2.LINE_AA)
    return img