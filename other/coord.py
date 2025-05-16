import copy

def clip01(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return x

def unmap_roi_point_inplace(roi, pt):
    pt[0]=roi[0]+pt[0]*(roi[2]-roi[0])
    pt[1]=roi[1]+pt[1]*(roi[3]-roi[1])

def unmap_roi_box_inplace(roi, box):
    box[0]=roi[0]+box[0]*(roi[2]-roi[0])
    box[1]=roi[1]+box[1]*(roi[3]-roi[1])
    box[2]=roi[0]+box[2]*(roi[2]-roi[0])
    box[3]=roi[1]+box[3]*(roi[3]-roi[1])

def unmap_roi_point(roi, pt):
    ret=[roi[0]+pt[0]*(roi[2]-roi[0]),
         roi[1]+pt[1]*(roi[3]-roi[1])]
    return ret

def unmap_roi_box(roi, box):
    ret=[roi[0]+box[0]*(roi[2]-roi[0]),
         roi[1]+box[1]*(roi[3]-roi[1]),
         roi[0]+box[2]*(roi[2]-roi[0]),
         roi[1]+box[3]*(roi[3]-roi[1])]
    return ret

def box_w(b1):
    """
    Return width of box
    """
    return b1[2]-b1[0]

def box_h(b1):
    """
    Return height of box
    """
    return b1[3]-b1[1]

def box_a(b1):
    """
    Return area of box
    """
    return (b1[3]-b1[1])*(b1[2]-b1[0])

def box_i(b1, b2):
    """
    Return area of box intersection
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    return ai

def box_iou(b1, b2):
    """
    Computes the iou between two boxes

    Args:
        b1, b2: list [x1,y1,x2,y2] in xyxy format
        
    Returns:
        float iou
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    if iw==0:
        return 0
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    if ai==0:
        return 0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    iou=(iw*ih)/(a1+a2-ai+1e-7)
    return iou

def box_iou_relaxed_person(b1, b2):
    iou=box_iou(b1,b2)
    if iou==0:
        return 0
    b1m=copy.copy(b1)
    b2m=copy.copy(b2)
    b1m[3]=min(b1[3], b1[1]+b1[2]-b1[0])
    b2m[3]=min(b2[3], b2[1]+b1[2]-b1[0])
    return 0.5*iou+0.5*box_iou(b1m, b2m)

def box_ioma(b1, b2):
    """
    Computes the ioma (intersection over minimum
    area) between two boxes

    Args:
        b1, b2: list [x1,y1,x2,y2] in xyxy format
        
    Returns:
        float iou
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    iou=(iw*ih)/(min(a1,a2)+1e-7)
    return iou

def point_in_box(pt, box):
    if pt[0]<box[0] or pt[0]>box[2]:
        return None
    if pt[1]<box[1] or pt[1]>box[3]:
        return None
    d=abs(pt[0]-0.5*(box[0]+box[2]))
    d+=abs(pt[1]-0.5*(box[1]+box[3]))
    return d

def interpolate2(x, y, f):
    if isinstance(x, float) or isinstance(x, int):
        return (1.0-f)*x+f*y
    if isinstance(x, list):
        assert len(x)==len(y)
        return [(1.0-f)*x[i]+f*y[i] for i in range(len(x))]
    print("interpolate: unsupported type")
    exit()

def interpolate(x, y, f):
    if isinstance(x, float) or isinstance(x, int):
        return (1.0-f)*x+f*y
    if isinstance(x, list):
        assert len(x)==len(y)
        for i in range(len(x)):
            x[i]=(1.0-f)*x[i]+f*y[i]
        return
    print("interpolate: unsupported type")
    exit()