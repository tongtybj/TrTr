import sys
import cv2
import numpy as np

test_flag = False

# Note: if you have a exemplar size with 127 and a search size with 255, then you need instance size of 511 for later data augmentation in dataset
INSTANCE_SIZE = 511
EXEMPLAR_SIZE = 127
CONTEXT_AMOUNT= 0.5

def get_instance_size():
    return INSTANCE_SIZE

def get_exemplar_size():
    return EXEMPLAR_SIZE

def get_context_amount():
    return CONTEXT_AMOUNT


def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()



def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0]) # TODO: check the one pixel operation i.e., -1
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def siamfc_like_scale(bbox):

    bb_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]

    wc_z = bb_size[0] + CONTEXT_AMOUNT * sum(bb_size)
    hc_z = bb_size[1] + CONTEXT_AMOUNT * sum(bb_size)

    s_z = np.sqrt(wc_z * hc_z)

    scale_z = EXEMPLAR_SIZE / s_z

    return s_z, scale_z

def crop_image(image, bbox, padding=(0, 0, 0), instance_size = INSTANCE_SIZE):

    def pos_s_2_bbox(pos, s):
        return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


    bb_center = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    s_z =  siamfc_like_scale(bbox)[0]
    s_x = instance_size / EXEMPLAR_SIZE  * s_z
    #print("instance_size: {}; EXEMPLAR_SIZE: {}; s_z: {}; s_x: {}".format(instance_size, EXEMPLAR_SIZE, s_z, s_x))

    z = crop_hwc(image, pos_s_2_bbox(bb_center, s_z), EXEMPLAR_SIZE, padding)
    x = crop_hwc(image, pos_s_2_bbox(bb_center, s_x), instance_size, padding) # crop a size of s_x, then resize to instance_size

    #z = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if test_flag:

        rec_image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
        cv2.imshow('raw_image', rec_image)
        cv2.imshow('z_image', z)
        cv2.imshow('x_image', x)

        k = cv2.waitKey(40)
        if k == 27:         # wait for ESC key to exit
          sys.exit()

    return z, x


# for heatmap (copy from Center/Corner net)
def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

# for heatmap (copy from Center/Corner net)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

# for heatmap (copy from Center/Corner net)
def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

