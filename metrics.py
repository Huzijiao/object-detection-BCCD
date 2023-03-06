# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

def IoU(a_, b_):
    '''
    Calculate le overlap
    input:
        a,b: labels with : index of image, i, j, h, l = a[0:5]
    return:
        fraction (intersect surface / Union surface)
    '''

    a_x_ = a_[1] + a_[3]
    a_y_ = a_[2] + a_[4]
    b_x_ = b_[1] + b_[3]
    b_y_ = b_[2] + b_[4]

    overlapX_ = b_[3] + a_[3] - (max(a_x_, b_x_) - min(a_[1], b_[1]))
    overlapY_ = b_[4] + a_[4] - (max(a_y_, b_y_) - min(a_[2], b_[2]))
    if overlapY_ == min(a_[2], b_[2]) and overlapX_ == min(a_[1], b_[1]):
        return 1
    if overlapY_ <= 0 or overlapX_ <= 0:
        Inter_ = 0
    else:
        Inter_ = overlapX_ * overlapY_
    Union_ = a_[3] * a_[4] + b_[3] * b_[4] - Inter_

    return Inter_ / Union_




import xml.etree.ElementTree as ET #读取xml文件
import os
import cPickle #序列化存储模块
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    # 解析xml文件，将GT框信息放入一个列表
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

# 单个计算AP的函数，输入参数为精确率和召回率，原理见上面
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 如果使用2017年的计算AP的方式(插值的方式)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
       # 使用2010年后的计算AP值的方式
        # 这里是新增一个(0,0)，方便计算
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 主函数
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: 产生的txt文件，里面是一张图片的各个检测框结果。
    annopath: xml 文件与对应的图像相呼应。
    imagesetfile: 一个txt文件，里面是每个图片的地址，每行一个地址。
    classname: 种类的名字，即类别。
    cachedir: 缓存标注的目录。
    [ovthresh]: IOU阈值，默认为0.5，即mAP50。
    [use_07_metric]: 是否使用2007的计算AP的方法，默认为Fasle
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # 首先加载Ground Truth标注信息。
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # 即将新建文件的路径
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # 读取文本里的所有图片路径
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    # 获取文件名，strip用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    imagenames = [x.strip() for x in lines]
    #如果cachefile文件不存在，则写入
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            #annopath.format(imagename): label的xml文件所在的路径
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            #写入cPickle文件里面。写入的是一个字典，左侧为xml文件名，右侧为文件里面个各个参数。
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {}# 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #  different基本都为0/False
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult) #自增，~difficult取反,统计样本个数
        # # 记录Ground Truth的内容
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets 读取某类别预测输出
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines] # 图片ID
    confidence = np.array([float(x[1]) for x in splitlines]) # IOU值
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # bounding box数值

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    #重排bbox，由大概率到小概率。
    BB = BB[sorted_ind, :]
    # 图片重排，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap