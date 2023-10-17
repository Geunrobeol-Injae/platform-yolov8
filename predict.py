# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)  # 색상 선정
data_deque = {}

deepsort = None

"""
1. count(founded_classes, im0)
- OpenCV 라이브러리의 함수들을 사용하여 이미지 im0에 여러 사각형과 텍스트를 그리는 작업
- 특정 객체(자동차,버스 등)의 수를 이미지에 표시함
"""
def count(founded_classes, im0):
    # founded_classes 딕셔너리의 아이템을 반복하며 각 클래스의 이름과 개수를 k, v에 담음. 반복문 내에서는 k의 값에 따라 다른 동작 수행
    for i, (k, v) in enumerate(founded_classes.items()):
        cnt_str = str(k) + ":" + str(v)
        height, width, _ = im0.shape
        # cv2.line(im0, (20,65+ (i*40)), (127,65+ (i*40)), [85,45,255], 30)

        # cv2.rectangle 함수를 사용하여 이미지에 사각형을 그리고
        # cv2.putText 함수를 사용하여 이미지에 텍스트를 그림
        if str(k) == 'helmet':
            i = 0
            cv2.rectangle(im0, (width - 190, 45 + (i * 40)), (width, 95 + (i * 40)), [85, 45, 255], -1, cv2.LINE_AA)
            cv2.putText(im0, cnt_str, (width - 190, 75 + (i * 40)), 0, 1, [255, 0, 0], thickness=2,
                        lineType=cv2.LINE_AA)
        elif str(k) == 'head':
            i = 1
            cv2.rectangle(im0, (width - 190, 45 + (i * 40)), (width, 95 + (i * 40)), [85, 45, 255], -1, cv2.LINE_AA)
            cv2.putText(im0, cnt_str, (width - 190, 75 + (i * 40)), 0, 1, [255, 0, 0], thickness=2,
                        lineType=cv2.LINE_AA)
        elif str(k) == 'person':
            i = 2
            cv2.rectangle(im0, (width - 190, 45 + (i * 40)), (width, 95 + (i * 40)), [85, 45, 255], -1, cv2.LINE_AA)
            cv2.putText(im0, cnt_str, (width - 190, 75 + (i * 40)), 0, 1, [255, 0, 0], thickness=2,
                        lineType=cv2.LINE_AA)

"""
2. init_tracker()
- 객체 추적 알고리즘인 DeepSort 초기화
"""
def init_tracker():
    global deepsort
    
    # 설정 불러오기
    cfg_deep = get_config()  # get_config 함수를 호출하여 cfg_deep 변수에 설정을 불러옴
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")  # cfg_deep 설정 객체에 ~deep_sort.yaml 파일의 설정을 병합함

    # deepsort 객체는 DeepSort 클래스의 인스턴스로 초기화됨
    # 초기화에 필요한 매개변수들은 cfg_deep 설정 객체에서 가져옴
    # REID_CKPT : 모델의 체크포인트 경로, MAX_DIST : 최대 거리 임계값
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


"""
3. xyxy_to_xywh(*xyxy), xyxy_to_tlwh(bbox_xyxy)
- 바운딩 박스(bounding box)의 형식 변환
- 바운딩 박스 : 객체 검출(object detection)에서 객체의 위치를 표현하는 데 사용되는 직사각형
- 바운딩 박스는 다양한 형식으로 표현될 수 있는데, 주로 사용되는 두 형식인 xyxy와 xywh (또는 tlwh) 사이의 변환을 다룸
"""
def xyxy_to_xywh(*xyxy):
    """ Calculates the relative bounding box from absolute pixel values. """
    """
    - 입력 : xyxy 형식의 바운딩 박스 (xyxy -박스의 왼쪽 상단 좌표 (x1, y1)과 오른쪽 하단 좌표 (x2, y2))
    - 작동 원리
        · 바운딩 박스의 왼쪽과 상단 좌표를 계산함
        · 바운딩 박수의 너비(bbox_w)와 높이(bbox_h)를 계산함
        · 중심점 좌표(x_c, y_c) 계산
    - 출력 : xywh 형식의 바운딩 박스 (xywh - 중심 좌표(x_c,y_c)와 너비 w, 높이 h를 의미)
    
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    """
        - 입력 : xyxy 형식의 바운딩 박스 (xyxy -박스의 왼쪽 상단 좌표 (x1, y1)과 오른쪽 하단 좌표 (x2, y2))
        - 작동 원리
            입력된 각 바운딩 박스에 대해 :
                · 각 좌표 값을 정수로 변환 
                · 상단(top)과 왼쪽(left) 좌표를 추출함
                · 너비(w)와 높이(h) 계산
                · tlwh 형식의 바운딩 박스를 생성하고 목록에 추가
        - 출력 : tlwh 형식의 바운딩 박스 (tlwh - 상단 좌측 좌표(top,left)와 너비 w, 높이 h를 의미)

        """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

"""
4. compute_color_for_labels(label)
- 입력받은 레이블 값에 따라 특정 색상을 반환
- 레이블 : 주로 객체 인식에서 각각의 객체 종류를 구분하기 위해 사용되는 정수값
"""
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # helmet
        color = (222, 82, 175)  # 핑크
    elif label == 1:  # head
        color = (0, 204, 255)  # 하늘
    elif label == 2:  # person
        color = (85, 45, 255) # 보라
    else:
        # 각 색상 채널 값은 (p * (label ** 2 - label + 1)) % 255를 통해 계산되며 p는 팔레트의 각 원소
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

"""
4. draw_border(img, pt1, pt2, color, thickness, r, d)
- 이미지에 둥근 모서리의 사각형 경계를 그리는 함수

# 파라미터
- img : 이미지 객체. 이 이미지에 사각형의 경계를 그림
- pt1, pt2: 사각형의 왼쪽 상단 꼭짓점(pt1)과 오른쪽 하단 꼭짓점(pt2)의 좌표
- color : 사각형의 색상
- thickness : 선의 굵기
- r : 모서리의 반지름
- d : 모서리의 길이를 조정하는 값
"""
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # cv2.line(img, 시작점, 끝점, 색상, 선의 굵기)
    # cv2.ellipse(img, 중심점, 축 길이, 각도, 시작 각도, 종료 각도, 색상, 선의 굵기)

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness) # 수평선
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness) # 수직선
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness) # 왼쪽 상단 모서리에 위치한 원의 1/4부분인 원호를 그림
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness) # 수평선
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness) # 수직선
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness) # 오른쪽 상단 모서리에 위치한 원의 1/4부분인 원호를 그림
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness) # 수평선
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness) # 수직선
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness) # 왼쪽 하단 모서리에 위치한 원의 1/4부분인 원호를 그림
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness) # 수평선
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness) # 수직선
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness) # 오른쪽 하단 모서리에 위치한 원의 1/4부분인 원호를 그림

    # 사각형의 상단, 중앙 부분
    # cv2.rectangle(img, 시작점 좌표, 종료점 좌표, 색상, 선의 굵기(-1은 사각형 채우기), 선의 유형)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    # 각 모서리에 작은 원을 그림 (for 디자인)
    # cv2.circle(img, 중심점 좌표(사각형의 네 모서리), 반지름, 색상, 선의 굵기)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img

"""
5. UI_box(x, img, color=None, label=None, line_thickness=None)
- 주어진 이미지 img 위에 바운딩 박스와 label을 그리는 역할

# 파라미터
- x : 바운딩 박스의 좌표를 나타내는 리스트 or 튜플 (x1, y1, x2, y2)의 형태여야 함
- img : 바운딩 박스가 그려질 이미지
- color : 바운딩 박스의 색상. 기본값은 랜덤 색상
- label : 바운딩 박스 위에 표시될 텍스트 레이블
- line_thickness : 바운딩 박스의 선 굵기. 기본값은 이미지의 크기에 따라 달라짐
"""
def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    
    # 선의 굵기를 계산하고 색상이 주어지지 않은 경우 랜덤한 색상 생성
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    # 바운딩 박스의 좌측 상단 좌표 c1과 우측 하단 좌표 c2를 정수형으로 변환
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # cv2.rectangle 함수는 img 이미지 위에 바운딩 박스를 그림
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # label이 주어진 경우, label의 텍스트 크기를 계산하고 label의 배경 상자를 그리며 cv2.putText 함수를 사용하여 label 텍스트를 이미지에 추가함
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, -1, cv2.LINE_AA)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

"""
6. draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0))
- 주어진 이미지에 여러 개의 바운딩 박스와 각 박스에 대한 label, 아이디, 객체의 경로(trail)를 그리는 역할
- 주로 객체 추적에서 사용됨

# 파라미터
- img : 바운딩 박스와 label이 그려질 이미지
- bbox : 각 객체의 바운딩 박스 좌표 리스트
- names : 객체의 이름을 나타내는 문자열 리스트
- object_id : 각 바운딩 박스에 대응하는 객체의 ID
- identities : 추적 중인 각 객체의 ID 리스트. 기본값은 None
- offset : 바운딩 박스의 좌표에 더해질 오프셋. 기본값은 (0,0)
"""
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # cv2.line(img, line[0], line[1], (46,162,112), 3)

    # 이미지의 높이와 너비 얻기
    height, width, _ = img.shape

    # data_deque는 각 객체의 이동 경로를 저장함. 이전에 추적하던 객체가 사라진 경우 data_deque에서 해당 객체의 정보를 삭제함
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # 이미지에 그릴 각 바운딩 박스에 대해 박스의 좌표를 가져오고 필요하다면 오프셋을 적용함
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # 바운딩 박스의 바닥 중앙점을 계산함. 객체의 이동 경로를 그릴 때 사용됨
        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # 객체의 ID를 가져옴. 바운딩 박스에 label을 부여할 때 사용됨
        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # 새로운 객체가 나타나면 해당 객체의 이동 경로를 저장할 새로운 deque를 생성함
        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        # 객체의 색상과 레이블을 계산함
        # compute_color_for_labels 함수를 통해서 색상을 얻고 객체의 이름과 ID를 결합하여 레이블 생성
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # add center to buffer
        # 계산된 중앙점을 deque에 추가하고 UI_box 함수를 사용하여 이미지에 바운딩 박스와 label을 그림
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # data_deque에 저장된 이동 경로를 사용하여 이미지에 객체의 trail을 그림. trail의 두께는 동적으로 계산되며 이동 경로의 각 선분을 그림
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

"""
6. DetectionPredictor(BasePredictor)
- 객체 탐지 모델의 결과를 전처리 및 후처리하고 탐지된 객체에 대한 정보를 기록하고 시각화함
"""
class DetectionPredictor(BasePredictor):

    # 주어진 이미지에 Annotator 객체를 초기화하여 반환함. Annotator는 이미지에 어노테이션을 추가하는 도구로 보임
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
    
    # 입력 이미지 전처리. 이미지는 PyTorch 텐서로 변환되고 정규화됨
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    # 모델의 예측 결과를 후처리함
    # non-maximum suppression이 적용되어 각 객체의 바운딩 박스가 추출됨. 의 크기그런 다음 이 박스들은 원본 이미지에 맞게 조정됨
    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    # 모델의 에측 결과를 처리하여 결과를 기록하고 이미지에 어노테이션을 추가함
    # 각 탐지된 객체의 클래스 및 수를 계산하고 로그 문자열에 추가함
    # 이미지의 각 객체에 대해 blurring 처리를 수행함
    # deepsort 알고리즘을 사용하여 객체의 위치를 추적하고 draw_boxes 함수를 사용하여 객체의 바운딩 박스를 이미지에 그림
    # 최종적으로 로그 문자열을 반환함
    def write_results(self, idx, preds, batch):
        p, im, im0 = batch  # 입력 배치에서 각 변수를 추출함 (im : 원본 이미지, im0 : 처리된 이미지)
        all_outputs = [] # 모든 출력을 저장할 리스트 초기화
        log_string = ""  # 로그 문자열 초기화
        if len(im.shape) == 3: # 이미지가 배치 차원을 가지지 않으면 추가함
            im = im[None]  # expand for batch dim
        self.seen += 1 # 처리된 프레임 수를 증가시킴
        im0 = im0.copy()  # 이미지의 복사본 생성
        
        # 웹캠이 사용되면 로그 문자열에 인덱스를 추가하고 프레임 카운트를 가져옴
        if self.webcam:  # batch_size >= 1  
            log_string += f'{idx}: ' # 로그 문자열에 인덱스 추가
            frame = self.dataset.count # 프레임 카운트 가져옴
        else:  
            frame = getattr(self.dataset, 'frame', 0) # 웹캠이 아닌 경우 데이터셋의 프레임 값을 가져옴

        # 데이터 경로 설정
        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg # 이미지 저장 경로 설정
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}') # 텍스트 경로 설정

        log_string += '%gx%g ' % im.shape[2:]  # print string

        self.annotator = self.get_annotator(im0)  # 어노테이션 설정
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names # 클래스 이름 가져오기

        # 객체 탐지 결과 처리 (각 탐지된 객체의 클래스별로 카운트를 하고 로그 문자열을 업데이트 함)
        det = preds[idx] # 현재 인덱스에 해당하는 탐지 결과 가져오기
        all_outputs.append(det) # 탐지 결과 저장
        if len(det) == 0:
            return log_string  # 탐지된 객체가 없으면 로그 문자열 반환
        
        # 객체 블러 처리 및 DeepSort 업데이트
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            class_index = int(c)
            count_of_object = int(n)
            founded_classes = {}
            founded_classes[names[class_index]] = int(n)
            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            count(founded_classes=founded_classes, im0=im0)

            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            class_name = names[int(cls)]
            if class_name == 'helmet':
            # Add Object Blurring Code
            # ..................................................................
                crop_obj = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                blur = cv2.blur(crop_obj, (20, 20))
                im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = blur
            # ..................................................................
            
            # 바운딩 박스 좌표 변환 및 저장
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
            
        # DeepSort 업데이트
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, im0)

        # 출력이 있을 경오 draw_boxes 함수를 호출하여 이미지에 박스를 그림
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg, model_name="bestv2.pt", source_name="0", show=True):
    init_tracker()
    cfg.model = model_name
    cfg.source = source_name
    cfg.show = show
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()