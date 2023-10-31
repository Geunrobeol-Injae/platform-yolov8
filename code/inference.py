import torch
import cv2
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import  ops
from ultralytics.yolo.utils.plotting import Annotator

import cv2
from utils import count, xyxy_to_xywh, draw_boxes, init_tracker
from utils import deepsort
from utils import init_tracker, get_deepsort

init_tracker()
deepsort = get_deepsort()

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

    global connected_clients


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
        # save_path = str(self.save_dir / p.name)  # im.jpg # 이미지 저장 경로 설정
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

        cv2.imshow('Processed Frame',im0)
        cv2.waitKey(1)

        return log_string