import os
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils import DEFAULT_CONFIG
from inference import DetectionPredictor
from omegaconf import OmegaConf


def predict(cfg):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_weights_path = os.path.join(current_dir, '..', 'bestv2.pt')

    cfg.model = cfg.model or model_weights_path
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = 0
    # cfg.source = "http://192.168.3.92:8002/video"
    predictor = DetectionPredictor(cfg)
    predictor()

# Hydra 대신 OmegaConf를 사용하여 설정을 로드합니다.
cfg = OmegaConf.load(str(DEFAULT_CONFIG))

# 함수 호출
predict(cfg)