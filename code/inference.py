import numpy as np
import torch, os, json, io, cv2, time
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_fn(model_dir):
    logging.info("Executing model_fn from inference.py ...")
    env = os.environ
    model = YOLO("/opt/ml/model/code/" + env['YOLOV8_MODEL'])
    return model

def input_fn(request_body, request_content_type):
    logging.info("Executing input_fn from inference.py ...")
    logging.info(f"Received request with Content-Type: {request_content_type}")
    if request_content_type == 'image/jpeg':
        img_as_np = np.frombuffer(request_body, dtype=np.uint8)
        img = cv2.imdecode(img_as_np, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode the image. Ensure that the input is a valid JPEG image.")

        img_resized = cv2.resize(img, (1024, 1024))

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        return img_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    logging.info("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result


def output_fn(prediction_output, content_type):
    try:
        logging.info("Executing output_fn from inference.py ...")
        
        if prediction_output is None:
            logging.error("Received 'None' as 'prediction_output'")
            return {}

        logging.info(f"Type of prediction_output: {type(prediction_output)}")
        logging.info(f"Attributes of prediction_output: {dir(prediction_output)}")

        infer = {}
        for result in prediction_output:
            logging.info(f"Type of result: {type(result)}")
            logging.info(f"Attributes of result: {dir(result)}")
            
            if hasattr(result, '_keys'):
                logging.info(f"Keys present in result: {result._keys}")
                if 'boxes' in result._keys:
                    try:
                        infer['boxes'] = result.boxes.numpy().data.tolist()
                    except AttributeError as e:
                        logging.error(f"Unable to convert 'boxes' to numpy array: {e}")
                else:
                    logging.warning("'boxes' not in result._keys")

                if 'masks' in result._keys:
                    try:
                        infer['masks'] = result.masks.numpy().data.tolist() if result.masks is not None else None
                    except AttributeError as e:
                        logging.error(f"Unable to convert 'masks' to numpy array: {e}")
                else:
                    logging.warning("'masks' not in result._keys")

                if 'probs' in result._keys:
                    try:
                        infer['probs'] = result.probs.numpy().data.tolist() if result.probs is not None else None
                    except AttributeError as e:
                        logging.error(f"Unable to convert 'probs' to numpy array: {e}")
                else:
                    logging.warning("'probs' not in result._keys")
            else:
                logging.error("'result' does not have '_keys' attribute")

        json_result = json.dumps(infer)
        logging.info(f"Inference results: {json_result}")

        return json_result
    
    except Exception as e:
        logging.error(f"Error in output_fn: {e}")
        raise

