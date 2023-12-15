# 🧑‍🔧 근로자의 영상정보 가명처리

### Reference
> Github

https://github.com/noorkhokhar99/YOLOv8-Object-Tracking-ID-Trails-Blurring-and-Counting

> Model

https://www.kaggle.com/code/princep/helmet-detection-with-yolov8 <br/>
https://www.kaggle.com/code/plasticglass/yolov8-safety-helmet-detection

&nbsp;

### Release Notes
```[v1.0.0] - 2023-09-26```
- Helmet 탐지 시 Helmet의 Bounding Box 블러 처리. Helmet이 탐지되지 않으면 블러 해제
- Model : YOLOv8 (bestv2.pt)
- Class : Helmet, Head, Person

  
| Class    | Images | Instances | Precision | Recall | mAP50  | mAP50-95 |
|----------|--------|-----------|-----------|--------|--------|----------|
| All      | 500    | 2503      | 0.945     | 0.591  | 0.634  | 0.418    |
| Helmet   | 500    | 1766      | 0.921     | 0.9    | 0.953  | 0.64     |
| Head     | 500    | 668       | 0.915     | 0.873  | 0.937  | 0.609    |
| Person   | 500    | 69        | 1         | 0      | 0.0138 | 0.00533  |


&nbsp;

### How To Use
TBU

## Contributors

* [Hongrok Lim](https://hongroklim.github.io/)
* [Won Gyeom](https://github.com/GyeomE)
* [Suhyun Yun](https://github.com/yun-suhyun)
* [Chaewon Lim](https://github.com/Chaewon-L)
* [Jungu Kang](https://github.com/KJunGu)
