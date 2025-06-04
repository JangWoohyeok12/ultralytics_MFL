from ultralytics import YOLO
import cv2
import torch
from torchvision import transforms
from PIL import Image

# ----------------- 모델 로딩 -----------------
# YOLO 모델
model = YOLO('yolov8n.pt')


# UNet 모델
# unet_model =

# ----------------- 설정 -----------------
# threshold 설정
CONF_THRESH = 0.5  # 50% 이상 확신한 탐지만 사용

# 프레임 n개마다 추론
SKIP_FRAME = 2

# ----------------- 실시간 영상 -----------------
# 영상
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

frame_count = 0

# Processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 프레임 스킵 적용
    if frame_count % SKIP_FRAME != 0:
        cv2.imshow("Real-Time Detection & Damage Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 영상 프레임으로 추론
    results = model(frame)
    
    # bounding box 추출 
    boxes = results[0].boxes.xyxy.tolist()
    
    # confidence 추출 
    confs = results[0].boxes.conf.tolist()
    
    # class 추출 
    classes = results[0].boxes.cls.tolist()

    # 프레임 속 bounding box 순회
    for i, box in enumerate(boxes):

        # bouding box의 conf
        score = confs[i]

        # conf가 0.5보다 낮으면
        if score < CONF_THRESH:
            continue  # 낮은 신뢰도 박스는 건너뜀

        try:
            x1, y1, x2, y2 = box
    
            # bounding box crop
            crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop_object.size == 0 or crop_object.shape[0] < 10 or crop_object.shape[1] < 10:
                continue

        except:
            print("crop 실패")
            continue

        # bouding box의 클래스 
        is_damaged = class[i]

        # 손상된 경우
        if is_damaged == 'damaged_sign':
            try:
                with torch.no_grad():
                    restored = unet_model(input_tensor)
                    
                # 복원된 tensor를 이미지로 변환
                restored_np = restored.squeeze(0).permute(1, 2, 0).numpy()
                restored_np = (restored_np * 255).clip(0, 255).astype("uint8")
                restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            except:
                continue
    
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
