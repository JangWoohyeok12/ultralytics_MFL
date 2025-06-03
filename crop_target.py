from ultralytics import YOLO
import cv2
import torch
from torchvision import transforms
from PIL import Image

# ----------------- 전처리 -----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ----------------- 모델 로딩 -----------------
# Load YOLO model
model = YOLO('yolov8n.pt')

# Load Classification CNN Model
# cnn_model = 

# CNN Input Preprocessing 

# Load UNet
# unet_model =

# ----------------- 설정 -----------------
# 신뢰도 threshold 설정
CONF_THRESH = 0.5  # 50% 이상 확신한 탐지만 사용

# n 프레임마다 추론
SKIP_FRAME = 2

# ----------------- 실시간 영상 -----------------
# Video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

frame_count = 0

# processing
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
    
    # Perform inference on an image
    results = model(frame)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.tolist()

    # Extract conf
    confs = results[0].boxes.conf.tolist()

    # Iterate through the bounding boxes
    for i, box in enumerate(boxes):
        
        score = confs[i]

        # less than 0.5 
        if score < CONF_THRESH:
            continue  # 낮은 신뢰도 박스는 건너뜀

        try:
            x1, y1, x2, y2 = box
    
            # Crop the object using the bounding box coordinates
            crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop_object.size == 0 or crop_object.shape[0] < 10 or crop_object.shape[1] < 10:
                continue

        except:
            print("crop 실패")
            continue

        try:
            # CNN 입력을 위해 포멧 변환 (BGR to RGB 후 PIL.Image 객체로 변환)
            img_pil = Image.fromarray(cv2.cvtColor(crop_object, cv2.COLOR_BGR2RGB))
    
            # Resize an Image & Convert to Tensor & Add batch dimension
            input_tensor = transform(img_pil).unsqueeze(0)

        except:
            print("전처리 실패")
            continue

        try:
        # Predict with CNN
            with torch.no_grad():
                output = cnn_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

        except:
            print("추론 실패")
            continue
            
        label = "정상" if pred == 0 else "손상"
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)

        # 손상된 경우
        if pred == 1:
            try:
                with torch.no_grad():
                    restored = unet_model(input_tensor)
                    
                # 복원된 tensor를 이미지로 변환
                restored_np = restored.squeeze(0).permute(1, 2, 0).numpy()
                restored_np = (restored_np * 255).clip(0, 255).astype("uint8")
                restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

                # 복원된 이미지 저장
                cv2.imwrite(f"restored_{frame_count}_{i}.jpg", restored_bgr)
            except:
                continue
        
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
