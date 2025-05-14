import cv2
import os
import time
import requests
import json

# Thư mục lưu ảnh chụp
SAVE_DIR = 'detected_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# URL API để nhận diện muỗi
API_URL = "https://mksol.vn/detect/v1/api/predict/img"

# Số frame bỏ qua trước khi gửi frame thứ 10
FRAME_SKIP = 10

# Dictionary để đếm số lượng từng loại muỗi
mosquito_count = {}

def detect_mosquito(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Lỗi khi gọi API: {response.status_code}")
                return None
    except Exception as e:
        print(f"Lỗi khi gửi ảnh đến API: {str(e)}")
        return None

def analyze_result(result):
    if not result:
        return "No result to be found", []

    if result.get('code') == 0:
        bboxes = []
        if 'infos' in result:
            for info in result['infos']:
                classname = info.get('classname', '')
                confidence = info.get('confidence', 0.0)
                if all(coord in info for coord in ['x1', 'y1', 'x2', 'y2']):
                    bboxes.append({
                        'x1': info['x1'],
                        'y1': info['y1'],
                        'x2': info['x2'],
                        'y2': info['y2'],
                        'classname': classname,
                        'confidence': confidence
                    })
                    # Cập nhật số lượng từng loại muỗi
                    if classname in mosquito_count:
                        mosquito_count[classname] += 1
                    else:
                        mosquito_count[classname] = 1
        return "Detection complete", bboxes
    else:
        return f"ERROR API: {result.get('message', 'error unknown')}", []

def calculate_iou(box1, box2):
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def match_mosquito(bbox, previous_bboxes, iou_threshold=0.5, conf_threshold=0.8):
    for prev_bbox in previous_bboxes:
        iou = calculate_iou(bbox, prev_bbox)
        conf_diff = abs(bbox['confidence'] - prev_bbox['confidence'])
        if iou >= iou_threshold and conf_diff <= conf_threshold:
            return True
    return False

def draw_detection(frame, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        classname = bbox['classname']
        confidence = bbox['confidence']
        label = f"{classname} ({confidence})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

def main():
    VIDEO_DIR = "F:/Deep learning/object_detect/data"
    video_paths = [os.path.join(VIDEO_DIR, f"video{i + 1}.mp4") for i in range(3)]
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    frame_count = 0
    previous_bboxes = []

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)

        if not frames:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            for idx, frame in enumerate(frames):
                img_path = os.path.join(SAVE_DIR, f'frame_{idx}_{frame_count}.jpg')
                cv2.imwrite(img_path, frame)

                result = detect_mosquito(img_path)
                _, bboxes = analyze_result(result)

                valid_bboxes = [bbox for bbox in bboxes if not match_mosquito(bbox, previous_bboxes)]
                previous_bboxes = valid_bboxes

                if valid_bboxes:
                    frame = draw_detection(frame, valid_bboxes)
                    detected_img_path = os.path.join(SAVE_DIR, f'detected_{idx}_{frame_count}.jpg')
                    cv2.imwrite(detected_img_path, frame)
                    print(f"Processed frame {frame_count} from video {idx + 1}")

                cv2.imshow(f"Video {idx + 1}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    # Hiển thị kết quả tổng kết số lượng muỗi đã phát hiện
    print("\n=== KẾT QUẢ PHÁT HIỆN MUỖI ===")
    for classname, count in mosquito_count.items():
        print(f"{classname}: {count} phát hiện")

if __name__ == '__main__':
    main()
