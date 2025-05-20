import cv2
import os
import time
import requests
import json
from skimage.metrics import structural_similarity as ssim

# Thư mục lưu ảnh chụp
SAVE_DIR = 'detected_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# URL API để nhận diện muỗi
API_URL = "https://mksol.vn/detect/v1/api/predict/img"

# Số frame bỏ qua trước khi gửi frame thứ 10
FRAME_SKIP = 10

# Ngưỡng SSIM để phát hiện thay đổi đáng kể
SSIM_THRESHOLD = 0.85

# Dictionary để đếm số lượng từng loại muỗi
mosquito_count = {}

previous_frames = [None, None, None]


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
                classname = info.get('detected_name', '')
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
                    if classname in mosquito_count:
                        mosquito_count[classname] += 1
                    else:
                        mosquito_count[classname] = 1
        return "Detection complete", bboxes
    else:
        return f"ERROR API: {result.get('message', 'error unknown')}", []


def draw_bounding_boxes(frame, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        classname = bbox['classname']
        confidence = bbox['confidence']
        label = f"{classname} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    score, _ = ssim(gray1, gray2, full=True)
    return score


def main():
    VIDEO_DIR = "F:/Deep learning/object_detect/data"
    video_paths = [os.path.join(VIDEO_DIR, f"video{i + 1}.mp4") for i in range(3)]
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    frame_count = 0

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
                previous_frame = previous_frames[idx]
                if previous_frame is not None:
                    ssim_score = calculate_ssim(previous_frame, frame)
                    print(f"SSIM for video {idx + 1}, frame {frame_count}: {ssim_score}")

                    if ssim_score < SSIM_THRESHOLD:
                        img_path = os.path.join(SAVE_DIR, f'frame_{idx}_{frame_count}.jpg')
                        cv2.imwrite(img_path, frame)
                        result = detect_mosquito(img_path)
                        _, bboxes = analyze_result(result)

                        # Vẽ bounding box
                        if bboxes:
                            frame = draw_bounding_boxes(frame, bboxes)

                        detected_img_path = os.path.join(SAVE_DIR, f'detected_{idx}_{frame_count}.jpg')
                        cv2.imwrite(detected_img_path, frame)
                        print(f"Processed frame {frame_count} from video {idx + 1}")

                previous_frames[idx] = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    print("\n=== KẾT QUẢ PHÁT HIỆN MUỖI ===")
    total_mosquitoes = sum(mosquito_count.values())
    print(f"Tổng số muỗi phát hiện: {total_mosquitoes}")
    for classname, count in mosquito_count.items():
        print(f"{classname}: {count} detected")

if __name__ == '__main__':
    main()
