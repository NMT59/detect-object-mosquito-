import cv2
import os
import time
import requests
import json
import numpy as np
import matplotlib.pyplot as plt

# Thư mục lưu ảnh chụp
SAVE_DIR = 'detected_FT_image'
os.makedirs(SAVE_DIR, exist_ok=True)

# URL API để nhận diện muỗi
API_URL = "https://mksol.vn/detect/v1/api/predict/img"

# Số frame bỏ qua trước khi gửi frame thứ 10
FRAME_SKIP = 10

# Ngưỡng cho sự khác biệt Fourier Transform
FOURIER_THRESHOLD = 3000000

# Dictionary để đếm số lượng từng loại muỗi
mosquito_count = {}

previous_frames = [None, None, None]
selected_points = []


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


def resize_images(frame1, frame2):
    h, w = frame1.shape[:2]
    frame2_resized = cv2.resize(frame2, (w, h))
    return frame1, frame2_resized


def calculate_fourier_difference(frame1, frame2):
    frame1, frame2 = resize_images(frame1, frame2)

    f1 = np.fft.fft2(frame1)
    f2 = np.fft.fft2(frame2)

    f1_shift = np.fft.fftshift(f1)
    f2_shift = np.fft.fftshift(f2)

    diff = np.abs(f1_shift - f2_shift)
    mse = np.mean(diff ** 2)

    # Lưu điểm đặc trưng nếu vượt ngưỡng
    if mse > FOURIER_THRESHOLD:
        selected_points.append(mse)

    return mse


def plot_fourier_transform():
    plt.figure(figsize=(12, 6))
    plt.plot(selected_points, marker='o', linestyle='-', color='r')
    plt.title("Fourier Transform - Feature Points")
    plt.xlabel("Detection Instance")
    plt.ylabel("MSE Value")
    plt.grid()
    plt.show()


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
                    fourier_diff = calculate_fourier_difference(previous_frame, frame)
                    print(f"Fourier Difference for video {idx + 1}, frame {frame_count}: {fourier_diff}")

                    if fourier_diff > FOURIER_THRESHOLD:
                        img_path = os.path.join(SAVE_DIR, f'frame_{idx}_{frame_count}.jpg')
                        cv2.imwrite(img_path, frame)
                        result = detect_mosquito(img_path)
                        _, bboxes = analyze_result(result)

                previous_frames[idx] = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    print("\n=== KẾT QUẢ PHÁT HIỆN MUỖI ===")
    print(f"Sum of all mosquito detected: {sum(mosquito_count.values())}")
    for classname, count in mosquito_count.items():
        print(f"{classname}Objects detected: {count} detected")

    # Hiển thị đồ thị sau khi kết thúc quá trình phát hiện
    plot_fourier_transform()

if __name__ == '__main__':
    main()
