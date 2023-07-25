import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

# 동영상을 InceptionV3 입력 크기에 맞게 변환하여 특징 벡터 생성
def create_inception_embedding(video_path, batch_size=32):
    
    # GPU 장치 사용 설정
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    # InceptionV3 모델 불러오기
    model = models.inception_v3()
    model.load_state_dict(torch.load("/workspace/Thin-Plate-Spline-Motion-Model/frechet_video_distance/inception_v3.pth"))

    # 드롭아웃과 fc layer를 그대로 통과시킴
    model.dropout = nn.Identity()
    model.fc = nn.Identity()

    # 모델의 avgpool layer까지만 유지
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (299, 299))  # InceptionV3의 입력 크기에 맞게 리사이징
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 RGB로 변환
        frames.append(frame_rgb)
    cap.release()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_data = []
    for frame in frames:
        video_data.append(transform(frame))

    video_embedding = []
    with torch.no_grad():
        for i in range(0, len(video_data), batch_size):
            batch_data = video_data[i:i+batch_size]
            batch_data = torch.stack(batch_data).to(device)  # InceptionV3의 입력 전처리
            batch_embedding = model(batch_data).detach().cpu().numpy()
            video_embedding.append(batch_embedding)

    video_embedding = np.concatenate(video_embedding, axis=0)
    video_embedding = video_embedding.flatten()

    return video_embedding

# FVD 계산 함수
def calculate_fvd(video1_path, video2_path):
    video1_embedding = create_inception_embedding(video1_path)
    video2_embedding = create_inception_embedding(video2_path)

    # FVD 계산
    fvd = np.linalg.norm(video1_embedding - video2_embedding)
    return fvd

def calculate_aed(img_path, tovideo_path):

    def extract_frames_from_video(video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

        cap.release()
        return frames

    # img 파일 경로
    image_path = img_path
    # mp4 파일 경로를 지정해주세요.
    video_path = tovideo_path

    # 프레임별로 변환된 이미지들을 리스트로 얻습니다.
    frames_list = extract_frames_from_video(video_path)

    # 리스트의 길이(프레임 수)를 확인해봅니다.
    # print("총 프레임 수:", len(frames_list))

    def calculate_euclidean_distance(image1, image2):
        diff = np.subtract(image1, image2)
        squared_diff = np.square(diff)
        sum_squared_diff = np.sum(squared_diff)
        euclidean_distance = np.sqrt(sum_squared_diff / image1.size)
        return euclidean_distance

    def calculate_average_euclidean_distance(path, image_list):
        total_distance = 0.0

        # 리스트의 첫 번째 이미지를 기준 이미지로 설정합니다.
        reference_image = cv2.imread(path)
        resized_image = cv2.resize(reference_image, (256, 256))

        for image in image_list:
            distance = calculate_euclidean_distance(resized_image, image)
            total_distance += distance

        # 이미지의 개수로 나눠서 평균을 계산합니다.
        average_distance = total_distance / len(image_list[1:])
        return average_distance

    # frames_list는 앞서 생성한 이미지 프레임들의 리스트입니다.
    # 이미지 간의 평균 유클리드 거리를 구합니다.
    aed = calculate_average_euclidean_distance(image_path, frames_list)
    
    return aed
    
def main():

    # 두 개의 MP4 파일 경로 (가정)
    video_path = "/workspace/Thin-Plate-Spline-Motion-Model/assets/driving.mp4"
    
    result_path = "/workspace/Thin-Plate-Spline-Motion-Model/result.mp4"

    # FVD 계산 실행
    fvd_result = calculate_fvd(video_path, result_path)
    print("FVD:", fvd_result)
    
    # AED 계산 실행
    aed_result = calculate_aed('/workspace/Thin-Plate-Spline-Motion-Model/assets/source.png', result_path) # 원본이미지 경로
    print("AED:", aed_result)
    
    return fvd_result, aed_result

if __name__ == '__main__' :

    main()
    
