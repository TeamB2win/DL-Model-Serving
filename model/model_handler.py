import sys
import yaml

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from scipy.spatial import ConvexHull

from model.modules.inpainting_network import InpaintingNetwork
from model.modules.keypoint_detector import KPDetector
from model.modules.dense_motion import DenseMotionNetwork
from model.modules.avd_network import AVDNetwork


class ModelHandler:
  def __init__(self):
    if sys.version_info[0] < 3:
      raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.config_path = '/workspace/model_file/config/vox-256.yaml'
    self.checkpoint_path = '/workspace/model_file/weights/vox.pth.tar'
    self.inception_path = '/workspace/model_file/weights/inception_v3.pth'
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    self.decay = np.array([0.2, 0.5, 0.3])  # FVD, AED, PSNR

  def load_checkpoints(self):
    """Pretrained 된 모델 가중치 불러오기"""
    with open(self.config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(self.device)
    dense_motion_network.to(self.device)
    inpainting.to(self.device)
    avd_network.to(self.device)
       
    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network

  def relative_kp(self, kp_source, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

  def make_animation(self, source, driving, inpainting_network, kp_detector, dense_motion_network, avd_network, mode='relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(self.device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = self.relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            
    return predictions

  def find_best_frame(self, source, driving):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(
      face_alignment.LandmarksType.TWO_D, flip_input=True,
      device=str(self.device)
    )
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in enumerate(driving[::2]):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num
  
  def extract_frames_from_video(self, video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

  def create_inception_embedding(self, frames, batch_size=32):
    # InceptionV3 모델 불러오기
    model = models.inception_v3()
    model.load_state_dict(torch.load(self.inception_path))
    # 드롭아웃과 fc layer를 그대로 통과시킴
    model.dropout = nn.Identity()
    model.fc = nn.Identity()

    # 모델의 avgpool layer까지만 유지
    model.to(self.device)
    model.eval()
    
    # frames = self.extract_frames_from_video(video_path)
    video_data = []
    for frame in frames:
        video_data.append(self.transform(frame))

    video_embedding = []
    with torch.no_grad():
        for i in range(0, len(video_data), batch_size):
            batch_data = video_data[i:i+batch_size]
            batch_data = torch.stack(batch_data).to(self.device)  # InceptionV3의 입력 전처리
            batch_embedding = model(batch_data).detach().cpu().numpy()
            video_embedding.append(batch_embedding)

    video_embedding = np.concatenate(video_embedding, axis=0)
    video_embedding = video_embedding.flatten()
    
    del model
    
    return video_embedding
  
  def calculate_fvd(self, actual_frames: np.ndarray, predicted_frames: np.ndarray) -> float:
    video1_embedding = self.create_inception_embedding(actual_frames)
    video2_embedding = self.create_inception_embedding(predicted_frames)

    # FVD 계산
    fvd = np.linalg.norm(video1_embedding - video2_embedding)
    return float(fvd)


  def calculate_aed(
    self, 
    image: np.ndarray, 
    video_frames: np.ndarray, 
  ) -> float:
    assert image.shape == video_frames.shape[1:], "Shapes of actual_frames and predicted_frames must be the same"

    # 프레임별로 변환된 이미지들을 리스트로 얻습니다.
    # frames_list = self.extract_frames_from_video(video_path)
    
    # 리스트의 길이(프레임 수)를 확인해봅니다.
    # print("총 프레임 수:", len(frames_list))

    def calculate_euclidean_distance(image1, image2):
      diff = np.subtract(image1, image2)
      squared_diff = np.square(diff)
      sum_squared_diff = np.sum(squared_diff)
      euclidean_distance = np.sqrt(sum_squared_diff / image1.size)
      return euclidean_distance

    def calculate_average_euclidean_distance(image, frames_list):
      total_distance = 0.0

      # 리스트의 첫 번째 이미지를 기준 이미지로 설정합니다.
      # reference_image = cv2.imread(path)
      # resized_image = cv2.resize(reference_image, (pixel, pixel))
      # resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 RGB로 변환
      
      for frame in frames_list:
          distance = calculate_euclidean_distance(image, frame)
          total_distance += distance

      # 이미지의 개수로 나눠서 평균을 계산합니다.
      average_distance = total_distance / len(frames_list[1:])
      return average_distance

    # frames_list는 앞서 생성한 이미지 프레임들의 리스트입니다.
    # 이미지 간의 평균 유클리드 거리를 구합니다.
    aed = calculate_average_euclidean_distance(image, video_frames)
    
    return float(aed)

  def _calculate_psnr(self, original: np.ndarray , generated: np.ndarray):
    mse = np.mean((original - generated) ** 2)
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
  
  def calculate_psnr(
    self, 
    actual_frames: np.ndarray, 
    predicted_frames: np.ndarray
  ) -> float:    
    # Calculate PSNR for each frame pair
    # Make sure the shapes of actual_frames and predicted_frames are the same
    assert actual_frames.shape[1:] == predicted_frames.shape[1:], "Shapes of actual_frames and predicted_frames must be the same"

    num_frames = actual_frames.shape[0]

    # Calculate PSNR for each frame pair
    psnr_total = 0.0
    for i in range(num_frames):
        actual_frame = actual_frames[i]
        predicted_frame = predicted_frames[i]
        psnr_value = self._calculate_psnr(actual_frame, predicted_frame)
        psnr_total += psnr_value

    # Calculate average PSNR for the video pair
    average_psnr = psnr_total / num_frames

    return float(average_psnr)

  def calculate_metrix(self, scores: list) -> int:
    # 3번째 열 psnr의 값을 정규화를 위해 전부 음수로 변환함.
    scores = np.array(scores)
    scores[:, 2] = -scores[:, 2]

    # Step 1: 열별로 최소값과 최대값을 계산
    min_arr = np.min(scores, axis=0)
    max_arr = np.max(scores, axis=0)
    # Step 2: 정규화
    normalized = (scores - min_arr) / (max_arr - min_arr)
    # Step 3: 가중합계
    result = np.sum(self.decay * normalized, axis=1)

    return int(np.argmin(result))