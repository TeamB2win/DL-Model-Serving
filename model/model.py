# 다른 디렉토리의 .py 파일을 불러올 경로를 지정합니다.
import os
import hashlib

import imageio
import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage.transform import resize
import torch

from model.model_handler import ModelHandler
from schema.priority_queue import RequestData

import warnings
warnings.filterwarnings("ignore")


class DLModelHandler(ModelHandler):
  def __init__(self):
    super().__init__()
    self.initialize()
    
  def initialize(self):
    self.dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
    self.predict_mode = 'relative' # ['standard', 'relative', 'avd']
    self.best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result
    self.pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
    if(self.dataset_name == 'ted'): # for ted, the resolution is 384*384
        self.pixel = 384
  
    # Load pretrained models
    self.inpainting, self.kp_detector, self.dense_motion_network, self.avd_network = self.load_checkpoints()
    
    # 비디오 파일 생성을 위한 Driving video paths
    self.driving_root = '/workspace/model_file/driving_videos'
    self.video_paths = [os.path.join(self.driving_root, file_name) for file_name in os.listdir(self.driving_root)]

    # 저장할 위치
    self.output_path = '/workspace/data/video'
    
  def preprocess_source(self, image_path: str) -> np.ndarray | None:
    # Load the source image
    try:
      source_image = imageio.imread(image_path)
      source_image = resize(source_image, (self.pixel, self.pixel))[..., :3]
      source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
      source = source.to(self.device)
      
      return source
    # 예외 처리 필요
    except:
      print("Occur Error during preprocessing the source image")
      return None
      
  def preprocess_driving_video(self, video_path: str) -> list[np.ndarray | None]:
    try:
      reader = imageio.get_reader(video_path)
      fps = reader.get_meta_data()['fps']
      driving_video = []
      try:
        for im in reader:
          driving_video.append(im)
      except RuntimeError:
        pass

      reader.close()    
      driving_video = [resize(frame, (self.pixel, self.pixel))[..., :3] for frame in driving_video]
      driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(self.device)

      return driving, fps
    
    except:
      print("Occur Error during preprocessing driving video")
      return None, None
    
  def inference(self, data: RequestData) -> dict:
    ret = {
      'id': data.id,
      'isErr': False,
      'errMsg': ""
    }
    output_paths = [] # 만들어진 동영상들의 path를 저장하는 리스트
    
    # Preprocess source image
    source_image_path = data.image_path
    source_image = self.preprocess_source(source_image_path)
    print("Getting source image from", source_image_path)
    # Except Can't get 
    if source_image is None:
      ret['isErr'] = True
      ret['errMsg'] = "Fail to get the source image"      
      return ret
      
    # 이전 생성된 driving video를 제외한 나머지 비디오만 추론
    driving_video_paths = self.video_paths
    if data.prev_driving_path != "":
      driving_video_paths = [driving_video_path for driving_video_path in driving_video_paths 
                             if driving_video_path != data.prev_driving_path] 

    video_index = 0
    video_performance = pd.DataFrame(columns=['fvd', 'aed'])
    
    for driving_video_path in driving_video_paths:
      try:
        # Create video path
        video_name = f'generated_from_{data.id}_{video_index}'
        m = hashlib.sha256(video_name.encode('utf-8'))
        output_path = os.path.join(self.output_path, m.hexdigest() + '.mp4')

        # Preprocess driving video
        driving_video, fps = self.preprocess_driving_video(driving_video_path)
        print("Getting driving video from", driving_video_path)
        if driving_video is None:
          ret['isErr'] = True
          ret['errMsg'] = "Fail to get the drving video"
          return ret
        
        if self.predict_mode == 'relative' and self.best_frame:
          # driving video에서 source image와 가장 잘 맞는 프레임을 찾는다.
          i = self.find_best_frame(source_image, driving_video)
          print ("Best frame: " + str(i))
          print(f"Getting start make animation using best frame {self.best_frame} video index {video_index}")
          driving_forward = driving_video[i:]               # 기준 프레임 뒷 부분 순방향으로 예측
          driving_backward = driving_video[:(i + 1)][::-1]  # 기준 프레이 앞 부분 역방향으로 예측
          predictions_forward = self.make_animation(
            source_image, 
            driving_forward, 
            inpainting_network=self.inpainting,
            kp_detector=self.kp_detector,
            dense_motion_network=self.dense_motion_network,
            avd_network=self.avd_network,
            mode=self.predict_mode
          )
          predictions_backward = self.make_animation(
            source_image, 
            driving_backward, 
            inpainting_network=self.inpainting,
            kp_detector=self.kp_detector,
            dense_motion_network=self.dense_motion_network,
            avd_network=self.avd_network,
            mode=self.predict_mode
          )
          predictions = predictions_backward[::-1] + predictions_forward[1:]
          # delete predictions in GPU
          del predictions_backward
          del predictions_forward
          
        else:
          print(f"Getting start make animation using best frame {self.best_frame} video index {video_index}")
          predictions = self.make_animation(
            source_image,
            driving_video,
            inpainting_network=self.inpainting,
            kp_detector=self.kp_detector,
            dense_motion_network=self.dense_motion_network,
            avd_network=self.avd_network,
            mode=self.predict_mode
          )
        
        # save resulting video
        output_paths.append(output_path)
        imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        # 디버깅 확인용 출력
        print(output_path, 'create completed!')
        print(driving_video_path)

        # FVD 계산 실행
        fvd_result = self.calculate_fvd(driving_video_path, output_path)
        print("FVD:", fvd_result)
        
        # AED 계산 실행
        aed_result = self.calculate_aed(source_image_path, output_path) # 원본이미지 경로
        print("AED:", aed_result)
        
        video_performance.loc[video_index] = [fvd_result, aed_result]

        video_index += 1
        # delete created video in GPU
        del predictions
    
      except:
        print("Occur Error for inference!!")
        for path in output_paths:
          if os.path.exists(path):
            os.remove(path)
        
        ret['isErr'] = True
        ret['errMsg'] = "Fail to create animation"
        return ret
          
    # 만들어진 동영상 중 평가지표 계산
    best_index, remove_idxs = self.calculate_metrix(video_performance)
    self.postprocess(remove_idxs, output_paths)
    
    # 평가 지표가 가장 높은 비디오 path 저장
    ret['videoSource'] = driving_video_paths[best_index]
    ret['video'] = output_paths[best_index]
    
    return ret
  
  def postprocess(self, remove_idxs, output_paths) -> None:
    # 만들어진 동영상 삭제
    print("Running postprocess")
    for i in list(remove_idxs):
      if os.path.exists(output_paths[i]):
        print(f"Delete created Video Path: {output_paths[i]}")
        os.remove(output_paths[i])