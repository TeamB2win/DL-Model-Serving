# 다른 디렉토리의 .py 파일을 불러올 경로를 지정합니다.
import os
import hashlib
import shutil
from datetime import datetime
from typing import Any

import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
import torch

from model.model_handler import ModelHandler
from schema.inference_schema import InferenceRequest

import warnings
warnings.filterwarnings("ignore")


class DLModelHandler(ModelHandler):
  def __init__(self):
    super().__init__()
    self.initialize()
    
  def initialize(self):
    self.dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
    self.predict_mode = 'relative' # ['standard', 'relative', 'avd']
    self.best_frame = True  # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result
    self.pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
    if(self.dataset_name == 'ted'): # for ted, the resolution is 384*384
        self.pixel = 384
  
    # Load pretrained models
    self.inpainting, self.kp_detector, self.dense_motion_network, self.avd_network = self.load_checkpoints()
    
    # 비디오 파일 생성을 위한 Driving video paths
    self.driving_root = '/workspace/model_file/driving_videos'
    self.video_paths = [os.path.join(self.driving_root, file_name) for file_name in os.listdir(self.driving_root)]

    # 저장할 위치
    self.output_path = os.environ['VIDEO_DIR']
    self.working_dir = os.environ['WORKING_DIR']
    
  def preprocess_source(self, image_path: str) -> torch.Tensor | None:
    try: # Load the source image
      source_image = imageio.imread(image_path)
      source_image = resize(source_image, (self.pixel, self.pixel))[..., :3]
      source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
      source = source.to(self.device)

      return source
    
    except:
      print("Occur Error during preprocessing the source image")
      return None
      
  def preprocess_driving_video(self, video_path: str) -> list[torch.Tensor | Any | None]:
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
      driving = torch.tensor(
        np.array(driving_video)[np.newaxis].astype(np.float32)
      ).permute(0, 4, 1, 2, 3).to(self.device)

      return driving, fps
    
    except:
      print("Occur Error during preprocessing driving video")
      return None, None
    
  def inference(self, data: InferenceRequest) -> dict:
    ret = {
      'id': data.id,
      'is_err': False,
      'err_msg': None,
    }
    
    performance = {
      'driving_videos': [],
      'fvd': [],
      'aed': [],
      'psnr': [],
      'video_path': [],
    }
    output_paths = [] # working storage 저장 리스트
    video_performance = [] # 생성된 비디오 평가지표 저장
    # video_performance = pd.DataFrame(columns=['fvd', 'aed'])

    # Preprocess source image
    source_image_path = data.image_path
    source_image = self.preprocess_source(source_image_path)
    print("Getting source image from", source_image_path)
    
    # Except Can't get the source image    
    if source_image is None:
      ret['is_err'] = True
      ret['err_msg'] = "Fail to get the source image"      
      return ret
      
    # 이전 생성된 driving video를 제외한 나머지 비디오만 추론
    driving_video_paths = self.video_paths
    if data.prev_driving_path != "":
      driving_video_paths = [driving_video_path for driving_video_path in driving_video_paths 
                             if driving_video_path != data.prev_driving_path]
    
    curtime = datetime.now().strftime("%Y%m%d%H%M%S")
    for idx, driving_video_path in enumerate(driving_video_paths):
      # Create video path
      m = hashlib.sha256(f'{curtime}_generated_from_{data.id}_{idx}'.encode('utf-8'))
      video_name = m.hexdigest() + '.mp4'
      output_path = os.path.join(self.working_dir, video_name)

      # Preprocess driving video
      print("Getting driving video from", driving_video_path)
      driving_video, fps = self.preprocess_driving_video(driving_video_path)

      if driving_video is None:
        print(f"Fail to get the drving video {driving_video_path}")
        if idx != len(driving_video_paths) - 1:
          continue
        
        if len(output_paths) == 0:
          ret['is_err'] = True
          ret['err_msg'] = "Fail to load driving video"
          return ret
      
      # 추론 시작
      predictions = self._inference_use_best_frame(source_image, driving_video)
      if predictions is None:
        predictions = self._inference(source_image, driving_video)
      if predictions is None:
        print("Occur Error for inference!!")
        if idx != len(driving_video_paths) - 1:
          continue
        
        # 모든 driving video에 대한 추론 실패
        if len(output_paths) == 0:
          ret['is_err'] = True
          ret['err_msg'] = "Fail to create animation"
          return ret

      try:
        # save resulting video
        imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        output_paths.append(output_path)
        print(output_path, 'create completed')
        
        np_pred = np.array(predictions)
        np_driving = driving_video.cpu().clone().squeeze(0).detach().numpy().transpose((1, 2, 3, 0))
        np_source = source_image.cpu().clone().squeeze(0).detach().numpy().transpose((1, 2, 0))
        
        # PSNR 계산 
        psnr_result = self.calculate_psnr(np_driving, np_pred)
        print(f"PSNR: {psnr_result}")        
        
        # FVD 계산
        fvd_result = self.calculate_fvd(np_driving, np_pred)
        print(f"FVD: {fvd_result}")
                
        # AED 계산
        aed_result = self.calculate_aed(np_source, np_pred)
        print(f"AED: {aed_result}")

        performance['video_path'].append(output_path)
        performance['driving_videos'].append(driving_video_path)
        performance['psnr'].append(psnr_result)
        performance['fvd'].append(fvd_result)
        performance['aed'].append(aed_result)
        video_performance.append([fvd_result, aed_result, psnr_result])
        
        # delete created video in GPU
        del predictions
      
      except:
        print("Occur Error for curculate metrix!!")
        if idx != len(driving_video_paths) - 1:
          if os.path.exists(output_path):
            os.remove(output_path)
          continue
        
        if len(output_paths) == 0:
          ret['is_err'] = True
          ret['err_msg'] = "Fail to curculate metrix"
          return ret

    # 만들어진 동영상 중 평가지표 계산
    best_idx = self.calculate_metrix(video_performance)

    performance['video_performance'] = video_performance
    performance['best_idx'] = best_idx
    
    video_path = self.postprocess(best_idx, output_paths)
    
    # # 평가 지표가 가장 높은 비디오 path 저장
    ret['driving_video'] = driving_video_paths[best_idx]
    ret['video'] = video_path
        
    result_path = os.path.join('/workspace/result_logs', f'{data.id}_result.txt')    
    with open(result_path, 'w') as file:
      import json
      file.write(json.dumps(performance, indent='\t'))
      
    return ret
  
  def postprocess(self, best_idx: int, output_paths: list[str]) -> str:
    print("Running postprocess")
    # 평가 기준 최고 best_idx 동영상 저장
    src = output_paths[best_idx]
    best_video_file_name = os.path.basename(src)
    dir = os.path.join(self.output_path, best_video_file_name)
    shutil.move(src, dir)   # best 동영상 이동
    
    print(f"Best animation file name: {best_video_file_name}")
    print(f"Move Best video file to {dir}")
    
    # working dir 파일들 전부 삭제
    filenames = os.listdir(self.working_dir)
    for filename in filenames:
      path = os.path.join(self.working_dir, filename)
      os.remove(path)

      print(f"Delete created Video Path: {path}")
    
    return dir
        
  def _inference_use_best_frame(
    self, 
    source_image: torch.Tensor, 
    driving_video: torch.Tensor, 
  ) -> list | None:
    print(f"Getting start make animation using best frame: {self.best_frame}")
    new_source_image = source_image.squeeze(0).permute(1, 2, 0)
    
    # driving video에서 source image와 가장 잘 맞는 프레임을 찾는다.
    print("Getting start find the best frame from source image to driving video frames")

    try: 
      driving_video_frames = driving_video.squeeze(0).permute(1, 2, 3, 0)
      i = self.find_best_frame(new_source_image, driving_video_frames)
      print ("Best frame: " + str(i))
      del driving_video_frames
      del new_source_image
      
    except:
      print(f"Fail to find the best frame")
      return None
        
    driving_forward = driving_video[:, :, i:, :, :]                           # 기준 프레임 뒷 부분 순방향으로 예측
    driving_backward = torch.flip(driving_video[:, :, :(i + 1), :, :], (0, )) # 기준 프레이 앞 부분 역방향으로 예측
    print(f"driving video shape: {driving_video.shape}")
    print(f"driving forward shape: {driving_forward.shape}")
    print(f"driving backward shape: {driving_backward.shape}")
    
    try:
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
    except:
      print("Fail to make animation")
      return None
    
    predictions = predictions_backward[::-1] + predictions_forward[1:]
    # delete predictions in GPU
    del predictions_backward
    del predictions_forward

    return predictions
    
  def _inference(self,
    source_image: torch.Tensor,
    driving_video: torch.Tensor
  ) -> list | None:
    print(f"Getting start make animation using best frame: False")
    try:
      predictions = self.make_animation(
        source_image,
        driving_video,
        inpainting_network=self.inpainting,
        kp_detector=self.kp_detector,
        dense_motion_network=self.dense_motion_network,
        avd_network=self.avd_network,
        mode=self.predict_mode
      )
    except:
      print("Fail to make animation not using best frame")
      return None
    
    return predictions