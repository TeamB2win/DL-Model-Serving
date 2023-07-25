# 다른 디렉토리의 .py 파일을 불러올 경로를 지정합니다.
import sys
other_directory_path = "/workspace/Thin-Plate-Spline-Motion-Model/frechet_video_distance"
sys.path.append(other_directory_path)
import torch
from demo import load_checkpoints
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")
from demo import make_animation
from skimage import img_as_ubyte
from score import calculate_fvd,calculate_aed
from sklearn.preprocessing import StandardScaler

def standard_scaling(df):

    # 표준화 객체 생성
    scaler = StandardScaler()

    # 'fvd' 열을 표준화하여 새로운 열로 추가
    df['fvd_standardized'] = scaler.fit_transform(df[['fvd']])

    # 'aed' 열을 표준화하여 새로운 열로 추가
    df['aed_standardized'] = scaler.fit_transform(df[['aed']])

    df['fvd_aed'] = abs(df['fvd_standardized'])+abs(df['aed_standardized'])

    df.to_csv('/eval_video/test_performance.csv',index=True)

    best_index = df.sort_values(by=['fvd_aed'], ascending=[True]).index[0]

    return best_index, list(df.drop(best_index).index)

# edit the config
device = torch.device('cuda:0')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
config_path = '/workspace/Thin-Plate-Spline-Motion-Model/config/vox-256.yaml'
checkpoint_path = '/workspace/Thin-Plate-Spline-Motion-Model/vox.pth.tar'
predict_mode = 'relative' # ['standard', 'relative', 'avd']
find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
if(dataset_name == 'ted'): # for ted, the resolution is 384*384
    pixel = 384

inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

# 영상 생성을 기존 사진 경로
source_image_path = '/workspace/Thin-Plate-Spline-Motion-Model/test.png'
image_name = source_image_path.split('/')[3].split('.')[0]

# 비디오 파일 생성을 위한 mp4 경로 리스트
video_paths = ['/root/mp4/id02019/P5L0zKBOFUg/00097.mp4','/root/mp4/id01822/fJhR7TsN6b8/00155.mp4',
                '/root/mp4/id00562/nb9rbeaDwr4/00207.mp4','/root/mp4/id00817/0ZfPJikvUaM/00004.mp4',
                '/root/mp4/id01509/kpICNY0gvYU/00346.mp4']

# fvd, aed threshold 설정(오름차순 정렬후 상위 50개값의 평균을 각각 구한 값)
# thres_fvd = 173.3
# thres_aed = 6.53


def make_video():

    video_index = 0

    # 영상 기준 판단을 위한 dataframe
    video_performance = pd.DataFrame(columns = ['fvd','aed'])

    for video_path in video_paths:

        # if video_index == 5:
        #     # 가장 근접했던 영상의 index 산출
        #     min_index = min(video_dict, key=lambda k: video_dict[k])

        #     video_path = video_paths[min_index]
        #     # video path는 가변
        #     driving_video_path = video_path
        #     # ouput video path는 video path와 동일하도록 설정
        #     output_video_path = video_path

        
        # video path는 가변
        driving_video_path = video_path
        # ouput video path는 video path와 동일하도록 설정
        output_video_path = video_path


        source_image = imageio.imread(source_image_path)
        reader = imageio.get_reader(driving_video_path)


        source_image = resize(source_image, (pixel, pixel))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]


        if predict_mode=='relative' and find_best_frame:
            from demo import find_best_frame as _find
            i = _find(source_image, driving_video, device.type=='cpu')
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

        # 5개의 영상이 기준치를 만족시키지 않아 그나마 나은 영상으로 재 생성하는 경우
        # if video_index == 5:

        #     #save resulting video
        #     imageio.mimsave(f'/eval_video/generated_from_{min_index}_{image_name}.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
        #     # 디버깅 확인용
        #     print(f'/eval_video/generated_from_{min_index}_{image_name}.mp4','completed!')

        #     print(driving_video_path)
        #     print(f'/eval_video/generated_from_{min_index}_{image_name}.mp4')


        
        #save resulting video
        imageio.mimsave(f'/eval_video/generated_from_{video_index}_{image_name}.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
        # 디버깅 확인용
        print(f'/eval_video/generated_from_{video_index}_{image_name}.mp4','completed!')
        
        print(driving_video_path)
        print(f'/eval_video/generated_from_{video_index}_{image_name}.mp4')

        # FVD 계산 실행
        fvd_result = calculate_fvd(driving_video_path, f'/eval_video/generated_from_{video_index}_{image_name}.mp4')
        print("FVD:", fvd_result)
        
        # AED 계산 실행
        aed_result = calculate_aed(source_image_path, f'/eval_video/generated_from_{video_index}_{image_name}.mp4') # 원본이미지 경로
        print("AED:", aed_result)
        
        video_performance.loc[video_index] = [fvd_result,aed_result]

        video_index += 1


    best_index, remove_index = standard_scaling(video_performance)
        
    print(f'{best_index}번째 동영상으로 비디오를 생성합니다.')


    # 만들어진 동영상 삭제
    for i in list(remove_index):
        os.remove(f'/eval_video/generated_from_{i}_{image_name}.mp4')

    # 영상 생성이 완료된 후 GPU 메모리 정리를 위해 empty_cache() 호출
    torch.cuda.empty_cache()
    


if __name__ == "__main__":
 
    make_video()
    
