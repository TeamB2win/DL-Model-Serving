<img src="src/passerby_service.gif"/>

# 🕵🏻 Passerby-Backend

### <div align="center"><b><i> Passerby, 여러분의 안전을 책임집니다. </i></b></div>

&nbsp; 

> Passerby project
> 
> launched at 2023.06
> 
> Programmers AI DEV-course 5th

&nbsp; 

🎥 **Passerby**는 공개수배자들의 이미지를 **AI 기술**을 통하여 비디오로 생성하여 여러분께 제공합니다.

🇰🇷 본 프로젝트를 통해 팀 b2win은 **더 안전한 대한민국, 더 건강한 대한민국**을 꿈꿉니다.

💾 본 리포지토리는 Passerby의 DL model serving Server 저장소입니다.
k11.6
🚀 본 리포지토리의 Image to Video 생성 모델은 pretrain된 TPSM 모델을 차용하였습니다. [(출처)](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)

&nbsp;

# ⚙️ Tech Stack

<div align="center">
<img src="https://img.shields.io/badge/Python-3776AB0?style=for-the-badge&logo=Python&logoColor=white"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"><img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=Pydantic&logoColor=white"><img src="https://img.shields.io/badge/Swagger-85EA2D?style=for-the-badge&logo=Swagger&logoColor=white"><img src="https://img.shields.io/badge/Amazon%20EC2-FF9900?style=for-the-badge&logo=Amazon%20EC2&logoColor=white"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">


</div>
&nbsp; 

# ❓ About B2win Team

<div align="center">
  
| [@hwaxrang](https://github.com/hwaxrang) | [@Magenta195](https://github.com/Magenta195) | [@moongni](https://github.com/moongni) | [@heehaaaheeeee](https://github.com/heehaaaheeeee) | [@ShinEunChae](https://github.com/ShinEunChae) | [@joseokjun](https://github.com/joseokjun) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="src/khr.png" width=200 /> | <img src="src/kth.jpeg" width=200 /> | <img src="src/mgh.png" width=200 /> | <img src="src/msh.jpg" width=200 /> | <img src="src/sec.jpeg" width=200 /> | <img src="src/jsj.jpg" width=200 /> |
| `권화랑`   | `김태형` | `문건희` | `문숙희` | `신은채` | `조석준`  |

</div>

<div align="center">
<a href = "https://github.com/TeamB2win"><img alt="GitHub" src ="https://img.shields.io/badge/GitHub-181717.svg?&style=for-the-badge&logo=GitHub&logoColor=white"/></a>
<a href = "https://www.notion.so/B2Win-Between-a9b09623b67243319d9bbce293bfa46b?pvs=4"><img alt="Notion" src ="https://img.shields.io/badge/Notion-eeeeee.svg?&style=for-the-badge&logo=Notion&logoColor=black"/></a>
</div>




&nbsp; 

# 🗝️ Key Service

💡 사용자는 passerby 웹 및 모바일 환경에 접속하여 **전체 공개수배자 신상정보**를 확인할 수 있습니다.

💡 또한, 공개수배자 정보와 대조하여 바로 제보 및 신고할 수 있도록 **관련 안내 정보**를 제공합니다.

💡 관리자는 passerby db를 통하여 공개수배자 **생성, 조회**, 이미지 및 비디오, 신상정보의 **수정, 삭제**를 수행할 수 있습니다.

&nbsp;

# 🧭 Structure

```bash
🗂️ DL-MODEL-SERVING
├── 📂 api
│   ├── 📂 errors
│   │   ├── 📄 __init__.py
│   │   └── 📄 http_errors.py
│   ├── 📂 routes
│   │   ├── 📄 __init__.py
│   │   └── 📄 inference.py
│   ├── 📄 __init__.py
│   └── 📄 endpoint.py
├── 📂 config
│   ├── 📄 __init__.py
│   └── 📄 app.py
├── 📂 core
│   ├── 📄 __init__.py
│   ├── 📄 background_task.py
│   ├── 📄 events.py
│   └── 📄 utils.py
├── 📄 main.py
├── 📂 model
│   ├── 📄 __init__.py
│   ├── 📄 model.py
│   ├── 📄 model_handler.py
│   └── 📂 modules
│       ├── 📄 __init__.py
│       ├── 📄 avd_network.py
│       ├── 📄 bg_motion_predictor.py
│       ├── 📄 dense_motion.py
│       ├── 📄 inpainting_network.py
│       ├── 📄 keypoint_detector.py
│       ├── 📄 model.py
│       └── 📄 util.py
├── 📂 resource
│   ├── 📄 __init__.py
│   └── 📄 strings.py
├── 📂 schema
│   ├── 📄 __init__.py
│   ├── 📄 base.py
│   ├── 📄 errors.py
│   ├── 📄 inference_schema.py
│   └── 📄 priority_queue.py
├── 📄 README.md
└── 📄 run.sh
```
&nbsp;

# 📚 API docs

👉 자세한 request 및 response 문법은 백엔드 내 swagger 및 redoc 문서를 참조해 주세요! (추후 문서분리 예정)


|Router|Method|Endpoint|Description|
|---|---|---|---|
| admin | `POST` | `/api/inference` | 지명수배자 이미지 정보를 토대로 ML 모델 작업 정보를 inference queue에 삽입 |




&nbsp;

# 📝 Tutorial

본 model serving 레포지토리는 다음과 같은 환경에서 동작을 확인하였습니다.

+ Python 3.10.8
+ CUDA 11.6
+ pytorch 1.13.1
+ Ubuntu linux 도커 인스턴스


### prerequisite

본 DL_Model_Server는 FastAPI 기반의 서버 형식으로 구동되며, 실제 추론을 위해서 API 형식으로 큐를 삽입해야 합니다.
또한, 최상위 디렉토리에 `.env` 파일 형식의 설정 파일이 필요합니다. 포함되어야 하는 변수명 및 정보는 다음과 같습니다.

```bash
model_dir: 모델 추론에 사용되는 source 저장 주소
backend_url: PasserBy Backend server 주소
video_dir: 최종 Video 생성 결과물이 저장되는 경로
working_dir: 임시 Video 생성 결과물들이 저장되는 경로
```

또한, 위의 `model_dir` 에 해당하는 경로에는 다음 디렉토리가 구성되어야 합니다. TPSM 모델 관련 체크포인트 및 config는 [여기](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)에서 찾을 수 있습니다.

```bash
🗂️ model_dir
├── 📂 model_file
│   ├── 📂 config
│   │   └── 📄 vox-256.yaml     // TPSM model vox config file (see )
│   ├── 📂 driving_videos        // driving video to generate result
│   └── 📂 weights
│       ├── 📄 inception_v3.pth  // pretrained Inception model checkpoint
│       └── 📄 vox.pth.tar      // pretrained TPSM model checkpoint
└── 📂 result_logs              // path to save logs of model generation
```

### run

다음 명령어를 실행해 주세요.

```bash
sh run.sh
```

본 DL_Model_Server는 FastAPI 기반의 서버 형식으로 구동되며, 실제 추론을 위해서 API 형식으로 큐를 삽입해야 합니다. 백엔드 서버와의 연동을 전제로 하고 있어 단일 서버 사용 시 정상작동을 보증하지 않습니다.

