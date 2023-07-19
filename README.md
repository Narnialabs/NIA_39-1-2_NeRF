
# NIA 39-1 3D 에셋-이미지쌍 데이터 (2023.01.31) [link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71412)


![image](https://user-images.githubusercontent.com/109494925/216935624-be690389-c06b-4fc4-bcdc-38064f479315.png)

이 저장소는 NIA 39-1 3D 에셋-이미지쌍 데이터의 View Synthesis 모델을 개발하는 데 사용됩니다. 해당 프로젝트는 Novel view synthesis 작업을 수행하는 NeRF 모델을 기반으로 합니다. 아래에서는 프로젝트의 구조, 작업환경 설정 방법, 데이터셋 목록, 실행 방법, 그리고 결과물에 대한 설명을 제공합니다.

## 작업환경

- CPU: AMD EPYC 7543 32 core * 128개
- RAM: 1TB
- GPU: NVIDIA A100 * 8 개
- CUDA 버전: 11.3
- OS: Ubuntu v18.04
- Python: 3.8
- PyTorch 딥러닝 프레임워크 버전: 1.11.0+cu113

## 학습 환경 설치 방법

1. 개발 환경 구축
   - 관련 코드 및 라이브러리 다운로드: `git clone https://github.com/Narnialabs/NIA_39-1-2_NeRF.git`
   - 도커 배치 파일 명령어 실행: `bash docker_run.sh`

2. 도커 컨테이너 내 모델 실행 방법
   - 도커 실행 및 패키지 설치: 
     ```bash
     docker start pytorch_nerf # pytorch_nerf 컨테이너가 실행하지 않은 경우
     docker exec -it pytorch_nerf bash # 컨테이너 실행 후 해당 명령어로 접속함
     # python 확인 후 버전이 2.x 버전일 시 아래와 같이 명령어를 입력함.
     alias python=python3
     cd /home/Nia_AI/
     pip install -r requirements.txt
     unzip dataset.zip # 데이터셋 압축 풀기
     ```

   - AI 모델 실행 방법:
     ```bash
     # --gpu_num은 내가 사용하고자 하는 GPU의 번호
     # --config는 타겟 에셋의 환경설정 파일
     # --training은 학습 시 True로 설정함
     # --testing은 학습된 모델로 테스트 시 True로 설정함
     # --rendering은 학습된 모델로 렌더링 시 True로 설정함
     # "asset_x_x.yaml" 파일에 들어가서 학습하고자 하는 에셋의 환경 설정을 변경할 수 있음

     # 모델 학습시
     python run_nerf.py --gpu_num 0 --config ./config/asset_1_1.yaml --training True
     # 학습된 모델 테스트시
     python run_nerf.py --gpu_num 0 --config ./config/asset_1_1.yaml --testing True

     # 학습된 모델로 새로운 뷰 예측 및 렌더링 (선택 사항)
     python run_nerf.py --gpu_num 0 --config ./config/asset_1_1.yaml --rendering True
     ```

3. Output
   - 학습이 완료된 후 logs 폴더에 모델, 체크포인트, 테스팅, 렌더링 폴더가 생성되며 결과물이 저장됩니다.
   ```
   └── logs
       ├── model
       │   └── tgt_class_model.tar # 학습 완료된 모델 저장
       ├── checkpoint
       │   └─── tgt_class_folder
       │       └─── result_0.png # 500 번에 한 번씩 체크포인트 결과 저장
       │       └─── result_500.png
       ├── testing
       │   └─── tgt_class_folder
       │       └─── TestSet_result.png # 테스트 결과 저장
       │       └─── ValSet_result.png # 검증 결과 저장
       ├── rendering
       │   └─── tgt_class_video.png # 360 각도 뷰 렌더링 결과 저장
   ```

## Utils

- `image_processing.py`: 정사각 이미지가 아닌 경우 이미지를 자르거나 배경을 블랙으로 처리하는 유틸리티입니다.
  ```bash
  python image_processing.py
  \ --src_path [이미지 소스 경로]
  \ --tgt_asset [타겟 에셋명]
  \ --cropping True
  \ --black True
  ```
- `colmap2nerf.py`: colmap을 사용하여 이미지를 추정하고 pose를 추정한 경우, colmap의 출력을 NeRF에서 사용할 수 있는 데이터로 변환하는 유틸리티입니다.
  ```bash
  python colmap2nerf.py
  ```
