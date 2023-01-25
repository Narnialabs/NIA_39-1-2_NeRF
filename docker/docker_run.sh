#도커 이미지 명 
#nia39-1/ubuntu/pytorch/nerf

#도커 파일 만들기 
docker build -t nia39-1/ubuntu/pytorch/nerf -< dockerfile

#도커 컨테이너 명 
#pytorch_nerf

#도커 컨테이너 만들기 
docker run --gpus all --name pytorch_nerf -v $(pwd):/home/Nia_AI -p 3901:8888 -it nia39-1/ubuntu/pytorch/nerf:latest /bin/bash

#도커 실행하기
docker start pytorch_nerf
docker exec -it pytorch_nerf bash