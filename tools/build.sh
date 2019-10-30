docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_wly:$1 -f docker/docker_wly/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_wly_gpu:$1 -f docker/docker_wly_gpu/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_wly:$1 -f docker/stream_wly/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_wly_gpu:$1 -f docker/stream_wly_gpu/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_wly:$1
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_wly_gpu:$1
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_wly:$1
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/stream_wly_gpu:$1