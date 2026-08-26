
## 浸泡测试

cd deploy 

docker compose --profile soak down
docker compose --profile soak up -d --build
               └──────┬────────┘ └┬┘ └──┬──┘
               激活soak组(拉起接收端)  后台  首次顺便构建镜像

## 部署

docker compose up -d --build --no-cache

docker compose up -d
docker compose down