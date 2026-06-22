# OpsMind RAG — 生产部署文档 (DEPLOYMENT)

**版本**: v0.1  
**日期**: 2026-06-20  
**状态**: Milvus Standalone → Cluster 升级指南

---

## 1. 概述

本文档描述如何将 OpsMind RAG 从 Demo 环境（Milvus Standalone）升级为生产级部署。

---

## 2. 架构升级: Demo → 生产

### 2.1 组件替换矩阵

| 组件 | Demo 实现 | 生产推荐 | 原因 |
|------|----------|---------|------|
| **向量数据库** | Milvus Standalone (Docker) | Milvus Cluster (Q/D/I 节点分离) | 水平扩展、强一致性、读写分离 |
| **消息/状态** | 无 (内存) | Redis Cluster + Streams | 持久化、多 Agent 通信、Consumer Group |
| **元数据存储** | Milvus 内嵌 (etcd) | PostgreSQL 14+ | 事务支持、SQL 审计查询 |
| **对象存储** | Milvus 内嵌 (MinIO) | 独立 MinIO Cluster / S3 | 高可用、多副本、版本管理 |
| **LLM 网关** | 直连 DeepSeek API | LiteLLM / 自建路由 | 多模型负载均衡、故障转移、成本控制 |
| **Embedding** | FastEmbed 本地 | BGE-M3 via ONNX 服务 / 专用 GPU 节点 | 更高精度(1024维)、批量推理加速 |
| **缓存** | 无 | Redis (L1) + 本地 LRU (L2) | 热点查询缓存、LLM 响应缓存 |
| **Web 服务器** | Uvicorn 直接 | Nginx → Gunicorn + Uvicorn Workers | 反向代理、SSL 终止、静态资源 |
| **前端** | Vite dev server | Nginx 静态托管 + CDN | 生产构建、Gzip、缓存策略 |

### 2.2 生产部署拓扑

```
                              ┌──────────────┐
                              │  Cloudflare  │
                              │   CDN + DNS  │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   Nginx      │
                              │  (Ingress)   │
                              └──┬───────┬───┘
                                 │       │
                    ┌────────────▼─┐  ┌──▼────────────┐
                    │ FastAPI × N  │  │ React Static   │
                    │ (Gunicorn)   │  │ (Nginx)        │
                    └──┬───┬───┬──┘  └───────────────┘
                       │   │   │
         ┌─────────────┘   │   └─────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐    ┌──────────────┐   ┌──────────────┐
   │  Redis   │    │   Milvus     │   │  PostgreSQL  │
   │  Cluster │    │   Cluster    │   │  + MinIO     │
   └──────────┘    └──────────────┘   └──────────────┘
```

---

## 3. Docker Compose 部署 (测试/预发布)

### 3.1 docker-compose.yml

```yaml
version: '3.8'
services:
  # === FastAPI Backend ===
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_BASE_URL=${LLM_BASE_URL:-https://api.deepseek.com/v1}
      - LLM_MODEL=${LLM_MODEL:-deepseek-v4-pro}
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
    depends_on:
      - redis
      - milvus
    restart: unless-stopped
    volumes:
      - ./data:/app/data

  # === Milvus Standalone ===
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.4.13-hotfix
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"

  # === Redis ===
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # === Frontend (Nginx) ===
  frontend:
    build:
      context: frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  etcd_data:
  minio_data:
  milvus_data:
  redis_data:
```

### 3.2 Dockerfile.api

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pymilvus \
    pydantic-settings openai fastembed httpx

COPY opmind/ ./opmind/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "opmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 Dockerfile.frontend

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 3.4 nginx.conf

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 4. Kubernetes 部署 (生产)

### 4.1 资源规划

| 组件 | Replicas | CPU req/limit | Memory req/limit | 说明 |
|------|----------|---------------|------------------|------|
| FastAPI | 3-5 | 2/4 core | 4/8 Gi | HPA 基于 CPU 70% |
| Milvus Proxy | 2 | 1/2 core | 2/4 Gi | 查询路由 + 负载均衡 |
| Milvus Query Node | 3+ | 4/8 core | 16/32 Gi | 向量检索主力 |
| Milvus Data Node | 2 | 2/4 core | 8/16 Gi | 数据写入 + 索引构建 |
| Milvus Index Node | 1 | 4/8 core | 16/32 Gi | 后台索引构建 |
| PostgreSQL | 2 (主从) | 2/4 core | 4/8 Gi | 元数据 + 审计日志 |
| Redis | 3 (Cluster) | 1/2 core | 2/4 Gi | 状态/消息/缓存 |
| MinIO | 4 | 1/2 core | 2/4 Gi | 原始文档存储 |
| Nginx Ingress | 2 | 1/2 core | 1/2 Gi | 入口网关 |

### 4.2 关键 K8s 对象

```yaml
# HPA 示例
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opsmind-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opsmind-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: opsmind-config
data:
  LLM_BASE_URL: "https://api.deepseek.com/v1"
  LLM_MODEL: "deepseek-v4-pro"
  EMBEDDING_MODEL: "BAAI/bge-m3"
  TOP_K: "10"
  MAX_ITERATIONS: "3"

# Secret (通过 External Secrets Operator 或 Vault 管理)
apiVersion: v1
kind: Secret
metadata:
  name: opsmind-secrets
type: Opaque
stringData:
  LLM_API_KEY: "<from-vault>"
  REDIS_PASSWORD: "<from-vault>"
  POSTGRES_PASSWORD: "<from-vault>"
```

---

## 5. 安全加固

### 5.1 认证与授权

| 层级 | 方案 | 实现 |
|------|------|------|
| 前端 → API | JWT + API Key | FastAPI middleware, HS256/RS256 |
| API → LLM | API Key (Secret) | K8s Secret + env var |
| API → Milvus | Token 认证 | Milvus `common.security.authorizationEnabled` |
| API → Redis | ACL + Password | Redis ACL 限制命令 |
| API → PostgreSQL | SSL + Password | 连接串含 sslmode=require |
| 运维 → K8s | RBAC + OIDC | IRSA for AWS, Workload Identity for GCP |

### 5.2 网络安全

- API 仅暴露 443 端口，内部服务通过 K8s Service 通信
- Nginx Ingress 做 TLS 终止 (Let's Encrypt / cert-manager)
- 所有内部组件启用 TLS (mTLS via Istio 可选)
- 出站 LLM API 调用通过 NAT Gateway

### 5.3 敏感数据处理

| 数据类型 | 处理策略 |
|----------|---------|
| API Key | K8s Secret / HashiCorp Vault，定期轮换 |
| 用户查询日志 | PII 脱敏（邮箱/电话打码），7 天自动归档 |
| 文档内容 | 访问控制：按团队/角色过滤 Milvus 查询 |
| 审计日志 | WORM 存储，保留 90 天，不可删除 |

---

## 6. 监控与告警

### 6.1 关键指标

| 指标 | 告警阈值 | 严重程度 |
|------|---------|---------|
| `opsmind_retrieval_latency_seconds` (p95) | > 500ms | Warning |
| `opsmind_retrieval_latency_seconds` (p95) | > 2s | Critical |
| `opsmind_agent_iterations_total` (avg) | > 3 | Warning |
| `opsmind_tool_execution_total{status="failure"}` | > 5% of total | Critical |
| `http_server_duration_ms` (p99) | > 5s | Critical |
| `milvus_num_entities` (delta) | < 0 for 10min | Warning (索引异常) |
| `redis_connected_clients` | > 100 | Warning |
| LLM API `4xx/5xx` 错误率 | > 1% | Critical |

### 6.2 日志

- 格式: 结构化 JSON
- 级别: INFO (默认), DEBUG (可动态开启)
- 聚合: Fluentd → Elasticsearch / Loki
- 保留: 热数据 3 天, 温数据 30 天, 冷归档 90 天

### 6.3 链路追踪

```python
# opmind/observability/tracer.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

导出到 Jaeger / Grafana Tempo，通过 OpenTelemetry Collector。

---

## 7. 灾备与恢复

### 7.1 备份策略

| 数据 | 备份方式 | 频率 | 保留 |
|------|---------|------|------|
| Milvus 向量数据 | Milvus 备份工具 → MinIO/S3 | 每日 | 30 天 |
| PostgreSQL | `pg_dump` → S3 | 每小时增量, 每日全量 | 90 天 |
| Redis | AOF + RDB 快照 → S3 | AOF 实时, RDB 每小时 | 7 天 |
| 原始文档 | S3 版本控制 | 实时 | 永久 |

### 7.2 恢复流程

```bash
# 1. 恢复 Milvus
docker compose restart milvus

# 2. 恢复 PostgreSQL
pg_restore -h $DB_HOST -U opsmind -d opsmind backup.dump

# 3. 恢复 Redis
docker compose stop redis
cp dump.rdb redis_data/
docker compose start redis

# 4. 冒烟测试
curl http://localhost:8000/health
python scripts/smoke_test.py
```

---

## 8. 容量规划

### 8.1 存储估算

| 数据类型 | 每 1000 文档 | 每 10000 文档 | 每 100000 文档 |
|----------|-------------|--------------|----------------|
| 原始文档 | ~50 MB | ~500 MB | ~5 GB |
| Milvus 向量 (384d) | ~100 MB | ~1 GB | ~10 GB |
| Milvus 向量 (1024d) | ~250 MB | ~2.5 GB | ~25 GB |
| PostgreSQL 元数据 | ~5 MB | ~50 MB | ~500 MB |
| 审计日志 (月) | ~10 MB | ~100 MB | ~1 GB |

### 8.2 性能基准 (生产目标)

| 指标 | 目标值 | 条件 |
|------|--------|------|
| 检索延迟 (p95) | < 200ms | Milvus, 100K chunks |
| 端到端延迟 (p95) | < 5s | 含 LLM 推理 |
| 并发用户 | 50+ | FastAPI × 5 replicas |
| 摄入吞吐 | >100 docs/min | 批量嵌入 |

---

## 9. 运维 Runbook

### 9.1 日常巡检

```bash
# 健康检查
curl http://localhost:8000/health

# Milvus 状态
curl http://localhost:9091/healthz
docker compose ps

# Redis 状态
redis-cli -h redis -p 6379 INFO stats

# 磁盘使用
df -h /data
```

### 9.2 LLM 故障切换

```bash
# 当 DeepSeek API 不可用时，自动/手动切换到备用模型
# 修改 ConfigMap 或环境变量
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
# 重启 API pods
kubectl rollout restart deployment/opsmind-api
```

### 9.3 索引重建

```bash
# 当向量质量下降或切换 Embedding 模型时
# 1. 清空旧索引
python -c "from opmind.retrieval.vector_store import VectorStore; VectorStore().clear()"

# 2. 重新摄入
python scripts/ingest.py

# 3. 冒烟验证
python scripts/smoke_test.py
```

---

## 10. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-06-20 | 初始版本: Docker Compose + K8s 部署方案、安全加固、监控告警、灾备恢复 |
