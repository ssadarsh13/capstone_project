# Deployment Guide — AWS ECS Fargate

Deploys the Hospital ML Prediction API as a serverless container behind an
Application Load Balancer (ALB), with CloudWatch Logs for the prediction audit
trail. No EC2 instances to manage.

```
Client ──HTTPS──▶ ALB ──HTTP:8000──▶ ECS Fargate Task (uvicorn, 2 workers)
                   │                         │
              ACM cert                   CloudWatch Logs  ◀── stdout audit lines
                                              │
                                          ECR image
```

---

## 0. Prerequisites

- AWS CLI v2 configured (`aws configure`) with permissions for ECR, ECS, IAM, ELB, CloudWatch.
- Docker installed and running.
- A VPC with at least two public (or private + NAT) subnets across two AZs.
- Set these shell variables once:

```bash
export AWS_REGION=ap-south-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=hospital-ml-api
export IMAGE_TAG=v2.0.0
export ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}
```

---

## 1. Build & push the image to ECR

```bash
# Create the repository (first time only)
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build for the Fargate platform (linux/amd64) and push
docker build --platform linux/amd64 -t ${ECR_URI}:${IMAGE_TAG} .
docker push ${ECR_URI}:${IMAGE_TAG}
```

> Fargate runs `linux/amd64`. If you build on Apple Silicon / ARM, the
> `--platform linux/amd64` flag is **required** or the task will fail to start.

---

## 2. IAM roles

The task needs two roles:

- **Task execution role** (`ecsTaskExecutionRole`) — lets ECS pull from ECR and write to CloudWatch. Attach the AWS-managed policy `AmazonECSTaskExecutionRolePolicy`.
- **Task role** — application-level permissions. This service needs none beyond logging, so an empty role is fine.

```bash
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

---

## 3. CloudWatch log group

```bash
aws logs create-log-group --log-group-name /ecs/hospital-ml-api --region $AWS_REGION
aws logs put-retention-policy --log-group-name /ecs/hospital-ml-api \
  --retention-in-days 90 --region $AWS_REGION   # audit retention
```

---

## 4. Task definition

Save as `taskdef.json` (substitute `${...}` or render with `envsubst`):

```json
{
  "family": "hospital-ml-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "hospital-ml-api",
      "image": "${ECR_URI}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
      "environment": [
        { "name": "MODEL_VERSION", "value": "v2.0.0" },
        { "name": "LOG_LEVEL", "value": "INFO" },
        { "name": "PREDICTION_LOG_PATH", "value": "/var/log/hospital-ml/predictions.log" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)\""],
        "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 20
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hospital-ml-api",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register it:

```bash
aws ecs register-task-definition --cli-input-json file://taskdef.json --region $AWS_REGION
```

> **Sizing:** 0.5 vCPU / 1 GB is sufficient (the two models load to ~30 MB RAM,
> uvicorn runs 2 workers). Scale `cpu`/`memory` up if p99 latency degrades.

---

## 5. Networking — ALB + target group

```bash
# Security groups
aws ec2 create-security-group --group-name hospital-ml-alb-sg \
  --description "ALB ingress 443" --vpc-id $VPC_ID
aws ec2 create-security-group --group-name hospital-ml-task-sg \
  --description "Task ingress from ALB" --vpc-id $VPC_ID
# ALB SG: allow 443 from the internet; Task SG: allow 8000 from the ALB SG only.

# Target group (target type 'ip' is required for Fargate awsvpc)
aws elbv2 create-target-group \
  --name hospital-ml-tg --protocol HTTP --port 8000 --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 --unhealthy-threshold-count 3
```

Create the ALB, an HTTPS:443 listener (with an ACM certificate) forwarding to
the target group, and redirect HTTP:80 → HTTPS.

---

## 6. ECS cluster & service

```bash
aws ecs create-cluster --cluster-name hospital-ml-cluster --region $AWS_REGION

aws ecs create-service \
  --cluster hospital-ml-cluster \
  --service-name hospital-ml-api \
  --task-definition hospital-ml-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_A,$SUBNET_B],securityGroups=[$TASK_SG],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=hospital-ml-api,containerPort=8000" \
  --health-check-grace-period-seconds 30 \
  --region $AWS_REGION
```

`--desired-count 2` spreads tasks across two AZs for high availability.
(Use `assignPublicIp=ENABLED` only with public subnets; with private subnets use
a NAT gateway and `DISABLED`.)

---

## 7. Auto-scaling (recommended)

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs --resource-id service/hospital-ml-cluster/hospital-ml-api \
  --scalable-dimension ecs:service:DesiredCount --min-capacity 2 --max-capacity 6

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs --resource-id service/hospital-ml-cluster/hospital-ml-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-target-70 --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
     "TargetValue": 70.0,
     "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}}'
```

---

## 8. Verify

```bash
ALB_DNS=$(aws elbv2 describe-load-balancers --names hospital-ml-alb \
  --query 'LoadBalancers[0].DNSName' --output text)

curl https://$ALB_DNS/health
curl -X POST https://$ALB_DNS/predict/risk -H "Content-Type: application/json" -d @risk_request.json
```

---

## 9. Rolling updates

```bash
docker build --platform linux/amd64 -t ${ECR_URI}:v2.1.0 .
docker push ${ECR_URI}:v2.1.0
# bump image tag in taskdef.json, register a new revision, then:
aws ecs update-service --cluster hospital-ml-cluster --service hospital-ml-api \
  --task-definition hospital-ml-api --region $AWS_REGION
```

ECS performs a rolling deploy (`minimumHealthyPercent=100`,
`maximumPercent=200` by default) — new tasks must pass the `/health` check
before old tasks drain. **Bump `MODEL_VERSION`** whenever the artifacts change so
the audit log distinguishes model generations.

---

# Operations Runbook

## Service summary
| Item | Value |
|------|-------|
| Service | `hospital-ml-api` (ECS Fargate) |
| Port | 8000 (behind ALB 443) |
| Health check | `GET /health` |
| Logs | CloudWatch `/ecs/hospital-ml-api` (90-day retention) |
| Min/Max tasks | 2 / 6 (CPU target-tracking 70%) |
| Model version | env `MODEL_VERSION` (currently `v2.0.0`) |

## Health & monitoring
- **Liveness/Readiness:** ALB target group polls `/health` every 30 s. A task is
  cycled out after 3 consecutive failures.
- **Key CloudWatch alarms to configure:**
  - ALB `HTTPCode_Target_5XX_Count` > 0 (sustained) → page.
  - ALB `TargetResponseTime` p99 > 1 s → investigate latency.
  - ECS `CPUUtilization` > 85% for 5 min → scaling pressure.
  - `UnHealthyHostCount` ≥ 1 → task failing health checks.

## Common incidents

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| All requests `503` | Models failed to load at startup | Check task logs for `feature_schema.json`/joblib load errors; verify the image contains `models/`. |
| Task stuck `PROVISIONING`/crash-loop | Wrong CPU arch (built on ARM) | Rebuild with `--platform linux/amd64`. |
| `500` on valid input | unpickle mismatch (sklearn/numpy drift) | Confirm `requirements.txt` pins match training versions (sklearn 1.8.0, numpy 2.3.4). |
| Predictions look wrong / shifted | model–schema mismatch | Verify `MODEL_VERSION` matches the deployed artifacts; check the `feature_hash` of a known input against a golden value. |
| Health check flapping | `startPeriod` too short for cold start | Increase task `healthCheck.startPeriod` and ALB grace period. |

## Rollback
```bash
# Re-point the service to the previous known-good task revision
aws ecs update-service --cluster hospital-ml-cluster --service hospital-ml-api \
  --task-definition hospital-ml-api:<PREVIOUS_REVISION> --region $AWS_REGION
```
ECS rolls back with the same health-gated rolling strategy.

## Audit & governance
- Every prediction logs `request_id`, `model_version`, `feature_hash`,
  `prediction`, `probabilities`, and `latency_ms` to CloudWatch.
- To trace a decision: query CloudWatch Logs Insights by `request_id`:
  ```
  fields @timestamp, model_name, prediction, feature_hash, latency_ms
  | filter request_id = "f47fec9d-b07b-441d-bc5e-3c262d73ba24"
  ```
- `feature_hash` proves the exact engineered inputs without storing PHI.
- Retain logs ≥ 90 days (extend per regulatory requirements).

## Scaling & capacity
- Throughput is CPU-bound on `predict_proba`. Add tasks (raise max capacity) or
  bump task `cpu` before increasing uvicorn `--workers`.
- Each worker holds its own copy of the models in memory (~30 MB); keep
  `memory` ≥ 512 MB per task.

## Routine maintenance
- **Model refresh:** retrain (Phase 3) → drop new `.joblib`s into `models/` →
  build new image with a bumped `MODEL_VERSION` → rolling update.
- **Dependency patching:** rebuild monthly to pick up base-image security fixes;
  re-run `pytest` before pushing.
- **DR:** ECR image + this repo are the only stateful artifacts; both are
  reproducible. No database to back up.
