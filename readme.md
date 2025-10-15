# 🚀 Hướng dẫn Deploy Ứng dụng FastAPI

---

## 🧩 1. Chạy Local / VPS

### ✅ Chuẩn bị quyền Docker
```bash
sudo usermod -aG docker $USER
````

> Sau đó **đăng xuất và đăng nhập lại** để nhóm quyền có hiệu lực.

---

### ✅ Chuẩn bị thư mục runtime

```bash
mkdir -p outputs tmp
sudo chown 1000:1000 outputs tmp
sudo chmod -R 755 outputs tmp
```

---

### ✅ Build và chạy container

```bash
docker compose up --build -d
```

Ứng dụng sẽ chạy tại:
👉 **[http://localhost:8001](http://localhost:8001)**

---

## ⚙️ 2. Thông tin bổ sung

### ✅ Cài đặt AWS CLI

```bash
pip install awscli
```

### ✅ Cấu hình AWS CLI

```bash
./aws_configure.sh
```

### ✅ Đảm bảo user hiện tại có quyền chạy Docker mà **không cần sudo**

```bash
sudo usermod -aG docker $USER
```

> Nếu sau khi thêm nhóm mà vẫn bị lỗi `permission denied`, hãy đăng xuất rồi đăng nhập lại.

---

### ✅ Tạo IAM Role nếu chưa có

Lambda cần role `lambda-execution-role` với quyền cơ bản.
Tạo **một lần duy nhất** bằng lệnh sau:

```bash
aws iam create-role \
--role-name lambda-execution-role \
--assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Effect": "Allow",
        "Principal": { "Service": "lambda.amazonaws.com" },
        "Action": "sts:AssumeRole"
    }
    ]
}'
```

---

## ☁️ 3. Cấu hình AWS Lambda (chạy bằng script)

### ✅ Gắn policy mặc định

```bash
aws iam attach-role-policy \
--role-name lambda-execution-role \
--policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

---

## 🧭 4. Deploy lên Lambda

Sau khi có role và cấu hình AWS CLI (`aws configure`), chạy script:

```bash
bash deploy_lambda.sh
```

---

## 🧪 5. Kiểm tra Lambda sau khi deploy thành công

```bash
aws lambda invoke --function-name extract-panel-api response.json
cat response.json
```

---

## ✅ Tóm tắt

| Môi trường     | Cách chạy                      | File sử dụng                            |
| -------------- | ------------------------------ | --------------------------------------- |
| **Local/VPS**  | `docker compose up --build -d` | `docker-compose.yaml`                   |
| **AWS Lambda** | `bash deploy_lambda.sh`        | `Dockerfile.lambda`, `deploy_lambda.sh` |

---

> 🟢 Sau khi chạy, hệ thống sẽ tự tạo thư mục `outputs` và `tmp`, gán quyền phù hợp, và khởi động ứng dụng.

````

---

### 💡 Nâng cấp gợi ý (nếu bạn muốn):
Bạn có thể thêm **mục 6 - Debug nhanh** vào cuối file:

```md
---

## 🧩 6. Debug nhanh

### Xem log container:
```bash
docker logs -f phamthang-api
````

### Dừng toàn bộ dịch vụ:

```bash
docker compose down
```

🧪 Test local
docker build -t extract-panel-api -f Dockerfile.lambda .
docker run -p 8080:8080 extract-panel-api