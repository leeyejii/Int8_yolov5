#모델 크기 비교
import os

# 파일 경로
fp32_model_path = 'runs/train/exp97/weights/best.pt'  # 기존 FP32 모델
int8_model_path = 'quantized_fin.pt'  # INT8 모델

# 파일 크기 계산
fp32_size = os.path.getsize(fp32_model_path) / (1024 * 1024)  # MB 단위
int8_size = os.path.getsize(int8_model_path) / (1024 * 1024)  # MB 단위

print(f"FP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Model Size: {int8_size:.2f} MB")