from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 训练模型
train_results = model.train(
    data="data.yaml",  # 数据集 YAML 路径
    epochs=120,  # 训练轮次
    imgsz=640,  # 训练图像尺寸
    batch=8,
    device="cpu",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)
# import torch
#
# # 检查 GPU 是否可用
# print("GPU 可用:", torch.cuda.is_available())
#
# # 检查 CUDA 版本
# print("CUDA 版本:", torch.version.cuda)
#
# # 检查 GPU 数量
# print("GPU 数量:", torch.cuda.device_count())
#
# # 检查当前 GPU 索引
# print("当前 GPU 索引:", torch.cuda.current_device())
#
# # 检查 GPU 名称
# print("GPU 名称:", torch.cuda.get_device_name(0))

# 评估模型在验证集上的性能
# metrics = model.val()
# print(metrics)

# 在图像上执行对象检测
# results = model("path/to/1.jpg")
# results[0].show()

# 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径