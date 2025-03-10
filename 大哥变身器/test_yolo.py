from ultralytics import YOLO

# 加载模型
model = YOLO("runs\\detect\\train\\weights\\best.pt")

# 预测 val 文件夹中的所有图片
results = model.predict(
    source="dataset\\images\\test",  # 设置为文件夹路径
    save=True,                      # 保存带检测框的图像
    conf=0.50                       # 置信度阈值
)

# 遍历并打印所有图片的检测结果
for result in results:
    print(f"图片: {result.path}")  # 打印当前处理的图片路径
    print(result.boxes)            # 打印检测框信息
    print("-" * 50)                # 分隔线，便于区分不同图片的结果
