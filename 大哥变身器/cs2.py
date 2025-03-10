import mss
import cv2
import numpy as np
import torch
import keyboard
from ultralytics import YOLO
import win32gui
import pydirectinput
from pynput.mouse import Button, Controller, Listener
import time
import ctypes



# 检查运行权限
print("运行权限:", "管理员" if ctypes.windll.shell32.IsUserAnAdmin() else "普通用户")
# 检查 CUDA 可用性
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(0))

# 加载 YOLOv11 模型（使用 GPU）
model = YOLO("runs/detect/train/weights/best.pt")  # 指定使用 CUDA
model.to("cuda")
print(model.device)

# 初始化 MSS 和鼠标控制器
sct = mss.mss()
mouse = Controller()

# 变量用于跟踪鼠标按钮状态
right_button_pressed = False

# 获取 CS:GO/Cs2 窗口（缩小捕获区域）
def get_csgo_window():
    hwnd = win32gui.FindWindow(None, "反恐精英：全球攻势") or win32gui.FindWindow(None, "Counter-Strike 2")
    if hwnd:
        rect = win32gui.GetWindowRect(hwnd)
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        # 捕获中心 1000x800 区域
        center_x, center_y = rect[0] + width // 2, rect[1] + height // 2
        capture_width, capture_height = 800, 600
        return {
            "top": center_y - capture_height // 2,
            "left": center_x - capture_width // 2,
            "width": capture_width,
            "height": capture_height
        }
    else:
        raise Exception("未找到 CS:GO 窗口。请确保游戏正在运行！")

# 鼠标事件处理程序
def on_click(x, y, button, pressed):
    global right_button_pressed
    if button == Button.left:  # 注意这里是左键
        right_button_pressed = pressed

# 启动鼠标监听器
listener = Listener(on_click=on_click)
listener.start()

# 设置灵敏度因子（根据 CS2 的灵敏度设置进行调整）
sensitivity_factor = 2.5  # 例如，如果灵敏度设置为 2.0，请根据实际情况进行调整

# 主循环
try:
    monitor = get_csgo_window()
    window_width, window_height = monitor["width"], monitor["height"]

    while True:
        start_time = time.time()

        # 捕获 CS:GO 窗口
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 将图像调整为模型输入大小
        img_resized = cv2.resize(img, (640, 640))

        # 模型推理
        results = model.predict(img_resized, conf=0.50)  # 持续的置信度阈值设为 0.50

        # 处理检测结果
        closest_box = None
        closest_distance = float('inf')
        center_x, center_y = window_width / 2 + monitor["left"], window_height / 2 + monitor["top"]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())  # 获取类别 ID
                if confidence > 0.30 :  # 仅在置信度高且类别为 CT 时处理  0:CT 1:Tand class_id == 1
                    # 获取检测框的中心坐标（归一化并映射到窗口坐标）
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    target_x = (x_min + x_max) / 2 * window_width / 640 + monitor["left"]
                    box_height = y_max - y_min  # 计算方框高度
                    target_y = (y_min + box_height * 0.1333) * window_height / 640 + monitor["top"]  # 头线

                    # 计算当前框的距离
                    distance = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)

                    # 更新最近的框
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_box = (target_x, target_y)

        if closest_box and (not keyboard.is_pressed('a') and not keyboard.is_pressed('d')and not keyboard.is_pressed('w')and not keyboard.is_pressed('s')): # or 1)

            # keyboard.press('ctrl')
            target_x, target_y = closest_box

            # 获取当前鼠标位置
            current_x, current_y = mouse.position

            # 计算相对移动增量并应用灵敏度因子
            delta_x = (target_x - current_x) * sensitivity_factor
            delta_y = (target_y - current_y) * sensitivity_factor

            # 使用 pydirectinput 相对移动鼠标
            pydirectinput.move(int(delta_x), int(delta_y), relative=True)

            # 可选：模拟射击（左键点击）
            if (not keyboard.is_pressed('a') and not keyboard.is_pressed('d')and not keyboard.is_pressed('w')and not keyboard.is_pressed('s')):

                mouse.press(Button.left)
                #for i in range(3):
                #    time.sleep(0.02)
                #    pydirectinput.move(int(0), int(15), relative=True)

                mouse.release(Button.left)
                time.sleep(0.05)


        ##else:
            ##keyboard.release('ctrl')

        # 显示结果
        annotated_img = results[0].plot()
        cv2.imshow("CS:GO 检测", annotated_img)

        # 计算 FPS
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(str(e))

# 停止监听器并清理
listener.stop()
cv2.destroyAllWindows()
print("Cleanup completed.")
