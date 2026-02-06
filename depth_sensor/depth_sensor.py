import numpy as np
import cv2
from openni import openni2
import time
# 初始化 OpenNI2
openni2.initialize()
# 打开设备
dev = openni2.Device.open_any()
# 创建深度流
depth_stream = dev.create_depth_stream()
# 创建新的视频模式
new_mode = depth_stream.get_video_mode()
new_mode.resolutionX = 640 # 修改深度相机深度图分辨率->查看自己相机设备是否兼容该分辨率
new_mode.resolutionY = 400
new_mode.fps = 60 # 修改深度图帧率->查看自己设备是否兼容
new_mode.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM # 设置深度数据格式
depth_stream.set_video_mode(new_mode) # 应用设置
depth_stream.start()
# 获取RGB彩图
cap = cv2.VideoCapture(0) #捕捉彩色图像->我的深度相机的彩色摄像头支持UVC协议，所以opencv可以直接调用彩色图，而深度图需要通过openni调用
cap.set(3,640)#设置RGB彩图分辨率->查看自己相机设备是否兼容该分辨率
cap.set(4,480)
fps = 0.0
while True:
	t1 = time.time()
	# 获取深度帧
	depth_frame = depth_stream.read_frame()
	depth_data = depth_frame.get_buffer_as_uint16()
	depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
	#归一化深度图以便于显示
	depth_image = cv2.flip(depth_image, 1)  # 参数1表示水平翻转
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
	ret, color_image = cap.read()
	depth_value = depth_image[320, 240] # 获取对应坐标的深度数据
	# 显示图像
	#depth_colormap = cv2.resize(depth_colormap, (640, 480)) # opencv拉伸深度图分辨率由640*400变为640*480，以满足下句代码和彩图640*480拼接显示
	#combined_image = np.hstack((color_image, depth_colormap)) # 拼接显示
	fps = (fps + (1. / (time.time() - t1))) / 2
	print(fps)
	print(depth_value)
	cv2.imshow("Color", color_image)
	cv2.imshow("Depth", depth_colormap)
	
	# 按下 'q' 键退出
	if cv2.waitKey(1) == ord('q'):
		break
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
