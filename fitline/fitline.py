import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式

def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

#用于判断电路连接中电流表电压表连接问题：一根导线和哪个接线柱子相连
#因为在算法处理时，如果导线检测在靠近接线柱的地方断开了，这时用欧式距离直接求距离判断是存在问题的
#必须求接线柱到曲线末端的二点的直线距离才能准确判断
def FitLineTest():
    img = cv2.imread('fitline_test_1.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #二值化
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #求轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cnt = cnt[:, 0, :]
    center_point = cnt[0]
    cnt_1 = contours[1]
    cnt_1 = cnt_1[:, 0, :]
    line_list = []
    for point in cnt_1:
        if math.hypot(point[0]-center_point[0],point[1]-center_point[1]) < 40:
            line_list.append(point)
    line_list = np.array(line_list)
    # 直线拟合
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(line_list, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = -float(vy)/float(vx)  # 直线斜率
    lefty = int((x*slope) + y)
    righty = int(((x-cols)*slope)+y)
    cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)
    #计算接线柱到曲线末端直线的距离
    distance = get_distance_from_point_to_line(center_point, (cols-1, righty), (0, lefty))
    #任意取一点到曲线末端的欧式距离比接线柱到曲线的距离要近就行
    temp_point = [70, 170]
    #计算该点到曲线末端直线的距离
    distance_1 = get_distance_from_point_to_line(temp_point, (cols-1, righty), (0, lefty))
    print('distance:', distance, 'distance_1:', distance_1)
    text1 = 'Center: (' + str(int(x)) + ', ' + str(int(y)) + ') '
    text2 = 'Slope: ' + str(round(slope, 2))
    #将曲线末端的直线画出
    cv2.putText(img, text1, (10, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
    cv2.putText(img, text2, (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
    #将任意取的一点在图中画出
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    cv2.circle(img, (70, 170), point_size, point_color, thickness)

    plt.subplot(236), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Line')
    plt.show()

if __name__ == "__main__":
    print('fit line start...')
    FitLineTest()