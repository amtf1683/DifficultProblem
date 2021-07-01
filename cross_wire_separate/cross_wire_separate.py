import math
import numpy as np
import cv2
import copy
import random
from skimage import morphology
#np.set_printoptions(threshold=np.inf)

def DistancePoint2Line(line_start, line_end, point):
    normalLength = math.hypot(line_end.x - line_start.x, line_end.y - line_start.y)
    distance = float((point.x - line_start.x) * (line_end.y - line_start.y) - (point.y - line_start.y) * (
                line_end.x - line_start.x)) / normalLength
    return abs(distance)

class Point():
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_
    def add(self, a, b, c):
        c.x = a.x + b.x
        c.y = a.y + b.y

#交叉导线分离，适用于十字交叉型曲线的分离
def cross_wire_separate():
    print('start separate....')
    data_list = []
    img_ = cv2.imread('./test.jpg', cv2.IMREAD_GRAYSCALE)
    #对于很短的曲线，进行扩大图像，处理完再缩小
    scale = 3
    img_ = cv2.resize(img_, (img_.shape[1]*scale, img_.shape[0]*scale))
    #二值化
    ret, binary = cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #提取曲线骨架
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8)
    cv2.imwrite("skeleton.png", skeleton*255)
    cluser_list = []
    point_list = []
    contour_flag = False
    #点到直线距离最大值(欧式距离)
    max_dis = 8
    final_count = 0
    print(skeleton.shape)
    # i：高 代表多少行，5：每一次移动多少行遍历，太大太小都不行,如果曲线是二条交叉的直线，取大取小都没问题
    # 当二条曲线是弯弯曲曲时这个取值就很关键，对于曲度变化的交叉曲线，此种方法不能很好的解决问题
    for i in range(0, skeleton.shape[0], 5):
        for j in range(0, skeleton.shape[1]):#j：宽 代表多少列
            if skeleton[i, j] == 1:#[i, j]多少行多少列的值为1说明是曲线
                point_list.append(Point(j, i))
                contour_flag = True
            else:
                if contour_flag == True:
                    #对于比较粗的曲线，每次取该行所有值的平均值，所以上面的轮廓提取不是必须的，不提取轮廓也能分离曲线
                    point_middle = Point(0.0, 0.0)
                    point_middle.x = 0.0
                    point_middle.y = 0.0
                    for point_obj in point_list:
                        point_middle.x = point_middle.x + point_obj.x
                        point_middle.y = point_middle.y + point_obj.y
                    point_middle.x = float(point_middle.x / (len(point_list)))
                    point_middle.y = float(point_middle.y / (len(point_list)))
                    point_list.clear()
                    contour_flag = False
                    best_dis = 10000000.0
                    best_cluster_number = -1
                    for k in range(len(cluser_list)):
                        dis2clusters = 1000000.0
                        if len(cluser_list[k]) == 1:
                            #如果只有一个点，直接求范式距离就行
                            temp_array = np.array([[point_middle.x, point_middle.y],
                                                   [cluser_list[k][0][0], cluser_list[k][0][1]]])
                            dis = np.linalg.norm(temp_array[0] - temp_array[1])
                            if dis < dis2clusters:
                                dis2clusters = dis
                        else:
                            #多余一个点时，必须求点到直线的距离，点到同一条曲线的直线距离最短
                            line_a = Point(cluser_list[k][len(cluser_list[k]) - 1][0],
                                           cluser_list[k][len(cluser_list[k]) - 1][1])
                            line_b = Point(cluser_list[k][len(cluser_list[k]) - 2][0],
                                           cluser_list[k][len(cluser_list[k]) - 2][1])
                            dis = DistancePoint2Line(line_a, line_b, point_middle)
                            if dis < dis2clusters:
                                dis2clusters = dis
                        if round(dis2clusters, 3) < round(max_dis, 3):
                            if round(dis2clusters, 3) < round(best_dis, 3):
                                best_dis = dis2clusters
                                best_cluster_number = k
                            elif round(dis2clusters, 3) == round(best_dis, 3):
                                best_dis = dis2clusters
                                best_cluster_number = -2
                    if best_cluster_number == -2:
                        continue
                    if best_cluster_number < 0:
                        temp_list_ = []
                        temp_list_.clear()
                        temp_list_.append(copy.deepcopy(point_middle.x))
                        temp_list_.append(copy.deepcopy(point_middle.y))
                        temp_list_list = []
                        temp_list_list.clear()
                        temp_list_list.append(temp_list_)
                        cluser_list.append(temp_list_list)
                    else:
                        temp_list_2 = []
                        temp_list_2.clear()
                        temp_list_2.append(copy.deepcopy(point_middle.x))
                        temp_list_2.append(copy.deepcopy(point_middle.y))
                        cluser_list[best_cluster_number].append(temp_list_2)
                    best_cluster_number = -1
    #打印有几条曲线
    print('len:', len(cluser_list), 'line:', cluser_list)
    #在一张空图上将分离出来的曲线画出
    img_test = np.zeros([skeleton.shape[0], skeleton.shape[1]], np.uint8)
    for p_p in range(len(cluser_list)):
        #小于四个点的曲线不要画
        if len(cluser_list[p_p]) < 4:
            continue
        color_current = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for w in range(1, len(cluser_list[p_p])):
            cv2.line(img_test, (int(cluser_list[p_p][w - 1][0]), int(cluser_list[p_p][w - 1][1])),
                     (int(cluser_list[p_p][w][0]), int(cluser_list[p_p][w][1])), color_current, thickness=5)
    new_path_name = str(final_count) + '_out.jpg'
    #缩放成原图大小
    img_test = cv2.resize(img_test, (int(img_test.shape[1]/scale), int(img_test.shape[0]/scale)))
    cv2.imwrite(new_path_name, img_test)

if __name__ == "__main__":
    print('test cross wire separate')
    cross_wire_separate()


