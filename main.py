import os
from Config import *

class monitor_mouse:
    def __init__(self, img, xyxy_list=None, xyxy_tree=None):
        self.lsPointsChoose = []  # 存入选择的点
        self.tpPointsChoose = []  # 存入选择的点
        self.pointsCount = 0  # 对鼠标按下的点计数(初始化)
        self.count = 0  # 统计量：鼠标按下的点计数(初始化)
        self.pointsMax = 2  # 最多点击多少下
        self.wim_name = 'src'  # 显示图像框名称
        self.img = img  # 转录输入图像
        self.img2 = img.copy()  # 对输入图像进行copy
        self.xyxy_list = xyxy_list  # 根据面积由小到大存储坐标框xyxy格式
        self.xyxy_tree = xyxy_tree  # sam结果存储到红黑树中便于查找

    def on_mouse(self, event, x, y, flags, param):
        # -------------------------左键单击事件---------------------------------------
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            if self.pointsCount == self.pointsMax:  # 设定了只允许两点成线
                self.clear_info()

            self.pointsCount += 1  # 当左键点击后，记录点击的次数
            cv2.circle(self.img2, (x, y), 2, (255, 0, 0), 2)  # 绘制了当前鼠标选取的点
            # 将选取的点保存到list列表里
            self.lsPointsChoose.append([x, y])  #
            self.tpPointsChoose.append((x, y))  # 用于画点
            # ----------------------------------------------------------------------
            # # 将鼠标选的点用直线连起来
            for i in range(len(self.tpPointsChoose) - 1):
                cv2.line(self.img2, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)

            cv2.imshow(self.wim_name, self.img2)

        # -------------------------右键单击事件-----------------------------
        if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
            self.axis_in_boxs()  # 输出单击选择的坐标集
            self.clear_info()
            cv2.imshow(self.wim_name, self.img2)

        # -------------------------左键双击事件：清空画布及控制台-----------------------------
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.img2 = self.img.copy()
            self.clear_info()
            self.clear_win()
            cv2.imshow(self.wim_name, self.img2)

    def clear_info(self):
        self.pointsCount = 0
        self.tpPointsChoose.clear()
        self.lsPointsChoose.clear()
        self.count = 0

    def clear_win(self):
        os.system('cls')

    def axis_in_boxs(self):
        for _, box in self.xyxy_tree.items():
            if is_box_containing_point(box, self.lsPointsChoose):
                cv2.rectangle(self.img2, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 2)
                print("pt1:", (box[0], box[1]), " ", "pt2:", (box[2], box[3]))
                break


if __name__ == "__main__":
    mask_generator = load_mask()  # 加载网络

    src_img = cv2.imread("car.jpg")
    masks = mask_generator.generate(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    xywh_list = mask_infer(masks)  # 通过mask得到面积和坐标
    xyxy_list = xywh2xyxy(xywh_list)  # xywh2xyxy
    xyxy_tree = build_box_tree(xyxy_list)  # 红黑树
    mm = monitor_mouse(src_img, xyxy_list, xyxy_tree)  # 启用鼠标监控

    cv2.namedWindow('src')
    cv2.setMouseCallback('src', mm.on_mouse)
    cv2.imshow('src', src_img)
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()
