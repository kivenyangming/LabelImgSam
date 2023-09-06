# LabelImgSam
这是一个通过mobileSam推理获取预测框进行筛选出目标框的脚本，协助标注人员快速标注工作，在小目标标注上具有明显优势\
# 流程：
1. CV2读取图像并转RGB格式
2. 调用mobileSam网络获取mask_generator
3. 筛选出图像推理结果中的area和bbox数组
4. 根据area数组中数据由小到大进行排序并将bbox数组内元素顺序同area顺序
5. 将bbox中的xywh格式数据更改为xyxy格式
6. 将xyxy数组转换为SortedDict格式数据
7. 两点法获取目标框最小外接矩形
