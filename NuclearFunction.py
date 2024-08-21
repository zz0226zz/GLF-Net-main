import os
import cv2
import numpy as np
from skimage import color, filters, morphology


def calculate_nuclei_score(patch):
    # 转换RGB图像到HED空间
    hed_image = color.rgb2hed(patch)

    # 提取Hematoxylin通道（即H通道）
    hematoxylin_channel = hed_image[:, :, 0]

    # 应用Yen阈值以生成初步的细胞核掩膜M1
    yen_threshold = filters.threshold_yen(hematoxylin_channel)
    yen_mask = hematoxylin_channel > yen_threshold
    # 反转yen_mask
    yen_mask_final = ~yen_mask

    # 对初步的细胞核掩膜M1应用White Top-Hat变换以生成清理后的掩膜M2
    selem = morphology.disk(2)
    white_tophat = morphology.white_tophat(yen_mask_final, selem)

    # 对White Top-Hat变换后的图像应用Yen阈值
    white_tophat_threshold = filters.threshold_yen(white_tophat)
    white_tophat_mask = white_tophat > white_tophat_threshold

    # 通过减去M2从M1中得到最终的细胞核掩膜
    nuclei_mask = yen_mask_final.astype(int) - white_tophat_mask.astype(int)
    nuclei_mask = np.clip(nuclei_mask, 0, 1)

    nuclei_ratio = np.sum(nuclei_mask) / nuclei_mask.size

    gray_image = color.rgb2gray(patch)

    # 使用Otsu阈值法进行二值化
    _, otsu_threshold = cv2.threshold((gray_image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反转二值化结果，使细胞核为白色
    otsu_threshold = cv2.bitwise_not(otsu_threshold)

    # 使用形态学膨胀操作来连接区域
    structuring_element = morphology.disk(5)  # 增大结构元素的大小
    binary_dilation = morphology.binary_dilation(otsu_threshold, structuring_element)

    # 去除小孔
    area_threshold = 18000  # 提高面积阈值以去除更多小孔
    tissue_mask = morphology.remove_small_holes(binary_dilation, area_threshold=area_threshold)

    # 计算Tt
    Tt = np.sum(tissue_mask) / tissue_mask.size

    nuclei_score = nuclei_ratio * np.tanh(Tt)

    return nuclei_score, otsu_threshold


def calculate_white_area_ratio(binary_image):
    # 计算白色区域占图像总面积的比例
    white_area = np.sum(binary_image > 0)
    total_area = binary_image.size
    return white_area / total_area


def process_images_in_directory(directory, top_k=100, white_area_threshold=0.8):
    image_scores = []

    # 遍历文件夹中的所有图片
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            patch = cv2.imread(file_path)
            nuclei_score, otsu_threshold = calculate_nuclei_score(patch)

            # 计算白色区域的比例
            white_area_ratio = calculate_white_area_ratio(otsu_threshold)

            # 仅处理白色区域占比小于指定阈值的图片
            if white_area_ratio < white_area_threshold:
                image_scores.append((filename, nuclei_score))

    # 按核分数降序排序
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # 保留核分数最高的前top_k个图片
    top_images = set([x[0] for x in image_scores[:top_k]])

    # 删除其他图片
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename not in top_images and os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    print(f"Kept top {top_k} images by nuclei score in {directory}.")


def process_all_subdirectories(parent_directory, top_k=100, white_area_threshold=0.8):
    # 遍历主文件夹中的所有子文件夹
    for subdirectory in os.listdir(parent_directory):
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            print(f"Processing directory: {subdirectory_path}")
            process_images_in_directory(subdirectory_path, top_k=top_k, white_area_threshold=white_area_threshold)


# 示例使用
parent_directory = 'D:/TCGA-LUAD/Regress'
process_all_subdirectories(parent_directory, top_k=100, white_area_threshold=0.8)

