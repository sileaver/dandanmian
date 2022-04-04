# -*- coding:utf8 -*-
import cv2
import os
import shutil


def get_frame_from_video(_video_name, num_s, _save_path):
    """
    :param _video_name: 输入视频路径
    :param num_s: 保存图片的帧率间隔
    :param _save_path: 抽出的图片保存的位置
    """

    # 保存图片的路径
    path = _video_name.split('.mp4')[0]
    file_name = path.split('/')[-1]
    print(file_name)

    is_exists = os.path.exists(_save_path)
    if not is_exists:
        os.makedirs(_save_path)
        print('path of %s is build' % _save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)
    #     print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    cap = cv2.VideoCapture(_video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    i = 0
    j = 0

    while True:
        success, frame = cap.read()
        i += 1
        if i % int(fps / num_s) == 0:
            # 保存图片
            try:
                j += 1
                save_name = _save_path + file_name + '_' + str(j).zfill(4) + '.jpeg'
                cv2.imwrite(save_name, frame)
            except:
                print('出现未知错误！跳过')

            print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break


if __name__ == '__main__':
    # 视频文件名字

    file_path = 'Car_Opencv/data/'
    save_path = 'Car_Opencv/'
    files = os.listdir(file_path)  # 采用listdir来读取所有文件
    files.sort()
    interval = 1  # 设置每秒抽多少帧
    for file_ in files:
        video_name = file_path + file_
        get_frame_from_video(video_name, interval, save_path)


