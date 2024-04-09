import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.plots import plot_lettuce_centercircle  
from utils.plots import plot_weed_center
from utils.plots import Lettuce_detection_box_sorting, weed_sort_ymin, weed_sort_Smax
from utils.plots import Region_Segmentation
from utils.plots import grading
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace   #参数重命名
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))    #在桌面上展示

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run 如果EXP已经存在就加1为exp1
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  如果保存save_txt 则存在exp labels文件夹中

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA  选择处理器

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model  加载模型
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size  验证图片像素是否是32的倍数

    if trace:
        model = TracedModel(model, device, opt.img_size)   #部署的桥梁

    if half:
        model.half()  # to FP16

    # Second-stage classifier  提高准确度的二阶段分类
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader  识别数据导入
    vid_path, vid_writer = None, None
    if webcam:  #摄像头
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]     #随机每个类别的颜色

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
#分级程序开始
        t4 = time_synchronized()    
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt 保存路径
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
############################ Write results#######################################################
        ################生菜杂草坐标提取程序###########################
                #lettuce_number = 0                
                #lettuce_wh = []
                lettuce_center = []
                weed_centerall = []  #杂草中心坐标存放处
                lettuce_xyall = []   #生菜四角坐标存放处
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 长宽中心点坐标 坐标在这里
                    #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #print(xyxy)
                    xyxy1 = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    #print(xyxy1)
                    #print(xywh)
                    if cls == 0:
                        lettuce_xy = []  #生菜中点坐标临时存放列表
                        x1y1 = []        #生菜矩形框左下角坐标
                        x1y2 = []        #生菜矩形框左上角坐标
                        x2y1 = []        #生菜矩形框右下角坐标
                        x2y2 = []        #生菜矩形框右上角坐标
                        lettuce_xy1 = []

                        #lettuce_number +=1

                        #lettuce_xy.append(int(xywh[0]*gn[0])) #生菜中心点x轴像素坐标
                        #lettuce_xy.append(int(xywh[1]*gn[1])) #生菜中心点y轴像素坐标
                        
                        lettuce_xy.append(int(xyxy1[0]+((xyxy1[2]-xyxy1[0])/2))) #生菜中心点x轴像素坐标
                        lettuce_xy.append(int(xyxy1[1]+((xyxy1[3]-xyxy1[1])/2))) #生菜中心点y轴像素坐标
                        lettuce_center.append(lettuce_xy)     #生菜中点坐标最终存放列表
                        #print(lettuce_center)
                        #将四角坐标提取出来写在对应的数组中
                        x1y1.append(int(xyxy1[0]))    
                        x1y1.append(int(xyxy1[3]))
                        x1y2.append(int(xyxy1[0]))
                        x1y2.append(int(xyxy1[1]))
                        x2y1.append(int(xyxy1[2]))
                        x2y1.append(int(xyxy1[3]))
                        x2y2.append(int(xyxy1[2]))
                        x2y2.append(int(xyxy1[1]))
                        #############################
                        #lettuce_wh.append(int(xywh[2]*gn[2])) #生菜矩形框宽
                        #lettuce_wh.append(int(xywh[3]*gn[3])) #生菜矩形框高
                        #x1y1.append(int(lettuce_xy[0]-0.5*lettuce_wh[0]))
                        #x1y1.append(int(lettuce_xy[1]+0.5*lettuce_wh[1]))
                        #x1y2.append(int(lettuce_xy[0]-0.5*lettuce_wh[0]))
                        #x1y2.append(int(lettuce_xy[1]-0.5*lettuce_wh[1]))
                        #x2y1.append(int(lettuce_xy[0]+0.5*lettuce_wh[0]))
                        #x2y1.append(int(lettuce_xy[1]+0.5*lettuce_wh[1]))
                        #x2y2.append(int(lettuce_xy[0]+0.5*lettuce_wh[0]))
                        #x2y2.append(int(lettuce_xy[1]-0.5*lettuce_wh[1]))
                        lettuce_xy1.append(x1y1)
                        lettuce_xy1.append(x1y2)
                        lettuce_xy1.append(x2y1)
                        lettuce_xy1.append(x2y2)
                        lettuce_xyall.append(lettuce_xy1)
                        #print("作物检测框坐标：", lettuce_xyall)
                        
                    elif cls != 0:
                        weed_center = []   #杂草中心点坐标和像素坐标的存在地方
                        weed_center.append(int(xyxy1[0]+((xyxy1[2]-xyxy1[0])/2)))  #杂草中心点坐标x轴像素坐标
                        weed_center.append(int(xyxy1[1]+((xyxy1[3]-xyxy1[1])/2)))  #杂草中心点坐标y轴像素坐标
                        weed_center.append(int((xyxy1[2]-xyxy1[0])*(xyxy1[3]-xyxy1[1])))  #杂草像素面积 s= w*h
                        weed_centerall.append(weed_center)
                ################生菜杂草坐标提取程序###########################
                        #print(weed_center)
                        #print("杂草中心坐标：", weed_centerall)
                    if save_img or view_img:  # Add bbox to image
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)
                        #plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                        
                        if cls == 0:
                            plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=15)
                            #plot_lettuce_centercircle(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                        else: 
                            plot_weed_center(xyxy, im0, color=colors[int(cls)], line_thickness=10)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
             # Stream results

            #print("作物坐标", lettuce_xyall) 
            #print("杂草坐标", weed_centerall)
            ###################################打印需要的字符在图片上##############################
            PIC = 'weeds:'   #图片中的杂草总数
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, PIC, (1, 100), font, 4, (66, 92, 238), 10)  #结果印在图片上
            num = str(len(weed_centerall))     #杂草数量
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, num, (430, 100), font, 4, (255, 0, 255), 10)  #结果印在图片上
            left = 'Left intra-row zone:'   #左安全区
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, left, (1, 250), font, 4, (66, 92, 238), 10)  #结果印在图片上
            right = 'Right intra-row zone:'   #右安全区
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, right, (1, 400), font, 4, (66, 92, 238), 10)  #结果印在图片上
            T = 'Classification times:'   #分级时间
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, T, (1, 550), font, 4, (66, 92, 238), 10)  #结果印在图片上
            unit= 'ms'   #单位ms
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0, unit, (1500, 550), font, 4, (255, 0, 255), 10)  #结果印在图片上
            if len(lettuce_xyall) == 0:
                #print("警告：未检测到生产，故不构成行内杂草分级前提条件!")
                continue
            else:
            # Print time (inference + NMS)
            #########################坐标排序#####################################
                lettuce_center.sort(key=lambda x:x[1] )     #根据y坐标大小，从小到大排序
                weed_sort_ymin(weed_centerall)
                #weed_centerall.sort(key=lambda x:x[1])     #根据y坐标大小，从小到大排序
                Lettuce_detection_box_sorting(lettuce_xyall)  #生菜检测框排序，从小到大
                #print("排序后作物检测框坐标：", lettuce_xyall)
                #print("排序后杂草坐标：", weed_centerall)
            ###########################end#######################################
                if len(lettuce_xyall) == 2:
                #中心线
                    center_start_point = [lettuce_center[0][0],lettuce_xyall[0][0][1]]   
                    center_end_point = [lettuce_center[0][0],lettuce_xyall[1][1][1]]
                    cv2.line(im0, center_start_point, center_end_point, (0, 255, 0), 20)
                #左边界线
                    left_start_point = [lettuce_xyall[0][0][0],lettuce_xyall[0][0][1]]   
                    left_end_point = [lettuce_xyall[0][0][0],lettuce_xyall[1][1][1]]
                    cv2.line(im0, left_start_point, left_end_point, (0, 255, 0), 20)
                #右边界线
                    right_start_point = [lettuce_xyall[0][2][0],lettuce_xyall[0][0][1]]   
                    right_end_point = [lettuce_xyall[0][2][0],lettuce_xyall[1][1][1]]
                    cv2.line(im0, right_start_point, right_end_point, (0, 255, 0), 20)
                #下边界线
                    botton_start_point = [lettuce_xyall[0][2][0], lettuce_xyall[1][1][1]]   #右边的点
                    botton_end_point = [lettuce_xyall[0][0][0], lettuce_xyall[1][1][1]]     #左边的点
                    cv2.line(im0, botton_start_point, botton_end_point, (0, 255, 0), 20)
                #上边界线
                    top_start_point = [lettuce_xyall[0][0][0],lettuce_xyall[0][0][1]]   #左边的点
                    top_end_point = [lettuce_xyall[0][2][0],lettuce_xyall[0][0][1]]     #右边的点
                    cv2.line(im0, top_start_point, top_end_point, (0, 255, 0), 20)
            ##################################杂草所属区域划分#####################################
                point_1 = [1300, 250]
                point_2 = [1400, 400]
                if len(weed_centerall) == 0:
                    #print("判定结果：无草级")
                    left1 = 'no weed'   #左安全区判定结果
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im0, left1, point_1, font, 4, (255, 0, 255), 10)  #结果印在图片上
                    right1 = 'no wee'   #右安全区判定结果
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im0, right1, point_2, font, 4, (255, 0, 255), 10)  #结果印在图片上
                    continue
                else:
                    Danger_left, Danger_right, Safe_left, Safe_right = Region_Segmentation(lettuce_center, weed_centerall, lettuce_xyall)
                    print("左危险区:", Danger_left, "右危险区:",Danger_right, "左安全区:",Safe_left, "右安全区:",Safe_right)
                    weed_area_left = []   #左安全区的所有杂草
                    weed_area_right = []   #右安全区的所有杂草
                    if len(Safe_left) != 0:
                        for i in range(len(Safe_left)):
                            weed_area_left.append(Safe_left[i][2])
                        weed_sort_Smax(weed_area_left)
                    if len(Safe_right) != 0:
                        for i in range(len(Safe_right)):
                            weed_area_right.append(Safe_right[i][2])
                        weed_sort_Smax(weed_area_right)
                    #print('左边区域大到小：', weed_area_left)
                    #print('右边区域大到小：', weed_area_right)
                    #print("左安全区:",Safe_left, "右安全区:",Safe_right)
            ###########################################end##########################################

            #######################行内杂草分级######################################################
            #生菜检测框内的杂草全部由激光除草 株间杂草分为无草级 机械级和激光级，机械级激光级主要根据杂草的数量以及空间分布和杂草的大小进行判定
            ii = 0
            if len(Safe_left) == 0:           #判定左安全区情况
                #print("判定结果：无草级")
                left1 = 'no weed'   #左安全区判定结果
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im0, left1, point_1, font, 4, (255, 0, 255), 10)  #结果印在图片上
            else:
                grading(Safe_left, ii, im0, weed_area_left)
            ii = ii+1
            if len(Safe_right) == 0:           #判定右安全区情况
                #print("判定结果：无草级")
                right1 = 'no weed'   #右安全区判定结果
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im0, right1, point_2, font, 4, (255, 0, 255), 10)  #结果印在图片上
            else:
                grading(Safe_right, ii, im0, weed_area_right)
            #################################################################################################
#分级程序结束
        t5 = time_synchronized()
        #打印分级时间
        print(f'Done. ({(1E3 * (t5 - t4)):.1f}ms) classification.')
        grading_t = round(1E3 * (t5 - t4), 1)   #分级时间保留一位小数
        grading_time = str(grading_t)    #分级时间转化为字符
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im0, grading_time, (1300, 550), font, 4, (255, 0, 255), 10)  #结果印在图片上

        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)   
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'runs\train\exp\weights\best.pt', help='model.pt path(s)')  #weights: 用于检测的模型路径
    parser.add_argument('--source', type=str, default='data/mydata/9', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  #置信度阈值，大于该阈值的框才会被保留
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')     #NMS的阈值，大于该阈值的框会被合并，小于该阈值的框会被保留，一般设置为0.45
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')   #每处理完一张图片是否展示
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')  #是否将检测结果保存为txt文件，包括类别，框的坐标，默认为False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')   #是否将检测结果保存为txt文件，包括类别，框的坐标，置信度，默认为False
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')  #有些类不需要 就不输出
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()  #导入参数
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():   #不计算梯度
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
