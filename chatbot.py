
import random
import sys
from pathlib import Path
import time
import torch
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore, QtGui, QtWidgets
from main import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap,QMovie
import threading

#import task2
import argparse
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
#import task 2
import torch
from gtts import gTTS
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import json
import speech_recognition as sr
from pyvi import ViTokenizer, ViPosTagger

thread2_running = False
thread3_running = False
# task1 hien thi gif mat
def task1():
    global p1,p2,p3_1,p3_2,ui,im0,user_input,out_text,c,thread3_running,thread2_running,d
    p1=1
    p2=1
    p3_1=2
    p3_2=2
    c=0
    d=2
    class MYUI(Ui_MainWindow):
        global p2,ui,im0
        def __init__(self):
            self.setupUi(MainWindow)
            self.stackedWidget.setCurrentWidget(self.page_1)
            self.start_animation()
            self.Page1.clicked.connect(lambda: self.changePage(1))
            self.Page2.clicked.connect(lambda: self.changePage(2))
            self.Page3.clicked.connect(lambda: self.changePage(3))
            self.start.clicked.connect(self.starlisten_speak)
            
        def start_animation(self):
            # Specify the path to the GIF file you want to display
            gif_path = "ezgif.com-gif-maker.gif"
            # Create a QMovie object and load the GIF file
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(self.label.size())  # Set the animation size to match the label size
            self.label.setMovie(self.movie)

            # Start the animation
            self.movie.start()
        def update_frame(self):
            global im0
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            height, width, channel = im0.shape
            bytesPerLine = 3 * width
            image = QImage(im0.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label_2.setPixmap(pixmap)
        def changePage(self, index):
            global p2,thread3_running
            if index==1:
                self.stackedWidget.setCurrentWidget(self.page_1)
                thread3_running=False
            elif index==2:
                self.stackedWidget.setCurrentWidget(self.page_2)
                p2=2
                thread3_running=False
            elif index==3:
                self.stackedWidget.setCurrentWidget(self.page_3)
                thread3_running=True
                thread3 = threading.Thread(target=task3)
                thread3.start()
                
                
        def starlisten_speak(self):
            global c,thread2_running,thread3_running
            c=c+1
            if c==1 or thread3_running:
                thread2_running = True
                thread2 = threading.Thread(target=task2)
                thread2.start()
            elif c==2:
                thread2_running = False
                print('c = 2')
            elif c==3:
                print('c = 1')
                c=0

        def change_gif(self, gif_path):
        # Dừng animation hiện tại (nếu có)
            self.movie.stop()
            
            # Thay đổi đường dẫn file GIF
            self.movie.setFileName(gif_path)
             # Đặt kích thước animation bằng kích thước cửa sổ
            
            # Bắt đầu animation mới
            self.movie.start()
        def animation_finished(self):
            # Animation has finished, change the gif_path back to "nhinthang.gif" after a delay
            QTimer.singleShot(5000, lambda: self.change_gif("4.gif"))
        def label5(self):
            self.label_5.setText(user_input)
        def label6(self):
            self.label_6.setText(out_text)
    

    #detect.py
    def run(
            weights=ROOT / 'best.pt',  # model path or triton URL
            source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        #đọc khung hình từ webcam
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        global ui, p2,im0
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                    
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                x=im.shape[3]
                y=im.shape[2]

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Thay đổi tỷ lệ các hộp từ kích thước img_size thành kích thước im0
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    boxes = det[:, :4].int()
                    labels = det[:, -1].int()
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box.tolist()
                        print(f'Label: {label}')
                        if label==0:
                            if 0 <= x1 <0.35*x:
                                print(f' x1: {x1}, y1: {y1}')
                                gif_path2 = "5.gif"
                                ui.change_gif(gif_path2)
                                print('đã thêm')
                            elif 0 <= y1 <0.35*y:
                                print(f' x1: {x1}, y1: {y1}')
                                gif_path2 = "6.gif"
                                ui.change_gif(gif_path2)
                                
                                print('đã thêm')
                            elif 0.65*y<= y1 <y:
                                print(f' x1: {x1}, y1: {y1}')
                                gif_path2 = "8.gif"
                                ui.change_gif(gif_path2)
                                
                                print('đã thêm')
                            elif 0.65*x<= x1 <x:
                                print(f' x1: {x1}, y1: {y1}')
                                gif_path2 = "7.gif"
                                ui.change_gif(gif_path2)
                                
                                print('đã thêm')
                            ui.animation_finished()

                    
                    # Thêm giá trị từ tensor boxes vào danh sách boxes_list

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                # Stream results
                im0 = annotator.result()
                global p2
                if p2==2:
                    ui.update_frame()
                else:
                    continue
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')

        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt
    if __name__ == "__main__":
        opt = parse_opt()
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = MYUI()
        MainWindow.show()
        run(**vars(opt))
        sys.exit(app.exec_())
#task3 nghe nói

#phận loại câu hỏi xem dùng model hay datachatbot
with open('datachatbot.json','r', encoding="utf-8") as file:
    data = json.load(file)
def compare(user_input, data):
    max_matched_words = 0
    best_answer = None
    tokens, tags = ViPosTagger.postagging(ViTokenizer.tokenize(user_input))
    tokens = set(tokens)

    for item in data:
        question_tokens, _ = ViPosTagger.postagging(ViTokenizer.tokenize(item['question']))
        matched_words = len(tokens.intersection(set(question_tokens)))

        if matched_words > max_matched_words:
            max_matched_words = matched_words
            best_answer = random.choice(item['answer'])
    return best_answer


def play_wav(wav_file_path, speed_factor=1.0):
    # Chuyển đổi tệp MP3 sang WAV
    audio = AudioSegment.from_mp3('output.mp3')
    audio.export(wav_file_path, format="wav")
    # Đọc tệp WAV với thư viện scipy
    fs, data = wavfile.read(wav_file_path)
    # Tăng tốc độ phát âm thanh và phát
    sd.play(data, speed_factor * fs)
    sd.wait()
    # Xóa dữ liệu trong tệp WAV và MP3 sau khi phát xong
    with open(wav_file_path, 'wb') as wav_file:
        wav_file.truncate(0)
    with open(wav_file_path.replace('.wav', '.mp3'), 'wb') as mp3_file:
        mp3_file.truncate(0)
def run2():
    global user_input,p3_1,p3_2,out_text,thread2_running,d
    wav_file_path = 'temp_audio1.wav'
    while thread2_running:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Đang nghe...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio, language="vi-VN")
            print("Đã nhận được: " + user_input)  
            p3_1=1 
        except sr.UnknownValueError:
            print("Không nhận dạng được giọng nói.")
        if user_input == 'kết thúc':
            break
        
        try:
            best_answer=compare(user_input,data)
        except:
            best_answer='xin lỗi tôi chưa hiểu bạn nói gì'
        print("tra loi:",best_answer)
        p3_2=1
        tts = gTTS(best_answer, lang='vi')
        tts.save('output.mp3')
        play_wav(wav_file_path, speed_factor=1.0)
def task2_contact_task1(t):
    global ui
    if t=='sang phải':
        gif_path2 = "quayphai2.gif"
        ui.change_gif(gif_path2)
        ui.animation_finished()
        print('đã thêm')
def task2():
    global thread2_running
    if __name__=='__main__':
        run2()
def task3():
    global user_input,p3_1,p3_2,ui,out_text,thread3_running
    while thread3_running:
        if p3_1 == 1:  
            ui.label5()
            p3_1=2
            print('thread3 dang chay')
        if p3_2 == 1:
            ui.label6()
            p3_2=2
        time.sleep(2)
# Tạo hai đối tượng Thread cho hai luồng
thread1 = threading.Thread(target=task1)
# Bắt đầu thực thi hai luồng
thread1.start()
# Chờ hai luồng kết thúc
thread1.join()
print("Chương trình đã kết thúc.")
