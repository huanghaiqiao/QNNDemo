# Ultralytics YOLO 🚀, AGPL-3.0 license
 
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import pickle
import json
import struct

CLASS_NAMES = {
    0: 'microphone' 
}
 
 
class YOLO11:
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.input_image = input_image #RGB_image path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = CLASS_NAMES
        self.color_palette = [(0, 255, 0)]
        
        self.detections = []  # 存储所有检测结果的列表
        self.detected_centers = []  # 专门存储中心坐标的列表
        self.debug_data = {}  # 存储调试数据

    def preprocess(self):
        self.img = cv2.imread(self.input_image)
        self.img_height, self.img_width = self.img.shape[:2]

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
 
        # 保持宽高比，进行 letterbox 填充
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
 
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data.tofile("tmp.raw")
        
        self.debug_data['preprocess'] = {
            'original_image_size': (self.img_width, self.img_height),
            'model_input_size': (self.input_width, self.input_height),
            'ratio': self.ratio,
            'dw': self.dw,
            'dh': self.dh
        }
        
        return image_data
 
    def letterbox(self, img, new_shape=(640, 480), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]  # 当前图像的宽高
 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
 
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
        dw /= 2  # padding 均分
        dh /= 2
       
        if shape[::-1] != new_unpad:  
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
 
        return img, (r, r), (dw, dh)
 
    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        self.detections = []
        self.detected_centers = []
        
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        
        self.debug_data['raw_output_data'] = output[0]  # [1, 5, 6300/8400]
        self.debug_data['raw_output_shape'] = output[0].shape  # [1, 5, 6300/8400]
        

        sample_conf_threshold = 0.422  
        high_conf_samples = []
        
        ratio = self.img_width / self.input_width, self.img_height / self.input_height
        
        all_candidates = []  #
        
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score > sample_conf_threshold:
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                high_conf_samples.append({
                    'index': i,
                    'cx': x,
                    'cy': y,
                    'w': w,
                    'h': h,
                    'conf': max_score
                })
            
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                
                all_candidates.append({
                    'index': int(i),
                    'original_cx': float(x),
                    'original_cy': float(y),
                    'original_w': float(w),
                    'original_h': float(h),
                    'score': float(max_score),
                    'class_id': int(class_id)
                })

                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  
                y -= self.dh
                x /= self.ratio[0]  
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                
                # 中心坐标
                center_x = int(left + width / 2)
                center_y = int(top + height / 2)
                
                detection_info = {
                    'bbox': [left, top, width, height],  # 边界框
                    'center': (center_x, center_y),      # 中心坐标
                    'score': max_score,                  # 置信度
                    'class_id': class_id,                # 类别ID
                    'class_name': self.classes[class_id] # 类别名称
                }
                
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
                self.detections.append(detection_info)
                self.detected_centers.append((center_x, center_y))
        
        self.debug_data['high_conf_samples'] = high_conf_samples
        
        self.debug_data['all_candidates'] = all_candidates
        
        # 保存NMS前的boxes和scores
        self.debug_data['before_nms'] = {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids
        }

        # 应用NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        
        filtered_detections = []
        filtered_centers = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                
                detection_info = self.detections[i]
                filtered_detections.append(detection_info)
                filtered_centers.append(detection_info['center'])
                
                self.draw_detections(input_image, box, score, class_id)
        
        self.detections = filtered_detections
        self.detected_centers = filtered_centers
        
        self.debug_data['final_detections'] = self.detections
        self.debug_data['final_centers'] = self.detected_centers
        

        print("\n" + "="*50)
        print("检测到的中心坐标:")
        print("="*50)
        if len(self.detected_centers) > 0:
            for i, center in enumerate(self.detected_centers):
                print(f"目标 {i+1}: ({center[0]}, {center[1]})")
        else:
            print("未检测到任何目标")
        
        return input_image
        
    def save_debug_files(self, output_dir="debug_data"):
        """保存调试文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        params_info = {
            'confidence_thres': self.confidence_thres,
            'iou_thres': self.iou_thres,
            'preprocess_info': self.debug_data.get('preprocess', {}),
            'raw_output_shape': self.debug_data.get('raw_output_shape', [])
        }
        
        with open(f"{output_dir}/params.json", 'w') as f:
            json.dump(params_info, f, indent=2)

        if 'raw_output_data' in self.debug_data:
            raw_output_data = self.debug_data['raw_output_data']  # [1, 5, 8400]
            raw_output_shape = raw_output_data.shape

            with open(f"{output_dir}/raw_output.bin", 'wb') as f:
 
                f.write(struct.pack('iii', 
                    raw_output_shape[0],  # batch_size = 1
                    raw_output_shape[1],  # channels = 5
                    raw_output_shape[2]   # num_predictions = 8400
                ))

                raw_output_data.flatten().tofile(f)
            
            print(f"  - raw_output.bin: 全部原始输出数据 ({raw_output_shape[0]}x{raw_output_shape[1]}x{raw_output_shape[2]})")
        

        sample_conf_threshold = 0.422  
        with open(f"{output_dir}/sample_output.txt", 'w') as f:
            if 'high_conf_samples' in self.debug_data:
                high_conf_samples = self.debug_data['high_conf_samples']
                
                f.write(f"High confidence samples (confidence > {sample_conf_threshold}):\n")
                f.write("=" * 80 + "\n")
                
                if high_conf_samples:
                    for sample in high_conf_samples:
                        f.write(f"Row {sample['index']}: ")
                        f.write(f"cx={sample['cx']:.4f}, ")
                        f.write(f"cy={sample['cy']:.4f}, ")
                        f.write(f"w={sample['w']:.4f}, ")
                        f.write(f"h={sample['h']:.4f}, ")
                        f.write(f"conf={sample['conf']:.4f}\n")
                    
                    f.write(f"\nTotal high confidence samples: {len(high_conf_samples)}\n")
                else:
                    f.write(f"No samples with confidence > {sample_conf_threshold}\n")
        
        with open(f"{output_dir}/candidates.json", 'w') as f:
            json.dump(self.debug_data.get('all_candidates', []), f, indent=2)
        
        with open(f"{output_dir}/final_results.txt", 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("Final Detection Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of detections: {len(self.detected_centers)}\n\n")
            
            for i, det in enumerate(self.detections):
                f.write(f"Target {i+1}:\n")
                f.write(f"  Class: {det['class_name']}\n")
                f.write(f"  Score: {det['score']:.4f}\n")
                f.write(f"  BBox: [x={det['bbox'][0]}, y={det['bbox'][1]}, "
                    f"w={det['bbox'][2]}, h={det['bbox'][3]}]\n")
                f.write(f"  Center: ({det['center'][0]}, {det['center'][1]})\n")
                f.write("-" * 40 + "\n")
            
            f.write("\nAll center coordinates:\n")
            for i, center in enumerate(self.detected_centers):
                f.write(f"  Target {i+1}: ({center[0]}, {center[1]})\n")
        
        print(f"\nDebug files saved to directory: {output_dir}/")
        print(f"  - params.json: 参数信息")
        print(f"  - raw_output.bin: 全部原始输出数据")
        print(f"  - sample_output.txt: 高置信度样本（> {sample_conf_threshold}）")
        print(f"  - candidates.json: 候选框信息")
        print(f"  - final_results.txt: 最终检测结果")
 
    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        
        center_x = int(x1 + w/2)
        center_y = int(y1 + h/2)
        
        cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
        coord_text = f"({center_x}, {center_y})"
        cv2.putText(img, coord_text, (center_x + 5, center_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def get_detection_summary(self):
        """获取检测结果摘要"""
        summary = f"\n检测结果摘要:\n"
        summary += f"检测到的目标数量: {len(self.detected_centers)}\n"
        
        if len(self.detected_centers) > 0:
            summary += "中心坐标列表:\n"
            for i, center in enumerate(self.detected_centers):
                summary += f"  目标 {i+1}: ({center[0]}, {center[1]})\n"
        
        return summary
 
    def main(self):
        session = ort.InferenceSession(
            self.onnx_model, 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
        )
        print("YOLO11 🚀 目标检测 ONNXRuntime")
        print("模型名称：", self.onnx_model)
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape 
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height} shape = {input_shape}")
 
        img_data = self.preprocess()
 
        outputs = session.run(None, {model_inputs[0].name: img_data})

        self.debug_data['raw_output_data'] = outputs[0]
        
        # outputs = [np.fromfile("output/Result_0/output0.raw", dtype=np.float32).reshape((1, 5, 6300))]
        output_image = self.postprocess(self.img, outputs)
        

        self.save_debug_files()
        
        return output_image, self.detected_centers
 
 
if __name__ == "__main__":
    model_path = r"model/best_sim.onnx"
    image_path = r"input_images/test_picture.jpg"
    conf_thres = 0.422
    iou_thres = 0.7
    
    detection = YOLO11(model_path, image_path, conf_thres, iou_thres)

    output_image, centers = detection.main()
    
    cv2.imwrite("debug_data/det_result_picture.jpg", output_image)
    print(f"\n检测图像已保存为: debug_data/det_result_picture.jpg")
