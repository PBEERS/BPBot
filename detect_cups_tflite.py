#!/usr/bin/env python3
import time
import cv2
import numpy as np

# Prefer tflite_runtime if installed, else fallback to full TF
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Config ---
MODEL_PATH = str("/home/pi/models/coco-ssd-mobilenet-v2/detect.tflite")
CONF_THRESHOLD = 0.5
TARGET_CLASS_NAME = "cup"
# COCO class index for cup is 47 in the standard 1-90 mapping.
# Many TFLite models use 1-based class IDs; we'll handle both.
CUP_IDS = {47}

def load_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

def get_io_details(interp):
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    in_h, in_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    return input_details, output_details, in_w, in_h

def preprocess(frame, in_w, in_h):
    # SSD expects 300x300 or 320x320. Our model is 300 or 320; we’ll just resize to model size.
    img = cv2.resize(frame, (in_w, in_h))
    # Quantized model usually takes uint8 [0,255]
    return np.expand_dims(img, axis=0).astype(np.uint8)

def parse_detections(frame, outputs, conf_thr):
    """
    TFLite SSD typically returns:
      boxes: [1, N, 4] in ymin, xmin, ymax, xmax (normalized)
      classes: [1, N]
      scores: [1, N]
      num: [1]
    """
    h, w = frame.shape[:2]
    boxes, classes, scores, num = outputs
    results = []
    count = int(num[0])
    for i in range(count):
        score = float(scores[0][i])
        if score < conf_thr:
            continue
        cls = int(classes[0][i])
        # Some models use 0-based, some 1-based class IDs; normalize to 1-based for COCO mapping
        cls_1based = cls if cls >= 1 else cls + 1
        if cls in CUP_IDS or cls_1based in CUP_IDS:
            ymin, xmin, ymax, xmax = boxes[0][i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            results.append((x1, y1, x2, y2, score))
    return results

def main():
    print("Loading model…")
    interp = load_interpreter(MODEL_PATH)
    input_details, output_details, in_w, in_h = get_io_details(interp)

    # Picamera2 + OpenCV capture (works on Bookworm)
    print("Starting camera… (press q to quit)")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Locate output tensor indices by name/order (model dependent)
    # Common order for SSD MobileNet tflite:
    #   boxes:   output_details[0]
    #   classes: output_details[1]
    #   scores:  output_details[2]
    #   num:     output_details[3]
    # We’ll just map by size to be safe.
    def pick(outputs):
        idx_boxes = idx_classes = idx_scores = idx_num = None
        for i, od in enumerate(outputs):
            shp = od['shape']
            if len(shp) == 3 and shp[-1] == 4:
                idx_boxes = i
            elif len(shp) == 2:
                # classes or scores: shape [1, N]
                # use dtype: classes=int, scores=float
                if np.issubdtype(od['dtype'], np.integer):
                    idx_classes = i
                else:
                    idx_scores = i
            elif len(shp) == 1:
                idx_num = i
        return idx_boxes, idx_classes, idx_scores, idx_num

    idx_boxes, idx_classes, idx_scores, idx_num = pick(output_details)

    fps_t0 = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            inp = preprocess(frame, in_w, in_h)
            # Set input
            interp.set_tensor(input_details[0]['index'], inp)
            interp.invoke()

            boxes   = interp.get_tensor(output_details[idx_boxes]['index'])
            classes = interp.get_tensor(output_details[idx_classes]['index'])
            scores  = interp.get_tensor(output_details[idx_scores]['index'])
            num     = interp.get_tensor(output_details[idx_num]['index'])

            cups = parse_detections(frame, (boxes, classes, scores, num), CONF_THRESHOLD)

            # Draw results
            for (x1, y1, x2, y2, sc) in cups:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"cup {sc:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            # Show FPS
            frames += 1
            if frames % 20 == 0:
                now = time.time()
                fps = 20.0 / (now - fps_t0)
                fps_t0 = now
                cv2.putText(frame, f"{fps:.1f} FPS", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Cup detector (TFLite)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()