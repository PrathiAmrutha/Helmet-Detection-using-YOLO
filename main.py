from ultralytics import YOLO
import cv2
import os

def draw_label(img, text, x, y, color):
    """
    Draws a filled label with white text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2

    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Keep label inside frame
    y = max(h + 8, y)

    # Filled rectangle
    cv2.rectangle(
        img,
        (x, y - h - 8),
        (x + w + 10, y + baseline - 2),
        color,
        -1
    )

    # White text
    cv2.putText(
        img,
        text,
        (x + 5, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

# -----------------------------
# Load YOLO Model
# -----------------------------
vehicle_model = YOLO("yolov8n.pt")          # Person + Motorcycle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
helmet_path = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train",
    "weights",
    "best.pt"
)
helmet_model = YOLO(helmet_path)

print("Helmet Classes:", helmet_model.names)

# -----------------------------
# Input Video
# -----------------------------
video_path = "traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

# -----------------------------
# Video Information
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Video Width :", width)
print("Video Height:", height)
print("FPS         :", fps)

# -----------------------------
# Read Frames
# -----------------------------q
PERSON = 0
MOTORCYCLE = 3
cv2.namedWindow("Traffic Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Video", 900,700)
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = vehicle_model(frame, verbose=False)

    persons = []
    motorcycles = []

    # -------------------------
    # Store detections
    # -------------------------
    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON:

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf
                })

            elif cls == MOTORCYCLE:

                motorcycles.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf
                })

    # -------------------------
    # Draw motorcycles
    # -------------------------
    # --------------------------------------------------
    # Associate rider with each motorcycle (Improved)
    # --------------------------------------------------
    for bike in motorcycles:

        bx1, by1, bx2, by2 = bike["box"]

        # Draw motorcycle
        cv2.rectangle(frame,
                    (bx1, by1),
                    (bx2, by2),
                    (255, 0, 0),
                    2)

        draw_label(frame, "Motorcycle", bx1, by1, (255, 0, 0))

        # --------------------------------------------------
        # Expand motorcycle region
        # --------------------------------------------------
        margin_x = int((bx2 - bx1) * 0.30)
        margin_top = int((by2 - by1) * 1.20)
        margin_bottom = int((by2 - by1) * 0.15)

        ex1 = max(0, bx1 - margin_x)
        ey1 = max(0, by1 - margin_top)
        ex2 = min(frame.shape[1], bx2 + margin_x)
        ey2 = min(frame.shape[0], by2 + margin_bottom)

        best_person = None
        best_score = -1

        bike_center = (bx1 + bx2) / 2

        for person in persons:

            px1, py1, px2, py2 = person["box"]

            # ----------------------------
            # Intersection area
            # ----------------------------
            ix1 = max(px1, ex1)
            iy1 = max(py1, ey1)
            ix2 = min(px2, ex2)
            iy2 = min(py2, ey2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (ix2 - ix1) * (iy2 - iy1)
            person_area = (px2 - px1) * (py2 - py1)

            overlap_ratio = intersection / person_area

            # ----------------------------
            # Horizontal alignment
            # ----------------------------
            person_center = (px1 + px2) / 2

            distance = abs(person_center - bike_center)

            alignment = 1 - (distance / frame.shape[1])

            alignment = max(0, alignment)

            # ----------------------------
            # Combined score
            # ----------------------------
            score = 0.7 * overlap_ratio + 0.3 * alignment

            if score > best_score:

                best_score = score
                best_person = person

        # --------------------------------------------------
        # Draw associated rider
        # --------------------------------------------------
        if best_person is not None:

            px1, py1, px2, py2 = best_person["box"]

            cv2.rectangle(frame,
                        (px1, py1),
                        (px2, py2),
                        (0, 255, 0),
                        2)

            draw_label(frame, f"Rider {best_score:.2f}", px1, py1, (0, 180, 0))
            # ------------------------------------
            # Crop rider for helmet detection
            # ------------------------------------

            crop = frame[py1:py2, px1:px2]

            if crop.size != 0:

                helmet_results = helmet_model(
                    crop,
                    verbose=False
                )

                helmet_found = False
                nohelmet_found = False

                for h_result in helmet_results:

                    for h_box in h_result.boxes:

                        h_cls = int(h_box.cls[0])
                        h_conf = float(h_box.conf[0])

                        if h_conf < 0.50:
                            continue

                        class_name = helmet_model.names[h_cls].lower()

                        if class_name == "helmet":
                            helmet_found = True

                        elif class_name == "no_helmet":
                            nohelmet_found = True

                # ----------------------------
                # Final decision
                # ----------------------------
                if nohelmet_found:

                    status = "HELMET"
                    color = (0,255,0)

                elif helmet_found:

                    status = "NO HELMET"
                    color = (0,0,255)

                else:

                    status = "UNKNOWN"
                    color = (0,255,255)

                draw_label(frame, status, px1, py2 + 30, color)

                        # ---------- Optional ----------
                        # Uncomment to visualize search area
                        #
                        # cv2.rectangle(frame,
                        #               (ex1, ey1),
                        #               (ex2, ey2),
                        #               (0,255,255),
                        #               1)
    cv2.imshow("Traffic Video", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()