import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


LABELS = ["neutral", "happy", "sad", "angry"]
IMG_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PROB_EMA = 0.65


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def rotate_image(image: np.ndarray, angle_deg: float, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, rot_mat


def transform_point(x: float, y: float, rot_mat: np.ndarray):
    px = rot_mat[0, 0] * x + rot_mat[0, 1] * y + rot_mat[0, 2]
    py = rot_mat[1, 0] * x + rot_mat[1, 1] * y + rot_mat[1, 2]
    return px, py


def make_square_box(x1, y1, x2, y2, frame_w, frame_h, scale=1.45):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(frame_w, nx2)
    ny2 = min(frame_h, ny2)
    return nx1, ny1, nx2, ny2


def preprocess_roi_bgr(roi_bgr: np.ndarray):
    roi_resized = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    x = roi_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.expand_dims(x, axis=0)
    return x, roi_resized


def get_mesh_aligned_roi(frame_bgr: np.ndarray, face_landmarks, scale=1.45):
    fh, fw = frame_bgr.shape[:2]

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    pts = []
    for lm in face_landmarks.landmark:
        x = lm.x * fw
        y = lm.y * fh
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    lx, ly = pts[LEFT_EYE_OUTER]
    rx, ry = pts[RIGHT_EYE_OUTER]

    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    rotated, rot_mat = rotate_image(frame_bgr, angle, eye_center)
    rot_pts = np.array([transform_point(x, y, rot_mat) for x, y in pts], dtype=np.float32)

    min_x = float(np.min(rot_pts[:, 0]))
    max_x = float(np.max(rot_pts[:, 0]))
    min_y = float(np.min(rot_pts[:, 1]))
    max_y = float(np.max(rot_pts[:, 1]))

    x1, y1, x2, y2 = make_square_box(min_x, min_y, max_x, max_y, fw, fh, scale=scale)
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None

    roi_bgr = rotated[y1:y2, x1:x2]
    return roi_bgr, (x1, y1, x2, y2), rotated, rot_pts


def draw_softmax_panel(canvas, probs, labels, x=10, y=10):
    if probs is None:
        cv2.putText(canvas, "Emotion: N/A", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        return

    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_conf = float(probs[top_idx])

    cv2.rectangle(canvas, (x, y), (x + 340, y + 155), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + 340, y + 155), (200, 200, 200), 1)

    cv2.putText(canvas, f"Emotion: {top_label}", (x + 10, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Conf: {top_conf:.3f}", (x + 200, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    bar_y = y + 50
    for i, (label, p) in enumerate(zip(labels, probs)):
        yy = bar_y + i * 24
        cv2.putText(canvas, f"{label:>7}", (x + 10, yy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + 95, yy), (x + 295, yy + 14), (80, 80, 80), -1)
        cv2.rectangle(canvas, (x + 95, yy), (x + 95 + int(200 * float(p)), yy + 14), (0, 200, 0), -1)
        cv2.putText(canvas, f"{float(p):.3f}", (x + 300, yy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def build_region_index_map(mp_face_mesh):
    region_map = {}

    def add_conn_set(name, conn_set):
        for a, b in conn_set:
            region_map.setdefault(a, name)
            region_map.setdefault(b, name)

    add_conn_set("face_oval", mp_face_mesh.FACEMESH_FACE_OVAL)
    add_conn_set("left_eyebrow", mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    add_conn_set("right_eyebrow", mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
    add_conn_set("left_eye", mp_face_mesh.FACEMESH_LEFT_EYE)
    add_conn_set("right_eye", mp_face_mesh.FACEMESH_RIGHT_EYE)
    add_conn_set("lips_outer", mp_face_mesh.FACEMESH_LIPS)
    add_conn_set("nose", mp_face_mesh.FACEMESH_NOSE)

    if hasattr(mp_face_mesh, "FACEMESH_LEFT_IRIS"):
        add_conn_set("left_iris", mp_face_mesh.FACEMESH_LEFT_IRIS)
    if hasattr(mp_face_mesh, "FACEMESH_RIGHT_IRIS"):
        add_conn_set("right_iris", mp_face_mesh.FACEMESH_RIGHT_IRIS)

    return region_map


REGION_COLORS = {
    "face_oval": (255, 255, 255),
    "left_eyebrow": (0, 165, 255),
    "right_eyebrow": (0, 255, 255),
    "left_eye": (255, 0, 0),
    "right_eye": (0, 255, 0),
    "lips_outer": (0, 0, 255),
    "nose": (255, 0, 255),
    "left_iris": (255, 255, 0),
    "right_iris": (255, 255, 0),
    "default": (180, 180, 180),
}

INNER_LIP_INDICES = {
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415
}


def get_region_name(idx, region_map):
    if idx in INNER_LIP_INDICES:
        return "lips_inner"
    return region_map.get(idx, "default")


def get_region_color(region_name):
    if region_name == "lips_inner":
        return (203, 192, 255)
    return REGION_COLORS.get(region_name, REGION_COLORS["default"])


def draw_region_connections(image, face_landmarks, mp_face_mesh):
    h, w = image.shape[:2]

    def draw_conn_set(conn_set, color, thickness=1):
        for start_idx, end_idx in conn_set:
            x1 = int(face_landmarks.landmark[start_idx].x * w)
            y1 = int(face_landmarks.landmark[start_idx].y * h)
            x2 = int(face_landmarks.landmark[end_idx].x * w)
            y2 = int(face_landmarks.landmark[end_idx].y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    draw_conn_set(mp_face_mesh.FACEMESH_FACE_OVAL, REGION_COLORS["face_oval"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_LEFT_EYEBROW, REGION_COLORS["left_eyebrow"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_RIGHT_EYEBROW, REGION_COLORS["right_eyebrow"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_LEFT_EYE, REGION_COLORS["left_eye"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_RIGHT_EYE, REGION_COLORS["right_eye"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_LIPS, REGION_COLORS["lips_outer"], 1)
    draw_conn_set(mp_face_mesh.FACEMESH_NOSE, REGION_COLORS["nose"], 1)

    if hasattr(mp_face_mesh, "FACEMESH_LEFT_IRIS"):
        draw_conn_set(mp_face_mesh.FACEMESH_LEFT_IRIS, REGION_COLORS["left_iris"], 1)
    if hasattr(mp_face_mesh, "FACEMESH_RIGHT_IRIS"):
        draw_conn_set(mp_face_mesh.FACEMESH_RIGHT_IRIS, REGION_COLORS["right_iris"], 1)


def draw_landmark_indices_colored(image, face_landmarks, region_map):
    h, w = image.shape[:2]
    for idx, lm in enumerate(face_landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        region_name = get_region_name(idx, region_map)
        color = get_region_color(region_name)

        cv2.circle(image, (x, y), 1, color, -1)
        cv2.putText(
            image,
            str(idx),
            (x + 1, y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_region_legend(image, x=10, y=175):
    legend_items = [
        ("face_oval", "Face Oval"),
        ("left_eyebrow", "Left Eyebrow"),
        ("right_eyebrow", "Right Eyebrow"),
        ("left_eye", "Left Eye"),
        ("right_eye", "Right Eye"),
        ("nose", "Nose"),
        ("lips_outer", "Lips Outer"),
        ("lips_inner", "Lips Inner"),
        ("left_iris", "Iris"),
    ]

    box_w = 220
    box_h = 220
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (200, 200, 200), 1)

    cv2.putText(image, "Landmark Regions", (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    yy = y + 45
    for region_key, label in legend_items:
        color = get_region_color(region_key)
        cv2.rectangle(image, (x + 10, yy - 10), (x + 25, yy + 5), color, -1)
        cv2.putText(image, label, (x + 35, yy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        yy += 20


def run_visualizer(
    tflite_path: str,
    cam_index: int = 0,
    cam_w: int = 1280,
    cam_h: int = 720,
    min_det_conf: float = 0.5,
    min_track_conf: float = 0.5,
):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    mp_face_mesh = mp.solutions.face_mesh
    region_map = build_region_index_map(mp_face_mesh)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    prob_ema = None
    show_indices = True
    show_connections = True
    show_roi = False
    show_legend = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            display = frame.copy()
            roi_disp = None
            probs_out = None
            draw_box = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                if show_connections:
                    draw_region_connections(display, face_landmarks, mp_face_mesh)

                if show_indices:
                    draw_landmark_indices_colored(display, face_landmarks, region_map)

                aligned_roi, draw_box, rotated_disp, rot_pts = get_mesh_aligned_roi(
                    frame, face_landmarks, scale=1.45
                )

                if aligned_roi is not None and aligned_roi.size > 0:
                    x_nhwc, roi_disp = preprocess_roi_bgr(aligned_roi)
                    x_in = x_nhwc.astype(input_details["dtype"])
                    interpreter.set_tensor(input_details["index"], x_in)
                    interpreter.invoke()
                    out = interpreter.get_tensor(output_details["index"])[0]
                    probs = softmax(out)

                    if prob_ema is None:
                        prob_ema = probs.copy()
                    else:
                        prob_ema = PROB_EMA * prob_ema + (1.0 - PROB_EMA) * probs

                    probs_out = prob_ema.copy()

            if draw_box is not None:
                x1, y1, x2, y2 = draw_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            draw_softmax_panel(display, probs_out, LABELS, x=10, y=10)

            if show_legend:
                draw_region_legend(display, x=10, y=175)

            cv2.putText(
                display,
                "keys: q quit | i index | c connections | r roi | l legend",
                (20, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("FER Visualizer", display)

            if show_roi and roi_disp is not None:
                cv2.imshow("Aligned ROI", roi_disp)
            else:
                try:
                    if cv2.getWindowProperty("Aligned ROI", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("Aligned ROI")
                except cv2.error:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                show_indices = not show_indices
            elif key == ord("c"):
                show_connections = not show_connections
            elif key == ord("r"):
                show_roi = not show_roi
            elif key == ord("l"):
                show_legend = not show_legend

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--min_det_conf", type=float, default=0.5)
    parser.add_argument("--min_track_conf", type=float, default=0.5)
    args = parser.parse_args()

    run_visualizer(
        tflite_path=args.tflite,
        cam_index=args.cam_index,
        cam_w=args.cam_w,
        cam_h=args.cam_h,
        min_det_conf=args.min_det_conf,
        min_track_conf=args.min_track_conf,
    )