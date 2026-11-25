"""
Распознавание (детекция) лиц человека с веб-камеры ноутбука.

Использует OpenCV и нейросетевой детектор лиц YuNet (FaceDetectorYN) —
он точнее и быстрее классических каскадов Хаара и поддерживается
в OpenCV 4.x и 5.x.

Установка зависимостей:
    pip install opencv-python

Модель YuNet (файл models/face_detection_yunet_2023mar.onnx) должна лежать
рядом со скриптом. Если её нет, программа подскажет команду для скачивания.

Запуск:
    python tech-vision.py

Управление:
    q / Esc  — выход
    s        — сохранить снимок в папку snapshots/
    m        — переключить зеркальный режим
"""

import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # 0 — встроенная камера ноутбука, 1 — внешняя и т.п.
MIRROR = True               # зеркальное отображение (как в зеркале)
WINDOW_NAME = "Face Detection - Tech Vision"
SNAPSHOT_DIR = "snapshots"
MODELS_DIR = "models"

# Нейросетевая модель YuNet
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/" + YUNET_FILENAME
)

# Порог уверенности детекции (0..1) и параметры NMS
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Резервный вариант для OpenCV 4.x — каскад Хаара
HAAR_FILENAME = "haarcascade_frontalface_default.xml"
HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "data/haarcascades/" + HAAR_FILENAME
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Загрузка детектора
# ---------------------------------------------------------------------------
def _model_path(filename: str) -> str:
    return os.path.join(PROJECT_DIR, MODELS_DIR, filename)


def _download(url: str, dest: str) -> None:
    """Скачивает файл модели, если его нет локально."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Скачивание модели: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Модель сохранена: {dest}")


def _ensure_model(filename: str, url: str) -> str:
    path = _model_path(filename)
    if not os.path.exists(path):
        try:
            _download(url, path)
        except Exception as exc:  # noqa: BLE001 - показываем пользователю причину
            raise FileNotFoundError(
                f"Не удалось скачать модель {filename}: {exc}\n"
                f"Скачайте её вручную в папку {MODELS_DIR}/:\n    {url}"
            ) from exc
    return path


class YuNetDetector:
    """Обёртка над cv2.FaceDetectorYN (OpenCV >= 4.5.4)."""

    def __init__(self, model_path: str) -> None:
        self._size = None
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._detector.setInputSize((w, h))
            self._size = (w, h)

        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        # Каждая строка: x, y, w, h, 5 пар координат ключевых точек, score
        return faces


class HaarDetector:
    """Резервный детектор на каскаде Хаара для OpenCV 4.x."""

    def __init__(self, cascade_path: str) -> None:
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Не удалось загрузить каскад: {cascade_path}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        # Приводим к формату YuNet: x, y, w, h, (без точек), score
        return [(*map(int, rect), 1.0) for rect in rects]


def build_detector():
    """Создаёт лучший доступный детектор лиц."""
    if hasattr(cv2, "FaceDetectorYN"):
        model_path = _ensure_model(YUNET_FILENAME, YUNET_URL)
        return YuNetDetector(model_path), "YuNet (DNN)"

    if hasattr(cv2, "CascadeClassifier"):
        cascade_path = _ensure_model(HAAR_FILENAME, HAAR_URL)
        return HaarDetector(cascade_path), "Haar cascade"

    raise RuntimeError(
        "Установленная версия OpenCV не поддерживает детекцию лиц.\n"
        "Обновите библиотеку: pip install --upgrade opencv-python"
    )


# ---------------------------------------------------------------------------
# Камера и отрисовка
# ---------------------------------------------------------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    """Открывает камеру с проверкой доступности."""
    # CAP_DSHOW ускоряет запуск камеры на Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть камеру с индексом {index}.\n"
            "Проверьте, что камера не занята другим приложением "
            "и разрешён доступ к ней."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def draw_face(frame, face) -> None:
    """Рисует рамку, ключевые точки и уверенность вокруг лица."""
    x, y, w, h = map(int, face[:4])
    score = float(face[-1])

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Ключевые точки (глаза, нос, уголки рта) — если детектор их вернул
    landmarks = face[4:14].reshape(-1, 2) if len(face) >= 14 else []
    for px, py in landmarks:
        if px > 0 and py > 0:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)

    label = f"Face {score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    cv2.rectangle(
        frame,
        (x, max(y - text_h - baseline - 6, 0)),
        (x + text_w + 8, max(y, text_h + baseline + 6)),
        (0, 255, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x + 4, max(y - baseline - 3, text_h + 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, fps: float, face_count: int, mirrored: bool, engine: str) -> None:
    """Отображает служебную информацию в углу кадра."""
    h, _ = frame.shape[:2]
    lines = [
        f"Engine: {engine}",
        f"FPS: {fps:5.1f}",
        f"Faces: {face_count}",
        f"Mirror: {'on' if mirrored else 'off'}",
        "q/Esc - exit, s - snapshot, m - mirror",
    ]
    for i, line in enumerate(lines):
        y = h - 10 - (len(lines) - 1 - i) * 22
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def save_snapshot(frame) -> str:
    """Сохраняет текущий кадр в папку snapshots."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    filename = datetime.now().strftime("face_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        detector, engine = build_detector()
        cap = open_camera(CAMERA_INDEX)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    mirrored = MIRROR
    prev_time = time.perf_counter()
    fps = 0.0
    last_saved = ""

    print(f"Детектор: {engine}")
    print("Камера запущена. Нажмите 'q' или Esc для выхода.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Не удалось получить кадр с камеры. Завершение.", file=sys.stderr)
                break

            if mirrored:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)
            for face in faces:
                draw_face(frame, face)

            # Расчёт FPS со сглаживанием
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            prev_time = now

            draw_hud(frame, fps, len(faces), mirrored, engine)

            if last_saved:
                cv2.putText(
                    frame,
                    f"Saved: {os.path.basename(last_saved)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' или Esc
                break
            if key == ord("s"):
                last_saved = save_snapshot(frame)
                print(f"Снимок сохранён: {last_saved}")
            elif key == ord("m"):
                mirrored = not mirrored
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
