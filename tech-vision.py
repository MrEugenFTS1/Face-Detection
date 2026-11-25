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

import ctypes
import os
import sys
import time
import urllib.request
from datetime import datetime

# Убираем служебные предупреждения OpenCV (должно быть до import cv2)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

try:
    cv2.setLogLevel(0)
except (AttributeError, cv2.error):
    pass

try:  # звуковой сигнал на Windows
    import winsound
except ImportError:  # другие ОС — используем системный "звонок" терминала
    winsound = None


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

# Тревога: если лиц в кадре больше этого числа — предупреждение и звук
MAX_ALLOWED_FACES = 1
ALERT_COOLDOWN = 2.0        # минимальный интервал между сигналами, сек
MINIMIZE_ON_ALERT = True    # сворачивать все окна (Win+D) при тревоге

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


def draw_hud(
    frame,
    fps: float,
    face_count: int,
    max_faces: int,
    mirrored: bool,
    engine: str,
) -> None:
    """Отображает служебную информацию в углу кадра."""
    h, _ = frame.shape[:2]
    lines = [
        f"Engine: {engine}",
        f"FPS: {fps:5.1f}",
        f"Faces: {face_count} / max {max_faces}",
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


def ask_max_faces(default: int = 1):
    """Окно выбора допустимого числа лиц перед запуском.

    Возвращает выбранное число или None, если пользователь отменил запуск.
    Если tkinter недоступен, возвращает значение по умолчанию.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter недоступен — используется значение по умолчанию.")
        return default

    result = {"value": None}

    root = tk.Tk()
    root.title("Tech Vision — настройки")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Сколько лиц допускается в кадре?",
        font=("Segoe UI", 11, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 6))

    ttk.Label(
        frame,
        text=(
            "Если лиц станет больше, сработает тревога:\n"
            "предупреждение на экране, звуковой сигнал\n"
            "и сворачивание всех окон (Win+D)."
        ),
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(0, 10))

    var = tk.IntVar(value=max(1, int(default)))
    spin = ttk.Spinbox(
        frame, from_=1, to=10, textvariable=var, width=6,
        font=("Segoe UI", 11),
    )
    spin.grid(row=2, column=0, sticky="w", pady=(0, 12))

    def on_start(*_args):
        try:
            value = int(var.get())
        except (tk.TclError, ValueError):
            value = max(1, int(default))
        result["value"] = max(1, value)
        root.destroy()

    def on_cancel(*_args):
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=3, column=0, sticky="ew")
    ttk.Button(buttons, text="Запустить", command=on_start).pack(side="left")
    ttk.Button(buttons, text="Отмена", command=on_cancel).pack(
        side="left", padx=(8, 0)
    )

    root.bind("<Return>", on_start)
    root.bind("<Escape>", on_cancel)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    spin.focus_set()

    root.mainloop()
    return result["value"]


def play_alert_sound() -> None:
    """Проигрывает короткий звуковой сигнал тревоги (не блокирует цикл)."""
    if winsound is not None:
        try:
            winsound.PlaySound(
                "SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC
            )
            return
        except RuntimeError:
            try:
                winsound.Beep(1000, 300)
                return
            except RuntimeError:
                pass
    # Запасной вариант — системный "звонок" терминала
    sys.stdout.write("\a")
    sys.stdout.flush()


def minimize_all_windows() -> bool:
    """Сворачивает все окна сочетанием Win+D (только Windows)."""
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        vk_lwin, vk_d, keyeventf_keyup = 0x5B, 0x44, 0x0002
        # Эмулируем нажатие Win+D
        user32.keybd_event(vk_lwin, 0, 0, 0)
        user32.keybd_event(vk_d, 0, 0, 0)
        user32.keybd_event(vk_d, 0, keyeventf_keyup, 0)
        user32.keybd_event(vk_lwin, 0, keyeventf_keyup, 0)
        return True
    except Exception:  # noqa: BLE001 - сворачивание не должно ломать цикл
        return False


def find_window_handle(title: str):
    """Возвращает HWND окна по заголовку (только Windows)."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.FindWindowW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        hwnd = user32.FindWindowW(None, title)
        return hwnd or None
    except Exception:  # noqa: BLE001
        return None


def is_window_minimized(hwnd) -> bool:
    """Проверяет, свёрнуто ли окно с данным HWND (только Windows)."""
    if not hwnd or os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.IsIconic.restype = ctypes.c_bool
        user32.IsIconic.argtypes = [ctypes.c_void_p]
        return bool(user32.IsIconic(hwnd))
    except Exception:  # noqa: BLE001
        return False


def draw_alert_banner(frame, face_count: int, max_faces: int) -> None:
    """Рисует мигающее красное предупреждение о лишних лицах в кадре."""
    if int(time.time() * 3) % 2 == 0:
        return  # мигание: показываем через такт

    h, w = frame.shape[:2]
    text = f"WARNING: MORE THAN {max_faces} FACE(S)! ({face_count})"
    scale, thickness = 0.9, 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max((w - text_w) // 2, 0)
    y = 20
    cv2.rectangle(
        frame,
        (x - 10, y),
        (x + text_w + 10, y + text_h + baseline + 10),
        (0, 0, 255),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x, y + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------
def main() -> int:
    # Перед запуском спрашиваем допустимое число лиц
    max_faces = ask_max_faces(MAX_ALLOWED_FACES)
    if max_faces is None:
        print("Запуск отменён пользователем.")
        return 0
    print(f"Допустимое число лиц в кадре: {max_faces}")

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
    last_alert = 0.0
    # Win+D — переключатель: повторный вызов развернул бы окна обратно.
    # Флаг сбрасывается, когда пользователь сам разворачивает окно программы.
    windows_minimized = False
    minimize_time = 0.0
    minimize_confirmed = False   # убедились, что окно действительно свёрнуто
    cv_hwnd = None               # HWND окна OpenCV (ищем после первого imshow)

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

            # Если окна были свёрнуты, проверяем, не развернул ли их пользователь.
            # Пока окно свёрнуто, повторный Win+D не отправляем.
            if windows_minimized:
                if cv_hwnd is None:
                    cv_hwnd = find_window_handle(WINDOW_NAME)
                if cv_hwnd is not None:
                    if not minimize_confirmed:
                        # Ждём, пока Win+D реально свернёт окно
                        if is_window_minimized(cv_hwnd):
                            minimize_confirmed = True
                        elif now - minimize_time > 1.5:
                            # Win+D не сработал — разрешаем попытку снова
                            windows_minimized = False
                    elif not is_window_minimized(cv_hwnd):
                        # Пользователь вручную развернул окно программы
                        windows_minimized = False
                        minimize_confirmed = False
                        print(
                            "Окно восстановлено — автосворачивание снова активно.",
                            file=sys.stderr,
                        )

            # Тревога: в кадре больше разрешённого числа лиц
            if len(faces) > max_faces:
                draw_alert_banner(frame, len(faces), max_faces)

                # Сворачиваем только если окна ещё не свёрнуты: повторный
                # Win+D развернул бы их обратно. Проверка вынесена из блока
                # со звуком, чтобы после ручного восстановления окна
                # сворачивание срабатывало сразу, не дожидаясь кулдауна.
                if (
                    MINIMIZE_ON_ALERT
                    and not windows_minimized
                    and minimize_all_windows()
                ):
                    windows_minimized = True
                    minimize_confirmed = False
                    minimize_time = now
                    print("Все окна свёрнуты (Win+D).", file=sys.stderr)

                if now - last_alert >= ALERT_COOLDOWN:
                    last_alert = now
                    play_alert_sound()
                    print(
                        f"ВНИМАНИЕ: в кадре {len(faces)} лиц(а), "
                        f"а допустимо {max_faces}!",
                        file=sys.stderr,
                    )

            draw_hud(frame, fps, len(faces), max_faces, mirrored, engine)

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
