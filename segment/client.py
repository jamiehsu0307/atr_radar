import argparse
import logging
import socket
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil
import main_single_image

tz_taiwan = timezone(timedelta(hours=8))
LOG_DIR = Path(__file__).resolve().parent / "logs" / 'log'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"client_{datetime.now(tz=tz_taiwan).strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Client] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
LOGGER = logging.getLogger("client")

DEFAULT_HOST = "127.0.0.1"  # 改成 server IP
DEFAULT_PORT = 55688
RECONNECT_DELAY = 5  # 重新連線延遲秒數


# ===== 動作函式區 =====
def alive_action(*arg):
    return


def img_action(conn, cmd):
    print("[Client] get IMG")
    LOGGER.info("get IMG")
    main_single_image.IMAGE_PATH = '/data/yolov9/images/' + '_'.join(cmd) + '.png'
    src_path = Path(main_single_image.IMAGE_PATH)
    dst_dir = Path("/data/yolov9/segment/logs/img_in")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    try:
        shutil.copy2(src_path, dst_path)
    except FileNotFoundError:
        LOGGER.error(f"[Client] image not found for backup: {src_path}")
    except Exception as err:
        LOGGER.exception(f"[Client] failed to copy original image: {err}")
    main_single_image.LABELS_DIR = '/data/yolov9/segment/txt_out/'
    main_single_image.STEP1_VIS_DIR = '/data/yolov9/segment/img_out/'
    main_single_image.BACKUP_LABELS_OUT_DIR = '/data/yolov9/segment/logs/txt_out/'
    main_single_image.BACKUP_VIS_DIR = '/data/yolov9/segment/logs/img_out/'
    # main_single_image.IMAGE_PATH = "/mnt/share/img/" + "_".join(cmd) + ".png"
    # main_single_image.LABELS_OUT_DIR = "/mnt/share/txt/"
    # main_single_image.STEP1_VIS_DIR = "/mnt/share/inf/"
    # ===== image process =====
    try:
        start_time = time.time()
        # main_single_image.run("sea")
        main_single_image.run("ship")
        end_time = time.time()
        LOGGER.info(f"[Client] use {end_time - start_time:.2f} seconds")
        conn.sendall(f"txt_{cmd[1]}".encode())
    except Exception as e:
        LOGGER.exception(f"[Client] img process error: {e}")

# ===== 指令對應表 =====
COMMAND_MAP = {"alive": alive_action, "img": img_action}


def run_action_safe(action, conn, cmd):
    try:
        action(conn, cmd)
    except Exception as err:
        LOGGER.exception(f"[Client] 執行指令時發生錯誤: {err}")


# ===== 接收主迴圈 =====
def listen_server(conn):
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                LOGGER.warning("[Client] ⚠️ 連線中斷")
                return

            msg = data.decode().strip().lower()
            cmd = msg.split("_")
            # 處理 ALIVE 心跳
            if msg == "hey! hey you! yeah you~":
                LOGGER.info("[Client] receive heart")
                continue

            LOGGER.info(f"[Client] receive message: {msg}")

            # 依指令執行對應函式
            action = COMMAND_MAP.get(cmd[0])
            if action:
                threading.Thread(
                    target=run_action_safe, args=(action, conn, cmd), daemon=True
                ).start()
            else:
                LOGGER.warning(f"[Client] ⚠ unknown command: {cmd[0]}")

        except ConnectionResetError:
            LOGGER.exception("[Client] ❌ 伺服端已斷線")
            return
        except Exception as e:
            LOGGER.exception(f"[Client] 錯誤: {e}")
            return


def start_client(host, port):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
                conn.connect((host, port))
                LOGGER.info(f"[Client] connection {host}:{port}")
                listen_server(conn)
        except KeyboardInterrupt:
            LOGGER.info("\n[Client] user interrupt, exiting...")
            break
        except socket.error as err:
            LOGGER.exception(f"[Client] ❌ 連線錯誤: {err}")

        LOGGER.info(f"[Client] {RECONNECT_DELAY} seconds later will try to reconnect...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple TCP client for command execution."
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="伺服器 IP，預設為 127.0.0.1"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="伺服器埠號，預設為 55688"
    )
    args = parser.parse_args()

    start_client(args.host, args.port)
