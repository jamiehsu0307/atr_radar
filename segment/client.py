import argparse
import logging
import socket
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil
import main_single_image

# ===== logging setting =====
def setup_logger():
    tz_taiwan = timezone(timedelta(hours=8))
    LOG_DIR = Path(__file__).resolve().parent / "logs" / 'log'
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"client_{datetime.now(tz=tz_taiwan).strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Client] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    return logging.getLogger("client")


# ===== backup path setting =====
def setup_backup_paths():
    main_single_image.BACKUP_LABELS_OUT_DIR = '/data/yolov9/segment/logs/txt_out/'
    main_single_image.BACKUP_VIS_DIR = '/data/yolov9/segment/logs/img_out/'
    backup_img_in_path = Path("/data/yolov9/segment/logs/img_in")
    backup_img_in_path.mkdir(parents=True, exist_ok=True)
    return backup_img_in_path


# ===== action list =====
def alive_action(*arg):
    return


# ===== backup original image =====
def backup_action(src_path, backup_img_in_path):
    src_path = Path(main_single_image.IMAGE_PATH)
    dst_path = backup_img_in_path / src_path.name
    try:
        shutil.copy2(src_path, dst_path)
    except FileNotFoundError:
        LOGGER.error(f"[Client] image not found for backup: {src_path}")
    except Exception as err:
        LOGGER.exception(f"[Client] failed to copy original image: {err}")


# ===== image process action =====
def img_action(conn, cmd):
    LOGGER.info("get IMG")
    # ===== path setting =====
    main_single_image.IMAGE_PATH = f"/data/yolov9/images/{'_'.join(cmd)}.png"
    main_single_image.LABELS_OUT_DIR = '/data/yolov9/segment/txt_out/'
    main_single_image.STEP1_VIS_DIR = '/data/yolov9/segment/img_out/'
    
    # main_single_image.IMAGE_PATH = f"/mnt/share/img/{"_".join(cmd)}.png"
    # main_single_image.LABELS_OUT_DIR = "/mnt/share/txt/"
    # main_single_image.STEP1_VIS_DIR = "/mnt/share/inf/"
    backup_action(main_single_image.IMAGE_PATH, backup_img_in_path)
    # ===== image process =====
    try:
        start_time = time.time()
        sea_inf = threading.Thread(
                    target=main_single_image.run, args=('sea',), daemon=True
                )
        ship_inf = threading.Thread(
                    target=main_single_image.run, args=('ship',), daemon=True
                )
        sea_inf.start()
        ship_inf.start()
        sea_inf.join()
        ship_inf.join()
        end_time = time.time()
        LOGGER.info(f"[Client] finish img process and use {end_time - start_time:.2f} seconds")
        conn.sendall(f"txt_{cmd[1]}".encode())
        LOGGER.info(f"[Client] send message: txt_{cmd[1]}")
    except Exception as e:
        LOGGER.exception(f"[Client] img process error: {e}")


# ===== command map =====
COMMAND_MAP = {"alive": alive_action, "img": img_action}


# ===== run action with exception safe =====
def run_action_safe(action, conn, cmd):
    try:
        action(conn, cmd)
    except Exception as err:
        LOGGER.exception(f"[Client] execute error: {err}")


# ===== listen msg and execute action =====
def listen_server(conn):
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                LOGGER.warning("[Client] ⚠️ no data received, host may have disconnected")
                return

            msg = data.decode().strip().lower()
            cmd = msg.split("_")
            # convert heart beat msg
            if msg == "hey! hey you! yeah you~":
                LOGGER.info("[Client] receive heart")
                continue

            LOGGER.info(f"[Client] receive message: {msg}")

            # execute action
            action = COMMAND_MAP.get(cmd[0])
            if action:
                threading.Thread(
                    target=run_action_safe, args=(action, conn, cmd), daemon=True
                ).start()
            else:
                LOGGER.warning(f"[Client] ⚠ unknown command: {cmd[0]}")

        except ConnectionResetError:
            LOGGER.exception("[Client] ❌ host disconnected")
            return
        except Exception as e:
            LOGGER.exception(f"[Client] listening error: {e}")
            return


# ===== connect to host and start listen =====
def start_client(host, port):
    RECONNECT_DELAY = 5  # reconnect delay in seconds
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
            LOGGER.exception(f"[Client] ❌ connect error: {err}")

        LOGGER.info(f"[Client] {RECONNECT_DELAY} seconds later will try to reconnect...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple TCP client for command execution."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="host IP， default IP 127.0.0.1"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="host port， default port 55688"
    )
    args = parser.parse_args()
    LOGGER = setup_logger()
    backup_img_in_path = setup_backup_paths()
    start_client(args.host, args.port)
