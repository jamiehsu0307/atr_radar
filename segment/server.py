import socket
import threading
import time

HOST = '127.0.0.1'
PORT = 55688
HEARTBEAT_INTERVAL = 5  # 每5秒發送一次 ALIVE

# === 心跳傳送執行緒 ===
def send_heartbeat(conn, stop_event):
    while not stop_event.is_set():
        try:
            time.sleep(HEARTBEAT_INTERVAL)
            conn.sendall(b"hey! hey you! yeah you~")
        except (BrokenPipeError, ConnectionResetError):
            print("[Server] Client disconnected during heartbeat.")
            break
        except Exception as e:
            print(f"[Server] Heartbeat error: {e}")
            break
    stop_event.set()

# === 客戶端訊息監聽執行緒 ===
def listen_client(conn, stop_event):
    while not stop_event.is_set():
        try:
            data = conn.recv(1024)
            if not data:
                print("[Server] Client closed the connection.")
                break

            msg = data.decode().strip()
            if not msg:
                continue

            if msg.upper() == "EXIT":
                print("[Server] Client requested shutdown.")
                stop_event.set()
                break

            print(f"[Server] Message from client: {msg}")
        except (BrokenPipeError, ConnectionResetError):
            print("[Server] Connection lost while receiving client message.")
            break
        except Exception as e:
            print(f"[Server] Client listener error: {e}")
            break

    stop_event.set()

# === 主伺服器 ===
def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT}. Waiting for clients...")

        try:
            while True:
                print("[Server] Waiting for client connection...")
                try:
                    conn, addr = s.accept()
                except KeyboardInterrupt:
                    print("\n[Server] Keyboard interrupt. Shutting down server.")
                    break

                with conn:
                    print(f"[Server] Connected by {addr}")
                    stop_event = threading.Event()

                    heartbeat_thread = threading.Thread(
                        target=send_heartbeat, args=(conn, stop_event), daemon=True
                    )
                    heartbeat_thread.start()

                    listener_thread = threading.Thread(
                        target=listen_client, args=(conn, stop_event), daemon=True
                    )
                    listener_thread.start()

                    try:
                        while not stop_event.is_set():
                            cmd = input("Press Enter command: ").strip()
                            if not cmd:
                                cmd = 'img_1141015101033'

                            cmd_upper = cmd.upper()

                            conn.sendall(cmd_upper.encode())
                    except (BrokenPipeError, ConnectionResetError):
                        print("[Server] Connection lost while sending command.")
                    except EOFError:
                        print("[Server] Input stream closed. Shutting down server.")
                        stop_event.set()
                        return
                    except KeyboardInterrupt:
                        print("\n[Server] Keyboard interrupt. Shutting down server.")
                        stop_event.set()
                        return
                    except Exception as e:
                        print(f"[Server] Command loop error: {e}")

                    stop_event.set()
                    listener_thread.join(timeout=1)
                    heartbeat_thread.join(timeout=1)

                print("[Server] Client disconnected. Waiting for next client...")
        except KeyboardInterrupt:
            print("\n[Server] Keyboard interrupt. Shutting down server.")

if __name__ == "__main__":
    start_server()
