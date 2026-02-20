import re
import socket
from threading import Thread
from time import sleep, time
from urllib.parse import urlparse

import requests

from app.translations import _


class TwitchChatListener:
    def __init__(
        self,
        client_id,
        token,
        channel,
        nickname,
        on_message,
        on_connect,
        on_disconnect,
        on_error,
        on_expiries_access,
        lang="en",
    ):
        self.client_id = client_id
        self.token = token
        self.channel = self._parse_channel(channel)
        self.nickname = str(nickname).lower()
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        self.on_expiries_access = on_expiries_access
        self.lang = lang

        self.sock = None
        self.is_connected = False
        self.listen_thread = None
        self.last_ping = time()
        self._is_stopping = False

    def _handle_expired_access(self):
        try:
            self.token = self.on_expiries_access()
            return True
        except Exception as e:
            self.on_error(f"{_(self.lang, 'connection_failed')}. {e}")
            return False

    def _send_command(self, command):
        try:
            self.sock.send(f"{command}\r\n".encode("utf-8"))
        except Exception as e:
            self.on_error(f"{_(self.lang, "Error send a command")} {command}. {e}")

    def _recv(self):
        try:
            return self.sock.recv(4096).decode("utf-8", errors="ignore")
        except Exception as e:
            self.on_error(f"{_(self.lang, "Error recv data")}. {e}")

    def _parse_channel(self, channel_input):
        if not channel_input:
            return None

        channel_input = channel_input.strip().lower()

        if channel_input.startswith(("http://", "https://")):
            parsed = urlparse(channel_input)
            path = parsed.path.strip("/")

            if "twitch.tv" in parsed.netloc or "twitch.tv" in channel_input:
                parts = path.split("/")
                if parts and parts[0]:
                    return parts[0]

        channel_input = channel_input.lstrip("@#")
        channel_input = re.sub(r"[^a-zA-Z0-9_]", "", channel_input)

        return channel_input

    def _connect(self):
        for attempt in range(2):
            try:
                server = "irc.chat.twitch.tv"
                port = 6667

                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10)
                self.sock.connect((server, port))

                self._send_command(
                    "CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership"
                )
                self._send_command(f"PASS oauth:{self.token}")
                self._send_command(f"NICK {self.nickname}")

                capabilities_confirmed = False
                start_time = time()
                timeout = 10

                while not capabilities_confirmed and (time() - start_time) < timeout:
                    try:
                        response = self.sock.recv(4096).decode("utf-8", errors="ignore")
                        if "Login authentication failed" in response:
                            if attempt == 0 and self._handle_expired_access():
                                self.sock.close()
                                self.sock = None
                                break
                            self.on_error(_(self.lang, "connection_failed"))
                            return

                        if "CAP * ACK" in response:
                            capabilities_confirmed = True
                            break

                    except socket.timeout:
                        continue

                if self.sock is None:
                    continue

                if not capabilities_confirmed:
                    print(
                        "[TwitchChatListener] Warning: Capabilities not confirmed, continuing..."
                    )

                self._send_command(f"JOIN #{self.channel}")

                connected = False
                start_time = time()

                while not connected and (time() - start_time) < timeout:
                    try:
                        response = self.sock.recv(4096).decode("utf-8", errors="ignore")
                        lines = response.strip().split("\r\n")
                        for line in lines:
                            if "Login authentication failed" in line:
                                if attempt == 0 and self._handle_expired_access():
                                    self.sock.close()
                                    self.sock = None
                                    break
                                self.on_error(_(self.lang, "connection_failed"))
                                return

                            if f"JOIN #{self.channel}" in line:
                                connected = True
                                break

                            if "466" in line:  # ERR_ERRONEOUSNICKNAME
                                self.on_error(_(self.lang, "Incorrect nickname format"))
                                return
                            if "433" in line:  # ERR_NICKNAMEINUSE
                                self.on_error(
                                    _(self.lang, "The nickname is already in use")
                                )
                                return
                        if self.sock is None:
                            break

                    except socket.timeout:
                        continue

                if self.sock is None:
                    continue

                if not connected:
                    self.on_error(_(self.lang, "connection_failed"))
                    return

                self.is_connected = True
                self.on_connect()
                return

            except Exception as e:
                self.on_error(f"{_(self.lang, "connection_failed")}. {e}")
                return

    def disconnect(self):
        was_connected = self.is_connected
        self._is_stopping = True
        self.is_connected = False
        if self.sock:
            try:
                self._send_command(f"PART #{self.channel}")
                self.sock.close()
            except:
                pass
            self.sock = None
        if was_connected:
            self.on_disconnect()

    def _fetch(self, endpoint, params=None):
        url = f"https://api.twitch.tv/helix/{endpoint}"
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f'Bearer {self.token.replace("oauth:", "")}',
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 401:
                if self._handle_expired_access():
                    headers["Authorization"] = (
                        f'Bearer {self.token.replace("oauth:", "")}'
                    )
                    response = requests.get(
                        url, headers=headers, params=params, timeout=10
                    )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[TwitchChatListener] API error: {e}")
            return None

    def _get_user_info(self, username):
        return self._fetch("users", {"login": username})

    def _parse_message(self, line):
        try:
            if line.startswith("@"):
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    return None

                tags_str = parts[0][1:]
                rest = parts[1]

                tag_dict = {}
                for tag in tags_str.split(";"):
                    if "=" in tag:
                        key, value = tag.split("=", 1)
                        value = (
                            value.replace("\\s", " ")
                            .replace("\\:", ";")
                            .replace("\\\\", "\\")
                        )
                        tag_dict[key] = value

                match = re.search(
                    r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.*)", rest
                )

                if match:
                    username = match.group(1)
                    message = match.group(2)

                    msg_id = tag_dict.get("id")
                    is_member = tag_dict.get("subscriber") == "1"
                    is_mod = tag_dict.get("mod") == "1"
                    is_vip = tag_dict.get("vip") == "1"
                    badges = tag_dict.get("badges", "")

                    return {
                        "id": msg_id,
                        "username": username,
                        "message": message,
                        "subscriber": is_member,
                        "mod": is_mod,
                        "vip": is_vip,
                        "badges": badges,
                        "tags": tag_dict,
                    }
            else:
                match = re.search(
                    r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.*)", line
                )
                if match:
                    username = match.group(1)
                    message = match.group(2)
                    return {
                        "id": None,
                        "username": username,
                        "message": message,
                        "subscriber": False,
                        "mod": False,
                        "vip": False,
                        "badges": "",
                        "tags": {},
                    }
        except Exception as e:
            self.on_error(f"{_(self.lang, "Error parsing message")}. {e}")

        return None

    def _listen_chat(self):
        errors = 0
        max_errors = 5
        buffer = ""
        self._is_stopping = False
        while self.is_connected and self.sock:
            if errors >= max_errors:
                self.disconnect()
                break

            try:
                data = self.sock.recv(4096).decode("utf-8", errors="ignore")

                if not data:
                    sleep(0.1)
                    continue

                buffer += data

                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)

                    if not line:
                        continue

                    if line.startswith("PING"):
                        self._send_command("PONG")
                        self.last_ping = time()
                        continue

                    msg_data = self._parse_message(line)

                    if msg_data:
                        self.on_message(
                            msg_data["id"],
                            msg_data["username"],
                            msg_data["message"],
                            msg_data["subscriber"],
                        )

                errors = 0

            except socket.timeout:
                if time() - self.last_ping > 60:
                    try:
                        self._send_command("PING")
                        self.last_ping = time()
                    except:
                        pass
                continue

            except ConnectionResetError:
                if self._is_stopping:
                    break
                self.on_error(_(self.lang, "Connection lost"))
                self.disconnect()
                break

            # except OSError as e:
            #     # Socket was closed from another thread during disconnect on Windows.
            #     if self._is_stopping or getattr(e, "winerror", None) == 10038:
            #         break
            #     self.on_error(f"{_(self.lang, "error_fetch_messages")}. {e}")
            #     errors += 1
            #     sleep(1 * errors)

            except Exception as e:
                if self._is_stopping:
                    break
                self.on_error(f"{_(self.lang, "error_fetch_messages")}. {e}")
                errors += 1
                sleep(1 * errors)
            finally:
                sleep(1)

    def run(self):
        try:
            self._connect()
            self.listen_thread = Thread(target=self._listen_chat, daemon=True)
            self.listen_thread.start()
            return True
        except Exception as e:
            self.on_error(f"{_(self.lang, "Runtime error")}. {e}")
            return False

    def stop(self):
        self.disconnect()
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=5)
