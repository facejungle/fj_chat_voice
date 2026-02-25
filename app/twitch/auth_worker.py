import requests
import webbrowser
import time

from PyQt6.QtCore import QThread, pyqtSignal

from app.translations import _, translate_text


class AuthWorker(QThread):
    user_code_signal = pyqtSignal(
        str, str, int
    )  # verification_uri, user_code, expires_in
    token_signal = pyqtSignal(str, str, str)  # token, refresh_token, nickname
    error_signal = pyqtSignal(str)

    def __init__(self, client_id, scopes="chat:read", lang="en"):
        super().__init__()
        self.client_id = client_id
        self.scopes = scopes
        self.lang = lang

    def run(self):
        try:
            device_response = requests.post(
                "https://id.twitch.tv/oauth2/device",
                data={"client_id": self.client_id, "scopes": self.scopes},
                timeout=30,
            )
            device_data = device_response.json()

            if device_response.status_code != 200:
                error_msg = device_data.get("message", "")
                self.error_signal.emit(
                    f"{_(self.lang, "Failed to get a code")}. {error_msg}"
                )
                return

            user_code = device_data["user_code"]
            verification_uri = device_data["verification_uri"]
            device_code = device_data["device_code"]
            interval = device_data.get("interval", 5)
            expires_in = device_data.get("expires_in", 1800)

            self.user_code_signal.emit(verification_uri, user_code, expires_in // 60)

            webbrowser.open(verification_uri)

            start_time = time.time()

            while time.time() - start_time < expires_in:
                time.sleep(interval)

                token_response = requests.post(
                    "https://id.twitch.tv/oauth2/token",
                    data={
                        "client_id": self.client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    timeout=30,
                )
                token_data = token_response.json()

                if token_response.status_code == 200:
                    access_token = token_data["access_token"]
                    refresh_token = token_data.get("refresh_token", "")
                    expires_in = token_data.get("expires_in", 0)
                    nickname = self.get_user_nickname(access_token)

                    self.token_signal.emit(access_token, refresh_token, nickname)
                    break

                elif token_data.get("status") == 400:
                    error_msg = token_data.get("message", "")

                    if error_msg == "authorization_pending":
                        continue

                    elif error_msg == "slow_down":
                        interval += 1
                        continue

                    elif error_msg == "expired_token":
                        self.error_signal.emit(
                            f"❌ {_(self.lang, 'Authorization timeout. Please start again.')}"
                        )
                        break

                    else:
                        self.error_signal.emit(
                            f"❌ {_(self.lang, 'Error')}: {error_msg}"
                        )
                        break
                else:
                    self.error_signal.emit(
                        f"❌ {_(self.lang, 'Unexpected error')}: {token_response.status_code}"
                    )
                    break
            else:
                self.error_signal.emit(
                    f"❌ {_(self.lang, 'Authorization wait time expired')}"
                )

        except requests.exceptions.Timeout:
            self.error_signal.emit(
                f"❌ {_(self.lang, 'Network timeout. Check your connection.')}"
            )
        except requests.exceptions.ConnectionError:
            self.error_signal.emit(f"❌ {_(self.lang, 'Connection error')}")
        except Exception as e:
            self.error_signal.emit(
                f"❌ {_(self.lang, 'Unexpected error')}: {translate_text(str(e), self.lang)}"
            )

    def get_user_nickname(self, access_token):
        try:
            response = requests.get(
                "https://api.twitch.tv/helix/users",
                headers={
                    "Client-ID": self.client_id,
                    "Authorization": f"Bearer {access_token}",
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    return str(data["data"][0]["login"])
            return None
        except Exception as e:
            self.error_signal.emit(
                f"{_(self.lang, 'Failed to get nickname')}: {translate_text(str(e), self.lang)}"
            )
            return None

    @staticmethod
    def refresh_access_token(client_id, refresh_token, lang="en"):
        response = requests.post(
            "https://id.twitch.tv/oauth2/token",
            data={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=30,
        )

        token_data = response.json()
        if response.status_code != 200:
            error_msg = token_data.get("message", token_data)
            raise RuntimeError(f"{_(lang, 'Failed to refresh token')}: {error_msg}")

        access_token = token_data.get("access_token")
        new_refresh_token = token_data.get("refresh_token", refresh_token)

        if not access_token:
            raise RuntimeError(
                f"{_(lang, 'Failed to refresh token')}: access_token {_(lang, 'missing')}"
            )

        return access_token, new_refresh_token
