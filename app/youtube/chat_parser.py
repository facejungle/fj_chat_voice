import re
from threading import Thread
from time import sleep
import httpx
import pytchat
from pytchat import util
import urllib

from app.translations import _, translate_text

_CHANNEL_PATTERNS = (
    re.compile(r'"channelId":"(UC[a-zA-Z0-9_-]{22})"'),
    re.compile(r'\\"channelId\\":\\"(UC[a-zA-Z0-9_-]{22})\\"'),
)


def _extract_channel_id(video_id: str) -> str:
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "accept-language": "en-US,en;q=0.9",
    }

    urls = (
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/embed/{video_id}",
        f"https://m.youtube.com/watch?v={video_id}",
    )

    with httpx.Client(
        http2=True, follow_redirects=True, timeout=20.0, headers=headers
    ) as client:
        for url in urls:
            text = client.get(url).text
            for pattern in _CHANNEL_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(1)

    raise pytchat.exceptions.InvalidVideoIdException(
        f"Cannot find channel id for video id:{video_id}."
    )


# Patch pytchat channel-id resolvers to avoid brittle built-in regex fallback.
util.get_channelid = lambda client, video_id: _extract_channel_id(video_id)
util.get_channelid_2nd = lambda client, video_id: _extract_channel_id(video_id)


class YouTubeChatParser:
    def __init__(
        self,
        url: str,
        on_message,
        on_connect,
        on_disconnect,
        on_error,
        lang: str = "en",
    ):
        super().__init__()
        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        self.lang = lang

        self.is_connected = False
        self.chat_id = None
        self.video_id = None

    def _connect(self):
        delay = 0.5
        errors = 0

        try:
            video_id = self._parse_video_id(self.url)
            chat = pytchat.create(video_id=video_id, interruptable=False)

            if chat.is_alive():
                self.is_connected = True
                self.on_connect()

            while chat.is_alive() and self.is_connected:
                try:
                    for message in chat.get().sync_items():
                        if message.type == "textMessage":
                            author_details = message.author
                            is_member = (
                                author_details.isChatSponsor
                                or author_details.isChatOwner
                                or author_details.isChatModerator
                            )
                            self.on_message(
                                message.id,
                                author_details.name,
                                message.message,
                                is_member,
                            )
                        errors = 0

                except Exception as e:
                    self.on_error(
                        f"{_(self.lang, "error_fetch_messages")}. {translate_text(str(e), self.lang)}"
                    )
                    errors += 1

                if errors >= 5:
                    self.disconnect()
                    return

                sleep(delay * max(1, errors))
            else:
                self.is_connected = False

        except Exception as e:
            self.on_error(
                f"{_(self.lang, "connection_failed")}. {translate_text(str(e), self.lang)}"
            )
        finally:
            self.disconnect()

    def disconnect(self):
        if self.is_connected:
            self.is_connected = False
            self.on_disconnect()

    def run(self):
        th = Thread(target=self._connect, daemon=True)
        th.start()

    def _parse_video_id(self, url):
        try:
            video_id = url
            if url.startswith("watch?v="):
                url = url.removeprefix("watch?v=")
            elif "youtube.com" in url or "youtu.be" in url:
                parsed = urllib.parse.urlparse(url)
                if "youtu.be" in parsed.netloc:
                    video_id = parsed.path[1:]
                elif "watch" in parsed.path:
                    query = urllib.parse.parse_qs(parsed.query)
                    video_id = query.get("v", [None])[0]
                elif "embed" in parsed.path:
                    video_id = parsed.path.split("/")[-1]
                elif "studio.youtube.com" in parsed.netloc and "/video/" in parsed.path:
                    path_parts = [part for part in parsed.path.split("/") if part]
                    if "video" in path_parts:
                        video_index = path_parts.index("video")
                        if video_index + 1 < len(path_parts):
                            video_id = path_parts[video_index + 1]

        except Exception as e:
            self.on_error(
                f"{_(self.lang, "not_determine_video_id")}. {translate_text(str(e), self.lang)}"
            )
            self.disconnect()
            return

        return video_id
