from time import sleep
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import urllib

from app.translations import _

YT_API_FIELDS = (
    "nextPageToken,pollingIntervalMillis,"
    "items(id,snippet(displayMessage),authorDetails(displayName,isChatOwner,isChatSponsor,isChatModerator))"
)


class YouTubeChatListener:
    def __init__(
        self,
        api_key: str,
        url: str,
        on_message,
        on_connect,
        on_disconnect,
        on_error,
        lang: str = "en",
    ):
        super().__init__()
        self._api_key_ = api_key
        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        self.lang = lang

        self.is_connected = False
        self.chat_id = None
        self.video_id = None

    def run(self):
        self.is_connected = self._connect()
        errors = 0
        delay = 3
        page_token = None

        try:
            while self.is_connected and bool(self.chat_id):
                page_token, is_error = self._fetch_messages(page_token)
                if is_error:
                    errors += 1
                else:
                    errors = 0

                if errors >= 5:
                    self.disconnect(_(self.lang, "many_errors"))
                    return

                sleep(delay * max(1, errors))

        except Exception as e:
            self.on_error(f"{_(self.lang, "chat_listener_error")}. {str(e)}")

        finally:
            self.disconnect(_(self.lang, "chat_listener_stopped"))

    def disconnect(self):
        if self.is_connected:
            self.is_connected = False
            self.on_disconnect()

    def _fetch_messages(self, page_token=None):
        """Fetch messages from YouTube chat"""
        if not self.is_connected:
            return
        next_token = None
        is_error = False

        try:
            response = (
                self.client.liveChatMessages()
                .list(
                    liveChatId=self.chat_id,
                    part="snippet,authorDetails",
                    pageToken=page_token,
                    fields=YT_API_FIELDS,
                )
                .execute()
            )

            next_token = response.get("nextPageToken")

            for item in response.get("items", []):
                msg_id = item["id"]
                snippet = item["snippet"]
                author_details = item.get("authorDetails", {})
                author = author_details.get(
                    "displayName",
                    snippet.get("authorDisplayName", _(self.lang, "Anonymous")),
                )

                if not author or author.strip() == "":
                    author = _(self.lang, "Anonymous")
                if str(author).startswith("@"):
                    author = str(author)[1:]

                message = snippet.get("displayMessage", "")
                is_member = (
                    author_details.get("isChatOwner", False)
                    or author_details.get("isChatSponsor", False)
                    or author_details.get("isChatModerator", False)
                )

                self.on_message(msg_id, author, message, is_member)

            poll_ms = response.get("pollingIntervalMillis")
            try:
                delay = max(1, int(poll_ms) / 1000) if poll_ms is not None else 1
                sleep(delay)
            except Exception:
                pass

        except HttpError as e:
            is_error = True
            self.on_error(f"{_(self.lang, "error_fetch_messages")}. {e.reason}")

        except Exception as e:
            is_error = True
            self.on_error(f"{_(self.lang, "error_fetch_messages")}. {str(e)}")

        return next_token, is_error

    def _connect(self):
        try:
            self.video_id = self._parse_video_id()
            if not self.video_id:
                self.on_error(_(self.lang, "not_determine_video_id"))
                return False

            self.client = build("youtube", "v3", developerKey=self._api_key_)
            self.chat_id = self._get_chat_id()
            if not self.chat_id:
                return False

            self.on_connect()
            return True

        except HttpError as e:
            self.on_error(f"{_(self.lang, "connection_failed")}. {e.reason}")
            return False

        except Exception as e:
            self.on_error(f"{_(self.lang, "connection_failed")}. {str(e)}")
            return False

    def _parse_video_id(self):
        try:
            video_id = self.url

            if "youtube.com" in self.url or "youtu.be" in self.url:
                parsed = urllib.parse.urlparse(self.url)
                if "youtu.be" in parsed.netloc:
                    video_id = parsed.path[1:]
                elif "watch" in parsed.path:
                    query = urllib.parse.parse_qs(parsed.query)
                    video_id = query.get("v", [None])[0]
                elif "embed" in parsed.path:
                    video_id = parsed.path.split("/")[-1]

        except Exception as e:
            self.on_error(f"{_(self.lang, "not_determine_video_id")}. {str(e)}")
            return

        return video_id

    def _get_chat_id(self):
        try:
            response = (
                self.client.videos()
                .list(part="liveStreamingDetails", id=self.video_id)
                .execute()
            )

            if response.get("items"):
                details = response["items"][0].get("liveStreamingDetails", {})
                return details.get("activeLiveChatId")
            else:
                self.on_error(_(self.lang, "video_not_found"))

        except HttpError as e:
            self.on_error(f"{_(self.lang, "not_determine_chat_id")}. {e.reason}")

        except Exception as e:
            self.on_error(f"{_(self.lang, "not_determine_chat_id")}. {str(e)}")
