from typing import TypedDict


class MessageStatsTD(TypedDict):
    messages_count: int
    spoken_count: int
    spam_count: int


class TwitchCredentialsTD(TypedDict):
    client_id: str
    access: str
    refresh: str
    nickname: str
