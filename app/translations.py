import asyncio
import inspect
import locale
import multiprocessing

from googletrans import Translator

TRANSLATIONS = {
    "en": {
        "app_title": "FJ Chat Voice - Silero TTS",
        "tab_chat": "Chat",
        "tab_settings": "Settings",
        "Messages": "Messages",
        "Spoken": "Spoken",
        "Spam": "Spam",
        "In queue": "In queue",
        "Volume": "Volume",
        "Speech rate": "Speech rate",
        "YouTube URL / ID": "YouTube URL / ID",
        "Connect": "Connect",
        "Disconnect": "Disconnect",
        "Clear log": "Clear log",
        "Export log": "Export log",
        "Message Log": "Message Log",
        "Auto-scroll": "Auto-scroll",
        "Language": "Language",
        "Credentials": "Credentials",
        "Google API Key": "Google API Key",
        "Another credentials": "Another credentials",
        "Silero model": "Silero model",
        "Silero not loaded": "Silero not loaded",
        "Chat language": "Chat language",
        "Voice": "Voice",
        "Add accents": "Add accents",
        "Replace e with yo (Russian)": "Replace e with yo (Russian)",
        "Message Queue": "Message Queue",
        "Queue depth": "Queue depth:",
        "Queue note": "(number of messages waiting to be spoken)",
        "Main filters": "Main filters",
        "Delay between messages": "Delay between messages (sec):",
        "Remove emojis": "Remove emojis",
        "Remove links": "Remove links",
        "Read author names": "Read author names",
        "Ignore system messages": "Ignore system messages",
        "Subscribers only": "Subscribers only",
        "Stop words": "Stop words (ignore messages)",
        "Apply": "Apply",
        "System": "System",
        "Error saving settings": "Error saving settings",
        "api_key_saved": "API key saved",
        "connection_success": "Connection success",
        "connection_failed": "Connection failed",
        "note_tts_not_loaded": "Silero TTS model is not loaded. Continue connecting to chat?",
        "Please enter video ID or URL": "Please enter video ID or URL",
        "Please enter API key": "Please enter API key",
        "silero_loading": "Loading Silero model (this may take 1-2 minutes)...",
        "silero_loaded": "Silero model loaded",
        "silero_failed": "Failed to load Silero model",
        "detoxify_loading": "Loading Detoxify model (this may take 1-2 minutes)...",
        "detoxify_loaded": "Detoxify model loaded",
        "detoxify_loading_failed": "Failed to load Detoxify model",
        "Warmup": "Warmup",
        "said": "said",
        "video_not_found": "Video not found or not a live stream",
        "Anonymous": "Anonymous",
        "error_fetch_messages": "Error fetching messages",
        "Audio queue error": "Audio queue error",
        "Audio playback error": "Audio playback error",
        "point": "point",
        "api_keys_help_text": "Creating and viewing API keys is available at the link",
        "not_determine_video_id": "Could not determine video ID",
        "not_determine_chat_id": "Could not determine chat ID",
        "queue_depth_desc": "Maximum number of messages in the voiceover queue",
        "chat_listener_stopped": "The chat listener has stopped",
        "chat_disconnected": "Disconnected from chat",
        "chat_connected": "Connected to chat",
        "many_errors": "Several errors in a row",
        "verification_uri": "If the browser does not open automatically, you need to follow this link",
        "expires_in": "Code validity period in minutes",
        "client_id_help_text": "You can get a CLIENT ID by creating an application using the link",
        "continue_authorize_browser": "Need to continue authorization in the browser",
    },
    "ru": {
        "app_title": "FJ Chat Voice - Silero TTS",
        "tab_chat": "Чат",
        "tab_settings": "Настройки",
        "Messages": "Сообщения",
        "Spoken": "Озвучено",
        "Spam": "Спам",
        "In queue": "В очереди",
        "Volume": "Громкость",
        "Speech rate": "Скорость озвучки",
        "YouTube URL / ID": "YouTube URL / ID",
        "Connect": "Подключиться",
        "Disconnect": "Отключиться",
        "Clear log": "Очистить лог",
        "Export log": "Экспорт лога",
        "Message Log": "Журнал сообщений",
        "Auto-scroll": "Автопрокрутка",
        "Language": "Язык",
        "Credentials": "Учётные данные",
        "Google API Key": "Ключ Google API",
        "Another credentials": "Другие данные",
        "Silero model": "Модель Silero",
        "Silero not loaded": "Silero не загружена",
        "Chat language": "Язык чата",
        "Voice": "Голос",
        "Add accents": "Добавлять акценты",
        "Replace e with yo (Russian)": "Заменять е на ё (русск.)",
        "Message Queue": "Очередь сообщений",
        "Queue depth": "Глубина очереди:",
        "Queue note": "(количество сообщений в очереди)",
        "Main filters": "Основные фильтры",
        "Min message length": "Минимальная длина сообщения",
        "Max message length": "Максимальная длина сообщения",
        "Delay between messages": "Задержка между сообщениями (сек)",
        "Remove emojis": "Удалять эмодзи",
        "Remove links": "Удалять ссылки",
        "Read author names": "Читать имена авторов",
        "Ignore system messages": "Игнорировать системные сообщения",
        "Subscribers only": "Только подписчики",
        "Stop words": "Стоп-слова",
        "Apply": "Применить",
        "System": "Система",
        "Error saving settings": "Ошибка сохранения настроек",
        "api_key_saved": "API Ключ сохранён",
        "connection_success": "Успешное подключение",
        "connection_failed": "Не удалось подключиться",
        "note_tts_not_loaded": "Модель Silero TTS не загружена. Продолжить подключение к чату?",
        "Please enter video ID or URL": "Пожалуйста, введите ID видео или URL",
        "Please enter API key": "Пожалуйста, введите ключ API",
        "silero_loading": "Загрузка модели Silero (это может занять 1-2 минуты)...",
        "silero_loaded": "Модель Silero загружена",
        "silero_failed": "Не удалось загрузить модель Silero",
        "detoxify_loading": "Загрузка модели Detoxify (это может занять 3-5 минут)...",
        "detoxify_loaded": "Модель Detoxify загружена",
        "detoxify_loading_failed": "Не удалось загрузить модель Detoxify",
        "Warmup": "Разогрев",
        "said": "сказал",
        "video_not_found": "Видео не найдено или не является прямой трансляцией",
        "Anonymous": "Аноним",
        "error_fetch_messages": "Ошибка при получении сообщений",
        "Audio queue error": "Ошибка очереди аудио",
        "Audio playback error": "Ошибка воспроизведения аудио",
        "Error convert text to speech": "Ошибка конвертирования текста в речь",
        "point": "точка",
        "English": "Английский",
        "Russian": "Русский",
        "Interface Language": "Язык интерфейса",
        "Speaker language": "Язык речи",
        "Speech configuration": "Настройки речи",
        "Message Settings": "Настройки сообщений",
        "Delays and processing": "Задержки и обработка",
        "Configure": "Настроить",
        "Connected": "Подключено",
        "Disconnected": "Отключено",
        "Resume": "Продолжить",
        "Pause": "Пауза",
        "Playback...": "Воспроизведение...",
        "Playback has been stopped": "Воспроизведение речи остановлено",
        "Speech playback continued...": "Воспроизведение речи продолжено",
        "Stopped": "Остановлено",
        "Enter your API key": "Введите ваш API ключ",
        "api_keys_help_text": "Создание и просмотр API ключей доступно по ссылке",
        "not_determine_video_id": "Не удалось определить ID видео",
        "not_determine_chat_id": "Не удалось определить ID чата",
        "Failed to translate text": "Не удалось перевести текст",
        "Clear queue": "Очистить очередь",
        "File": "Файл",
        "File saved": "Файл сохранён",
        "Log cleared": "Лог очищен",
        "Queue cleared": "Очередь очищена",
        "Saved": "Сохранено",
        "stop words": "стоп-слов/а",
        "Settings saved": "Настройки сохранены",
        "queue_depth_desc": "Максимальное количество сообщений в очереди на озвучку",
        "chat_listener_stopped": "Слушатель чата остановлен",
        "chat_disconnected": "Отключен от чата",
        "chat_connected": "Подключен к чату",
        "many_errors": "Несколько ошибок подряд",
        "Failed to get a code": "Не удалось получить код",
        "Success authorized": "Успешная авторизация",
        "verification_uri": "Если браузер не открылся автоматически, вам необходимо перейти по этой ссылке",
        "expires_in": "Срок действия кода в минутах",
        "Twitch account": "Twitch аккаунт",
        "client_id_help_text": "Получить CLIENT ID можно создав приложение по ссылке",
        "continue_authorize_browser": "Необходимо продолжить авторизацию в браузере",
        "Error send a command": "Ошибка отправки команды",
        "Incorrect nickname format": "Неправильный формат ника",
        "The nickname is already in use": "Ник уже используется",
        "Connection lost": "Соединение потеряно",
        "Authorization timeout. Please start again.": "Время ожидания авторизации истекло. Начните заново.",
        "Error": "Ошибка",
        "Unexpected error": "Неожиданная ошибка",
        "Authorization wait time expired": "Истекло время ожидания авторизации",
        "Network timeout. Check your connection.": "Таймаут сети. Проверьте подключение.",
        "Connection error": "Ошибка подключения",
        "Failed to get nickname": "Ошибка получения никнейма",
        "Failed to refresh token": "Не удалось обновить Twitch токен",
        "missing": "отсутствует",
        "Translate messages": "Перевод сообщений",
        "Edit": "Редактировать",
        "Save": "Сохранить",
        "Read platform name": "Озвучивать название платформы",
        "youtube": "Ютуб",
        "twitch": "Твич",
        "Message from": "Сообщение из",
        "random": "случайно",
        "Toxicity threshold (1 = OFF)": "Порог токсичности (1 = OFF)",
        "Toxicity": "Токсичность",
        "Severe toxicity": "Тяжелая токсичность",
        "Obscene": "Непристойное",
        "Identity attack": "Атака на личность",
        "Insult": "Оскорбление",
        "Threat": "Угроза",
        "Sexual explicit": "Сексуальный контент",
        "The file is corrupted": "Файл повреждён",
        "Link": "Ссылка",
        "Filtered": "Отфильтровано",
        "List of banned": "Список заблокированных",
        "The number of toxic messages leading to ban": "Количество токсичных сообщений для блокировки",
        "Banned": "Заблокирован",
    },
}

GUI_LANGUAGES = tuple(TRANSLATIONS.keys())
SYS_LOCALE = locale.getlocale()

DEFAULT_LANGUAGE = (
    "ru"
    if SYS_LOCALE
    and SYS_LOCALE[0]
    and (SYS_LOCALE[0].startswith("Russian") or SYS_LOCALE[0].startswith("ru_"))
    else "en"
)
LANG_CODES = {"en": "English", "ru": "Russian"}

_CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

_LAT_TO_CYR = {
    "a": "а",
    "b": "б",
    "c": "ц",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "ы",
    "z": "з",
}


def _map_char_with_case(ch: str, mapping: dict[str, str]) -> str:
    base = mapping.get(ch.lower())
    if base is None:
        return ch
    if not ch.isalpha():
        return base
    if ch.isupper():
        return base[:1].upper() + base[1:]
    return base


def transliteration(text: str, lang: str) -> str:
    """
    Transliterate text to target language.

    lang='en': Cyrillic -> Latin (Привет -> Privet)
    lang='ru': Latin -> Cyrillic (john -> йохн)
    """
    src = str(text or "")
    target = str(lang or "").strip().lower()
    if target == "en":
        return "".join(_map_char_with_case(ch, _CYR_TO_LAT) for ch in src)
    if target == "ru":
        return "".join(_map_char_with_case(ch, _LAT_TO_CYR) for ch in src)
    return src


def _(lang, key):
    """Simple translation helper"""
    return TRANSLATIONS.get(lang, {}).get(key, key)


def _proc_translate_external(q, txt, dst):
    """Module-level worker for multiprocessing spawn on Windows."""
    try:
        tr = Translator()
        r = tr.translate(txt, dest=dst)
        if inspect.isawaitable(r):
            r = asyncio.run(r)
        q.put(getattr(r, "text", txt))
    except Exception as e:
        try:
            q.put({"__err__": str(e)})
        except Exception:
            pass


def translate_text(text, dest):
    try:
        q = multiprocessing.Queue()
        _proc_translate_external(q, text, dest)
        return q.get_nowait()
    except Exception as e:
        # self.add_sys_message(
        #     author="_translate_text()",
        #     text=f"{_(self.language, 'Failed to translate text')}. {e}",
        #     status="error",
        # )
        return text
