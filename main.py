from torch import hub, no_grad, set_grad_enabled, set_num_threads
import sys
from collections import defaultdict, deque
import asyncio
from datetime import datetime
from functools import lru_cache
import gc
import json
import html
import re
import threading
import inspect
import multiprocessing
from time import sleep
from typing import TypedDict
import hashlib
import colorsys
import urllib

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from num2words import num2words
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QHBoxLayout,
    QSlider,
    QComboBox,
    QDialog,
    QMenuBar,
    QGridLayout,
    QLineEdit,
    QCheckBox,
    QTextEdit,
    QFileDialog,
    QMenu,
    QPlainTextEdit,
)
from PyQt6.QtCore import Qt, QTimer, QFile, QIODevice
from PyQt6.QtGui import QFont, QAction, QPalette, QTextCursor, QIcon
from scipy.signal import resample
import numpy as np
from googletrans import Translator

from translations import DEFAULT_LANGUAGE, LANG_CODES, _


PADDING = 20
DEFAULT_BUFFER_SIZE = 5
VOICES = {
    "Russian": ("xenia", "aidar", "baya", "kseniya", "eugene"),
    "English": (
        "random",
        "en_0",
        "en_1",
        "en_2",
        "en_3",
        "en_4",
        "en_5",
        "en_6",
        "en_7",
        "en_8",
        "en_9",
        "en_10",
        "en_11",
        "en_12",
        "en_13",
        "en_14",
        "en_15",
        "en_16",
        "en_17",
        "en_18",
        "en_19",
        "en_20",
    ),
}

YT_API_FIELDS = (
    "nextPageToken,pollingIntervalMillis,"
    "items(id,snippet(displayMessage),authorDetails(displayName,isChatOwner,isChatSponsor,isChatModerator))"
)
APP_VERSION = "0.0.5"


class MessageStatsTD(TypedDict):
    messages_count: int
    spoken_count: int
    spam_count: int


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        icon = QIcon("img/icon.png")
        self.setWindowIcon(icon)
        self.setWindowTitle("FJ Chat Voice")
        self.setMinimumSize(1200, 600)

        self.root_widget = QWidget()
        self.setCentralWidget(self.root_widget)
        self.root_layout = QVBoxLayout(self.root_widget)

        self.gui_language = DEFAULT_LANGUAGE
        self.voice_language = DEFAULT_LANGUAGE
        self.voice = VOICES[self.voice_language][0]
        self.volume = 100
        self.speech_rate = 1.00
        self.speech_delay = 1.5
        self.is_paused = False
        self.max_msg_length = 200

        self.auto_scroll = True
        self.add_accents = True
        self.remove_emojis = True
        self.remove_links = True
        self.filter_repeats = True
        self.read_author_names = False
        self.ignore_system_messages = True
        self.subscribers_only = False
        self.stop_words = []

        # Connections
        self.youtube = None
        self.twitch = None
        self.yt_chat_id = ""
        self.yt_video_id = ""
        self.yt_is_connected = False
        self.yt_credentials = None
        self.twitch_id = ""
        self.twitch_is_connected = False
        self.twitch_credentials = None

        self.messages_stats: MessageStatsTD = defaultdict(int)

        # Message queue
        self.buffer_maxsize = DEFAULT_BUFFER_SIZE
        self.message_buffer = deque(maxlen=self.buffer_maxsize)
        self.audio_queue = deque(maxlen=self.buffer_maxsize)
        self._pending_messages = deque()
        self.spam_hash_set = set()
        self.processed_messages = set()

        self.load_settings()
        self.setup_ui()

        self.silero_model = None
        self.translator = Translator()
        self.translator_lock = threading.Lock()

        set_num_threads(2)
        threading.Thread(target=self.init_silero, daemon=True).start()
        threading.Thread(target=self.process_audio_loop, daemon=True).start()
        threading.Thread(target=self.process_msg_buffer_loop, daemon=True).start()

    # === UI setup ===

    def setup_voice_menu(self):
        self.voice_menu.clear()
        for voice_lang in VOICES.keys():
            voice_lang_menu = self.voice_menu.addMenu(_(self.gui_language, voice_lang))
            for voice in VOICES[voice_lang]:
                voice_action = QAction(voice, self)
                voice_action.setCheckable(True)
                voice_action.setChecked(voice == self.voice)
                voice_action.triggered.connect(lambda checked, l=voice_lang, v=voice: self.voice_changed(l, v))
                voice_lang_menu.addAction(voice_action)

        add_accents_action = QAction(_(self.gui_language, "Add accents"), self)
        add_accents_action.setCheckable(True)
        add_accents_action.setChecked(self.add_accents)
        add_accents_action.triggered.connect(self.toggle_add_accents)
        self.voice_menu.addAction(add_accents_action)

    def setup_language_menu(self):
        self.language_menu.clear()
        for lang in VOICES.keys():
            lang_action = QAction(_(self.gui_language, lang), self)
            lang_action.setCheckable(True)
            lang_action.setChecked(lang == self.gui_language)
            lang_action.triggered.connect(lambda checked, l=lang: self.language_changed(l))
            self.language_menu.addAction(lang_action)

    def setup_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu(_(self.gui_language, "File"))

        export_log_action = QMenu(_(self.gui_language, "Export log"), file_menu)
        file_menu.addMenu(export_log_action)
        msg_log_html_action = QAction("Html", export_log_action)
        msg_log_html_action.triggered.connect(lambda: self.export_log("html"))
        export_log_action.addAction(msg_log_html_action)
        msg_log_md_action = QAction("Markdown", export_log_action)
        msg_log_md_action.triggered.connect(lambda: self.export_log("md"))
        export_log_action.addAction(msg_log_md_action)
        msg_log_text_action = QAction("Text", export_log_action)
        msg_log_text_action.triggered.connect(lambda: self.export_log("text"))
        export_log_action.addAction(msg_log_text_action)

        self.language_menu = menu_bar.addMenu(_(self.gui_language, "Language"))
        self.setup_language_menu()

        self.voice_menu = menu_bar.addMenu(_(self.gui_language, "Speech configuration"))
        self.setup_voice_menu()

        msg_settings_menu = menu_bar.addMenu(_(self.gui_language, "Message Settings"))

        stop_words_action = QAction(_(self.gui_language, "Stop words"), msg_settings_menu)
        stop_words_action.triggered.connect(self.on_stop_words_action)
        msg_settings_menu.addAction(stop_words_action)

        delays_action = QAction(_(self.gui_language, "Delays and processing"), msg_settings_menu)
        delays_action.triggered.connect(self.on_delays_settings_action)
        msg_settings_menu.addAction(delays_action)

        filter_repeats_action = QAction(_(self.gui_language, "Filter repeats"), msg_settings_menu)
        filter_repeats_action.setCheckable(True)
        filter_repeats_action.setChecked(self.filter_repeats)
        filter_repeats_action.triggered.connect(self.toggle_filter_repeats)
        msg_settings_menu.addAction(filter_repeats_action)

        read_authors_action = QAction(_(self.gui_language, "Read author names"), msg_settings_menu)
        read_authors_action.setCheckable(True)
        read_authors_action.setChecked(self.read_author_names)
        read_authors_action.triggered.connect(self.toggle_read_author_names)
        msg_settings_menu.addAction(read_authors_action)

        subscribers_only_action = QAction(_(self.gui_language, "Subscribers only"), msg_settings_menu)
        subscribers_only_action.setCheckable(True)
        subscribers_only_action.setChecked(self.subscribers_only)
        subscribers_only_action.triggered.connect(self.toggle_subscribers_only)
        msg_settings_menu.addAction(subscribers_only_action)

    def setup_dlg_yt_conf(self):
        dlg = QDialog(self)
        dlg.setMinimumSize(600, 100)
        dlg.setWindowTitle(_(self.gui_language, "Google API Key"))

        root_widget = QWidget(dlg)
        root_widget.setMinimumSize(600, 100)
        root_layout = QVBoxLayout(root_widget)

        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        root_layout.addLayout(entry_layout)
        self.google_api_key_input = QLineEdit()
        entry_layout.addWidget(self.google_api_key_input, 1)

        self.google_api_key_input.setPlaceholderText(_(self.gui_language, "Enter your API key"))
        if self.yt_credentials is None:
            self.google_api_key_input.returnPressed.connect(self.on_click_yt_save_settings)
            self.google_api_key_button = QPushButton(_(self.gui_language, "Save"))
            self.google_api_key_button.clicked.connect(self.on_click_yt_save_settings)
            entry_layout.addWidget(self.google_api_key_button)
        else:
            self.google_api_key_input.setText("*" * len(self.yt_credentials))
            self.google_api_key_input.setReadOnly(True)
            self.google_api_key_button = QPushButton(_(self.gui_language, "Edit"))
            self.google_api_key_button.clicked.connect(self.on_click_yt_edit_credential)
            entry_layout.addWidget(self.google_api_key_button)

        api_keys_help_text = QLabel(
            f'{_(self.gui_language, "api_keys_help_text")}: <a href="https://console.cloud.google.com/">https://console.cloud.google.com/</a>'
        )
        api_keys_help_text.setOpenExternalLinks(True)
        root_layout.addWidget(api_keys_help_text)

        dlg.exec()

    def setup_connections_grid(self):
        connections_grid = QGridLayout()
        self.root_layout.addLayout(connections_grid)

        yt_layout = QHBoxLayout()
        yt_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        yt_label = QLabel("YouTube")
        yt_layout.addWidget(yt_label)
        self.yt_video_input = QLineEdit()
        self.yt_video_input.returnPressed.connect(self.on_click_yt_connect)
        self.yt_video_input.setPlaceholderText("https://www.youtube.com/watch?v=VIDEO_ID or VIDEO_ID")
        yt_layout.addWidget(self.yt_video_input)
        self.connect_yt_button = QPushButton(_(self.gui_language, "Connect"))
        self.connect_yt_button.clicked.connect(self.on_click_yt_connect)
        yt_layout.addWidget(self.connect_yt_button)
        self.configure_yt_button = QPushButton(_(self.gui_language, "Configure"))
        self.configure_yt_button.clicked.connect(self.on_configure_yt)
        yt_layout.addWidget(self.configure_yt_button)
        connections_grid.addLayout(yt_layout, 0, 0)

        twitch_layout = QHBoxLayout()
        twitch_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        twitch_label = QLabel("Twitch")
        twitch_layout.addWidget(twitch_label)
        self.twitch_input = QLineEdit()
        self.twitch_input.returnPressed.connect(self.on_connect_twitch)
        self.twitch_input.setPlaceholderText("https://www.twitch.tv/CHANNEL_NAME or CHANNEL_NAME")
        twitch_layout.addWidget(self.twitch_input)
        self.connect_twitch_button = QPushButton(_(self.gui_language, "Connect"))
        self.connect_twitch_button.clicked.connect(self.on_connect_twitch)
        twitch_layout.addWidget(self.connect_twitch_button)
        self.configure_twitch_button = QPushButton(_(self.gui_language, "Configure"))
        self.configure_twitch_button.clicked.connect(self.on_configure_twitch)
        twitch_layout.addWidget(self.configure_twitch_button)
        connections_grid.addLayout(twitch_layout, 0, 1)

    def setup_pause_button_color(self):
        palette = self.pause_button.palette()
        if self.is_paused:
            palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkRed)
            self.pause_button.setPalette(palette)
        else:
            self.pause_button.setPalette(self.style().standardPalette())

    def setup_central_widget(self):
        chat_header_layout = QHBoxLayout()
        chat_header_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        self.chat_header_label = QLabel(_(self.gui_language, "Message Log"))
        self.chat_header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        chat_header_layout.addWidget(self.chat_header_label, 1)

        control_layout = QHBoxLayout()
        chat_header_layout.addLayout(control_layout)
        control_layout.setContentsMargins(PADDING, 0, PADDING, 0)
        self.audio_indicator = QLabel("🟢")
        control_layout.addWidget(self.audio_indicator)
        self.pause_button = QPushButton(
            f"⏸️ {_(self.gui_language, "Stopped")}" if self.is_paused else f"▶️ {_(self.gui_language, "Playback...")}"
        )
        self.setup_pause_button_color()
        self.pause_button.clicked.connect(self.on_pause_clicked)
        control_layout.addWidget(self.pause_button)
        self.clr_queue_button = QPushButton(_(self.gui_language, "Clear queue"))
        self.clr_queue_button.clicked.connect(self.on_clear_queue)
        control_layout.addWidget(self.clr_queue_button)

        self.auto_scroll_checkbox = QCheckBox(_(self.gui_language, "Auto-scroll"))
        self.auto_scroll_checkbox.setChecked(self.auto_scroll)
        self.auto_scroll_checkbox.clicked.connect(self.toggle_auto_scroll)
        chat_header_layout.addWidget(self.auto_scroll_checkbox)

        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems([str(s) for s in range(8, 24, 2)])
        self.font_size_combo.setCurrentIndex(2)
        self.font_size_combo.currentIndexChanged.connect(self.font_size_changed)
        chat_header_layout.addWidget(self.font_size_combo)

        self.clear_log_button = QPushButton(_(self.gui_language, "Clear log"))
        self.clear_log_button.clicked.connect(self.clear_log)
        chat_header_layout.addWidget(self.clear_log_button)

        self.root_layout.addLayout(chat_header_layout)

        # Chat log (rich text for messenger-like messages)
        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        self.chat_text.setAcceptRichText(True)
        self.root_layout.addWidget(self.chat_text)

        # Timer to flush messages added from background threads
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush_pending_messages)
        self._flush_timer.start(100)

    def setup_status_bar(self):
        self.version_label = QLabel(f"v{APP_VERSION}")
        self.version_label.setContentsMargins(PADDING, 0, PADDING, 10)
        self.statusBar().addWidget(self.version_label)

        # == Stats labels ==
        self.stats_label = QLabel(self.stats_text())
        self.stats_label.setContentsMargins(PADDING, 0, PADDING, 10)
        self.statusBar().addWidget(self.stats_label, 1)

        self.voice_label = QLabel(self.status_voice_text())
        self.voice_label.setContentsMargins(PADDING, 0, 0, 5)
        self.statusBar().addWidget(self.voice_label)

        # == Volume slider ==
        self.vol_label = QLabel(_(self.gui_language, "Volume"))
        self.vol_label.setContentsMargins(PADDING, 0, 0, 5)
        self.statusBar().addWidget(self.vol_label)

        vol_slider = QSlider(Qt.Orientation.Horizontal)
        vol_slider.setMinimum(0)
        vol_slider.setMaximum(200)
        vol_slider.setValue(self.volume)
        vol_slider.valueChanged.connect(self.on_change_volume)
        self.statusBar().addWidget(vol_slider)

        self.vol_label_value = QLabel(f"{self.volume}%")
        self.vol_label_value.setContentsMargins(0, 0, PADDING, 5)
        self.statusBar().addWidget(self.vol_label_value)

        # == Speech rate slider ==
        self.speech_rate_label = QLabel(_(self.gui_language, "Speech rate"))
        self.speech_rate_label.setContentsMargins(0, 0, 0, 5)
        self.statusBar().addWidget(self.speech_rate_label)

        speech_rate_slider = QSlider(Qt.Orientation.Horizontal)
        speech_rate_slider.setMinimum(50)
        speech_rate_slider.setMaximum(150)
        speech_rate_slider.setValue(int(self.speech_rate * 100))  # Convert to 50-150 range
        speech_rate_slider.valueChanged.connect(self.speech_rate_changed)
        self.statusBar().addWidget(speech_rate_slider)

        self.speech_rate_label_value = QLabel(f"{self.speech_rate:.2f}x")
        self.speech_rate_label_value.setContentsMargins(0, 0, 0, 5)
        self.statusBar().addWidget(self.speech_rate_label_value)

    def setup_ui(self):
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections_grid()
        self.setup_central_widget()
        self.font_size_changed(2)

    # === UI event handlers ===

    def on_pause_clicked(self):
        self.is_paused = not self.is_paused
        self.pause_button.setText(
            f"⏸️ {_(self.gui_language, 'Stopped')}" if self.is_paused else f"▶️ {_(self.gui_language, 'Playback...')}"
        )
        self.setup_pause_button_color()
        if self.is_paused:
            sd.stop()
            self.statusBar().showMessage(_(self.gui_language, "Playback has been stopped"), 3000)
        else:
            self.statusBar().showMessage(_(self.gui_language, "Speech playback continued..."), 3000)

    def language_changed(self, lang):
        self.gui_language = lang
        self.save_settings()
        self.setup_menu_bar()
        self.on_change_stats()
        self.speech_rate_label.setText(_(self.gui_language, "Speech rate"))
        self.vol_label.setText(_(self.gui_language, "Volume"))
        self.voice_label.setText(self.status_voice_text())

        self.connect_yt_button.setText(_(self.gui_language, "Connected" if self.yt_is_connected else "Connect"))
        self.configure_yt_button.setText(_(self.gui_language, "Configure"))
        self.connect_twitch_button.setText(_(self.gui_language, "Connected" if self.twitch_is_connected else "Connect"))
        self.configure_twitch_button.setText(_(self.gui_language, "Configure"))

        self.chat_header_label.setText(_(self.gui_language, "Message Log"))
        self.auto_scroll_checkbox.setText(_(self.gui_language, "Auto-scroll"))
        self.clear_log_button.setText(_(self.gui_language, "Clear log"))
        self.pause_button.setText(
            f"⏸️ {_(self.gui_language, "Stopped")}" if self.is_paused else f"▶️ {_(self.gui_language, "Playback...")}"
        )
        self.clr_queue_button.setText((_(self.gui_language, "Clear queue")))

    def voice_changed(self, lang, voice):
        self.voice_language = lang
        self.voice = voice
        self.save_settings()
        self.setup_voice_menu()
        self.voice_label.setText(self.status_voice_text())
        self.init_silero()

    def speech_rate_changed(self, value):
        self.speech_rate = value / 100.0  # Convert to 0.50 - 1.50 range
        self.speech_rate_label_value.setText(f"{self.speech_rate:.2f}x")

    def on_change_volume(self, value):
        self.volume = value
        self.vol_label_value.setText(f"{self.volume}%")

    def font_size_changed(self, index):
        font_size = int(self.font_size_combo.currentText())
        font = self.chat_text.font()
        font.setPointSize(font_size)
        self.chat_text.setFont(font)

    def on_configure_twitch(self):
        QMessageBox.information(self, "Info", "Configure Twitch connection (not implemented yet)")

    def on_connect_twitch(self):
        if self.twitch_is_connected:
            self.twitch_is_connected = False
            self.connect_twitch_button.setText(_(self.gui_language, "Connect"))
            self.connect_twitch_button.setPalette(self.style().standardPalette())
        else:
            self.twitch_is_connected = True
            self.connect_twitch_button.setText(_(self.gui_language, "Connected"))
            palette = self.connect_twitch_button.palette()
            palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGreen)
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            self.connect_twitch_button.setPalette(palette)

    def on_configure_yt(self):
        self.setup_dlg_yt_conf()

    def on_click_yt_connect(self):
        if self.yt_is_connected:
            self.on_disconnect_yt()
        else:
            self.on_connect_yt()

    def on_connect_yt(self):
        if self.yt_credentials is None:
            self.on_click_yt_edit_credential()
            return

        self.parse_yt_video_id()

        if not self.yt_video_id:
            return

        if not self.silero_model:
            res = QMessageBox.question(
                self, _(self.gui_language, "Silero not loaded"), _(self.gui_language, "note_tts_not_loaded")
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        try:
            self.youtube = build("youtube", "v3", developerKey=self.yt_credentials)
            self.yt_chat_id = self.get_chat_id_yt()
            if not self.yt_chat_id:
                return

            self.add_sys_message(
                author="YouTube",
                text=_(self.gui_language, "connection_success"),
                status="success",
            )
            self.yt_is_connected = True
            loop_yt = threading.Thread(target=self.connection_loop_yt, daemon=True)
            loop_yt.start()

        except Exception as e:
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "connection_failed")}. {str(e)}",
                status="error",
            )
            return

        if self.yt_is_connected:
            self.connect_yt_button.setText(_(self.gui_language, "Connected"))
            palette = self.connect_yt_button.palette()
            palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGreen)
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            self.connect_yt_button.setPalette(palette)

    def on_disconnect_yt(self):
        self.yt_is_connected = False
        self.connect_yt_button.setText(_(self.gui_language, "Connect"))
        self.connect_yt_button.setPalette(self.style().standardPalette())

    def on_stop_words_action(self):
        dlg = QDialog(self)
        dlg.setMinimumSize(600, 250)
        dlg.setWindowTitle(_(self.gui_language, "Stop words"))

        root_widget = QWidget(dlg)
        root_widget.setMinimumSize(600, 100)
        root_layout = QVBoxLayout(root_widget)

        self.stop_words_text = QPlainTextEdit()
        self.stop_words_text.setPlainText("\n".join(self.stop_words))
        root_layout.addWidget(self.stop_words_text)
        save_stop_words_btn = QPushButton(_(self.gui_language, "Save stop words"))
        save_stop_words_btn.clicked.connect(self.on_save_stop_words)
        root_layout.addWidget(save_stop_words_btn)

        dlg.exec()

    def on_save_stop_words(self):
        content = self.stop_words_text.toPlainText().strip()
        self.stop_words = tuple([w.strip() for w in content.splitlines() if w.strip()])
        self.save_settings()
        self.statusBar().showMessage(
            f"{_(self.gui_language, "Saved")} {len(self.stop_words)} {_(self.gui_language, "stop words")}", 3000
        )

    def on_click_yt_edit_credential(self):
        self.google_api_key_input.returnPressed.connect(self.on_click_yt_save_settings)
        self.google_api_key_input.setText(self.yt_credentials)
        self.google_api_key_input.setReadOnly(False)
        self.google_api_key_button.clicked.disconnect()
        self.google_api_key_button.clicked.connect(self.on_click_yt_save_settings)
        self.google_api_key_button.setText(_(self.gui_language, "Save"))

    def on_click_yt_save_settings(self):
        self.yt_credentials = f"{str(self.google_api_key_input.text())}"
        self.save_settings()
        self.google_api_key_input.setText("*" * len(self.yt_credentials))
        self.google_api_key_input.setReadOnly(True)
        self.google_api_key_button.clicked.disconnect()
        self.google_api_key_button.clicked.connect(self.on_click_yt_edit_credential)
        self.google_api_key_button.setText(_(self.gui_language, "Edit"))

    def on_delays_settings_action(self):
        dlg = QDialog(self)
        # dlg.setMinimumSize(600, 300)
        dlg.setWindowTitle(_(self.gui_language, "Delays and processing"))

        root_widget = QWidget(dlg)
        root_widget.setMinimumSize(600, 300)
        root_layout = QVBoxLayout(root_widget)

        # Queue depth

        queue_depth_v_layout = QVBoxLayout()
        queue_depth_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(queue_depth_v_layout)

        self.queue_depth_label_desc = QLabel(_(self.gui_language, "queue_depth_desc"))
        queue_depth_v_layout.addWidget(self.queue_depth_label_desc)

        queue_depth_layout = QHBoxLayout()
        queue_depth_v_layout.addLayout(queue_depth_layout)

        queue_depth_slider = QSlider(Qt.Orientation.Horizontal)
        queue_depth_layout.addWidget(queue_depth_slider)
        queue_depth_slider.setMinimum(1)
        queue_depth_slider.setMaximum(100)
        queue_depth_slider.setValue(self.buffer_maxsize)
        queue_depth_slider.valueChanged.connect(self.on_change_queue_depth)

        self.queue_depth_label_value = QLabel(str(self.buffer_maxsize))
        queue_depth_layout.addWidget(self.queue_depth_label_value)

        # Max message length

        msg_len_v_layout = QVBoxLayout()
        msg_len_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(msg_len_v_layout)

        self.msg_len_label_desc = QLabel(_(self.gui_language, "Max message length"))
        msg_len_v_layout.addWidget(self.msg_len_label_desc)

        msg_len_layout = QHBoxLayout()
        msg_len_v_layout.addLayout(msg_len_layout)

        msg_len_slider = QSlider(Qt.Orientation.Horizontal)
        msg_len_layout.addWidget(msg_len_slider)
        msg_len_slider.setMinimum(10)
        msg_len_slider.setMaximum(1000)
        msg_len_slider.setValue(self.max_msg_length)
        msg_len_slider.valueChanged.connect(self.on_change_queue_max_msg_len)

        self.msg_len_label_value = QLabel(str(self.max_msg_length))
        msg_len_layout.addWidget(self.msg_len_label_value)

        # Speech delay

        speech_delay_v_layout = QVBoxLayout()
        # speech_delay_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(speech_delay_v_layout)

        self.speech_delay_label_desc = QLabel(_(self.gui_language, "Delay between messages"))
        speech_delay_v_layout.addWidget(self.speech_delay_label_desc)

        speech_delay_layout = QHBoxLayout()
        speech_delay_v_layout.addLayout(speech_delay_layout)

        speech_delay_slider = QSlider(Qt.Orientation.Horizontal)
        speech_delay_layout.addWidget(speech_delay_slider)
        speech_delay_slider.setMinimum(5)
        speech_delay_slider.setMaximum(50)
        speech_delay_slider.setValue(int(self.speech_delay * 10))
        speech_delay_slider.valueChanged.connect(self.on_change_queue_speech_delay)

        self.speech_delay_label_value = QLabel(str(self.speech_delay))
        speech_delay_layout.addWidget(self.speech_delay_label_value)

        dlg.finished.connect(self.save_settings)
        dlg.exec()

    def on_change_queue_speech_delay(self, value):
        self.speech_delay = value / 10
        self.speech_delay_label_value.setText(f"{float(self.speech_delay):.2f}")

    def on_change_queue_max_msg_len(self, value):
        self.max_msg_length = value
        self.msg_len_label_value.setText(str(self.max_msg_length))

    def on_change_queue_depth(self, value):
        self.buffer_maxsize = value
        self.queue_depth_label_value.setText(str(self.buffer_maxsize))

    def on_change_stats(self):
        self.stats_label.setText(self.stats_text())

    def on_clear_queue(self):
        self.audio_queue.clear()
        self.on_change_stats()
        self.statusBar().showMessage(_(self.gui_language, "Queue cleared"), 3000)

    def toggle_add_accents(self, checked):
        self.add_accents = checked

    def toggle_filter_repeats(self, checked):
        self.filter_repeats = checked

    def toggle_read_author_names(self, checked):
        self.read_author_names = checked

    def toggle_subscribers_only(self, checked):
        self.subscribers_only = checked

    def toggle_auto_scroll(self, checked):
        self.auto_scroll = checked

    def clear_log(self):
        self.chat_text.clear()
        self.statusBar().showMessage(_(self.gui_language, "Log cleared"), 3000)

    def export_log(self, choice):
        if choice == "html":
            log = self.chat_text.toHtml()
            res = "Html Files (*.html);;All Files (*)"
        elif choice == "md":
            log = self.chat_text.toPlainText()
            res = "Markdown Files (*.md);;All Files (*)"
        else:
            log = self.chat_text.toPlainText()
            res = "Text Files (*.txt);;All Files (*)"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", res)

        if file_path:
            file = QFile(file_path)
            if file.open(QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Text):
                file.write(log.encode("utf-8"))
                file.close()
                self.statusBar().showMessage(f"{_(self.gui_language, "File saved")}: {file_path}", 3000)

    # === Helper methods ===

    def stats_text(self):
        return (
            f"{_(self.gui_language, 'Messages')}: {self.messages_stats['messages_count']} | "
            f"{_(self.gui_language, 'Spoken')}: {self.messages_stats['spoken_count']} | "
            f"{_(self.gui_language, 'Spam')}: {self.messages_stats['spam_count']} | "
            f"{_(self.gui_language, 'In queue')}: {len(self.audio_queue)}"
        )

    def status_voice_text(self):
        return f"{_(self.gui_language, 'Voice')}: {_(self.gui_language, self.voice_language)} - {self.voice}"

    def colored_text(self, text, color=None, background=None):
        style = ""
        if color:
            style += f"color: {color};"
        if background:
            style += f"background-color: {background};"
        safe_text = html.escape(str(text)).replace("\n", "<br>")
        return f'<span style="{style}">{safe_text}</span>'

    def add_sys_message(self, author, text, status="default"):
        status_colors = {
            "default": None,
            "warning": "orange",
            "error": "darkRed",
            "success": "darkGreen",
        }

        system_text = _(self.gui_language, "System")
        return self.add_message(platform=system_text, author=author, text=text, background=status_colors[status])

    def add_message(self, platform, author, text, color=None, background=None):
        self.messages_stats["messages_count"] += 1
        self.on_change_stats()

        # If called from a non-main thread, enqueue for the GUI flush timer
        if threading.current_thread() is not threading.main_thread():
            self._pending_messages.append((platform, author, text, color, background))
            return

        # On main thread, insert immediately
        self._insert_message(platform, author, text, color, background)

    def _flush_pending_messages(self):
        while len(self._pending_messages) > 0:
            platform, author, text, color, background = self._pending_messages.popleft()
            self._insert_message(platform, author, text, color, background)

    def _insert_message(self, platform, author, text, color=None, background=None):
        time_str = datetime.now().strftime("%H:%M:%S")
        safe_author = html.escape(str(author))
        avatar_text = safe_author[:1].upper() if safe_author else "?"
        avatar_bg, avatar_fg = _avatar_colors_from_name(safe_author)
        white_color = "#fff"

        message = f"""
<table cellpadding="5" cellspacing="10" width="100%">
    <tr>
        <td width="48" valign="top">
            <div
            style="
                width: 40px;
                height: 40px;
                max-height: 40px;
                text-align: center;
                vertical-align: middle;
                font-weight: bold;
                line-height: 40px;
                border-radius: 20px;
                background:{avatar_bg};
                color:{avatar_fg};
                overflow: hidden;
            "
            >
                <span style="font-size: 40px; color: {color or white_color};">{avatar_text}</span>
            </div>
        </td>
        <td valign="middle" bgcolor="{background or "#444"}">
            <div>
                <div>
                    <b style="color: {color or white_color};">{safe_author}</b>
                    <span style="font-weight: normal; color: {color or white_color};">[{time_str}] [{platform}]</span>
                </div>
                {self.colored_text(text, color=color or white_color)}
            </div>
        </td>
    </tr>
</table>
        """

        try:
            was_readonly = self.chat_text.isReadOnly()
            if was_readonly:
                self.chat_text.setReadOnly(False)

            cursor = self.chat_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml(message)
            cursor.insertBlock()
            self.chat_text.setTextCursor(cursor)

            if was_readonly:
                self.chat_text.setReadOnly(True)

            if self.auto_scroll:
                self.chat_text.verticalScrollBar().setValue(self.chat_text.verticalScrollBar().maximum())
        except Exception:
            pass

    def save_settings(self):
        settings = {
            "gui_language": self.gui_language,
            "voice_language": self.voice_language,
            "voice": self.voice,
            "volume": self.volume,
            "speech_rate": self.speech_rate,
            "speech_delay": self.speech_delay,
            "auto_scroll": self.auto_scroll,
            "add_accents": self.add_accents,
            "remove_emojis": self.remove_emojis,
            "remove_links": self.remove_links,
            "filter_repeats": self.filter_repeats,
            "read_author_names": self.read_author_names,
            "ignore_system_messages": self.ignore_system_messages,
            "subscribers_only": self.subscribers_only,
            "max_msg_length": self.max_msg_length,
            "buffer_maxsize": self.buffer_maxsize,
            "yt_credentials": self.yt_credentials,
            "twitch_credentials": self.twitch_credentials,
            "stop_words": self.stop_words,
        }
        with open("settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(_(self.gui_language, "Settings saved"), 3000)

    def load_settings(self):
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
                self.gui_language = settings.get("gui_language", self.gui_language)
                self.voice_language = settings.get("voice_language", self.voice_language)
                self.voice = settings.get("voice", self.voice)
                self.volume = settings.get("volume", self.volume)
                self.speech_rate = settings.get("speech_rate", self.speech_rate)
                self.speech_delay = settings.get("speech_delay", self.speech_delay)
                self.auto_scroll = settings.get("auto_scroll", self.auto_scroll)
                self.add_accents = settings.get("add_accents", self.add_accents)
                self.remove_emojis = settings.get("remove_emojis", self.remove_emojis)
                self.remove_links = settings.get("remove_links", self.remove_links)
                self.filter_repeats = settings.get("filter_repeats", self.filter_repeats)
                self.read_author_names = settings.get("read_author_names", self.read_author_names)
                self.ignore_system_messages = settings.get("ignore_system_messages", self.ignore_system_messages)
                self.subscribers_only = settings.get("subscribers_only", self.subscribers_only)
                self.buffer_maxsize = settings.get("buffer_maxsize", self.buffer_maxsize)
                self.max_msg_length = settings.get("max_msg_length", self.max_msg_length)
                self.yt_credentials = settings.get("yt_credentials", None)
                self.twitch_credentials = settings.get("twitch_credentials", None)
                self.stop_words = tuple(settings.get("stop_words", []))
        except FileNotFoundError:
            pass  # No settings file, use defaults

    def closeEvent(self, event):
        self.save_settings()
        sd.stop()
        super().closeEvent(event)

    def init_silero(self):
        self.add_sys_message(author="Silero", text=_(self.gui_language, "silero_loading"))
        try:
            set_grad_enabled(False)
            if self.voice_language == "Russian":
                self.silero_model, example_text = hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v5_ru",
                    trust_repo=True,
                )
            else:
                self.silero_model, example_text = hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="en",
                    speaker="v3_en",
                    trust_repo=True,
                )
            self.add_sys_message(author="Silero", text=_(self.gui_language, "silero_loaded"), status="success")
            self.speak(_(self.voice_language, "silero_loaded"))

        except Exception as e:
            import traceback

            traceback.print_exception(e)
            self.add_sys_message(author="Silero", text=f"{_(self.gui_language, "silero_failed")}. {e}", status="error")
            return False

    def clean_message(self, text):
        """Clean message from garbage"""

        if hasattr(self, "filter_links_var") and self.filter_links_var.get():
            text = re.sub(r"https?://\S+", "", text)
            text = re.sub(r"www\.\S+", "", text)

        if hasattr(self, "filter_emojis_var") and self.filter_emojis_var.get():
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"
                "\U0001f300-\U0001f5ff"
                "\U0001f680-\U0001f6ff"
                "\U0001f1e0-\U0001f1ff"
                "\U00002702-\U000027b0"
                "\U000024c2-\U0001f251"
                "]+",
                flags=re.UNICODE,
            )
            text = emoji_pattern.sub(r"", text)
            text = re.sub(r"[^\w\s\.\,\!\?\-\:\'\"\(\)]", " ", text)

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        try:
            min_len = int(self.min_length_var.get())
            max_len = int(self.max_length_var.get())
        except:
            min_len = 2
            max_len = 200

        if len(text) < min_len:
            return

        if len(text) > max_len:
            text = text[:max_len] + "..."

        return text

    def get_msg_hash(self, platform, author, message):
        return hashlib.md5(f"{platform}:{author}:{message}".encode()).hexdigest()

    def is_spam(self, platform, author, message):
        """Check for spam"""
        if not self.filter_repeats:
            return False

        message_hash = self.get_msg_hash(platform, author, message)

        if message_hash in self.spam_hash_set:
            self.messages_stats["spam_count"] += 1
            return True

        if len(self.spam_hash_set) > 500:
            self.spam_hash_set = set(list(self.spam_hash_set)[-500:])

        return False

    def contains_stop_words(self, text):
        """Check for stop words"""
        text_lower = text.lower()
        for word in self.stop_words:
            if word.lower() in text_lower:
                return True
        return False

    def convert_numbers_to_words(self, text):
        """Convert numbers to text representation"""

        def replace_number(match):
            num = match.group()
            try:
                if "." in num:
                    parts = num.split(".")
                    integer_part = num2words(int(parts[0]), lang=LANG_CODES.get(self.voice_language, "en"))
                    fractional_part = num2words(int(parts[1]), lang=LANG_CODES.get(self.voice_language, "en"))
                    point_word = "point" if self.voice_language == "en" else "запятая"
                    return f"{integer_part} {point_word} {fractional_part}"
                else:
                    return num2words(int(num), lang=LANG_CODES.get(self.voice_language, "en"))
            except Exception as e:
                self.add_sys_message(
                    author="_translate_text()",
                    text=f"{_(self.gui_language, 'Error convert num to word')}. {e}",
                    status="error",
                )
                return num

        number_pattern = r"-?\d+(?:[.,]\d+)?"
        converted_text = re.sub(number_pattern, replace_number, text)
        return converted_text

    def process_chat_message(self, msg_id, platform, author, message):
        if msg_id in self.processed_messages:
            return
        self.processed_messages.add(str(msg_id))

        cleaned_text = self.clean_message(message)
        if not cleaned_text:
            return

        if self.contains_stop_words(cleaned_text):
            self.add_message(platform=platform, author=author, text=message, color="gray")
            return

        if self.is_spam(platform, author, message):
            self.add_message(platform=platform, author=author, text=message, color="gray")
            return

        cleaned_text = self._translate_text(message, LANG_CODES[self.voice_language])

        self.add_message(platform=platform, author=author, text=cleaned_text)

        if len(cleaned_text) > self.max_msg_length:
            cleaned_text = cleaned_text[: self.max_msg_length] + "..."
        cleaned_text = self.convert_numbers_to_words(cleaned_text)

        if self.read_author_names:
            cleaned_text = f"{author} {_(self.voice_language, 'said')}: {cleaned_text}"

        self.message_buffer.append(cleaned_text)

    def _translate_text(self, text, dest):
        try:
            q = multiprocessing.Queue()
            _proc_translate_external(q, text, dest)
            return q.get_nowait()
        except Exception as e:
            self.add_sys_message(
                author="_translate_text()",
                text=f"{_(self.gui_language, 'Failed to translate text')}. {e}",
                status="error",
            )
            return text

    # == Audio processing ==

    def text_to_speech(self, text):
        """Convert text to speech using Silero"""
        try:
            if self.silero_model is not None:
                with no_grad():
                    return self.silero_model.apply_tts(
                        text=text,
                        speaker=self.voice,
                        # sample_rate=self.sample_rate,
                        put_accent=self.add_accents,
                        # put_yo=self.put_yo,
                    )
        except Exception as e:
            self.add_sys_message(
                author="text_to_speech()",
                text=f"{_(self.gui_language, "Error convert message")}. {e}",
                status="error",
            )

    def postprocess_audio(self, audio):
        """Postprocess audio: convert to numpy, normalize, apply volume and speed"""
        try:
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            else:
                audio = np.asarray(audio)
        except Exception:
            audio = np.asarray(audio)

        if audio.size == 0:
            return audio

        max_abs = float(np.max(np.abs(audio))) if np.any(np.abs(audio)) else 1.0
        if max_abs == 0:
            max_abs = 1.0
        audio = audio / max_abs

        # Apply volume (single scaling)
        audio = audio * (self.volume / 100.0)

        if self.speech_rate != 1.0:
            num_samples = max(1, int(len(audio) / self.speech_rate))
            audio = resample(audio, num_samples)

        return audio

    def speak(self, text):
        """Main TTS method"""
        try:
            audio = self.text_to_speech(text)
            if audio is None:
                return
            audio_numpy = self.postprocess_audio(audio)
            if len(audio_numpy) > 0:
                self.audio_queue.append(audio_numpy)
                return True
            return False
        except Exception as e:
            self.add_sys_message(
                author="speak()", text=f"{_(self.gui_language, "Audio playback error")}. {e}", status="error"
            )

    def play_audio(self, audio_to_play):
        try:
            self.audio_indicator.setText("🔴")
            sd.play(audio_to_play)
            sd.wait()
        except Exception as e:
            self.add_sys_message(
                author="play_audio()",
                text=f"{_(self.gui_language, 'Audio playback error')}. {e}",
                status="error",
            )
        finally:
            self.audio_indicator.setText("🟢")

    def process_msg_buffer_loop(self):
        while True:
            try:
                if len(self.message_buffer) > 0:
                    text = self.message_buffer.popleft()
                    self.speak(text)
            except Exception as e:
                self.add_sys_message(
                    author="process_msg_buffer_loop()",
                    text=f"{_(self.gui_language, "Error process message buffer")}. {e}",
                    status="error",
                )
            finally:
                sleep(0.2)

    def process_audio_loop(self):
        """Main loop to process audio queue"""
        while True:
            try:
                if len(self.audio_queue) > 0 and self.is_paused is False:
                    audio_data = self.audio_queue.popleft()
                    self.play_audio(audio_data)

                    del audio_data
                    gc.collect()

                    self.messages_stats["spoken_count"] += 1
                    self.on_change_stats()

            except Exception as e:
                self.add_sys_message(
                    author="process_audio_loop()",
                    text=f"{_(self.gui_language, "Audio queue error")}. {e}",
                    status="error",
                )
            finally:
                sleep(self.speech_delay)

            sleep(0.1)

    # == YouTube chat handling ==

    def connection_loop_yt(self):
        """Loop to maintain connection to YouTube chat"""

        errors = 0
        page_token = None
        try:
            while self.yt_is_connected and bool(self.yt_chat_id):
                if len(self.audio_queue) > 0:
                    sleep(3)
                    continue
                page_token, is_error = self.fetch_messages_yt(page_token)
                if is_error:
                    errors += 1
                else:
                    errors = 0

                if errors >= 5:
                    self.on_disconnect_yt()
        except Exception as e:
            self.add_sys_message(
                author="YouTube",
                text=f"Connection loop error. {str(e)}",
                status="error",
            )
        finally:
            self.on_disconnect_yt()

    def parse_yt_video_id(self):
        self.yt_video_id = self.yt_video_input.text()
        try:
            if not self.yt_video_id:
                QMessageBox.warning(self, "", _(self.gui_language, "Please enter video ID or URL"))
                return

            if "youtube.com" in self.yt_video_id or "youtu.be" in self.yt_video_id:
                parsed = urllib.parse.urlparse(self.yt_video_id)
                if "youtu.be" in parsed.netloc:
                    self.yt_video_id = parsed.path[1:]
                elif "watch" in parsed.path:
                    query = urllib.parse.parse_qs(parsed.query)
                    self.yt_video_id = query.get("v", [None])[0]
                elif "embed" in parsed.path:
                    self.yt_video_id = parsed.path.split("/")[-1]
        except Exception as e:
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "not_determine_video_id")}. {str(e)}",
                status="error",
            )
        if not self.yt_video_id:
            self.add_sys_message(
                author="YouTube",
                text=_(self.gui_language, "not_determine_video_id"),
                status="error",
            )
        return self.yt_video_id

    def get_chat_id_yt(self):
        """Get chat ID"""
        try:
            response = self.youtube.videos().list(part="liveStreamingDetails", id=self.yt_video_id).execute()

            if response.get("items"):
                details = response["items"][0].get("liveStreamingDetails", {})
                return details.get("activeLiveChatId")
            else:
                self.add_sys_message(
                    author="YouTube",
                    text=f"{_(self.gui_language, "video_not_found")}",
                    status="error",
                )
                self.on_disconnect_yt()

        except HttpError as e:
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "not_determine_chat_id")}. {e.reason}",
                status="error",
            )
            self.on_disconnect_yt()
        except Exception as e:
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "not_determine_chat_id")}. {str(e)}",
                status="error",
            )
            self.on_disconnect_yt()

    def fetch_messages_yt(self, page_token=None):
        """Fetch messages from YouTube chat"""
        if not self.yt_is_connected:
            return
        self.is_fetching = True
        next_token = None
        is_error = False
        delay = 3

        try:
            response = (
                self.youtube.liveChatMessages()
                .list(
                    liveChatId=self.yt_chat_id,
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
                    "displayName", snippet.get("authorDisplayName", _(self.gui_language, "Anonymous"))
                )

                if not author or author.strip() == "":
                    author = _(self.gui_language, "Anonymous")
                if str(author).startswith("@"):
                    author = str(author)[1:]

                message = snippet.get("displayMessage", "")
                is_member = (
                    author_details.get("isChatOwner", False)
                    or author_details.get("isChatSponsor", False)
                    or author_details.get("isChatModerator", False)
                )
                if self.subscribers_only and not is_member:
                    continue

                self.process_chat_message(msg_id=msg_id, platform="YouTube", author=author, message=message)

            # respect polling interval returned by API when available
            poll_ms = response.get("pollingIntervalMillis")
            try:
                delay = delay + max(1, int(poll_ms) / 1000) if poll_ms is not None else 5
            except Exception:
                pass

        except HttpError as e:
            is_error = True
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "error_fetch_messages")}. {e.reason}",
                status="error",
            )
        except Exception as e:
            is_error = True
            self.add_sys_message(
                author="YouTube",
                text=f"{_(self.gui_language, "error_fetch_messages")}. {str(e)}",
                status="error",
            )

        finally:
            self.is_fetching = False
            sleep(delay)

        return next_token, is_error


@lru_cache
def _avatar_colors_from_name(name: str):
    """Deterministic avatar background and foreground color from author name."""
    if not name:
        return "#777777", "#ffffff"

    # Use MD5 hash to get stable value from name
    digest = hashlib.md5(name.encode("utf-8")).digest()
    # Take 3 bytes to form a value for hue
    val = int.from_bytes(digest[:3], "big")
    hue = val % 360

    # HLS -> colorsys uses H,L,S where H in [0,1]
    h = hue / 360.0
    l = 0.50
    s = 0.65
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r_i, g_i, b_i = int(r * 255), int(g * 255), int(b * 255)
    bg = f"#{r_i:02x}{g_i:02x}{b_i:02x}"

    # Choose contrasting text color based on luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#000000" if lum > 0.6 else "#ffffff"
    return bg, fg


def main():
    app = QApplication(sys.argv)

    # Optional: Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":

    main()
