import os
from random import randint
import sys
from collections import defaultdict, deque
from datetime import datetime
import gc
import json
import html
import re
import threading
from time import sleep
import hashlib

from detoxify import Detoxify
from num2words import num2words
import sounddevice as sd
from torch import hub, no_grad, set_grad_enabled, set_num_threads
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
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QFile, QIODevice
from PyQt6.QtGui import QFont, QAction, QPalette, QTextCursor, QIcon
from PyQt6.QtGui import QShortcut, QKeySequence
from scipy.signal import resample
import numpy as np
from googletrans import Translator

from app.schema import MessageStatsTD, TwitchCredentialsTD
from app.translations import DEFAULT_LANGUAGE, TRANSLATIONS, _, translate_text
from app.twitch.auth_worker import AuthWorker
from app.twitch.chat_listener import TwitchChatListener
from app.constants import (
    APP_VERSION,
    APP_NAME,
    PADDING,
    DEFAULT_BUFFER_SIZE,
    VOICES,
    MODELS,
)
from app.utils import (
    avatar_colors_from_name,
    clean_message,
    clear_detoxify_checkpoint_cache,
    configure_torch_hub_cache,
    ensure_stdio_streams,
    find_cached_detoxify_checkpoint,
    find_cached_silero_repo,
    get_settings_path,
    prefer_cached_silero_package,
    resource_path,
)
from app.youtube.chat_parser import YouTubeChatParser

size_policy_fixed = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
window_flag_fixed = (
    Qt.WindowType.Window
    | Qt.WindowType.CustomizeWindowHint
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowCloseButtonHint
)
twitch_default_credentials = TwitchCredentialsTD(
    client_id=None, access=None, refresh=None, nickname=None
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        icon = QIcon(resource_path("img/icon.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1200, 600)

        self.root_widget = QWidget()
        self.setCentralWidget(self.root_widget)
        self.root_layout = QVBoxLayout(self.root_widget)

        self.language = DEFAULT_LANGUAGE
        self.voice_language = DEFAULT_LANGUAGE
        self.voice = "random"
        self.volume = 100
        self.speech_rate = 1.00
        self.speech_delay = 1.5
        self.is_paused = False
        self.min_msg_length = 5
        self.max_msg_length = 200
        self.toxic_sense = 0.25

        self.auto_scroll = True
        self.add_accents = True
        self.filter_repeats = True
        self.read_author_names = False
        self.read_platform_names = False
        self.subscribers_only = False
        self.auto_translate = False
        self.stop_words = tuple()
        self.chat_only_mode = False

        # Connections
        self.youtube = None
        self.yt_credentials = None
        self.yt_is_connected = False

        self.twitch = None
        self.twitch_credentials: TwitchCredentialsTD = twitch_default_credentials
        self.twitch_is_connected = False

        self.messages_stats: MessageStatsTD = defaultdict(int)

        # Message queue
        self.buffer_maxsize = DEFAULT_BUFFER_SIZE
        self._pending_messages = deque()
        self.spam_hash_set = set()
        self.spam_hash_queue = deque(maxlen=500)
        self.processed_messages = set()

        self.load_settings()

        self.message_buffer = deque(maxlen=self.buffer_maxsize)
        self.audio_queue = deque(maxlen=self.buffer_maxsize)

        self.setup_ui()

        self.detox_model = None
        self.model = None
        self.model_lock = threading.Lock()
        self.translator = Translator()
        self.translator_lock = threading.Lock()

        configure_torch_hub_cache()
        set_num_threads(2)
        set_grad_enabled(False)

        threading.Thread(target=self.init_silero, daemon=True).start()
        if self.toxic_sense < 1.0:
            threading.Thread(target=self.init_detoxify, daemon=True).start()
        threading.Thread(target=self.process_audio_loop, daemon=True).start()
        threading.Thread(target=self.process_msg_buffer_loop, daemon=True).start()

    # === UI setup ===

    def setup_voice_menu(self):
        self.voice_menu.clear()
        for voice_lang in VOICES.keys():
            voice_lang_menu = self.voice_menu.addMenu(voice_lang)

            voices = ["random"] + list(VOICES[voice_lang])
            for voice in voices:
                voice_action = QAction(voice, self)
                voice_action.setCheckable(True)
                voice_action.setChecked(
                    voice == self.voice and voice_lang == self.voice_language
                )
                voice_action.triggered.connect(
                    lambda checked, l=voice_lang, v=voice: self.voice_changed(l, v)
                )
                voice_lang_menu.addAction(voice_action)

        add_accents_action = QAction(_(self.language, "Add accents"), self)
        add_accents_action.setCheckable(True)
        add_accents_action.setChecked(self.add_accents)
        add_accents_action.triggered.connect(self.toggle_add_accents)
        self.voice_menu.addAction(add_accents_action)

    def setup_language_menu(self):
        self.language_menu.clear()
        for lang in TRANSLATIONS.keys():
            lang_action = QAction(lang, self)
            lang_action.setCheckable(True)
            lang_action.setChecked(lang == self.language)
            lang_action.triggered.connect(
                lambda checked, l=lang: self.language_changed(l)
            )
            self.language_menu.addAction(lang_action)

    def setup_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu(_(self.language, "File"))

        export_log_action = QMenu(_(self.language, "Export log"), file_menu)
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

        self.language_menu = menu_bar.addMenu(_(self.language, "Language"))
        self.setup_language_menu()

        self.voice_menu = menu_bar.addMenu(_(self.language, "Speech configuration"))
        self.setup_voice_menu()

        msg_settings_menu = menu_bar.addMenu(_(self.language, "Message Settings"))

        stop_words_action = QAction(_(self.language, "Stop words"), msg_settings_menu)
        stop_words_action.triggered.connect(self.on_stop_words_action)
        msg_settings_menu.addAction(stop_words_action)

        delays_action = QAction(
            _(self.language, "Delays and processing"), msg_settings_menu
        )
        delays_action.triggered.connect(self.on_delays_settings_action)
        msg_settings_menu.addAction(delays_action)

        filter_repeats_action = QAction(
            _(self.language, "Filter repeats"), msg_settings_menu
        )
        filter_repeats_action.setCheckable(True)
        filter_repeats_action.setChecked(self.filter_repeats)
        filter_repeats_action.triggered.connect(self.toggle_filter_repeats)
        msg_settings_menu.addAction(filter_repeats_action)

        read_authors_action = QAction(
            _(self.language, "Read author names"), msg_settings_menu
        )
        read_authors_action.setCheckable(True)
        read_authors_action.setChecked(self.read_author_names)
        read_authors_action.triggered.connect(self.toggle_read_author_names)
        msg_settings_menu.addAction(read_authors_action)

        read_platform_action = QAction(
            _(self.language, "Read platform name"), msg_settings_menu
        )
        read_platform_action.setCheckable(True)
        read_platform_action.setChecked(self.read_platform_names)
        read_platform_action.triggered.connect(self.toggle_read_platform_names)
        msg_settings_menu.addAction(read_platform_action)

        subscribers_only_action = QAction(
            _(self.language, "Subscribers only"), msg_settings_menu
        )
        subscribers_only_action.setCheckable(True)
        subscribers_only_action.setChecked(self.subscribers_only)
        subscribers_only_action.triggered.connect(self.toggle_subscribers_only)
        msg_settings_menu.addAction(subscribers_only_action)

        auto_translate_action = QAction(
            _(self.language, "Translate messages"), msg_settings_menu
        )
        auto_translate_action.setCheckable(True)
        auto_translate_action.setChecked(self.auto_translate)
        auto_translate_action.triggered.connect(self.toggle_auto_translate)
        msg_settings_menu.addAction(auto_translate_action)

        # view_menu = menu_bar.addMenu("View")
        # self.chat_only_action = QAction("Chat only mode", self)
        # self.chat_only_action.setCheckable(True)
        # self.chat_only_action.setChecked(self.chat_only_mode)
        # self.chat_only_action.triggered.connect(self.toggle_chat_only_mode)
        # view_menu.addAction(self.chat_only_action)

    def setup_connections_grid(self):
        connections_grid = QGridLayout()
        self.connections_widget = QWidget()
        self.connections_widget.setLayout(connections_grid)
        self.root_layout.addWidget(self.connections_widget)

        yt_layout = QHBoxLayout()
        yt_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        yt_label = QLabel("YouTube")
        yt_layout.addWidget(yt_label)
        self.yt_video_input = QLineEdit()
        self.yt_video_input.returnPressed.connect(self.on_click_yt_connect)
        self.yt_video_input.setPlaceholderText(
            "https://www.youtube.com/watch?v=VIDEO_ID or VIDEO_ID"
        )
        yt_layout.addWidget(self.yt_video_input)
        self.connect_yt_button = QPushButton(_(self.language, "Connect"))
        self.connect_yt_button.clicked.connect(self.on_click_yt_connect)
        yt_layout.addWidget(self.connect_yt_button)
        # self.configure_yt_button = QPushButton(_(self.language, "Configure"))
        # self.configure_yt_button.clicked.connect(self.on_configure_yt)
        # yt_layout.addWidget(self.configure_yt_button)
        connections_grid.addLayout(yt_layout, 0, 0)

        twitch_layout = QHBoxLayout()
        twitch_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        twitch_label = QLabel("Twitch")
        twitch_layout.addWidget(twitch_label)
        self.twitch_input = QLineEdit()
        self.twitch_input.returnPressed.connect(self.on_click_connect_twitch)
        self.twitch_input.setPlaceholderText(
            "https://www.twitch.tv/CHANNEL_NAME or CHANNEL_NAME"
        )
        twitch_layout.addWidget(self.twitch_input)
        self.connect_twitch_button = QPushButton(_(self.language, "Connect"))
        self.connect_twitch_button.clicked.connect(self.on_click_connect_twitch)
        twitch_layout.addWidget(self.connect_twitch_button)
        # self.configure_twitch_button = QPushButton(_(self.language, "Configure"))
        # self.configure_twitch_button.clicked.connect(self.on_configure_twitch)
        # twitch_layout.addWidget(self.configure_twitch_button)
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
        self.chat_header_label = QLabel(_(self.language, "Message Log"))
        self.chat_header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        chat_header_layout.addWidget(self.chat_header_label, 1)

        control_layout = QHBoxLayout()
        chat_header_layout.addLayout(control_layout)
        control_layout.setContentsMargins(PADDING, 0, PADDING, 0)
        self.audio_indicator = QLabel("🟢")
        control_layout.addWidget(self.audio_indicator)
        self.pause_button = QPushButton(
            _(self.language, "Stopped")
            if self.is_paused
            else _(self.language, "Playback...")
        )
        self.setup_pause_button_color()
        self.pause_button.clicked.connect(self.on_pause_clicked)
        control_layout.addWidget(self.pause_button)
        self.clr_queue_button = QPushButton(_(self.language, "Clear queue"))
        self.clr_queue_button.clicked.connect(self.on_clear_queue)
        control_layout.addWidget(self.clr_queue_button)

        self.auto_scroll_checkbox = QCheckBox(_(self.language, "Auto-scroll"))
        self.auto_scroll_checkbox.setChecked(self.auto_scroll)
        self.auto_scroll_checkbox.clicked.connect(self.toggle_auto_scroll)
        chat_header_layout.addWidget(self.auto_scroll_checkbox)

        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems([str(s) for s in range(8, 24, 2)])
        self.font_size_combo.setCurrentIndex(2)
        self.font_size_combo.currentIndexChanged.connect(self.font_size_changed)
        chat_header_layout.addWidget(self.font_size_combo)

        self.clear_log_button = QPushButton(_(self.language, "Clear log"))
        self.clear_log_button.clicked.connect(self.clear_log)
        chat_header_layout.addWidget(self.clear_log_button)

        self.chat_header_widget = QWidget()
        self.chat_header_widget.setLayout(chat_header_layout)
        self.root_layout.addWidget(self.chat_header_widget)

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
        self.vol_label = QLabel(_(self.language, "Volume"))
        self.vol_label.setContentsMargins(PADDING, 0, 0, 5)
        self.statusBar().addWidget(self.vol_label)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setMinimum(0)
        self.vol_slider.setMaximum(200)
        self.vol_slider.setValue(self.volume)
        self.vol_slider.valueChanged.connect(self.on_change_volume)
        self.statusBar().addWidget(self.vol_slider)

        self.vol_label_value = QLabel(f"{self.volume}%")
        self.vol_label_value.setContentsMargins(0, 0, PADDING, 5)
        self.statusBar().addWidget(self.vol_label_value)

        # == Speech rate slider ==
        self.speech_rate_label = QLabel(_(self.language, "Speech rate"))
        self.speech_rate_label.setContentsMargins(0, 0, 0, 5)
        self.statusBar().addWidget(self.speech_rate_label)

        self.speech_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.speech_rate_slider.setMinimum(50)
        self.speech_rate_slider.setMaximum(150)
        self.speech_rate_slider.setValue(
            int(self.speech_rate * 100)
        )  # Convert to 50-150 range
        self.speech_rate_slider.valueChanged.connect(self.speech_rate_changed)
        self.statusBar().addWidget(self.speech_rate_slider)

        self.speech_rate_label_value = QLabel(f"{self.speech_rate:.2f}x")
        self.speech_rate_label_value.setContentsMargins(0, 0, 0, 5)
        self.statusBar().addWidget(self.speech_rate_label_value)

    def setup_ui(self):
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections_grid()
        self.setup_central_widget()
        self.exit_chat_only_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_chat_only_shortcut.activated.connect(self.exit_chat_only_mode)
        self.toggle_chat_only_shortcut = QShortcut(QKeySequence("F11"), self)
        self.toggle_chat_only_shortcut.activated.connect(
            lambda: self.toggle_chat_only_mode(not self.chat_only_mode)
        )
        self.pause_play_shortcut = QShortcut(QKeySequence("Space"), self)
        self.pause_play_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.pause_play_shortcut.activated.connect(self.on_pause_clicked)
        self.volume_up_shortcut = QShortcut(QKeySequence("Up"), self)
        self.volume_up_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.volume_up_shortcut.activated.connect(lambda: self.change_volume_by_step(5))
        self.volume_down_shortcut = QShortcut(QKeySequence("Down"), self)
        self.volume_down_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.volume_down_shortcut.activated.connect(
            lambda: self.change_volume_by_step(-5)
        )
        self.speech_rate_up_shortcut = QShortcut(QKeySequence("Right"), self)
        self.speech_rate_up_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.speech_rate_up_shortcut.activated.connect(
            lambda: self.change_speech_rate_by_step(5)
        )
        self.speech_rate_down_shortcut = QShortcut(QKeySequence("Left"), self)
        self.speech_rate_down_shortcut.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.speech_rate_down_shortcut.activated.connect(
            lambda: self.change_speech_rate_by_step(-5)
        )
        self.font_size_changed(2)
        self.apply_chat_only_mode()

    # === UI event handlers ===

    def change_volume_by_step(self, delta):
        self.vol_slider.setValue(
            max(
                self.vol_slider.minimum(),
                min(self.vol_slider.maximum(), self.vol_slider.value() + delta),
            )
        )

    def change_speech_rate_by_step(self, delta):
        self.speech_rate_slider.setValue(
            max(
                self.speech_rate_slider.minimum(),
                min(
                    self.speech_rate_slider.maximum(),
                    self.speech_rate_slider.value() + delta,
                ),
            )
        )

    def on_pause_clicked(self):
        self.is_paused = not self.is_paused
        self.pause_button.setText(
            _(self.language, "Stopped")
            if self.is_paused
            else _(self.language, "Playback...")
        )
        self.setup_pause_button_color()
        if self.is_paused:
            sd.stop()
            self.statusBar().showMessage(
                _(self.language, "Playback has been stopped"), 3000
            )
        else:
            self.statusBar().showMessage(
                _(self.language, "Speech playback continued..."), 3000
            )

    def language_changed(self, lang):
        self.language = lang
        self.save_settings()
        self.setup_menu_bar()
        self.apply_chat_only_mode()
        self.on_change_stats()
        self.speech_rate_label.setText(_(self.language, "Speech rate"))
        self.vol_label.setText(_(self.language, "Volume"))
        self.voice_label.setText(self.status_voice_text())

        self.connect_yt_button.setText(
            _(self.language, "Connected" if self.yt_is_connected else "Connect")
        )
        # self.configure_yt_button.setText(_(self.language, "Configure"))
        self.connect_twitch_button.setText(
            _(self.language, "Connected" if self.twitch_is_connected else "Connect")
        )
        # self.configure_twitch_button.setText(_(self.language, "Configure"))

        self.chat_header_label.setText(_(self.language, "Message Log"))
        self.auto_scroll_checkbox.setText(_(self.language, "Auto-scroll"))
        self.clear_log_button.setText(_(self.language, "Clear log"))
        self.pause_button.setText(
            f"⏸️ {_(self.language, "Stopped")}"
            if self.is_paused
            else f"▶️ {_(self.language, "Playback...")}"
        )
        self.clr_queue_button.setText((_(self.language, "Clear queue")))

    def voice_changed(self, lang, voice):
        self.voice = voice
        if self.voice_language != lang:
            self.voice_language = lang
            threading.Thread(target=self.init_silero, daemon=True).start()
            self.stop_words = self.load_stop_words(self.voice_language)

        self.save_settings()
        self.setup_voice_menu()
        self.voice_label.setText(self.status_voice_text())

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
        if getattr(self, "dlg", False) and self.dlg:
            self.dlg.close()

        self.dlg = QDialog(self)
        self.dlg.setFixedSize(600, 100)
        self.dlg.setSizePolicy(size_policy_fixed)
        self.dlg.setWindowFlags(window_flag_fixed)
        self.dlg.setWindowTitle(_(self.language, "Twitch account"))
        root_widget = QWidget(self.dlg)
        root_widget.setMinimumSize(600, 100)
        root_layout = QVBoxLayout(root_widget)

        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        root_layout.addLayout(entry_layout)
        self.twitch_client_id_input = QLineEdit()
        entry_layout.addWidget(self.twitch_client_id_input, 1)

        self.twitch_client_id_input.setPlaceholderText(
            _(self.language, "Enter your client id")
        )
        if self.twitch_credentials["client_id"] is None:
            self.twitch_client_id_input.returnPressed.connect(
                self.on_click_twitch_save_settings
            )
            self.twitch_client_id_button = QPushButton(_(self.language, "Save"))
            self.twitch_client_id_button.clicked.connect(
                self.on_click_twitch_save_settings
            )
            entry_layout.addWidget(self.twitch_client_id_button)
        else:
            self.twitch_client_id_input.setText(
                "*" * len(self.twitch_credentials["client_id"])
            )
            self.twitch_client_id_input.setReadOnly(True)
            self.twitch_client_id_button = QPushButton(_(self.language, "Edit"))
            self.twitch_client_id_button.clicked.connect(
                self.on_click_twitch_edit_credential
            )
            entry_layout.addWidget(self.twitch_client_id_button)

        client_id_help_text = QLabel(
            f'{_(self.language, "client_id_help_text")}: <a href="https://dev.twitch.tv/console/apps/">https://dev.twitch.tv/console/apps/</a>'
        )
        client_id_help_text.setOpenExternalLinks(True)
        root_layout.addWidget(client_id_help_text)

        self.dlg.exec()

    def on_click_twitch_save_settings(self):
        # if self.twitch_credentials["client_id"] is not None:
        #     res = QMessageBox.question(
        #         self,
        #         _(self.language, "Twitch account"),
        #         _(self.language, "twitch_save_settings_warning")
        #     )
        #     if res != QMessageBox.StandardButton.Yes:
        #         return

        client_id = self.twitch_client_id_input.text()

        def on_finish():
            if getattr(self, "dlg", False) and self.dlg:
                self.dlg.close()

        def on_error(err):
            self.add_sys_message(author="Twitch", text=err, status="error")

        def on_user_code(verification_uri, user_code, expires_in):
            on_finish()

            self.dlg = QDialog(self)
            self.dlg.setWindowTitle(_(self.language, "Twitch account"))
            self.dlg.setFixedSize(600, 250)
            self.dlg.setSizePolicy(size_policy_fixed)
            self.dlg.setWindowFlags(window_flag_fixed)
            root_widget = QWidget(self.dlg)
            root_widget.setMinimumSize(600, 250)
            root_layout = QVBoxLayout(root_widget)
            root_widget.setContentsMargins(PADDING, PADDING, PADDING, PADDING)

            continue_authorize_browser_text = QLabel(
                _(self.language, "continue_authorize_browser")
            )
            root_layout.addWidget(continue_authorize_browser_text)

            user_code_text = QLabel(str(user_code))
            user_code_text.setFont(QFont("Arial", 25, 900))
            user_code_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root_layout.addWidget(user_code_text, 1)

            expires_in_text = QLabel(f"{_(self.language, "expires_in")}: {expires_in}")
            root_layout.addWidget(expires_in_text)
            root_layout.setContentsMargins(0, 0, 0, PADDING)

            verification_uri_text = QLabel(
                f'{_(self.language, "verification_uri")}: <a href="{verification_uri}">{verification_uri}</a>'
            )
            verification_uri_text.setWordWrap(True)
            verification_uri_text.setOpenExternalLinks(True)
            root_layout.addWidget(verification_uri_text)
            self.dlg.exec()

        def on_token_received(token, refresh_token, nickname):
            self.twitch_credentials["access"] = token
            self.twitch_credentials["refresh"] = refresh_token
            self.twitch_credentials["client_id"] = client_id
            self.twitch_credentials["nickname"] = nickname
            self.save_settings()
            self.add_sys_message(
                author="Twitch",
                text=_(self.language, "Success authorized"),
                status="success",
            )
            on_finish()

        self.twitch_client_id_input.setText("*" * len(client_id))
        self.twitch_client_id_input.setReadOnly(True)
        self.twitch_client_id_input.returnPressed.disconnect()
        self.twitch_client_id_button.clicked.disconnect()
        self.twitch_client_id_button.clicked.connect(
            self.on_click_twitch_edit_credential
        )
        self.twitch_client_id_button.setText(_(self.language, "Edit"))

        self.worker = AuthWorker(client_id=client_id, lang=self.language)
        self.worker.user_code_signal.connect(on_user_code)
        self.worker.token_signal.connect(on_token_received)
        self.worker.error_signal.connect(on_error)
        self.worker.finished.connect(on_finish)
        self.worker.start()

    def on_click_twitch_edit_credential(self):
        self.twitch_client_id_input.returnPressed.connect(
            self.on_click_twitch_save_settings
        )
        self.twitch_client_id_input.setText(self.twitch_credentials["client_id"])
        self.twitch_client_id_input.setReadOnly(False)
        self.twitch_client_id_button.clicked.disconnect()
        self.twitch_client_id_button.clicked.connect(self.on_click_twitch_save_settings)
        self.twitch_client_id_button.setText(_(self.language, "Save"))

    def on_click_connect_twitch(self):
        self.connect_twitch_button.setEnabled(False)
        try:
            if self.twitch:

                def _stop_twitch_async():
                    self.twitch.stop()
                    self.twitch = None

                threading.Thread(target=_stop_twitch_async, daemon=True).start()
            else:
                video_id = self.twitch_input.text()
                if not video_id:
                    QMessageBox.warning(
                        self,
                        "URL / ID",
                        _(self.language, "Please enter video ID or URL"),
                    )
                    return

                if (
                    self.twitch_credentials["client_id"] is None
                    or self.twitch_credentials["access"] is None
                ):
                    self.on_configure_twitch()
                    return

                def on_expiries_access():
                    access, refresh = AuthWorker.refresh_access_token(
                        self.twitch_credentials["client_id"],
                        self.twitch_credentials["refresh"],
                        self.language,
                    )
                    self.twitch_credentials["access"] = access
                    self.twitch_credentials["refresh"] = refresh
                    return access

                def on_msg(msg_id, author, msg, is_member):
                    if self.subscribers_only and not is_member:
                        return
                    self.process_chat_message(
                        msg_id=msg_id,
                        platform="Twitch",
                        author=author,
                        message=msg,
                        is_member=is_member,
                    )

                def on_error(err):
                    self.add_sys_message(author="Twitch", text=err, status="error")

                self.twitch = TwitchChatListener(
                    client_id=self.twitch_credentials["client_id"],
                    token=self.twitch_credentials["access"],
                    nickname=self.twitch_credentials["nickname"],
                    channel=video_id,
                    on_connect=self.on_connect_twitch,
                    on_disconnect=self.on_disconnect_twitch,
                    on_error=on_error,
                    on_message=on_msg,
                    on_expiries_access=on_expiries_access,
                    lang=self.language,
                )
                threading.Thread(target=self.twitch.run).start()
        finally:
            self.connect_twitch_button.setEnabled(True)

    def on_disconnect_twitch(self):
        self.twitch_is_connected = False
        self.twitch_input.setReadOnly(False)

        self.connect_twitch_button.setText(_(self.language, "Connect"))
        self.connect_twitch_button.setPalette(self.style().standardPalette())
        self.add_sys_message(
            author="Twitch",
            text=_(self.language, "chat_disconnected"),
            status="success",
        )

    def on_connect_twitch(self):
        self.twitch_is_connected = True
        self.twitch_input.setReadOnly(True)

        self.connect_twitch_button.setText(_(self.language, "Connected"))
        palette = self.connect_twitch_button.palette()
        palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGreen)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.connect_twitch_button.setPalette(palette)
        self.add_sys_message(
            author="Twitch", text=_(self.language, "chat_connected"), status="success"
        )

    def on_configure_yt(self):
        dlg = QDialog(self)
        dlg.setFixedSize(600, 100)
        dlg.setSizePolicy(size_policy_fixed)
        dlg.setWindowFlags(window_flag_fixed)
        dlg.setWindowTitle(_(self.language, "Google API Key"))

        root_widget = QWidget(dlg)
        root_widget.setMinimumSize(600, 100)
        root_layout = QVBoxLayout(root_widget)

        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        root_layout.addLayout(entry_layout)
        self.google_api_key_input = QLineEdit()
        entry_layout.addWidget(self.google_api_key_input, 1)

        self.google_api_key_input.setPlaceholderText(
            _(self.language, "Enter your API key")
        )
        if self.yt_credentials is None:
            self.google_api_key_input.returnPressed.connect(
                self.on_click_yt_save_settings
            )
            self.google_api_key_button = QPushButton(_(self.language, "Save"))
            self.google_api_key_button.clicked.connect(self.on_click_yt_save_settings)
            entry_layout.addWidget(self.google_api_key_button)
        else:
            self.google_api_key_input.setText("*" * len(self.yt_credentials))
            self.google_api_key_input.setReadOnly(True)
            self.google_api_key_button = QPushButton(_(self.language, "Edit"))
            self.google_api_key_button.clicked.connect(self.on_click_yt_edit_credential)
            entry_layout.addWidget(self.google_api_key_button)

        api_keys_help_text = QLabel(
            f'{_(self.language, "api_keys_help_text")}: <a href="https://console.cloud.google.com/">https://console.cloud.google.com/</a>'
        )
        api_keys_help_text.setOpenExternalLinks(True)
        root_layout.addWidget(api_keys_help_text)

        dlg.exec()

    def on_click_yt_connect(self):
        self.connect_yt_button.setEnabled(False)
        try:
            if self.youtube:
                self.youtube.disconnect()
                self.youtube = None
            else:
                video_id = self.yt_video_input.text()
                if not video_id:
                    QMessageBox.warning(
                        self,
                        "URL / ID",
                        _(self.language, "Please enter video ID or URL"),
                    )
                    return

                # if self.yt_credentials is None:
                #     self.on_configure_yt()
                #     return

                if not self.model:
                    res = QMessageBox.question(
                        self,
                        _(self.language, "Silero not loaded"),
                        _(self.language, "note_tts_not_loaded"),
                    )
                    if res != QMessageBox.StandardButton.Yes:
                        return

                def on_msg(msg_id, author, msg, is_member):
                    if self.subscribers_only and not is_member:
                        return
                    self.process_chat_message(
                        msg_id=msg_id,
                        platform="YouTube",
                        author=author,
                        message=msg,
                        is_member=is_member,
                    )

                def on_error(err):
                    self.add_sys_message(author="YouTube", text=err, status="error")

                self.youtube = YouTubeChatParser(
                    url=video_id,
                    on_connect=self.on_connect_yt,
                    on_disconnect=self.on_disconnect_yt,
                    on_message=on_msg,
                    on_error=on_error,
                    lang=self.language,
                )

                self.youtube.run()
        finally:
            self.connect_yt_button.setEnabled(True)

    def on_connect_yt(self):
        self.yt_is_connected = True
        self.yt_video_input.setReadOnly(True)

        self.add_sys_message(
            author="YouTube", text=_(self.language, "chat_connected"), status="success"
        )
        self.connect_yt_button.setText(_(self.language, "Connected"))
        palette = self.connect_yt_button.palette()
        palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGreen)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.connect_yt_button.setPalette(palette)

    def on_disconnect_yt(self):
        self.yt_is_connected = False
        self.yt_video_input.setReadOnly(False)

        self.add_sys_message(
            author="YouTube",
            text=_(self.language, "chat_disconnected"),
            status="success",
        )
        self.connect_yt_button.setText(_(self.language, "Connect"))
        self.connect_yt_button.setPalette(self.style().standardPalette())

    def on_stop_words_action(self):
        dlg = QDialog(self)
        dlg.setMinimumSize(600, 300)
        dlg.setWindowTitle(_(self.language, "Stop words"))

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)

        self.stop_words_text = QPlainTextEdit()
        self.stop_words_text.setPlainText("\n".join(self.stop_words))

        self.stop_words_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout.addWidget(self.stop_words_text, 1)

        save_stop_words_btn = QPushButton(_(self.language, "Save stop words"))
        save_stop_words_btn.clicked.connect(self.on_save_stop_words)

        layout.addWidget(save_stop_words_btn)

        dlg.exec()

    def on_save_stop_words(self):
        content = self.stop_words_text.toPlainText().strip()
        self.stop_words = sorted(
            tuple(set([w.strip() for w in content.splitlines() if w.strip()]))
        )

        with open(
            resource_path(f"spam_filter/{self.voice_language}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(self.stop_words) + "\n")

        self.statusBar().showMessage(
            f"{_(self.language, "Saved")} {len(self.stop_words)} {_(self.language, "stop words")}",
            3000,
        )

    def on_click_yt_edit_credential(self):
        self.google_api_key_input.returnPressed.connect(self.on_click_yt_save_settings)
        self.google_api_key_input.setText(self.yt_credentials)
        self.google_api_key_input.setReadOnly(False)
        self.google_api_key_button.clicked.disconnect()
        self.google_api_key_button.clicked.connect(self.on_click_yt_save_settings)
        self.google_api_key_button.setText(_(self.language, "Save"))

    def on_click_yt_save_settings(self):
        self.yt_credentials = self.google_api_key_input.text()
        self.save_settings()
        self.google_api_key_input.setText("*" * len(self.yt_credentials))
        self.google_api_key_input.setReadOnly(True)
        self.google_api_key_input.returnPressed.disconnect()
        self.google_api_key_button.clicked.disconnect()
        self.google_api_key_button.clicked.connect(self.on_click_yt_edit_credential)
        self.google_api_key_button.setText(_(self.language, "Edit"))

    def on_change_toxic_sense(self, value):
        self.toxic_sense = value / 100.0
        self.toxic_sense_label_value.setText(f"{self.toxic_sense:.2f}")

    def on_delays_settings_action(self):
        dlg = QDialog(self)
        dlg.setSizePolicy(size_policy_fixed)
        dlg.setWindowFlags(window_flag_fixed)
        dlg.setWindowTitle(_(self.language, "Delays and processing"))

        root_layout = QVBoxLayout(dlg)
        root_layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)

        # Toxicity threshold

        toxic_sense_v_layout = QVBoxLayout()
        toxic_sense_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(toxic_sense_v_layout)

        self.toxic_sense_label_desc = QLabel(
            _(self.language, "Toxicity threshold (1 = OFF)")
        )
        toxic_sense_v_layout.addWidget(self.toxic_sense_label_desc)

        toxic_sense_layout = QHBoxLayout()
        toxic_sense_v_layout.addLayout(toxic_sense_layout)

        toxic_sense_slider = QSlider(Qt.Orientation.Horizontal)
        toxic_sense_layout.addWidget(toxic_sense_slider)
        toxic_sense_slider.setMinimum(10)
        toxic_sense_slider.setMaximum(100)
        toxic_sense_slider.setValue(int(self.toxic_sense * 100))
        toxic_sense_slider.valueChanged.connect(self.on_change_toxic_sense)

        self.toxic_sense_label_value = QLabel(str(self.toxic_sense))
        toxic_sense_layout.addWidget(self.toxic_sense_label_value)

        # Queue depth

        queue_depth_v_layout = QVBoxLayout()
        queue_depth_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(queue_depth_v_layout)

        self.queue_depth_label_desc = QLabel(_(self.language, "queue_depth_desc"))
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

        # Min message length

        min_msg_len_v_layout = QVBoxLayout()
        min_msg_len_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(min_msg_len_v_layout)

        self.min_msg_len_label_desc = QLabel(_(self.language, "Min message length"))
        min_msg_len_v_layout.addWidget(self.min_msg_len_label_desc)

        min_msg_len_layout = QHBoxLayout()
        min_msg_len_v_layout.addLayout(min_msg_len_layout)

        min_msg_len_slider = QSlider(Qt.Orientation.Horizontal)
        min_msg_len_layout.addWidget(min_msg_len_slider)
        min_msg_len_slider.setMinimum(2)
        min_msg_len_slider.setMaximum(50)
        min_msg_len_slider.setValue(self.min_msg_length)
        min_msg_len_slider.valueChanged.connect(self.on_change_min_msg_len)

        self.min_msg_len_label_value = QLabel(str(self.min_msg_length))
        min_msg_len_layout.addWidget(self.min_msg_len_label_value)

        # Max message length

        msg_len_v_layout = QVBoxLayout()
        msg_len_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(msg_len_v_layout)

        self.msg_len_label_desc = QLabel(_(self.language, "Max message length"))
        msg_len_v_layout.addWidget(self.msg_len_label_desc)

        msg_len_layout = QHBoxLayout()
        msg_len_v_layout.addLayout(msg_len_layout)

        msg_len_slider = QSlider(Qt.Orientation.Horizontal)
        msg_len_layout.addWidget(msg_len_slider)
        msg_len_slider.setMinimum(50)
        msg_len_slider.setMaximum(300)
        msg_len_slider.setValue(self.max_msg_length)
        msg_len_slider.valueChanged.connect(self.on_change_max_msg_len)

        self.msg_len_label_value = QLabel(str(self.max_msg_length))
        msg_len_layout.addWidget(self.msg_len_label_value)

        # Speech delay

        speech_delay_v_layout = QVBoxLayout()
        # speech_delay_v_layout.setContentsMargins(0, 0, 0, PADDING)
        root_layout.addLayout(speech_delay_v_layout)

        self.speech_delay_label_desc = QLabel(
            _(self.language, "Delay between messages")
        )
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

        dlg.adjustSize()
        dlg.setFixedSize(dlg.sizeHint())
        dlg.finished.connect(self.save_settings)
        dlg.exec()

    def on_change_queue_speech_delay(self, value):
        self.speech_delay = value / 10
        self.speech_delay_label_value.setText(f"{float(self.speech_delay):.2f}")

    def on_change_min_msg_len(self, value):
        self.min_msg_length = value
        self.min_msg_len_label_value.setText(str(self.min_msg_length))

    def on_change_max_msg_len(self, value):
        self.max_msg_length = value
        self.msg_len_label_value.setText(str(self.max_msg_length))

    def on_change_queue_depth(self, value):
        self.buffer_maxsize = value
        self.queue_depth_label_value.setText(str(self.buffer_maxsize))
        self.message_buffer = deque(maxlen=self.buffer_maxsize)
        self.audio_queue = deque(maxlen=self.buffer_maxsize)

    def on_change_stats(self):
        self.stats_label.setText(self.stats_text())

    def on_clear_queue(self):
        self.audio_queue.clear()
        self.on_change_stats()
        self.statusBar().showMessage(_(self.language, "Queue cleared"), 3000)

    def toggle_add_accents(self, checked):
        self.add_accents = checked

    def toggle_filter_repeats(self, checked):
        self.filter_repeats = checked

    def toggle_read_author_names(self, checked):
        self.read_author_names = checked

    def toggle_read_platform_names(self, checked):
        self.read_platform_names = checked

    def toggle_subscribers_only(self, checked):
        self.subscribers_only = checked

    def toggle_auto_translate(self, checked):
        self.auto_translate = checked

    def toggle_chat_only_mode(self, checked):
        self.chat_only_mode = checked
        self.apply_chat_only_mode()

    def exit_chat_only_mode(self):
        if self.chat_only_mode:
            self.toggle_chat_only_mode(False)

    def apply_chat_only_mode(self):
        chat_only = self.chat_only_mode
        menu_bar = self.menuBar()
        if menu_bar is not None:
            menu_bar.setVisible(not chat_only)
        self.statusBar().setVisible(not chat_only)
        if hasattr(self, "connections_widget"):
            self.connections_widget.setVisible(not chat_only)
        if hasattr(self, "chat_header_widget"):
            self.chat_header_widget.setVisible(not chat_only)
        if (
            hasattr(self, "chat_only_action")
            and self.chat_only_action.isChecked() != chat_only
        ):
            self.chat_only_action.setChecked(chat_only)

    def toggle_auto_scroll(self, checked):
        self.auto_scroll = checked

    def clear_log(self):
        self.chat_text.clear()
        self.statusBar().showMessage(_(self.language, "Log cleared"), 3000)

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
            if file.open(
                QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Text
            ):
                file.write(log.encode("utf-8"))
                file.close()
                self.statusBar().showMessage(
                    f"{_(self.language, "File saved")}: {file_path}", 3000
                )

    # === Helper methods ===

    def stats_text(self):
        return (
            f"{_(self.language, 'Messages')}: {self.messages_stats['messages_count']} | "
            f"{_(self.language, 'Spoken')}: {self.messages_stats['spoken_count']} | "
            f"{_(self.language, 'Spam')}: {self.messages_stats['spam_count']} | "
            f"{_(self.language, 'In queue')}: {len(self.audio_queue)}"
        )

    def status_voice_text(self):
        return f"{_(self.language, 'Voice')}: {_(self.language, self.voice_language)} - {self.voice}"

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

        system_text = _(self.language, "System")
        return self.add_message(
            platform=system_text,
            author=author,
            text=text,
            background=status_colors[status],
        )

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
        avatar_bg, avatar_fg = avatar_colors_from_name(safe_author)
        white_color = "#fff"
        scrollbar = self.chat_text.verticalScrollBar()
        prev_scroll_value = scrollbar.value()

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
                scrollbar.setValue(scrollbar.maximum())
            else:
                scrollbar.setValue(prev_scroll_value)
        except Exception:
            pass

    def save_settings(self):
        settings = {
            "language": self.language,
            "voice_language": self.voice_language,
            "voice": self.voice,
            "volume": self.volume,
            "speech_rate": self.speech_rate,
            "speech_delay": self.speech_delay,
            "auto_scroll": self.auto_scroll,
            "add_accents": self.add_accents,
            "filter_repeats": self.filter_repeats,
            "read_author_names": self.read_author_names,
            "read_platform_names": self.read_platform_names,
            "subscribers_only": self.subscribers_only,
            "toxic_sense": self.toxic_sense,
            "auto_translate": self.auto_translate,
            "min_msg_length": self.min_msg_length,
            "max_msg_length": self.max_msg_length,
            "buffer_maxsize": self.buffer_maxsize,
            "yt_credentials": self.yt_credentials,
            "twitch_credentials": self.twitch_credentials,
        }
        with open(get_settings_path(), "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

        self.statusBar().showMessage(_(self.language, "Settings saved"), 3000)

    def load_settings(self):
        settings_path = get_settings_path()
        legacy_path = resource_path("settings.json")
        path_candidates = [settings_path]
        if legacy_path != settings_path:
            path_candidates.append(legacy_path)

        try:
            settings = None
            for path in path_candidates:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                    break

            if settings is None:
                return

            self.language = settings.get("language", self.language)
            self.voice_language = settings.get("voice_language", self.voice_language)
            self.voice = settings.get("voice", self.voice)
            self.volume = settings.get("volume", self.volume)
            self.speech_rate = settings.get("speech_rate", self.speech_rate)
            self.speech_delay = settings.get("speech_delay", self.speech_delay)
            self.auto_scroll = settings.get("auto_scroll", self.auto_scroll)
            self.add_accents = settings.get("add_accents", self.add_accents)
            self.filter_repeats = settings.get("filter_repeats", self.filter_repeats)
            self.read_author_names = settings.get(
                "read_author_names", self.read_author_names
            )
            self.read_platform_names = settings.get(
                "read_platform_names", self.read_platform_names
            )
            self.subscribers_only = settings.get(
                "subscribers_only", self.subscribers_only
            )
            self.toxic_sense = settings.get("toxic_sense", self.toxic_sense)
            self.auto_translate = settings.get("auto_translate", self.auto_translate)
            self.buffer_maxsize = settings.get("buffer_maxsize", self.buffer_maxsize)
            self.min_msg_length = settings.get("min_msg_length", self.min_msg_length)
            self.max_msg_length = settings.get("max_msg_length", self.max_msg_length)
            self.yt_credentials = settings.get("yt_credentials", self.yt_credentials)
            self.twitch_credentials = settings.get(
                "twitch_credentials", twitch_default_credentials
            )
            if not isinstance(self.twitch_credentials, dict):
                self.twitch_credentials = twitch_default_credentials

            self.stop_words = self.load_stop_words(self.voice_language)

        except FileNotFoundError:
            pass  # No settings file, use defaults

    def load_stop_words(self, lang):
        source_path = resource_path(f"spam_filter/{lang}.txt")
        try:
            with open(source_path, "r", encoding="utf-8") as file:
                return tuple(line.strip() for line in file if line.strip())
        except FileNotFoundError:
            self.add_sys_message(
                author="load_stop_words()",
                text=f"{_(self.language, "File with stop-words not found")}: ({source_path})",
                status="error",
            )
        except Exception as e:
            self.add_sys_message(
                author="load_stop_words()",
                text=f"{_(self.language, "Failed to read stop-words file")}: {translate_text(str(e), self.language)}",
                status="error",
            )
        return tuple()

    def closeEvent(self, event):
        self.save_settings()
        sd.stop()
        super().closeEvent(event)

    def init_detoxify(self):
        cached_checkpoint = None
        try:
            ensure_stdio_streams()
            self.add_sys_message(
                author="Detoxify", text=_(self.language, "detoxify_loading")
            )
            cached_checkpoint = find_cached_detoxify_checkpoint("multilingual")
            if cached_checkpoint:
                self.detox_model = Detoxify(
                    model_type="multilingual",
                    checkpoint=cached_checkpoint,
                )
            else:
                self.detox_model = Detoxify("multilingual")
            self.add_sys_message(
                author="Detoxify",
                text=_(self.language, "detoxify_loaded"),
                status="success",
            )

        except Exception as e:
            error_text = str(e)
            if (
                "PytorchStreamReader failed reading zip archive: failed finding central directory"
                in error_text
            ):
                if cached_checkpoint and os.path.isfile(cached_checkpoint):
                    try:
                        self.add_sys_message(
                            author="Detoxify",
                            text=f"{_(self.language, "detoxify_loading_failed")}. {_(self.language, "The file is corrupted")}",
                            status="error",
                        )
                        os.remove(cached_checkpoint)
                        return
                    except OSError:
                        pass
                clear_detoxify_checkpoint_cache("multilingual")
            self.add_sys_message(
                author="Detoxify",
                text=f"{_(self.language, "detoxify_loading_failed")}. {translate_text(str(e), self.language)}",
                status="error",
            )

    def init_silero(self):
        self.add_sys_message(author="Silero", text=_(self.language, "silero_loading"))
        try:
            with self.model_lock:
                if getattr(sys, "frozen", False):
                    cached_repo = find_cached_silero_repo()
                    if cached_repo:
                        prefer_cached_silero_package(cached_repo)
                        self.model, txt = hub.load(
                            repo_or_dir=cached_repo,
                            source="local",
                            model="silero_tts",
                            language=self.voice_language,
                            speaker=MODELS[self.voice_language],
                            trust_repo=True,
                            force_reload=False,
                            verbose=False,
                        )
                    else:
                        self.model, txt = hub.load(
                            repo_or_dir="snakers4/silero-models",
                            source="github",
                            model="silero_tts",
                            language=self.voice_language,
                            speaker=MODELS[self.voice_language],
                            trust_repo=True,
                            force_reload=False,
                            verbose=False,
                        )
                else:
                    self.model, txt = hub.load(
                        repo_or_dir="snakers4/silero-models",
                        model="silero_tts",
                        language=self.voice_language,
                        speaker=MODELS[self.voice_language],
                        trust_repo=True,
                        force_reload=False,
                        verbose=False,
                    )

            self.add_sys_message(
                author="Silero",
                text=_(self.language, "silero_loaded"),
                status="success",
            )
            # self.speak(_(self.voice_language, "silero_loaded"))

        except Exception as e:
            self.add_sys_message(
                author="Silero",
                text=f"{_(self.language, "silero_failed")}. {translate_text(str(e), self.language)}",
                status="error",
            )
            return False
        finally:
            try:
                silero_catalog_path = os.path.join(
                    os.getcwd(), "latest_silero_models.yml"
                )
                os.remove(silero_catalog_path)
            except OSError:
                pass

    def get_msg_hash(self, platform, author, message):
        return hashlib.md5(f"{platform}:{author}:{message}".encode()).hexdigest()

    def is_spam(self, platform, author, message):
        """Check for spam"""
        if self.filter_repeats is False:
            return False

        message_hash = self.get_msg_hash(platform, author, message)

        if message_hash in self.spam_hash_set:
            self.messages_stats["spam_count"] += 1
            return True

        if len(self.spam_hash_queue) == self.spam_hash_queue.maxlen:
            oldest = self.spam_hash_queue.popleft()
            self.spam_hash_set.discard(oldest)

        self.spam_hash_queue.append(message_hash)
        self.spam_hash_set.add(message_hash)

        return False

    def contains_stop_words(self, text: str):
        """Check for stop words"""
        text_words = text.split(" ")
        for word in text_words:
            if word in self.stop_words:
                return True
        return False

    def convert_numbers_to_words(self, text):
        """Convert numbers to text representation"""

        def replace_number(match):
            num = match.group()
            try:
                if "." in num:
                    parts = num.split(".")
                    integer_part = num2words(int(parts[0]), lang=self.voice_language)
                    fractional_part = num2words(int(parts[1]), lang=self.voice_language)
                    return f"{integer_part} {_(self.voice_language, "point")} {fractional_part}"
                else:
                    return num2words(int(num), lang=self.voice_language)
            except Exception as e:
                # self.add_sys_message(
                #     author="_translate_text()",
                #     text=f"{_(self.language, 'Error convert num to word')}. {e}",
                #     status="error",
                # )
                return num

        number_pattern = r"-?\d+(?:[.,]\d+)?"
        converted_text = re.sub(number_pattern, replace_number, text)
        return converted_text

    def process_chat_message(self, msg_id, platform, author, message, is_member):
        msg_id = str(msg_id)
        if msg_id in self.processed_messages:
            return
        self.processed_messages.add(msg_id)

        cleaned_text = clean_message(message, self.language)
        if not cleaned_text:
            return

        if self.contains_stop_words(cleaned_text):
            self.add_message(
                platform=platform,
                author=author,
                text=f"[{_(self.language, "Stop words")}] {message}",
                color="gray",
            )
            return

        if is_member is False and self.is_spam(platform, author, cleaned_text):
            self.add_message(
                platform=platform,
                author=author,
                text=f"[{_(self.language, "SPAM")}] {message}",
                color="gray",
            )
            return

        if self.detox_model:
            sentiment = self.detox_model.predict(cleaned_text)
            detox_key = max(sentiment, key=sentiment.get)
            detox_value = sentiment[detox_key]
            if detox_value > self.toxic_sense:
                self.add_message(
                    platform=platform,
                    author=author,
                    text=f"[{_(self.language, str(detox_key).replace("_", " ").capitalize())}] {message}",
                    color="gray",
                )
                return

        if self.auto_translate:
            cleaned_text = translate_text(cleaned_text, self.voice_language)
            cleaned_text = clean_message(cleaned_text, self.language)
            if not cleaned_text:
                return

        self.add_message(platform=platform, author=author, text=cleaned_text)

        if len(cleaned_text) < self.min_msg_length:
            return
        if len(cleaned_text) > self.max_msg_length:
            cleaned_text = cleaned_text[: self.max_msg_length] + "..."

        cleaned_text = self.convert_numbers_to_words(cleaned_text)

        if self.read_author_names:
            cleaned_text = f"{author} {_(self.voice_language, 'said')} - {cleaned_text}"

        if self.read_platform_names:
            cleaned_text = f"{_(self.voice_language, "Message from")} {_(self.voice_language, str(platform).lower())}: {cleaned_text}"

        self.message_buffer.append(cleaned_text)

    # == Audio processing ==

    def text_to_speech(self, text):
        """Convert text to speech using Silero"""
        try:
            if self.model is not None:
                with self.model_lock:
                    with no_grad():
                        if self.voice == "random":
                            num = randint(0, len(VOICES[self.voice_language]) - 1)
                            voice = VOICES[self.voice_language][num]
                        else:
                            voice = self.voice

                        return self.model.apply_tts(
                            text=text,
                            speaker=voice,
                            # sample_rate=self.sample_rate,
                            put_accent=self.add_accents,
                            # put_yo=self.put_yo,
                        )
        except Exception as e:
            self.add_sys_message(
                author="text_to_speech()",
                text=f"{_(self.language, "Error convert text to speech")}, {translate_text(str(e), self.language)}. TEXT: {text}",
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
                author="speak()",
                text=f"{_(self.language, "Audio playback error")}. {translate_text(str(e), self.language)}",
                status="error",
            )

    def play_audio(self, audio_to_play):
        try:
            self.audio_indicator.setText("🔴")
            sd.play(audio_to_play)
            sd.wait()
        except Exception as e:
            self.add_sys_message(
                author="play_audio()",
                text=f"{_(self.language, 'Audio playback error')}. {translate_text(str(e), self.language)}",
                status="error",
            )
        finally:
            self.audio_indicator.setText("🟢")

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
                    text=f"{_(self.language, "Audio queue error")}. {translate_text(str(e), self.language)}",
                    status="error",
                )
            finally:
                sleep(self.speech_delay)

            sleep(0.1)

    # == Audio processing ==

    def process_msg_buffer_loop(self):
        while True:
            try:
                if len(self.message_buffer) > 0:
                    text = self.message_buffer.popleft()
                    self.speak(text)
            except Exception as e:
                self.add_sys_message(
                    author="process_msg_buffer_loop()",
                    text=f"{_(self.language, "Error process message buffer")}. {translate_text(str(e), self.language)}",
                    status="error",
                )
            finally:
                sleep(0.2)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
