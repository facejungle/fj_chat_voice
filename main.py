from pathlib import Path
import sys
import time
import re
import json
import threading
from datetime import datetime
import os
import urllib.parse
import queue
import gc
import hashlib
import warnings

from googleapiclient.discovery import build
import torch
import sounddevice as sd
import numpy as np
import customtkinter as ctk
from tkinter import messagebox, filedialog
from collections import deque
from num2words import num2words

warnings.filterwarnings("ignore", category=UserWarning)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
BUFFER_SIZE = 15
PAD = 10
PAD_L = 20
FONT_SZ = 12
FONT_SZ_L = 18
BG_COLOR = "#000"


class FJChatVoice:
    def __init__(self):
        self.window = ctk.CTk(fg_color=BG_COLOR)
        self.window.title("FJ Chat Voice - Silero TTS")
        self.window.geometry("1200x800")
        self.window.resizable(False, False)

        # State variables
        self.is_running_yt = False
        self.is_tts_ready = False
        self.chat_thread = None
        self.message_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.silero_loaded = False
        self.is_fetching = False

        # Thread safety locks
        self.speech_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.model_lock = threading.Lock()

        # YouTube API variables
        self.api_key_yt = ""
        self.video_id_yt = ""
        self.youtube = None
        self.processed_messages = set()
        self.chat_id = None

        # Silero TTS variables
        self.silero_model = None
        self.silero_available = False
        self.device = torch.device("cpu")
        self.sample_rate = 48000
        self.speaker = "xenia"
        self.put_accent = True
        self.put_yo = True
        self.speech_rate = 1.0
        self.volume = 1.0

        # Statistics
        self.messages_count = 0
        self.spoken_count = 0
        self.spam_count = 0
        self.start_time = None
        self.last_message_time = {}
        self.message_hash_set = set()

        # API backoff (reduce quota usage on errors / long polling)
        self.api_backoff = 1
        self.api_backoff_max = 60

        # Message queue
        self.message_buffer = None
        self.last_speak_time = 0
        self.buffer_maxsize = BUFFER_SIZE

        # Stop words
        self.stop_words = []

        # Load settings
        self.load_settings()

        # Initialize buffer with loaded size
        self.message_buffer = deque(maxlen=self.buffer_maxsize)

        # Create interface
        self.setup_ui()

        # Automatically check for cached model
        self.window.after(1000, self.check_cached_model)

        # Start queue processing
        self.process_audio_queue()
        self.process_speech_queue()

    def setup_ui(self):
        """Create user interface"""
        # Main container
        self.main_container = ctk.CTkFrame(self.window, bg_color=BG_COLOR, fg_color=BG_COLOR)
        self.main_container.pack(fill="both", expand=True)

        # Create tabs
        self.tabview = ctk.CTkTabview(self.main_container, fg_color=BG_COLOR, bg_color=BG_COLOR)
        self.tabview.pack(fill="both", expand=True)

        # Tabs
        self.tab_main = self.tabview.add("📺 Chat")
        self.tab_settings = self.tabview.add("⚙️ Settings")

        self.setup_status_bar()

        self.setup_main_tab()
        self.setup_settings_tab()

    def setup_status_bar(self):
        self.status_bar = ctk.CTkFrame(self.window, bg_color=BG_COLOR, fg_color=BG_COLOR, height=15)
        self.status_bar.pack(fill="x")

        # Real-time statistics
        self.stats_label = ctk.CTkLabel(
            self.status_bar, text="📊 Messages: 0 | Spoken: 0 | Spam: 0 | In queue: 0", font=ctk.CTkFont(size=FONT_SZ)
        )
        self.stats_label.pack(side="left", pady=(0, 5), padx=PAD)

        # Audio indicator

        self.audio_indicator = ctk.CTkLabel(self.status_bar, text="🟢", font=ctk.CTkFont(size=20), text_color="white")
        self.audio_indicator.pack(side="right", pady=(0, 5), padx=PAD)

        # Volume

        volume_frame = ctk.CTkFrame(self.status_bar, bg_color=BG_COLOR, fg_color=BG_COLOR)
        volume_frame.pack(side="right", padx=(PAD, 0))

        ctk.CTkLabel(volume_frame, text="Volume:", font=ctk.CTkFont(size=FONT_SZ)).pack(side="left")

        self.volume_var = ctk.DoubleVar(value=self.volume)
        self.volume_slider = ctk.CTkSlider(
            volume_frame, from_=0.0, to=2.0, variable=self.volume_var, command=self.change_volume, width=200
        )
        self.volume_slider.pack(side="left", padx=PAD)

        self.volume_label = ctk.CTkLabel(volume_frame, text=f"{self.volume:.0%}", font=ctk.CTkFont(size=FONT_SZ))
        self.volume_label.pack(side="left")

        # Speech rate

        speed_frame = ctk.CTkFrame(self.status_bar, bg_color=BG_COLOR, fg_color=BG_COLOR)
        speed_frame.pack(side="right")

        ctk.CTkLabel(speed_frame, text="Speech rate:", font=ctk.CTkFont(size=FONT_SZ)).pack(side="left")

        self.speed_var = ctk.DoubleVar(value=self.speech_rate)
        self.speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.50,
            to=1.50,
            number_of_steps=100,
            variable=self.speed_var,
            command=self.change_speed,
            width=200,
        )
        self.speed_slider.pack(side="left", padx=PAD)

        self.speed_label = ctk.CTkLabel(speed_frame, text=f"{self.speech_rate:.2f}x", font=ctk.CTkFont(size=FONT_SZ))
        self.speed_label.pack(side="left")

    def setup_main_tab(self):
        """Setup main tab"""
        self.top_frame = ctk.CTkFrame(self.tab_main)
        self.top_frame.pack(fill="x", pady=(0, PAD))

        # Connection
        connection_frame = ctk.CTkFrame(self.top_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        connection_frame.pack(fill="x")

        ctk.CTkLabel(connection_frame, text="YouTube URL / ID", font=ctk.CTkFont(size=FONT_SZ), width=150).pack(
            side="left", padx=PAD
        )

        self.video_entry_yt = ctk.CTkEntry(
            connection_frame, width=500, placeholder_text="https://youtube.com/watch?v=... or video ID"
        )
        self.video_entry_yt.pack(side="left", padx=PAD, fill="x", expand=True)
        self.video_entry_yt.insert(0, self.video_id_yt)

        self.connect_btn_yt = ctk.CTkButton(
            connection_frame,
            text="Connect",
            width=100,
            command=self.toggle_connection_yt,
            # fg_color="#28a745",
            # hover_color="#218838",
        )
        self.connect_btn_yt.pack(side="left")

        # Connection status
        self.connection_status_yt = ctk.CTkLabel(
            connection_frame, text="🔴", font=ctk.CTkFont(size=20), text_color="red"
        )
        self.connection_status_yt.pack(side="right", padx=PAD)

        # === Chat panel ===
        self.chat_frame = ctk.CTkFrame(self.tab_main, fg_color=BG_COLOR, bg_color=BG_COLOR)
        self.chat_frame.pack(fill="both", expand=True)

        chat_header_frame = ctk.CTkFrame(self.chat_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        chat_header_frame.pack(fill="x", padx=PAD, pady=PAD)

        ctk.CTkLabel(chat_header_frame, text="💬 Message Log", font=ctk.CTkFont(size=FONT_SZ, weight="bold")).pack(
            side="left"
        )

        # Auto-scroll checkbox
        self.auto_scroll_var = ctk.BooleanVar(value=self.auto_scroll)
        self.auto_scroll_check = ctk.CTkCheckBox(chat_header_frame, text="Auto-scroll", variable=self.auto_scroll_var)
        self.auto_scroll_check.pack(side="left", padx=(PAD, 0))

        # Bottom panel
        bottom_frame = ctk.CTkFrame(chat_header_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        bottom_frame.pack(fill="x", side="right")

        self.clear_btn = ctk.CTkButton(
            bottom_frame,
            text="🧹 Clear log",
            width=100,
            command=self.clear_chat,
            # fg_color="#6c757d",
            # hover_color="#5a6268",
        )
        self.clear_btn.pack(side="left")

        self.export_btn = ctk.CTkButton(
            bottom_frame,
            text="💾 Export log",
            width=100,
            command=self.export_chat_log,
            # fg_color="#17a2b8",
            # hover_color="#138496",
        )
        self.export_btn.pack(side="left", padx=(PAD, 0))

        # Chat text box
        self.chat_text = ctk.CTkTextbox(self.chat_frame, wrap="word", font=ctk.CTkFont(size=FONT_SZ), state="disabled")
        self.chat_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Tag configuration for colored text
        self.chat_text.tag_config("system", foreground="#6c757d")
        self.chat_text.tag_config("error", foreground="#dc3545")
        self.chat_text.tag_config("success", foreground="#28a745")
        self.chat_text.tag_config("message", foreground="#ffffff")
        self.chat_text.tag_config("author", foreground="#17a2b8")
        self.chat_text.tag_config("spam", foreground="#ffc107", background="#343a40")
        self.chat_text.tag_config("paused", foreground="#ffc107")
        self.chat_text.tag_config("youtube", foreground="#9c1111")

    def setup_settings_tab(self):
        left_frame = ctk.CTkFrame(self.tab_settings, fg_color=BG_COLOR, bg_color=BG_COLOR)
        left_frame.pack(fill="both", expand=True, side="left", padx=PAD)

        language_frame = ctk.CTkFrame(left_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        language_frame.pack(pady=(PAD, PAD_L), padx=PAD, fill="x", side="top", anchor="n")
        ctk.CTkLabel(language_frame, text="Language", font=ctk.CTkFont(size=FONT_SZ_L)).pack(side="left")
        language_select = ctk.CTkOptionMenu(language_frame, values=["English", "Russian"])
        language_select.set("English")
        language_select.pack(side="right")

        credentials_frame = ctk.CTkFrame(left_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        credentials_frame.pack(pady=(PAD, PAD_L), padx=PAD, fill="x", side="top", anchor="n")
        ctk.CTkLabel(credentials_frame, text="Credentials", font=ctk.CTkFont(size=FONT_SZ_L), anchor="nw").pack(
            expand=True, fill="x", anchor="n"
        )

        yt_cred_frame = ctk.CTkFrame(credentials_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        yt_cred_frame.pack(fill="x", expand=True, side="top", anchor="n", pady=(0, PAD))
        ctk.CTkLabel(yt_cred_frame, text="Google API Key", font=ctk.CTkFont(size=FONT_SZ), width=150).pack(side="left")
        self.api_entry_yt = ctk.CTkEntry(yt_cred_frame, width=200, placeholder_text="Enter your API key")
        self.api_entry_yt.pack(side="left", fill="x", expand=True)
        self.api_entry_yt.insert(0, self.api_key_yt)
        self.save_api_btn_yt = ctk.CTkButton(yt_cred_frame, text="Save", width=100, command=self.save_yt_api_key)
        self.save_api_btn_yt.pack(side="right", padx=(PAD, 0))

        tw_cred_frame = ctk.CTkFrame(credentials_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        tw_cred_frame.pack(fill="x", expand=True, side="top", anchor="n")
        ctk.CTkLabel(tw_cred_frame, text="Another credentials", font=ctk.CTkFont(size=FONT_SZ), width=150).pack(
            side="left"
        )
        self.tw_api_entry = ctk.CTkEntry(tw_cred_frame, width=200, placeholder_text="Enter your API key")
        self.tw_api_entry.pack(side="left", fill="x", expand=True)
        self.tw_api_entry.insert(0, self.api_key_yt)
        self.save_tw_api_btn = ctk.CTkButton(tw_cred_frame, text="Save", width=100, command=self.save_yt_api_key)
        self.save_tw_api_btn.pack(side="right", padx=(PAD, 0))

        # = Silero model =

        silero_frame = ctk.CTkFrame(left_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        silero_frame.pack(pady=PAD, padx=PAD, fill="x", side="top", anchor="n")

        silero_label_frame = ctk.CTkFrame(silero_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        silero_label_frame.pack(fill="x", pady=(0, PAD), expand=True)

        ctk.CTkLabel(silero_label_frame, text="Silero model", font=ctk.CTkFont(size=FONT_SZ_L)).pack(side="left")
        self.tts_status_label = ctk.CTkLabel(
            silero_label_frame, text="⚪ Silero not loaded", font=ctk.CTkFont(size=FONT_SZ)
        )
        self.tts_status_label.pack(side="right")
        # self.init_tts_btn = ctk.CTkButton(
        #     silero_frame,
        #     text="📥 Load Silero model",
        #     command=self.init_silero,
        #     width=100,
        #     font=ctk.CTkFont(size=FONT_SZ),
        # )
        # self.init_tts_btn.pack(side="left", fill="x")

        model_cfg_frame = ctk.CTkFrame(silero_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        model_cfg_frame.pack(fill="x", pady=(0, PAD))
        # Language selection
        model_language_select_frame = ctk.CTkFrame(model_cfg_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        model_language_select_frame.pack(side="left")
        ctk.CTkLabel(model_language_select_frame, text="Chat language", font=ctk.CTkFont(size=FONT_SZ), width=150).pack(
            side="left"
        )
        model_language_select = ctk.CTkOptionMenu(model_language_select_frame, values=["English", "Russian"])
        model_language_select.set("Russian")
        model_language_select.pack(side="left")
        # Voice selection
        voice_select_frame = ctk.CTkFrame(model_cfg_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        voice_select_frame.pack(side="left")
        ctk.CTkLabel(voice_select_frame, text="Voice", font=ctk.CTkFont(size=FONT_SZ), width=150).pack(side="left")
        self.voice_var = ctk.StringVar(value=self.speaker)
        self.voice_combo = ctk.CTkOptionMenu(
            voice_select_frame,
            values=["xenia", "aidar", "baya", "kseniya", "eugene"],
            variable=self.voice_var,
            command=self.change_voice,
        )
        self.voice_combo.pack(side="left")

        options_frame = ctk.CTkFrame(silero_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        options_frame.pack(fill="x", pady=(0, PAD))

        self.put_accent_var = ctk.BooleanVar(value=self.put_accent)
        self.put_accent_check = ctk.CTkCheckBox(
            options_frame, text="Add accents", variable=self.put_accent_var, command=self.toggle_accent
        )
        self.put_accent_check.pack(side="left", padx=(0, PAD))

        self.put_yo_var = ctk.BooleanVar(value=self.put_yo)
        self.put_yo_check = ctk.CTkCheckBox(
            options_frame, text="Replace e with yo (Russian)", variable=self.put_yo_var, command=self.toggle_yo
        )
        self.put_yo_check.pack(side="left")

        # = Message queue settings =

        buffer_frame = ctk.CTkFrame(left_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        buffer_frame.pack(pady=PAD, padx=PAD, fill="x", side="top", anchor="n")

        ctk.CTkLabel(buffer_frame, text="Message Queue", font=ctk.CTkFont(size=FONT_SZ_L), anchor="nw").pack(
            expand=True, fill="x", anchor="n"
        )

        buffer_size_frame = ctk.CTkFrame(buffer_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        buffer_size_frame.pack(fill="x")

        ctk.CTkLabel(buffer_size_frame, text="Queue depth:", font=ctk.CTkFont(size=13)).pack(side="left")

        self.buffer_size_var = ctk.StringVar(value=str(self.buffer_maxsize))

        spinbox_container = ctk.CTkFrame(buffer_size_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        spinbox_container.pack(side="left", padx=10)

        self.buffer_size_entry = ctk.CTkEntry(spinbox_container, width=80, textvariable=self.buffer_size_var)
        self.buffer_size_entry.pack(side="left", padx=(0, 5))

        button_frame = ctk.CTkFrame(spinbox_container, fg_color=BG_COLOR, bg_color=BG_COLOR)
        button_frame.pack(side="left")

        self.buffer_up_btn = ctk.CTkButton(
            button_frame, text="▲", width=30, height=20, command=self.increase_buffer_size
        )
        self.buffer_up_btn.pack(side="top", pady=(0, 2))

        self.buffer_down_btn = ctk.CTkButton(
            button_frame, text="▼", width=30, height=20, command=self.decrease_buffer_size
        )
        self.buffer_down_btn.pack(side="bottom")

        self.save_buffer_btn = ctk.CTkButton(buffer_size_frame, text="Apply", width=100, command=self.save_buffer_size)
        self.save_buffer_btn.pack(side="left", padx=10)

        ctk.CTkLabel(
            buffer_size_frame,
            text="(number of messages waiting to be spoken)",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(side="left", padx=5)

        # === RIGHT FRAME ===

        right_frame = ctk.CTkFrame(self.tab_settings, fg_color=BG_COLOR, bg_color=BG_COLOR)
        right_frame.pack(fill="both", expand=True, side="right", padx=PAD)

        # Main filters
        main_filters_frame = ctk.CTkFrame(right_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        main_filters_frame.pack(pady=(PAD, PAD_L), padx=PAD, fill="x", side="top", anchor="n")
        ctk.CTkLabel(main_filters_frame, text="Main filters", font=ctk.CTkFont(size=FONT_SZ_L), anchor="nw").pack(
            expand=True, fill="x", anchor="n"
        )
        filters_grid = ctk.CTkFrame(main_filters_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        filters_grid.pack(fill="x")

        # Minimum length
        ctk.CTkLabel(filters_grid, text="Min message length:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.min_length_var = ctk.StringVar(value=str(self.min_length))
        self.min_length_entry = ctk.CTkEntry(filters_grid, width=80, textvariable=self.min_length_var)
        self.min_length_entry.grid(row=0, column=1, padx=5, pady=5)

        # Maximum length
        ctk.CTkLabel(filters_grid, text="Max message length:").grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")
        self.max_length_var = ctk.StringVar(value=str(self.max_length))
        self.max_length_entry = ctk.CTkEntry(filters_grid, width=80, textvariable=self.max_length_var)
        self.max_length_entry.grid(row=0, column=3, padx=5, pady=5)

        # Delay
        ctk.CTkLabel(filters_grid, text="Delay between messages (sec):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.delay_var = ctk.StringVar(value=str(self.speak_delay))
        self.delay_entry = ctk.CTkEntry(filters_grid, width=80, textvariable=self.delay_var)
        self.delay_entry.grid(row=1, column=1, padx=5, pady=5)

        # Checkboxes
        self.filter_emojis_var = ctk.BooleanVar(value=self.filter_emojis)
        self.filter_emojis_check = ctk.CTkCheckBox(filters_grid, text="Remove emojis", variable=self.filter_emojis_var)
        self.filter_emojis_check.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.filter_links_var = ctk.BooleanVar(value=self.filter_links)
        self.filter_links_check = ctk.CTkCheckBox(filters_grid, text="Remove links", variable=self.filter_links_var)
        self.filter_links_check.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="w")

        self.filter_repeats_var = ctk.BooleanVar(value=self.filter_repeats)
        self.filter_repeats_check = ctk.CTkCheckBox(
            filters_grid, text="Filter repeats (anti-spam)", variable=self.filter_repeats_var
        )
        self.filter_repeats_check.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.read_names_var = ctk.BooleanVar(value=self.read_names)
        self.read_names_check = ctk.CTkCheckBox(filters_grid, text="Read author names", variable=self.read_names_var)
        self.read_names_check.grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky="w")

        self.ignore_system_var = ctk.BooleanVar(value=self.ignore_system)
        self.ignore_system_check = ctk.CTkCheckBox(
            filters_grid, text="Ignore system messages", variable=self.ignore_system_var
        )
        self.ignore_system_check.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.subscribers_only_var = ctk.BooleanVar(value=self.subscribers_only)
        self.subscribers_only_check = ctk.CTkCheckBox(
            filters_grid, text="Subscribers only", variable=self.subscribers_only_var
        )
        self.subscribers_only_check.grid(row=4, column=2, columnspan=2, padx=5, pady=5, sticky="w")

        # == Stop words ==

        stop_words_frame = ctk.CTkFrame(right_frame, fg_color=BG_COLOR, bg_color=BG_COLOR)
        stop_words_frame.pack(fill="x")

        ctk.CTkLabel(
            stop_words_frame, text="Stop words (ignore messages)", font=ctk.CTkFont(size=FONT_SZ_L), anchor="nw"
        ).pack(expand=True, fill="x", anchor="n")

        self.stop_words_text = ctk.CTkTextbox(stop_words_frame)
        self.stop_words_text.pack(fill="both")
        self.stop_words_text.insert("1.0", "\n".join(self.stop_words))

        save_stop_words_btn = ctk.CTkButton(
            stop_words_frame, text="💾 Save stop words", width=200, command=self.save_stop_words
        )
        save_stop_words_btn.pack(pady=10)

    def increase_buffer_size(self):
        try:
            current = int(self.buffer_size_var.get())
            new_value = min(current + 5, 200)
            self.buffer_size_var.set(str(new_value))
            self.save_buffer_size()
        except ValueError:
            self.buffer_size_var.set(str(self.buffer_maxsize))

    def decrease_buffer_size(self):
        try:
            current = int(self.buffer_size_var.get())
            new_value = max(current - 5, 1)
            self.buffer_size_var.set(str(new_value))
            self.save_buffer_size()
        except ValueError:
            self.buffer_size_var.set(str(self.buffer_maxsize))

    def check_cached_model(self):
        """Check for cached model and auto-load"""

        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-models_master"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            if os.path.exists(cache_dir):
                self.display_system_message(
                    "Found cached Silero model, loading... (this may take 1-2 minutes)", "system"
                )
                self.init_silero()
            else:
                self.display_system_message("Silero model not found in cache, click the load button", "system")
        except Exception as e:
            pass

    def init_silero(self):
        """Initialize Silero TTS model"""

        def init_thread():
            with self.model_lock:
                try:
                    self.window.after(
                        0,
                        lambda: self.tts_status_label.configure(
                            text="🔄 Loading Silero model (this may take 1-2 minutes)..."
                        ),
                    )

                    # Configure CPU threads
                    torch.set_num_threads(2)

                    # Clear memory before loading
                    gc.collect()

                    # Load model
                    self.silero_model, example_text = torch.hub.load(
                        repo_or_dir="snakers4/silero-models",
                        model="silero_tts",
                        language="ru",
                        speaker="v5_ru",
                        trust_repo=True,
                    )

                    torch.set_grad_enabled(False)

                    self.speak_silero("Warmup")

                    self.silero_available = True
                    self.is_tts_ready = True
                    self.silero_loaded = True

                    self.window.after(
                        0,
                        lambda: self.tts_status_label.configure(text=f"✅ Silero TTS ready!", text_color="#28a745"),
                    )

                    self.window.after(
                        0, lambda: self.display_system_message("Silero TTS successfully loaded", "success")
                    )

                except Exception as e:
                    error_msg = str(e)
                    self.window.after(
                        0,
                        lambda err=error_msg: self.tts_status_label.configure(
                            text=f"❌ Loading error: {err}", text_color="#dc3545"
                        ),
                    )
                    self.window.after(
                        0, lambda err=error_msg: self.display_system_message(f"Error loading Silero: {err}", "error")
                    )
                finally:
                    gc.collect()

        threading.Thread(target=init_thread, daemon=True).start()

    def convert_numbers_to_words(self, text):
        """Convert numbers to text representation"""

        def replace_number(match):
            num = match.group()
            try:
                if "." in num:
                    parts = num.split(".")
                    integer_part = num2words(int(parts[0]), lang="en")
                    fractional_part = num2words(int(parts[1]), lang="en")
                    return f"{integer_part} point {fractional_part}"
                else:
                    return num2words(int(num), lang="en")
            except:
                return num

        number_pattern = r"\b\d+(?:\.\d+)?\b"
        converted_text = re.sub(number_pattern, replace_number, text)
        return converted_text

    def speak_silero(self, text):
        """Speak text through Silero"""
        if not self.silero_available or not self.silero_model:
            return False

        with self.tts_lock:
            try:
                # Convert numbers to words
                text = self.convert_numbers_to_words(text)

                # Trim long text
                if len(text) > 490:
                    text = text[:487] + "..."

                # Generate audio
                with torch.no_grad():
                    audio = self.silero_model.apply_tts(
                        text=text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate,
                        put_accent=self.put_accent,
                        put_yo=self.put_yo,
                    )

                # Convert to numpy
                if torch.is_tensor(audio):
                    audio_numpy = audio.cpu().numpy()
                else:
                    audio_numpy = np.array(audio)

                # Delete tensor
                del audio

                # Normalize
                max_val = np.max(np.abs(audio_numpy))
                if max_val > 0:
                    audio_numpy = audio_numpy / max_val
                else:
                    audio_numpy = np.zeros(1000)

                # Apply volume
                audio_numpy = audio_numpy * self.volume

                # Apply speed (without scipy)
                if self.speech_rate != 1.0 and len(audio_numpy) > 0:
                    new_length = max(1, int(len(audio_numpy) / self.speech_rate))
                    indices = np.linspace(0, len(audio_numpy) - 1, new_length)
                    audio_numpy = np.interp(indices, np.arange(len(audio_numpy)), audio_numpy)

                # Add to queue
                if len(audio_numpy) > 0:
                    self.audio_queue.put(audio_numpy)
                    return True
                else:
                    return False

            except Exception as e:
                error_msg = str(e)
                self.window.after(0, lambda err=error_msg: self.display_system_message(f"TTS error: {err}", "error"))
                return False

    def process_speech_queue(self):
        """Process message queue for TTS"""
        if self.is_running_yt and self.silero_available and not self.is_speaking:
            try:
                with self.speech_lock:
                    # Check delay
                    try:
                        delay = float(self.delay_var.get())
                    except:
                        delay = 1.5

                    current_time = time.time()

                    if self.message_buffer and (current_time - self.last_speak_time) >= delay and not self.is_speaking:
                        author, message = self.message_buffer.popleft()

                        if self.read_names_var.get():
                            speak_text = f"{author} said: {message}"
                        else:
                            speak_text = message

                        self.is_speaking = True
                        self.last_speak_time = current_time

                        def speak_and_continue():
                            try:
                                success = self.speak_silero(speak_text)
                                if success:
                                    self.spoken_count += 1
                                    self.window.after(0, self.update_stats)
                            except Exception as e:
                                error_msg = str(e)
                                self.window.after(
                                    0, lambda err=error_msg: self.display_system_message(f"TTS error: {err}", "error")
                                )
                            finally:
                                self.is_speaking = False

                        threading.Thread(target=speak_and_continue, daemon=True).start()

            except Exception as e:
                self.display_system_message(f"Error in speech queue: {e}", "error")
                self.is_speaking = False

        self.window.after(200, self.process_speech_queue)

    def process_audio_queue(self):
        """Process audio queue"""
        try:
            with self.audio_lock:
                if not self.audio_queue.empty() and not self.is_speaking:
                    audio_data = self.audio_queue.get()
                    self.is_speaking = True

                    self.window.after(0, lambda: self.audio_indicator.configure(text="🔴", text_color="#dc3545"))

                    def play_audio(audio_to_play):
                        try:
                            sd.play(audio_to_play, self.sample_rate)
                            sd.wait()
                        except Exception as e:
                            print(f"Audio playback error: {e}")
                        finally:
                            self.is_speaking = False
                            self.window.after(0, lambda: self.audio_indicator.configure(text="🔴", text_color="white"))
                            # Delete audio after playback
                            try:
                                del audio_to_play
                            except:
                                pass
                            gc.collect()

                    # Start thread with audio_data passed
                    threading.Thread(target=play_audio, args=(audio_data,), daemon=True).start()
        except Exception as e:
            self.is_speaking = False
            self.window.after(0, lambda: self.audio_indicator.configure(text="🔴", text_color="white"))
            print(f"Audio queue error: {e}")

        self.window.after(100, self.process_audio_queue)

    def speak(self, text):
        """Main TTS method"""
        if text and self.silero_available:
            success = self.speak_silero(text)
            if success:
                self.spoken_count += 1
                self.window.after(0, self.update_stats)

    def clean_message(self, text):
        """Clean message from garbage"""
        original = text

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

        return text

    def is_spam(self, author, message):
        """Check for spam"""
        if not hasattr(self, "filter_repeats_var") or not self.filter_repeats_var.get():
            return False

        current_time = time.time()
        message_hash = hashlib.md5(f"{author}:{message}".encode()).hexdigest()

        if message_hash in self.message_hash_set:
            self.spam_count += 1
            return True

        self.last_message_time[author] = current_time
        self.message_hash_set.add(message_hash)

        if len(self.message_hash_set) > 100:
            self.message_hash_set = set(list(self.message_hash_set)[-100:])

        return False

    def contains_stop_words(self, text):
        """Check for stop words"""
        text_lower = text.lower()
        for word in self.stop_words:
            if word.lower() in text_lower:
                return True
        return False

    def get_chat_id(self):
        """Get chat ID"""
        try:
            response = self.youtube.videos().list(part="liveStreamingDetails", id=self.video_id_yt).execute()

            if response.get("items"):
                details = response["items"][0].get("liveStreamingDetails", {})
                return details.get("activeLiveChatId")
            else:
                self.display_system_message("Video not found or not a live stream", "error")
        except Exception as e:
            self.display_system_message(f"Error getting chat: {e}", "error")
        return None

    def fetch_messages_yt(self):
        """Fetch messages from YouTube chat"""
        self.chat_id = self.get_chat_id()
        if not self.chat_id:
            self.window.after(
                0,
                lambda: self.display_system_message("YouTube chat not found. Make sure the stream is active.", "error"),
            )
            self.window.after(0, self.toggle_connection_yt)
            return

        self.window.after(
            0, lambda: self.display_system_message("Connected to YouTube chat! Waiting for messages...", "success")
        )

        next_token = None

        while self.is_running_yt:
            if self.is_speaking or not self.audio_queue.empty():
                time.sleep(1)
                continue
            try:
                self.is_fetching = True

                # Request only necessary fields to minimise payload
                fields = (
                    "nextPageToken,pollingIntervalMillis,"
                    "items(id,snippet(displayMessage),authorDetails(displayName,isChatOwner,isChatSponsor,isChatModerator))"
                )

                response = (
                    self.youtube.liveChatMessages()
                    .list(
                        liveChatId=self.chat_id,
                        part="snippet,authorDetails",
                        pageToken=next_token,
                        fields=fields,
                    )
                    .execute()
                )

                next_token = response.get("nextPageToken")

                for item in response.get("items", []):
                    if not self.is_running_yt:
                        break

                    msg_id = item["id"]

                    if msg_id not in self.processed_messages:
                        self.processed_messages.add(msg_id)
                        self.messages_count += 1

                        snippet = item["snippet"]
                        author_details = item.get("authorDetails", {})

                        author = author_details.get("displayName", snippet.get("authorDisplayName", "Anonymous"))
                        if not author or author.strip() == "":
                            author = "Anonymous"

                        message = snippet.get("displayMessage", "")
                        is_member = (
                            author_details.get("isChatOwner", False)
                            or author_details.get("isChatSponsor", False)
                            or author_details.get("isChatModerator", False)
                        )

                        if hasattr(self, "subscribers_only_var") and self.subscribers_only_var.get() and not is_member:
                            continue

                        if hasattr(self, "ignore_system_var") and self.ignore_system_var.get():
                            if message.startswith(("subscribed", "donated", "became a member")):
                                continue

                        cleaned = self.clean_message(message)

                        try:
                            min_len = int(self.min_length_var.get())
                            max_len = int(self.max_length_var.get())
                        except:
                            min_len = 2
                            max_len = 200

                        if len(cleaned) < min_len:
                            continue

                        if len(cleaned) > max_len:
                            cleaned = cleaned[:max_len] + "..."

                        if self.is_spam(author, cleaned):
                            self.window.after(0, self.display_spam_message, "youtube", author, cleaned)
                            continue

                        if self.contains_stop_words(cleaned):
                            continue

                        if cleaned:
                            self.window.after(0, self.display_message, "youtube", author, cleaned)

                            if self.silero_available and self.is_running_yt:
                                self.message_buffer.append((author, cleaned))

                self.window.after(0, self.update_stats)
                # reset backoff on success
                self.api_backoff = 1
                self.is_fetching = False

                # respect polling interval returned by API when available
                poll_ms = response.get("pollingIntervalMillis")
                try:
                    sleep_seconds = max(1, int(poll_ms) / 1000) if poll_ms is not None else 5
                except Exception:
                    sleep_seconds = 5

                time.sleep(sleep_seconds)
                if len(self.processed_messages) > 1000:
                    self.processed_messages = set(list(self.processed_messages)[-500:])

            except Exception as e:
                error_msg = str(e)
                self.is_fetching = False
                if self.is_running_yt:
                    self.window.after(
                        0,
                        lambda err=error_msg: self.display_system_message(
                            f"Error fetching YouTube messages: {err}", "error"
                        ),
                    )
                    time.sleep(5)

    def display_message(self, platform, author, message):
        """Display message in log"""
        time_str = datetime.now().strftime("%H:%M:%S")

        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"[{time_str}] ")
        self.chat_text.insert("end", f"[{platform}] ", platform)
        self.chat_text.insert("end", f"{author}: ", "author")
        self.chat_text.insert("end", f"{message}\n", "message")
        self.chat_text.configure(state="disabled")

        if self.auto_scroll_var.get():
            self.chat_text.see("end")

    def display_spam_message(self, platform, author, message):
        """Display spam message"""
        time_str = datetime.now().strftime("%H:%M:%S")

        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"[{time_str}] ")
        self.chat_text.insert("end", f"[{platform}] ", platform)
        self.chat_text.insert("end", f"{author}: ", "author")
        self.chat_text.insert("end", f"{message} ", "spam")
        self.chat_text.insert("end", f"[SPAM]\n", "error")
        self.chat_text.configure(state="disabled")

        if self.auto_scroll_var.get():
            self.chat_text.see("end")

    def display_system_message(self, message, tag="system"):
        """Add system message"""
        time_str = datetime.now().strftime("%H:%M:%S")

        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"[{time_str}] [System] {message}\n", tag)
        self.chat_text.configure(state="disabled")

        if self.auto_scroll_var.get():
            self.chat_text.see("end")

    def update_stats(self):
        """Update statistics"""
        queue_size = len(self.message_buffer) if self.message_buffer else 0
        self.stats_label.configure(
            text=f"📊 Messages: {self.messages_count} | Spoken: {self.spoken_count} | Spam: {self.spam_count} | In queue: {queue_size}"
        )

    def reset_stats(self):
        """Reset statistics"""
        self.messages_count = 0
        self.spoken_count = 0
        self.spam_count = 0
        self.processed_messages.clear()
        if self.message_buffer:
            self.message_buffer.clear()
        self.message_hash_set.clear()
        self.last_message_time.clear()
        self.start_time = datetime.now()
        self.update_stats()
        self.display_system_message("Statistics reset")

    def clear_chat(self):
        """Clear chat log"""
        self.chat_text.configure(state="normal")
        self.chat_text.delete("0.0", "end")
        self.chat_text.configure(state="disabled")
        self.display_system_message("Log cleared")

    def export_chat_log(self):
        """Export chat log to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        if filename:
            try:
                self.chat_text.configure(state="normal")
                content = self.chat_text.get("0.0", "end")
                self.chat_text.configure(state="disabled")

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                self.display_system_message(f"Log exported to {filename}", "success")
            except Exception as e:
                self.display_system_message(f"Export error: {e}", "error")

    def change_voice(self, choice):
        """Change voice"""
        self.speaker = choice
        if self.silero_available:
            self.display_system_message(f"Voice changed to: {choice}")
        self.save_settings()

    def change_speed(self, value):
        """Change speech rate"""
        self.speech_rate = float(value)
        self.speed_label.configure(text=f"{self.speech_rate:.2f}x")
        self.save_settings()

    def change_volume(self, value):
        """Change volume"""
        self.volume = float(value)
        self.volume_label.configure(text=f"{self.volume:.0%}")
        self.save_settings()

    def toggle_accent(self):
        """Toggle accents"""
        self.put_accent = self.put_accent_var.get()
        self.save_settings()

    def toggle_yo(self):
        """Toggle yo replacement"""
        self.put_yo = self.put_yo_var.get()
        self.save_settings()

    def save_buffer_size(self):
        """Save queue depth"""
        try:
            value_str = self.buffer_size_var.get()

            if not value_str or value_str.strip() == "":
                self.buffer_size_var.set(str(self.buffer_maxsize))
                self.display_system_message("Please enter a number", "error")
                return

            new_size = int(float(value_str))

            if new_size < 1:
                new_size = 10
                self.display_system_message("Queue depth cannot be less than 1, set to 10", "warning")
            if new_size > 200:
                new_size = 200
                self.display_system_message("Maximum queue depth is 200", "warning")

            self.buffer_maxsize = new_size
            old_buffer = list(self.message_buffer) if self.message_buffer else []
            self.message_buffer = deque(maxlen=self.buffer_maxsize)
            for item in old_buffer:
                if len(self.message_buffer) < self.buffer_maxsize:
                    self.message_buffer.append(item)

            self.buffer_size_var.set(str(self.buffer_maxsize))
            self.display_system_message(f"Queue depth changed to: {self.buffer_maxsize}", "success")
            self.save_settings()
        except ValueError:
            self.buffer_size_var.set(str(self.buffer_maxsize))
            self.display_system_message("Error: please enter a valid number", "error")

    def save_stop_words(self):
        """Save stop words"""
        content = self.stop_words_text.get("1.0", "end").strip()
        self.stop_words = [word.strip() for word in content.split("\n") if word.strip()]
        self.display_system_message(f"Saved {len(self.stop_words)} stop words", "success")
        self.save_settings()

    def toggle_connection_yt(self):
        """Connect/disconnect from chat"""
        if not self.is_running_yt:
            # Connect
            if not self.api_entry_yt.get():
                messagebox.showwarning("Warning", "Please enter API key")
                return

            if not self.video_entry_yt.get():
                messagebox.showwarning("Warning", "Please enter video ID or URL")
                return

            if not self.silero_available:
                result = messagebox.askyesno("TTS not loaded", "Silero model not loaded. Continue without TTS?")
                if not result:
                    return

            self.api_key_yt = self.api_entry_yt.get()
            self.video_id_yt = self.video_entry_yt.get()

            if "youtube.com" in self.video_id_yt or "youtu.be" in self.video_id_yt:
                parsed = urllib.parse.urlparse(self.video_id_yt)
                if "youtu.be" in parsed.netloc:
                    self.video_id_yt = parsed.path[1:]
                elif "watch" in parsed.path:
                    query = urllib.parse.parse_qs(parsed.query)
                    self.video_id_yt = query.get("v", [None])[0]
                elif "embed" in parsed.path:
                    self.video_id_yt = parsed.path.split("/")[-1]

            if not self.video_id_yt:
                messagebox.showerror("Error", "Could not determine video ID")
                return

            try:
                self.youtube = build("youtube", "v3", developerKey=self.api_key_yt)
                self.display_system_message("YouTube API connected", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Could not connect to YouTube API: {e}")
                return

            self.is_running_yt = True
            self.start_time = datetime.now()
            self.chat_thread = threading.Thread(target=self.fetch_messages_yt, daemon=True)
            self.chat_thread.start()

            self.connect_btn_yt.configure(text="Disconnect", fg_color="#dc3545", hover_color="#c82333")
            self.connection_status_yt.configure(text="🟢", text_color="#218838")
            self.save_api_btn_yt.configure(state="disabled")

        else:
            self.is_running_yt = False
            self.is_fetching = False
            self.connect_btn_yt.configure(text="Connect", fg_color="#28a745", hover_color="#218838")
            self.connection_status_yt.configure(text="🔴", text_color="red")
            self.save_api_btn_yt.configure(state="normal")

            self.display_system_message("Disconnected from chat")

    def save_yt_api_key(self):
        """Save API key"""
        self.api_key_yt = self.api_entry_yt.get()
        self.save_settings()
        self.display_system_message("YouTube API key saved", "success")

    def save_settings(self):
        """Save settings to file"""
        settings = {
            "api_key": self.api_key_yt,
            "video_id": self.video_id_yt,
            "silero_speaker": self.speaker,
            "speech_rate": self.speech_rate,
            "volume": self.volume,
            "put_accent": self.put_accent,
            "put_yo": self.put_yo,
            "min_length": self.min_length_var.get() if hasattr(self, "min_length_var") else "2",
            "max_length": self.max_length_var.get() if hasattr(self, "max_length_var") else "200",
            "delay": self.delay_var.get() if hasattr(self, "delay_var") else "1.5",
            "filter_emojis": self.filter_emojis_var.get() if hasattr(self, "filter_emojis_var") else True,
            "filter_links": self.filter_links_var.get() if hasattr(self, "filter_links_var") else True,
            "filter_repeats": self.filter_repeats_var.get() if hasattr(self, "filter_repeats_var") else True,
            "ignore_system": self.ignore_system_var.get() if hasattr(self, "ignore_system_var") else True,
            "subscribers_only": self.subscribers_only_var.get() if hasattr(self, "subscribers_only_var") else False,
            "read_names": self.read_names_var.get() if hasattr(self, "read_names_var") else False,
            "auto_scroll": self.auto_scroll_var.get() if hasattr(self, "auto_scroll_var") else True,
            "stop_words": self.stop_words,
            "buffer_size": self.buffer_maxsize,
        }

        try:
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if hasattr(self, "display_system_message"):
                self.display_system_message(f"Error saving settings: {e}", "error")

    def load_settings(self):
        """Load settings from file"""
        self.min_length = 2
        self.max_length = 200
        self.speak_delay = 1.5
        self.filter_emojis = True
        self.filter_links = True
        self.filter_repeats = True
        self.ignore_system = True
        self.subscribers_only = False
        self.read_names = False
        self.auto_scroll = True
        self.stop_words = []
        self.buffer_maxsize = BUFFER_SIZE

        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
                self.api_key_yt = settings.get("api_key", "")
                self.video_id_yt = settings.get("video_id", "")
                self.speaker = settings.get("silero_speaker", "xenia")
                self.speech_rate = settings.get("speech_rate", 1.0)
                self.volume = settings.get("volume", 1.0)
                self.put_accent = settings.get("put_accent", True)
                self.put_yo = settings.get("put_yo", True)

                self.min_length = int(settings.get("min_length", 2))
                self.max_length = int(settings.get("max_length", 200))
                self.speak_delay = float(settings.get("delay", 1.5))
                self.filter_emojis = settings.get("filter_emojis", True)
                self.filter_links = settings.get("filter_links", True)
                self.filter_repeats = settings.get("filter_repeats", True)
                self.ignore_system = settings.get("ignore_system", True)
                self.subscribers_only = settings.get("subscribers_only", False)
                self.read_names = settings.get("read_names", False)
                self.auto_scroll = settings.get("auto_scroll", True)
                self.stop_words = settings.get("stop_words", [])

                buffer_size = settings.get("buffer_size", BUFFER_SIZE)
                try:
                    self.buffer_maxsize = int(buffer_size)
                    if self.buffer_maxsize < 1:
                        self.buffer_maxsize = 10
                    elif self.buffer_maxsize > 200:
                        self.buffer_maxsize = 200
                except (ValueError, TypeError):
                    self.buffer_maxsize = BUFFER_SIZE

        except FileNotFoundError:
            pass

    def on_closing(self):
        """Handle window closing"""
        self.is_running_yt = False
        self.is_fetching = False

        # Clear Silero model
        if self.silero_model is not None:
            try:
                del self.silero_model
                self.silero_model = None
            except:
                pass

        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass

        # Stop audio
        if "sd" in sys.modules:
            try:
                sd.stop()
            except:
                pass

        # Garbage collection
        gc.collect()

        # Save settings
        self.save_settings()

        self.window.destroy()

    def run(self):
        """Run application"""
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()


if __name__ == "__main__":
    app = FJChatVoice()
    app.run()
