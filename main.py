#!/usr/bin/env python3
import os
import sys

# GPU Detection moved to after config load

# --- CONFIGURATION FOR FULL SCREEN HMI ---
# Must be done before other Kivy imports
from kivy.config import Config
# Config.set('graphics', 'fullscreen', 'auto') # Force full screen
# Config.set('graphics', 'show_cursor', '1')   # Show mouse cursor (set to '0' for touch-only)
Config.write()
# -----------------------------------------


import cv2
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import requests
import uuid
import sqlite3

# Setup main app logger
import logging
main_logger = logging.getLogger("MainApp")
main_logger.setLevel(logging.DEBUG)
if not main_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    main_logger.addHandler(_handler)

# Kivy Imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.texture import Texture
import logging

# Custom Modules
from modules.camera import CameraManager
from modules.database import DatabaseManager
from modules.api_sender import APISender
from modules.process_worker import submit_batch
from modules.utils import setup_logging, create_timestamp, save_last_batch, load_last_batch

# Lazy import QRDetector to avoid torch/torchvision issues at startup
_QRDetector = None

def get_qr_detector_class():
    """Lazy load QRDetector to handle import errors gracefully."""
    global _QRDetector
    if _QRDetector is None:
        try:
            from modules.detector import QRDetector
            _QRDetector = QRDetector
            print("[MAIN] QRDetector loaded successfully")
        except Exception as e:
            print(f"[WARNING] QRDetector import failed: {e}")
            # Create a dummy detector that returns empty results
            class DummyQRDetector:
                def __init__(self):
                    print("[WARNING] Using dummy QR detector - no detection will occur")
                def detect_and_decode(self, frame):
                    return [], 0
            _QRDetector = DummyQRDetector
    return _QRDetector

from modules.utils import setup_logging, create_timestamp, save_last_batch, load_last_batch
from config import (
    CAMERA_CONFIG, DEFAULT_KEG_COUNT, MAX_KEG_COUNT, SAVE_FOLDER,
    MIN_KEG_COUNT, STABILITY_THRESHOLD, COLOR_SCHEME,
    FOV_ENABLED, FOV_BOUNDARY_RATIO,
    CLOUD_CONFIG_ENDPOINT, CLOUD_SYNC_INTERVAL, CAMERA_MAC_ID, GPU_CONFIG
)

# ===== GPU DETECTION (Conditional) =====
print("\n" + "="*60)
print("   PALLETIZATION SYSTEM")
print("="*60 + "\n")

try:
    if not GPU_CONFIG.get('force_cpu', False):
        from modules.gpu_utils import detect_gpu, warm_up_gpu, GPU_STATUS
        detect_gpu()
        if GPU_STATUS.get('cuda_available'):
            print("GPU MODE ENABLED")
            warm_up_gpu()
        else:
            print("CPU MODE (No GPU detected)")
    else:
        print("CPU MODE (Forced by Config)")
        # Define dummy GPU_STATUS if needed
        GPU_STATUS = {'cuda_available': False}
except Exception as e:
    print(f"GPU Detection skipped: {e}")
    GPU_STATUS = {'cuda_available': False}
# =======================================

# Setup logging
logger = setup_logging()

# Color Constants
COLOR_BG_DARK = (1, 1, 1, 1)  # White
COLOR_PANEL_BG = (0.95, 0.95, 0.98, 1)  # Very light blue-white
COLOR_HIGHLIGHT = (0.0, 0.45, 0.85, 1)  # Brighter Blue
COLOR_TEXT_LIGHT = (0.1, 0.1, 0.1, 1)  # Dark Grey (for contrast on white)
COLOR_ALERT_RED = (0.9, 0.2, 0.2, 1)
COLOR_STATUS_GREEN = (0.1, 0.7, 0.2, 1)
COLOR_STATUS_ORANGE = (1, 0.6, 0, 1)
COLOR_STATUS_BLUE = (0.2, 0.6, 0.9, 1)
COLOR_BUTTON_NORMAL = (0.9, 0.92, 0.96, 1)  # Light Blueish Grey

def hex_color(rgb_tuple):
    """Convert RGB tuple to hex color string"""
    return f'#{int(rgb_tuple[0]*255):02x}{int(rgb_tuple[1]*255):02x}{int(rgb_tuple[2]*255):02x}{int(rgb_tuple[3]*255):02x}'

class ToastMessage(ModalView):
    """Toast-like popup for brief user messages"""
    def __init__(self, message, msg_type="info", duration=3, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.6, None)
        self.height = 60
        self.pos_hint = {'center_x': 0.5, 'top': 0.95}
        self.background_color = [0, 0, 0, 0]
        self.auto_dismiss = False
        
        # Choose color based on type
        if msg_type == "success":
            bg_color = COLOR_STATUS_GREEN
        elif msg_type == "error":
            bg_color = COLOR_ALERT_RED
        elif msg_type == "warning":
            bg_color = COLOR_STATUS_ORANGE
        else:
            bg_color = COLOR_STATUS_BLUE
        
        container = BoxLayout(padding=10)
        with container.canvas.before:
            Color(*bg_color)
            self.rect = Rectangle(pos=container.pos, size=container.size)
        container.bind(pos=lambda *x: setattr(self.rect, 'pos', container.pos))
        container.bind(size=lambda *x: setattr(self.rect, 'size', container.size))
        
        label = Label(
            text=message,
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        container.add_widget(label)
        self.add_widget(container)
        
        # Auto-dismiss after duration
        Clock.schedule_once(lambda dt: self.dismiss(), duration)


class ConfirmationModal(ModalView):
    """Modal for user confirmations"""
    def __init__(self, title, message, on_confirm, on_cancel=None, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.6, 0.35)
        self.background_color = hex_color((1, 1, 1, 0.95))
        self.on_confirm_callback = on_confirm
        self.on_cancel_callback = on_cancel
        self.auto_dismiss = False
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with layout.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # Solid light background
            self.rect = Rectangle(pos=layout.pos, size=layout.size)
        layout.bind(pos=lambda *x: setattr(self.rect, 'pos', layout.pos))
        layout.bind(size=lambda *x: setattr(self.rect, 'size', layout.size))
        
        # Title
        title_label = Label(
            text=title,
            size_hint_y=None,
            height=30,
            font_size='18sp',
            bold=True,
            color=hex_color(COLOR_HIGHLIGHT)
        )
        
        # Message
        message_label = Label(
            text=message,
            size_hint_y=None,
            height=60,
            font_size='14sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            halign='center',
            text_size=(380, None)
        )
        
        # Buttons
        btn_layout = BoxLayout(size_hint_y=None, height=45, spacing=15)
        
        cancel_btn = Button(
            text='CANCEL',
            font_size='14sp',
            background_color=hex_color((0.4, 0.4, 0.4, 1)),
            background_normal='',
            on_press=self.cancel
        )
        
        confirm_btn = Button(
            text='CONFIRM',
            font_size='14sp',
            bold=True,
            background_color=hex_color(COLOR_STATUS_GREEN),
            background_normal='',
            on_press=self.confirm
        )
        
        btn_layout.add_widget(cancel_btn)
        btn_layout.add_widget(confirm_btn)
        
        layout.add_widget(title_label)
        layout.add_widget(message_label)
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def confirm(self, instance):
        self.dismiss()
        if self.on_confirm_callback:
            self.on_confirm_callback()
    
    def cancel(self, instance):
        self.dismiss()
        if self.on_cancel_callback:
            self.on_cancel_callback()


class CountModal(ModalView):
    def __init__(self, on_confirm, current_count, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.5, 0.6)  # Taller for keypad
        self.background_color = hex_color((0.95, 0.95, 0.95, 1)) # Explicit Light Grey for contrast
        self.on_confirm = on_confirm
        self.current_value = str(current_count)
        self.reset_input = True # Flag to overwrite value on first keypress
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with layout.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # Solid light background
            self.bg_rect = Rectangle(pos=layout.pos, size=layout.size)
        layout.bind(pos=lambda *x: setattr(self.bg_rect, 'pos', layout.pos))
        layout.bind(size=lambda *x: setattr(self.bg_rect, 'size', layout.size))
        
        # Title
        title = Label(
            text='Set Target Keg Count',
            size_hint_y=None,
            height=40,
            font_size='22sp',
            bold=True,
            color=hex_color(COLOR_HIGHLIGHT)
        )
        
        # Display Area
        self.display_label = Label(
            text=self.current_value,
            size_hint_y=None,
            height=60,
            font_size='40sp',
            bold=True,
            color=hex_color(COLOR_TEXT_LIGHT),
            halign='center'
        )
        
        # Keypad Grid
        grid = GridLayout(cols=3, spacing=10, size_hint_y=1)
        
        # Keys 1-9
        for i in range(1, 10):
            btn = Button(
                text=str(i),
                font_size='24sp',
                bold=True,
                background_color=hex_color(COLOR_BUTTON_NORMAL),
                color=hex_color(COLOR_TEXT_LIGHT),
                background_normal=''
            )
            btn.bind(on_press=self.on_key_press)
            grid.add_widget(btn)
        
        # Bottom Row: CLEAR, 0, OK
        clear_btn = Button(
            text='CLR',
            font_size='20sp',
            bold=True,
            background_color=hex_color(COLOR_ALERT_RED),
            color=hex_color((1, 1, 1, 1)), # White text for Red button
            background_normal=''
        )
        clear_btn.bind(on_press=self.on_clear)
        grid.add_widget(clear_btn)
        
        zero_btn = Button(
            text='0',
            font_size='24sp',
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_TEXT_LIGHT),
            background_normal=''
        )
        zero_btn.bind(on_press=self.on_key_press)
        grid.add_widget(zero_btn)
        
        ok_btn = Button(
            text='OK',
            font_size='20sp',
            bold=True,
            background_color=hex_color(COLOR_STATUS_GREEN),
            color=hex_color((1, 1, 1, 1)), # White text for Green button
            background_normal=''
        )
        ok_btn.bind(on_press=self.confirm)
        grid.add_widget(ok_btn)
        
        layout.add_widget(title)
        layout.add_widget(self.display_label)
        layout.add_widget(grid)
        
        # Cancel Button at very bottom
        cancel_btn = Button(
            text='CANCEL',
            size_hint_y=None,
            height=50,
            font_size='16sp',
            background_color=hex_color((0.4, 0.4, 0.4, 1)),
            background_normal='',
            on_press=lambda x: self.dismiss()
        )
        layout.add_widget(cancel_btn)
        
        self.add_widget(layout)
    
    def on_key_press(self, instance):
        if self.reset_input:
            self.current_value = instance.text
            self.reset_input = False
        else:
            if self.current_value == '0':
                self.current_value = instance.text
            else:
                if len(self.current_value) < 2: # Limit to 2 digits (max 99)
                    self.current_value += instance.text
        self.display_label.text = self.current_value
        
    def on_clear(self, instance):
        self.current_value = '0'
        self.reset_input = True # Allow fresh start after clear
        self.display_label.text = self.current_value
    
    def confirm(self, instance):
        try:
            count = int(self.current_value)
            if count > 0:
                self.on_confirm(count)
                self.dismiss()
            else:
                 # Provide Feedback: Shake or Toast? Since this is a modal, label flash is easiest
                 original_text = self.display_label.text
                 self.display_label.text = "INVALID (>0)"
                 self.display_label.color = hex_color(COLOR_ALERT_RED)
                 
                 def reset_display(dt):
                     self.display_label.text = self.current_value
                     self.display_label.color = hex_color(COLOR_TEXT_LIGHT)
                 
                 Clock.schedule_once(reset_display, 1)
        except ValueError:
            pass

class BatchModal(ModalView):
    def __init__(self, on_confirm, current_batch, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.6, 0.55)
        self.background_color = hex_color((1, 1, 1, 0.95))
        self.on_confirm = on_confirm
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=12)
        with layout.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # Solid light background
            self.rect = Rectangle(pos=layout.pos, size=layout.size)
        layout.bind(pos=lambda *x: setattr(self.rect, 'pos', layout.pos))
        layout.bind(size=lambda *x: setattr(self.rect, 'size', layout.size))
        
        # Title
        title = Label(
            text='Batch Details',
            size_hint_y=None,
            height=30,
            font_size='18sp',
            bold=True,
            color=hex_color(COLOR_HIGHLIGHT)
        )
        
        # Batch Input
        batch_label = Label(text='Batch Number:', size_hint_y=None, height=20, halign='left', text_size=(400, None))
        self.input_field = TextInput(
            text=current_batch if current_batch else 'BATCH-',
            multiline=False,
            font_size='18sp',
            halign='center',
            write_tab=False,
            background_color=hex_color((0.95, 0.95, 0.95, 1)),
            foreground_color=hex_color(COLOR_TEXT_LIGHT),
            cursor_color=hex_color(COLOR_HIGHLIGHT),
            size_hint_y=None,
            height=40
        )
        
        # Format hint label
        format_hint = Label(
            text='Format: BATCH-XXX',
            size_hint_y=None,
            height=20,
            font_size='11sp',
            color=hex_color(COLOR_STATUS_BLUE),
            halign='center'
        )
        
        # Error message label
        self.error_label = Label(
            text='',
            size_hint_y=None,
            height=25,
            font_size='12sp',
            color=hex_color(COLOR_ALERT_RED)
        )
        
        # Action buttons
        btn_layout = BoxLayout(size_hint_y=None, height=40, spacing=10)
        cancel_btn = Button(
            text='Cancel',
            background_color=hex_color((0.4, 0.4, 0.4, 1)),
            on_press=lambda x: self.dismiss()
        )
        ok_btn = Button(
            text='SET BATCH',
            font_size='14sp',
            bold=True,
            background_color=hex_color(COLOR_HIGHLIGHT),
            on_press=self.confirm
        )
        
        btn_layout.add_widget(cancel_btn)
        btn_layout.add_widget(ok_btn)
        
        layout.add_widget(title)
        layout.add_widget(batch_label)
        layout.add_widget(self.input_field)
        layout.add_widget(format_hint)
        layout.add_widget(self.error_label)
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def show_error(self, message):
        self.error_label.text = f" {message}"
    
    def clear_error(self):
        self.error_label.text = ''
    
    def confirm(self, instance):
        import re
        self.clear_error()
        batch = self.input_field.text.strip()
        
        if not batch:
            self.show_error("Batch number is required!")
            return
        
        # Check for lowercase letters - reject them
        if batch != batch.upper():
            self.show_error("Use UPPERCASE only! (e.g., BATCH-001)")
            return
        
        # Validate format: BATCH-XXX where XXX is only numbers
        pattern = r'^BATCH-\d+$'
        if not re.match(pattern, batch):
            self.show_error("Invalid format! Use BATCH-XXX (numbers only)")
            return
        
        # Extract the number part and validate it's not empty
        number_part = batch.replace('BATCH-', '')
        if not number_part or int(number_part) <= 0:
            self.show_error("Enter a valid batch number (e.g., BATCH-001)")
            return
        
        self.on_confirm(batch)
        self.dismiss()

class SimpleKegHMI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [10, 10, 10, 10]
        self.spacing = 10
        
        Window.clearcolor = COLOR_BG_DARK
        
        main_logger.info("Initializing SimpleKegHMI...")
        
        # Initialize components
        self.camera = None
        self.database = DatabaseManager()
        self.api_sender = APISender()
        # Initialize QR Detector (lazy loaded)
        QRDetectorClass = get_qr_detector_class()
        self.qr_detector = QRDetectorClass()
        
        # Async Detection Setup
        self.detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LiveDetector")
        self.current_detection_task = None
        self.latest_qr_results = []
        
        # Track last captured QR codes to detect same kegs
        self.last_captured_qr_set = set()
        
        # System state
        self.detection_active = False
        self.required_keg_count = DEFAULT_KEG_COUNT
        self.current_count = 0
        self.prev_count = 0
        self.stability_counter = 0
        self.is_auto_mode = True  # Start with auto mode by default
        self.processing = False
        self.auto_confirm_pending = False
        self.auto_pending_frame = None  # Store frame during confirmation dialog
        self.auto_capture_cooldown = False
        self.beer_types = ["Loading..."]  # Wait for API
        self.beer_type_map = {} # Mapping name -> id
        
        # === PERFORMANCE OPTIMIZATIONS ===
        # Reusable texture for preview (avoids creating new texture every frame)
        self._preview_texture = None
        self._texture_size = (0, 0)
        
        # Data ready for sending state
        self.data_ready_to_send = False
        self._pending_capture = None
        
        # Performance tracking
        self._frame_times = []
        self._last_perf_log = time.time()
        
        # Detection resize (process smaller frames for faster detection)
        # This doesn't affect preview - preview shows full resolution
        # NOTE: Set to None to use full 4K resolution for maximum detection accuracy
        self._detection_resize = None  # Full 4K - no resize, best accuracy
        
        main_logger.info("Performance optimizations enabled: texture reuse, full 4K detection")


        # Load last batch number
        self.last_batch_number = load_last_batch()
        
        # Logs
        self.workflow_logs = []
        
        # Build UI
        self.build_ui()
        
        # Start systems
        self.start_detection_system()
        
        # Schedule tasks - Preview at 30 FPS for smooth display
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS for smooth preview
        Clock.schedule_interval(self.check_network, 30)
        Clock.schedule_interval(self.update_filling_date, 60)  # Update filling date every minute
        Clock.schedule_once(self.recover_batches, 1)
        Clock.schedule_once(lambda dt: self.sync_cloud(), 2)
        Clock.schedule_once(self.fetch_beer_types, 3)
    
    def build_ui(self):
        """Build simplified and clean UI layout"""
        
        # Top row: Title and status
        top_bar = BoxLayout(size_hint_y=None, height=40, spacing=10)
        
        self.title_label = Label(
            text='[b]KEG COUNTING SYSTEM[/b]',
            markup=True,
            font_size='20sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            halign='left'
        )
        
        self.system_status = Label(
            text='[b]● READY[/b]',
            markup=True,
            font_size='14sp',
            color=hex_color(COLOR_STATUS_GREEN),
            halign='right'
        )
        
        top_bar.add_widget(self.title_label)
        top_bar.add_widget(self.system_status)
        
        # Main content area
        main_content = BoxLayout(orientation='horizontal', spacing=10)
        
        # Left panel - Camera (60%)
        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=10)
        
        # Camera preview with border
        cam_container = BoxLayout(padding=2)
        with cam_container.canvas.before:
            Color(*COLOR_BG_DARK)
            Rectangle(pos=cam_container.pos, size=cam_container.size)
            Color(0.3, 0.3, 0.3, 1)
            Line(rectangle=(cam_container.pos[0], cam_container.pos[1], 
                          cam_container.size[0], cam_container.size[1]), width=1.5)
        
        # Fix deprecated properties
        self.preview_image = Image()
        self.preview_image.fit_mode = 'contain'  # Use fit_mode instead of allow_stretch and keep_ratio
        cam_container.add_widget(self.preview_image)
        left_panel.add_widget(cam_container)
        
        # Detection status below camera
        status_box = GridLayout(cols=2, rows=1, spacing=5, size_hint_y=None, height=40)
        
        # Target count
        target_label = Label(
            text='TARGET COUNT:',
            font_size='12sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            halign='left'
        )
        self.target_display = Label(
            text=str(self.required_keg_count),
            font_size='24sp',
            bold=True,
            color=hex_color(COLOR_HIGHLIGHT),
            halign='right'
        )
        
        self.current_display = Label(opacity=0) # Dummy
        
        status_box.add_widget(target_label)
        status_box.add_widget(self.target_display)
        
        left_panel.add_widget(status_box)
        
        # Right panel - Controls (40%)
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=10)
        
        # Mode selector - Larger buttons
        mode_box = BoxLayout(orientation='vertical', size_hint_y=None, height=80, spacing=5)
        mode_label = Label(
            #text='OPERATION MODE:',
            font_size='12sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            size_hint_y=None,
            height=20
        )
        
        mode_buttons = BoxLayout(spacing=5)
        self.auto_btn = ToggleButton(
            text='AUTO',
            group='mode',
            state='down',
            font_size='14sp',
            background_normal='',
            color=hex_color(COLOR_TEXT_LIGHT),
            background_color=hex_color(COLOR_HIGHLIGHT)
        )
        self.auto_btn.bind(on_press=self.set_auto_mode)
        
        self.manual_btn = ToggleButton(
            text='MANUAL',
            group='mode',
            state='normal',
            font_size='14sp',
            background_normal='',
            background_color=hex_color(COLOR_BUTTON_NORMAL)
        )
        self.manual_btn.bind(on_press=self.set_manual_mode)
        
        mode_buttons.add_widget(self.auto_btn)
        mode_buttons.add_widget(self.manual_btn)
        
        mode_box.add_widget(mode_label)
        mode_box.add_widget(mode_buttons)
        right_panel.add_widget(mode_box)
        
        # --- CONFIGURATION PANEL (UPDATED FOR SYMMETRY) ---
        config_box = GridLayout(cols=2, rows=4, spacing=10, size_hint_y=None, height=350)
        
        # Helper for symmetric styling
        def create_config_label(text):
            return Button(
                text=text,
                font_size='16sp',
                bold=True,
                color=hex_color(COLOR_HIGHLIGHT),
                background_color=hex_color(COLOR_BUTTON_NORMAL),
                background_normal='',
                size_hint_x=0.5
            )

        # 1. Beer Type
        config_box.add_widget(create_config_label('BEER TYPE:'))
        
        self.beer_display = Spinner(
            text='Loading...',
            values=[],
            font_size='16sp',
            bold=True,  # Match label
            color=hex_color(COLOR_HIGHLIGHT), # Match label color
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            background_normal='',
            size_hint_x=0.5,
            option_cls=lambda **kwargs: Button(
                **kwargs, 
                size_hint_y=None, 
                height=50,
                font_size='16sp',
                background_color=hex_color(COLOR_PANEL_BG),
                color=hex_color(COLOR_TEXT_LIGHT),
                background_normal=''
            )
        )
        self.beer_display.bind(text=self.on_beer_type_select)
        config_box.add_widget(self.beer_display)
        
        # 2. Target Count
        config_box.add_widget(create_config_label('TARGET COUNT:'))
        
        self.count_button = Button(
            text=str(self.required_keg_count),
            font_size='24sp', # Value slightly larger for emphasis, but box same
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_HIGHLIGHT), # Match label color
            background_normal='',
            size_hint_x=0.5,
            on_press=self.change_count
        )
        config_box.add_widget(self.count_button)
        
        # 3. Batch Number
        config_box.add_widget(create_config_label('BATCH NO:'))
        
        self.batch_display = Button(
            text=self.last_batch_number,
            font_size='16sp',
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_HIGHLIGHT), # Match label color
            background_normal='',
            size_hint_x=0.5,
            on_press=self.edit_batch
        )
        config_box.add_widget(self.batch_display)
        
        # 4. Filling Date
        config_box.add_widget(create_config_label('FILLING DATE:'))
        
        self.filling_date_display = Button(
            text=datetime.now().strftime('%d-%m-%Y %H:%M'),
            font_size='16sp',
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_HIGHLIGHT), # Match label color
            background_normal='',
            size_hint_x=0.5
        )
        config_box.add_widget(self.filling_date_display)
        

        
        right_panel.add_widget(config_box)
        # -----------------------------------------------------------
        
        # Status indicator - Reduced height to save space
        status_box = BoxLayout(orientation='vertical', spacing=3, size_hint_y=None, height=40)
        
        self.status_label = Label(
            text='',
            font_size='14sp',
            bold=True,
            color=hex_color(COLOR_TEXT_LIGHT),
            size_hint_y=None,
            height=30
        )
        
        # Duplicate batch warning label
        self.duplicate_warning = Label(
            text='',
            font_size='12sp',
            color=hex_color(COLOR_STATUS_ORANGE),
            size_hint_y=None,
            height=20
        )
        
        status_box.add_widget(self.status_label)
        status_box.add_widget(self.duplicate_warning)
        
        # STATUS DISPLAY AREA - For showing pallet ID (compact)
        status_display_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=55,
            spacing=1,
            padding=[5, 1, 5, 1]
        )

        # Success Message (for pallet creation)
        self.success_display = Label(
            text='',
            font_size='13sp',
            bold=True,
            color=hex_color(COLOR_STATUS_GREEN),
            size_hint_y=None,
            height=22,
            halign='center',
            text_size=(380, None)
        )

        # Pallet ID Display
        self.pallet_display = Label(
            text='',
            font_size='14sp',
            bold=True,
            color=hex_color(COLOR_HIGHLIGHT),
            size_hint_y=None,
            height=28,
            halign='center',
            text_size=(380, None)
        )

        # Batch Info
        self.batch_info_label = Label(
            text='',
            font_size='11sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            size_hint_y=None,
            height=18,
            halign='center',
            text_size=(380, None)
        )

        status_display_box.add_widget(self.success_display)
        status_display_box.add_widget(self.pallet_display)
        status_display_box.add_widget(self.batch_info_label)

        # Add pallet display box first
        right_panel.add_widget(status_display_box)
        
        # Add status_box (Target Achieved) right before action buttons - closer to SYNC/LOGS
        right_panel.add_widget(status_box)

        # Initialize as empty
        self.clear_status_display()
        
        # === PROCESS STATUS DISPLAY ===
        process_status_box = BoxLayout(orientation='vertical', size_hint_y=None, height=70, spacing=2, padding=[5, 5, 5, 5])
        with process_status_box.canvas.before:
            Color(0.12, 0.14, 0.18, 1)  # Dark background
            self._process_bg = Rectangle(pos=process_status_box.pos, size=process_status_box.size)
        process_status_box.bind(pos=lambda i, v: setattr(self._process_bg, 'pos', v))
        process_status_box.bind(size=lambda i, v: setattr(self._process_bg, 'size', v))
        
        self.process_title = Label(
            text='PROCESS STATUS',
            font_size='11sp',
            color=(0.6, 0.6, 0.6, 1),
            size_hint_y=None,
            height=16,
            halign='center'
        )
        
        self.process_status_label = Label(
            text='⏳ Waiting for kegs...',
            font_size='14sp',
            bold=True,
            color=(0.5, 0.8, 1, 1),
            size_hint_y=None,
            height=24,
            halign='center'
        )
        
        self.process_detail_label = Label(
            text='Place kegs under camera',
            font_size='11sp',
            color=(0.7, 0.7, 0.7, 1),
            size_hint_y=None,
            height=18,
            halign='center'
        )
        
        process_status_box.add_widget(self.process_title)
        process_status_box.add_widget(self.process_status_label)
        process_status_box.add_widget(self.process_detail_label)
        right_panel.add_widget(process_status_box)
        
        # === SEND TO SERVER BUTTON (Large, prominent) ===
        self.send_btn = Button(
            text='📤  SEND TO SERVER',
            font_size='20sp',
            bold=True,
            background_normal='',
            background_color=(0.3, 0.3, 0.35, 1),  # Grey when disabled
            color=(0.5, 0.5, 0.5, 1),
            size_hint_y=None,
            height=60,
            disabled=True
        )
        self.send_btn.bind(on_press=self.send_to_server)
        right_panel.add_widget(self.send_btn)
        
        # Action buttons - Smaller grid now
        action_grid = GridLayout(cols=4, rows=1, spacing=8, size_hint_y=None, height=50)
        
        sync_btn = Button(
            text='SYNC',
            font_size='16sp',
            bold=True,
            background_color=hex_color(COLOR_STATUS_BLUE),
            background_normal='',
            on_press=lambda x: self.sync_cloud()
        )
        
        logs_btn = Button(
            text='LOGS',
            font_size='16sp',
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_TEXT_LIGHT),
            background_normal='',
            on_press=self.show_logs
        )
        
        self.capture_btn = Button(
            text='CAPTURE',
            font_size='18sp',
            bold=True,
            background_color=hex_color(COLOR_BUTTON_NORMAL),
            color=hex_color(COLOR_TEXT_LIGHT),
            background_normal='',
            disabled=True,
            on_press=self.force_capture
        )
        
        exit_btn = Button(
            text='EXIT',
            font_size='16sp',
            bold=True,
            background_color=hex_color(COLOR_ALERT_RED),
            background_normal='',
            on_press=self.confirm_exit
        )
        
        action_grid.add_widget(sync_btn)
        action_grid.add_widget(logs_btn)
        action_grid.add_widget(self.capture_btn)
        action_grid.add_widget(exit_btn)
        
        right_panel.add_widget(action_grid)
        
        # Network status
        network_box = BoxLayout(size_hint_y=None, height=30)
        self.network_status = Label(
            text='● ONLINE',
            font_size='12sp',
            color=hex_color(COLOR_STATUS_GREEN),
            halign='left'
        )
        
        network_box.add_widget(self.network_status)
        right_panel.add_widget(network_box)
        
        main_content.add_widget(left_panel)
        main_content.add_widget(right_panel)
        
        # Add everything to main layout
        self.add_widget(top_bar)
        self.add_widget(main_content)
        
        # Add initial log
        self.add_log("System started - Manual mode")
    
    def show_pallet_created(self, pallet_id, batch_id):
        """Show pallet creation confirmation in the UI display"""
        print(f"\n[SHOW_PALLET_CREATED] Displaying pallet: {pallet_id} for batch: {batch_id}")
        # Format the text
        self.success_display.text = 'PALLET CREATED'
        self.pallet_display.text = pallet_id
        self.batch_info_label.text = f'Batch: {batch_id}'
        
        # Make it visible
        self.success_display.opacity = 1
        self.pallet_display.opacity = 1
        self.batch_info_label.opacity = 1
        
        print(f"[SHOW_PALLET_CREATED] UI updated - success_display: '{self.success_display.text}'")
        print(f"[SHOW_PALLET_CREATED] UI updated - pallet_display: '{self.pallet_display.text}'")
        
        # Show Success Popup
        from kivy.uix.popup import Popup
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text="Successfully sent to server!", font_size='20sp', bold=True, color=hex_color(COLOR_STATUS_GREEN)))
        content.add_widget(Label(text=f"Pallet ID: {pallet_id}", font_size='16sp'))
        
        popup = Popup(title='Success', content=content, size_hint=(0.6, 0.4))
        popup.open()
        
        # Close popup automatically after 3 seconds
        Clock.schedule_once(popup.dismiss, 3)
        
        # Auto-clear after 30 seconds
        Clock.schedule_once(lambda dt: self.clear_status_display(), 30)

    def show_status_message(self, message, msg_type="info"):
        """Show a status message in the display area"""
        if msg_type == "success":
            color = COLOR_STATUS_GREEN
        elif msg_type == "error":
            color = COLOR_ALERT_RED
        else:
            color = COLOR_STATUS_BLUE
        
        self.success_display.text = message
        self.success_display.color = hex_color(color)
        self.pallet_display.text = ''
        self.batch_info_label.text = ''
        
        # Make it visible
        self.success_display.opacity = 1
        self.pallet_display.opacity = 0
        self.batch_info_label.opacity = 0
        
        # Auto-clear after 10 seconds
        Clock.schedule_once(lambda dt: self.clear_status_display(), 10)

    def clear_status_display(self):
        """Clear the status display area"""
        self.success_display.text = ''
        self.pallet_display.text = ''
        self.batch_info_label.text = ''
        self.success_display.opacity = 0.5
        self.pallet_display.opacity = 0.5
        self.batch_info_label.opacity = 0.5
    
    def update_process_status(self, status_type, main_text, detail_text=''):
        """Update the process status display on home screen
        
        status_type: 'waiting', 'detecting', 'decoding', 'ready', 'sending', 'success', 'error'
        """
        status_config = {
            'waiting': ('⏳', (0.5, 0.8, 1, 1)),      # Light blue
            'detecting': ('🔍', (1, 0.85, 0.4, 1)),   # Yellow/orange
            'decoding': ('📝', (0.9, 0.6, 1, 1)),     # Purple
            'ready': ('✅', (0.3, 0.9, 0.4, 1)),      # Green
            'sending': ('📤', (0.4, 0.7, 1, 1)),      # Blue
            'success': ('✓', (0.2, 0.9, 0.3, 1)),    # Bright green
            'error': ('❌', (0.9, 0.3, 0.3, 1)),      # Red
        }
        
        icon, color = status_config.get(status_type, ('●', (0.7, 0.7, 0.7, 1)))
        self.process_status_label.text = f'{icon} {main_text}'
        self.process_status_label.color = color
        self.process_detail_label.text = detail_text
        
        # Log the status update
        self.add_log(f"Status: {main_text}")
    
    def _enable_send_button(self):
        """Enable the SEND TO SERVER button with visual feedback"""
        self.send_btn.disabled = False
        self.send_btn.background_color = (0.2, 0.75, 0.35, 1)  # Vibrant green
        self.send_btn.color = (1, 1, 1, 1)
        self.send_btn.text = '📤  SEND TO SERVER'
    
    def _disable_send_button(self):
        """Disable the SEND TO SERVER button"""
        self.send_btn.disabled = True
        self.send_btn.background_color = (0.3, 0.3, 0.35, 1)  # Grey
        self.send_btn.color = (0.5, 0.5, 0.5, 1)
        self.send_btn.text = '📤  SEND TO SERVER'
    
    def send_to_server(self, instance):
        """Handle SEND TO SERVER button press"""
        if not self.data_ready_to_send or not self._pending_capture:
            self.show_toast("No data ready to send!", "error")
            return
        
        # Disable button immediately to prevent double-clicks
        self._disable_send_button()
        self.data_ready_to_send = False
        self.processing = True
        
        # Update status
        batch = self._pending_capture.get('batch', 'Unknown')
        self.update_process_status('sending', 'Sending to server...', f'Batch: {batch}')
        
        # Save current kegs to prevent re-triggering for same kegs
        self.last_captured_qr_set = set(qr['data'] for qr in self.latest_qr_results) if self.latest_qr_results else set()
        
        # Submit the data
        self._submit_pending_capture()
    
    def fetch_beer_types(self, dt=None):
        """Fetch beer types from cloud API (Background Thread)"""
        threading.Thread(target=self._fetch_beer_types_thread, daemon=True).start()

    def _fetch_beer_types_thread(self):
        try:
            beer_types = self.api_sender.get_beer_types()
            if beer_types:
                Clock.schedule_once(lambda dt: self._update_beer_types(beer_types))
            else:
                self.add_log("Using default beer types")
        except Exception as e:
            self.add_log(f"Failed to fetch beer types: {str(e)[:50]}")

    def _update_beer_types(self, beer_types_data):
        # Handle list of dicts [{'name': '...', 'id': '...'}]
        self.beer_type_map = {}
        names = []
        
        # Preserve current selection
        current_selection = self.beer_display.text
        
        for item in beer_types_data:
            if isinstance(item, dict):
                name = item.get('name', 'Unknown')
                # Cloud uses _id, fallback to id, then name
                bid = item.get('_id', item.get('id', name))
                self.beer_type_map[name] = bid
                names.append(name)
            else:
                # Fallback for strings
                name = str(item)
                self.beer_type_map[name] = name
                names.append(name)
                
        self.beer_types = names if names else ["Lager"]
        
        # Update spinner values
        self.beer_display.values = self.beer_types
        
        # Restore selection if it exists in new list, otherwise default to first
        if current_selection in self.beer_types:
            self.beer_display.text = current_selection
        else:
            self.beer_display.text = self.beer_types[0]
            
        self.add_log(f"Beer types loaded: {len(names)} types")
    
    def set_auto_mode(self, instance):
        self.is_auto_mode = True
        self.auto_btn.background_color = hex_color(COLOR_HIGHLIGHT)
        self.manual_btn.background_color = hex_color(COLOR_BUTTON_NORMAL)
        self.add_log("Auto mode: Syncing with cloud...")
        self.sync_cloud()
    
    def set_manual_mode(self, instance):
        self.is_auto_mode = False
        self.manual_btn.background_color = hex_color(COLOR_HIGHLIGHT)
        self.auto_btn.background_color = hex_color(COLOR_BUTTON_NORMAL)
        self.add_log("Manual mode: Set keg count manually")
    
    def show_auto_confirm_popup(self, frame):
        """Show confirmation dialog before sending to API in Auto Mode"""
        # Prevent multiple popups
        if self.auto_confirm_pending:
            return
        
        self.auto_confirm_pending = True
        self.auto_pending_frame = frame.copy()  # Store the frame
        
        # Create modal
        modal = ModalView(size_hint=(0.6, 0.4), auto_dismiss=False)
        
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Store reference to background rectangle for updates
        self._modal_bg_rect = None
        with content.canvas.before:
            Color(0.15, 0.15, 0.2, 1)
            self._modal_bg_rect = Rectangle(pos=content.pos, size=content.size)
        
        # Bind to update background position/size when content changes
        def update_bg(instance, value):
            if self._modal_bg_rect:
                self._modal_bg_rect.pos = instance.pos
                self._modal_bg_rect.size = instance.size
        content.bind(pos=update_bg, size=update_bg)
        
        # Title
        title = Label(
            text=f'[b]SEND TO API?[/b]',
            markup=True,
            font_size='22sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            size_hint_y=0.3
        )
        
        # Info
        info = Label(
            text=f'Detected {len(self.latest_qr_results)} QR codes.\nBatch: {self.batch_display.text}',
            font_size='16sp',
            color=hex_color(COLOR_TEXT_LIGHT),
            size_hint_y=0.3,
            halign='center'
        )
        
        # Buttons
        btn_box = BoxLayout(spacing=20, size_hint_y=0.4)
        
        yes_btn = Button(
            text='YES - SEND',
            font_size='18sp',
            background_normal='',
            background_color=hex_color(COLOR_STATUS_GREEN),
            color=(1, 1, 1, 1)
        )
        
        no_btn = Button(
            text='NO - CANCEL',
            font_size='18sp',
            background_normal='',
            background_color=hex_color(COLOR_ALERT_RED),
            color=(1, 1, 1, 1)
        )
        
        def on_yes(instance):
            modal.dismiss()
            self.auto_confirm_pending = False
            if self.auto_pending_frame is not None:
                main_logger.info("Auto-confirm: User approved, triggering capture")
                self.trigger_capture(self.auto_pending_frame)
                self.auto_pending_frame = None
        
        def on_no(instance):
            modal.dismiss()
            self.auto_confirm_pending = False
            self.auto_pending_frame = None
            self.stability_counter = 0  # Reset stability counter
            main_logger.info("Auto-confirm: User cancelled")
        
        yes_btn.bind(on_press=on_yes)
        no_btn.bind(on_press=on_no)
        
        btn_box.add_widget(yes_btn)
        btn_box.add_widget(no_btn)
        
        content.add_widget(title)
        content.add_widget(info)
        content.add_widget(btn_box)
        
        modal.add_widget(content)
        modal.open()
        
        # Play a sound or visual cue (optional)
        self.show_toast("Confirm to send to API", "info", 2)
    
    def _update_modal_bg(self, widget):
        """Helper to update modal background on resize"""
        widget.canvas.before.clear()
        with widget.canvas.before:
            Color(0.15, 0.15, 0.2, 1)
            Rectangle(pos=widget.pos, size=widget.size)
    
    def on_beer_type_select(self, spinner, text):
        """Handle beer type selection"""
        self.add_log(f"Beer type selected: {text}")
    
    def update_filling_date(self, dt=None):
        """Update the filling date display"""
        self.filling_date_display.text = datetime.now().strftime('%d-%m-%Y %H:%M')
    
    def change_count(self, instance):
        """Open modal to change target keg count"""
        modal = CountModal(self.set_count, self.required_keg_count)
        modal.open()
    
    def set_count(self, count):
        """Set the target keg count"""
        self.required_keg_count = count
        self.count_button.text = str(count)
        self.target_display.text = str(count)
        self.add_log(f"Target count set to: {count}")
        self.show_toast(f"Target count set to {count}", "success")
    
    def edit_batch(self, instance):
        """Edit batch number"""
        modal = BatchModal(self.set_batch, self.batch_display.text)
        modal.open()
    
    def set_batch(self, batch):
        """Set batch number and save it"""
        self.batch_display.text = batch
        self.last_batch_number = batch
        save_last_batch(batch)
        log_msg = f"Batch: {batch}"
        self.add_log(log_msg)
        self.show_toast(f"Batch number set to: {batch}", "success")
        
        # Clear duplicate warning when batch number changes
        self.duplicate_warning.text = ''
        self.status_label.text = 'Waiting for kegs...'
        self.status_label.color = hex_color(COLOR_TEXT_LIGHT)

    def increment_batch_number(self):
        """Auto-increment batch number after success"""
        current = self.batch_display.text
        # Try to find trailing number
        import re
        match = re.search(r'(\d+)$', current)
        if match:
            num_str = match.group(1)
            num_len = len(num_str)
            new_num = int(num_str) + 1
            new_batch = current[:match.start()] + str(new_num).zfill(num_len)
            self.set_batch(new_batch)
        else:
            # Append -1 if no number
            self.set_batch(current + "-1")
    
    def sync_cloud(self):
        """Sync configuration with cloud (Non-blocking)"""
        # Avoid starting multiple sync threads
        if hasattr(self, 'syncing') and self.syncing:
            self.add_log("Sync already in progress...")
            return

        if self.is_auto_mode:
            self.syncing = True
            self.add_log("Syncing with cloud...")
            threading.Thread(target=self._sync_thread, daemon=True).start()
        else:
            self.add_log("Manual mode - using local config")
    
    def _sync_thread(self):
        """Background thread for cloud sync"""
        try:
            # Try to get configuration from cloud
            mac_address = CAMERA_MAC_ID
            if not mac_address or mac_address == "3C:6D:66:01:5A:F0":
                # Get actual MAC address
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                               for elements in range(0, 2*6, 2)][::-1])
                mac_address = mac.upper()
            
            payload = {"macId": mac_address}
            
            # Use configured endpoint
            endpoints = [CLOUD_CONFIG_ENDPOINT]
            
            success = False
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=5,
                        verify=False
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Parse configuration from response
                        keg_type = data.get("keg_type", "30L")
                        count = data.get("keg_count", DEFAULT_KEG_COUNT)
                        
                        # Apply updates on main thread
                        Clock.schedule_once(lambda dt: self._apply_sync_success(count, keg_type))
                        success = True
                        break
                except:
                    continue
            
            if not success:
                 Clock.schedule_once(lambda dt: self._apply_sync_fail("Cloud sync failed: No valid response"))

        except Exception as e:
            err_msg = str(e)
            Clock.schedule_once(lambda dt: self._apply_sync_fail(f"Sync error: {err_msg[:50]}"))
        finally:
            self.syncing = False

    def _apply_sync_success(self, count, keg_type):
        """Update UI with sync results (Main Thread)"""
        self.required_keg_count = count
        self.count_button.text = str(count)
        self.target_display.text = str(count)
        self.add_log(f"Cloud sync: {count} {keg_type} kegs")
        
    def _apply_sync_fail(self, error_msg):
        """Log sync failure (Main Thread)"""
        self.add_log(error_msg)
    
    def update_frame(self, dt):
        """
        Update camera frame and process detection.
        OPTIMIZED: Uses background camera thread, reusable texture, and frame resize.
        """
        frame_start = time.perf_counter()
        
        if not self.detection_active or not self.camera:
            # Show camera not initialized
            try:
                img_h, img_w = 480, 640
                vis_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                text = "Camera Not Initialized"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
                text_x = (img_w - text_size[0]) // 2
                text_y = (img_h + text_size[1]) // 2
                cv2.putText(vis_frame, text, (text_x, text_y), font, 1.0, (0, 0, 255), 2)
                
                # Use reusable texture
                self._update_preview_texture(vis_frame)
            except Exception as e:
                main_logger.warning(f"Preview error: {e}")
            return

        if self.processing:
            return
        
        # === NON-BLOCKING: Get frame from background capture thread ===
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            main_logger.debug("No frame available")
            return
        
        # Create visualization frame (reference, not copy for display)
        vis_frame = frame.copy()
        
        # === ASYNC DETECTION with resized frame ===
        try:
            # 1. Check if previous detection finished
            if self.current_detection_task and self.current_detection_task.done():
                try:
                    results = self.current_detection_task.result()
                    if results:
                        qr_list = results[0]  # Index 0 is the list of QRs
                        # Scale bboxes back up if we resized
                        if self._detection_resize and qr_list:
                            orig_h, orig_w = frame.shape[:2]
                            det_w, det_h = self._detection_resize
                            scale_x = orig_w / det_w
                            scale_y = orig_h / det_h
                            for qr in qr_list:
                                x1, y1, x2, y2 = qr['bbox']
                                qr['bbox'] = (
                                    int(x1 * scale_x), int(y1 * scale_y),
                                    int(x2 * scale_x), int(y2 * scale_y)
                                )
                        self.latest_qr_results = qr_list
                        if qr_list:
                            main_logger.info(f"Detection result: {len(qr_list)} QR codes found - will draw bboxes")
                            
                            # === AUTO-TRIGGER LOGIC ===
                            if self.is_auto_mode and not self.processing and not self.auto_confirm_pending:
                                if len(qr_list) == self.required_keg_count:
                                    self.stability_counter += 1
                                    if self.stability_counter >= 5: # Stable for ~5 detection cycles
                                        main_logger.info("Auto-trigger condition met! Capturing...")
                                        self.trigger_capture(frame)
                                        self.stability_counter = 0
                                else:
                                    self.stability_counter = 0
                            # ==========================
                except Exception as e:
                    main_logger.warning(f"Detection result error: {e}")
                finally:
                    self.current_detection_task = None

            # 2. Submit new detection if idle
            if self.current_detection_task is None:
                # === OPTIMIZATION: Resize frame for faster detection ===
                if self._detection_resize:
                    det_frame = cv2.resize(frame, self._detection_resize, interpolation=cv2.INTER_AREA)
                else:
                    det_frame = frame.copy()
                
                # Submit detection task (use_qreader=False for live preview)
                self.current_detection_task = self.detector_executor.submit(
                    self.qr_detector.detect_and_decode, 
                    det_frame,
                    False  # use_qreader=False for speed
                )

            # 3. Draw LATEST known results (Visual Feedback)
            for qr in self.latest_qr_results:
                x1, y1, x2, y2 = qr['bbox']
                # Draw green boundary
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw text label background
                label = qr['data'][:15] + '...' if len(qr['data']) > 15 else qr['data']
                source = qr.get('source', 'unknown')
                label_full = f"{label} [{source}]"
                t_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (x1, y1-25), (x1+t_size[0]+4, y1), (0, 255, 0), -1)
                cv2.putText(vis_frame, label_full, (x1+2, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        except Exception as e:
            main_logger.warning(f"QR detection error: {e}")

        # === OPTIMIZED: Update preview with reusable texture ===
        self._update_preview_texture(vis_frame)
        
        # Process frame for keg detection logic
        self.process_frame(frame)
        
        # === Performance logging ===
        frame_time = (time.perf_counter() - frame_start) * 1000
        self._frame_times.append(frame_time)
        
        # Log every 2 seconds
        now = time.time()
        if now - self._last_perf_log >= 2.0:
            if self._frame_times:
                avg_time = sum(self._frame_times) / len(self._frame_times)
                fps = 1000 / avg_time if avg_time > 0 else 0
                cam_fps = self.camera.get_fps() if hasattr(self.camera, 'get_fps') else 0
                main_logger.info(f"UI FPS: {fps:.1f} | Frame time: {avg_time:.1f}ms | Camera FPS: {cam_fps:.1f}")
                self._frame_times.clear()
            self._last_perf_log = now

    def _update_preview_texture(self, frame):
        """
        Update preview image texture efficiently.
        REUSES texture object when size matches to avoid allocation overhead.
        """
        try:
            h, w = frame.shape[:2]
            
            # Create or reuse texture
            if self._preview_texture is None or self._texture_size != (w, h):
                main_logger.debug(f"Creating new texture: {w}x{h}")
                self._preview_texture = Texture.create(size=(w, h), colorfmt='bgr')
                self._texture_size = (w, h)
            
            # Flip frame for OpenGL coordinate system and blit
            buf = cv2.flip(frame, 0).tobytes()
            self._preview_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.preview_image.texture = self._preview_texture
            
        except Exception as e:
            main_logger.warning(f"Texture update error: {e}")

    
    def process_frame(self, frame):
        """Process frame for keg detection"""
        try:
            # detection = self.keg_detector.detect(frame)
            # new_count = detection['count']
            
            # User Change: Rely ONLY on QR code count
            new_count = len(self.latest_qr_results) if hasattr(self, 'latest_qr_results') else 0
            
            # Check stability
            if new_count == self.prev_count:
                self.stability_counter = min(self.stability_counter + 1, STABILITY_THRESHOLD)
            else:
                self.stability_counter = max(self.stability_counter - 2, 0)
            
            self.prev_count = new_count
            self.current_count = new_count
            
            # Update UI
            self.update_display()
            
            # Get effective count (best of keg or QR detection)
            qr_count = len(self.latest_qr_results) if hasattr(self, 'latest_qr_results') else 0
            effective_count = max(self.current_count, qr_count)
            
            # Auto-capture when stable, count EXACTLY matches target, and not empty
            # USER REQUEST: Skip auto-capture confirmation, go directly to capture
            # The API sending confirmation dialog will still show before sending to cloud
            if (self.is_auto_mode and
                self.stability_counter >= STABILITY_THRESHOLD and 
                not self.processing and 
                not getattr(self, 'auto_confirm_pending', False) and
                not getattr(self, 'auto_capture_cooldown', False) and
                qr_count > 0 and
                qr_count == self.required_keg_count):
                # Check if same kegs are still under camera (prevent double capture)
                current_qr_set = set(qr['data'] for qr in self.latest_qr_results) if self.latest_qr_results else set()
                should_capture = True
                
                if current_qr_set and self.last_captured_qr_set:
                    overlap = len(current_qr_set & self.last_captured_qr_set)
                    if overlap > 0 and overlap >= len(current_qr_set) * 0.5:
                        should_capture = False
                        self.duplicate_warning.text = "Same kegs detected - waiting for new pallet"
                        self.duplicate_warning.color = hex_color(COLOR_STATUS_ORANGE)
                
                if should_capture:
                    # Validate beer type
                    if self.beer_display.text in ['Loading...', '', None]:
                        self.show_toast("Please select a beer type first!", "error")
                    # Validate batch number
                    elif not self.batch_display.text.strip() or self.batch_display.text == 'BATCH-':
                        self.show_toast("Please enter a valid batch number!", "error")
                    else:
                        # Direct capture - API confirmation will show before sending to cloud
                        self.duplicate_warning.text = ""
                        self.last_captured_qr_set = current_qr_set
                        self.add_log("Target reached - capturing...")
                        self.trigger_capture(frame)
                
        except Exception as e:
            self.add_log(f"Detect error: {str(e)[:50]}")
    
    def update_display(self):
        """Update all UI displays"""
        # Update count displays
        self.current_display.text = str(self.current_count)
        
        # Get QR count from live detection
        qr_count = len(self.latest_qr_results) if hasattr(self, 'latest_qr_results') else 0
        
        # Use the better of keg count or QR count for target status
        effective_count = max(self.current_count, qr_count)
        
        # Update colors based on status
        if self.processing:
            self.current_display.color = hex_color(COLOR_STATUS_ORANGE)
            self.system_status.text = '[b]● PROCESSING[/b]'
            self.system_status.color = hex_color(COLOR_STATUS_ORANGE)
            self.status_label.text = "Processing batch..."
            self.status_label.color = hex_color(COLOR_STATUS_ORANGE)
        elif self.data_ready_to_send:
            # Data is ready - waiting for user to click SEND
            self.current_display.color = hex_color(COLOR_STATUS_GREEN)
            self.system_status.text = '[b]● READY TO SEND[/b]'
            self.system_status.color = hex_color(COLOR_STATUS_GREEN)
            self.status_label.text = f"Data ready! Click SEND TO SERVER"
            self.status_label.color = hex_color(COLOR_STATUS_GREEN)
        else:
            # Update status label based on detected vs target count
            # Also update process status display
            if effective_count == 0:
                self.status_label.text = "Waiting for kegs..."
                self.status_label.color = hex_color(COLOR_STATUS_BLUE)
                self.system_status.text = '[b]● WAITING[/b]'
                self.system_status.color = hex_color(COLOR_STATUS_BLUE)
                # Update process status
                if not hasattr(self, '_last_process_update') or self._last_process_update != 'waiting':
                    self.process_status_label.text = '⏳ Waiting for kegs...'
                    self.process_status_label.color = (0.5, 0.8, 1, 1)
                    self.process_detail_label.text = 'Place kegs under camera'
                    self._last_process_update = 'waiting'
            elif effective_count < self.required_keg_count:
                self.status_label.text = f"Target Not Achieved ({effective_count}/{self.required_keg_count})"
                self.status_label.color = hex_color(COLOR_STATUS_ORANGE)
                self.system_status.text = '[b]● DETECTING[/b]'
                self.system_status.color = hex_color(COLOR_STATUS_BLUE)
                # Update process status
                if not hasattr(self, '_last_process_update') or self._last_process_update != 'detecting':
                    self.process_status_label.text = f'🔍 Detecting... ({effective_count}/{self.required_keg_count})'
                    self.process_status_label.color = (1, 0.85, 0.4, 1)
                    self.process_detail_label.text = 'Scanning QR codes...'
                    self._last_process_update = 'detecting'
                else:
                    # Just update the count
                    self.process_status_label.text = f'🔍 Detecting... ({effective_count}/{self.required_keg_count})'
            elif effective_count == self.required_keg_count:
                self.status_label.text = f"Target Achieved ({effective_count}/{self.required_keg_count})"
                self.status_label.color = hex_color(COLOR_STATUS_GREEN)
                self.system_status.text = '[b]● READY TO CAPTURE[/b]'
                self.system_status.color = hex_color(COLOR_STATUS_GREEN)
                # Update process status
                if not hasattr(self, '_last_process_update') or self._last_process_update != 'target_met':
                    self.process_status_label.text = f'📝 Decoding QR codes...'
                    self.process_status_label.color = (0.9, 0.6, 1, 1)
                    self.process_detail_label.text = f'{effective_count} kegs detected, processing...'
                    self._last_process_update = 'target_met'
            else:  # effective_count > required_keg_count
                self.status_label.text = f"WARNING: Count exceeds target! ({effective_count}/{self.required_keg_count})"
                self.status_label.color = hex_color(COLOR_ALERT_RED)
                self.system_status.text = '[b]● CHECK COUNT[/b]'
                self.system_status.color = hex_color(COLOR_ALERT_RED)
                # Update process status
                self.process_status_label.text = f'⚠️ Too many kegs ({effective_count}/{self.required_keg_count})'
                self.process_status_label.color = (0.9, 0.3, 0.3, 1)
                self.process_detail_label.text = 'Remove extra kegs'
                self._last_process_update = 'error'
            
        # Button Logic - Override for Manual Mode
        if not self.processing:
            if not self.is_auto_mode:
                # Manual Mode: Enable capture ONLY if we see EXACTLY the target count
                if effective_count == self.required_keg_count:
                    self.capture_btn.disabled = False
                    self.capture_btn.background_color = hex_color(COLOR_HIGHLIGHT)
                else:
                    self.capture_btn.disabled = True
                    self.capture_btn.background_color = hex_color(COLOR_BUTTON_NORMAL)
            else:
                # Auto Mode: Capture button always disabled - use auto-trigger only
                self.capture_btn.disabled = True
                self.capture_btn.background_color = hex_color(COLOR_BUTTON_NORMAL)
                
                # Auto Trigger Logic
                if not self.auto_confirm_pending and effective_count == self.required_keg_count:
                    # Check if we should auto-trigger
                    # Needs to be stable for X frames
                    self.stability_counter += 1
                    if self.stability_counter > 15: # Approx 0.5s at 30fps
                         # Check if same kegs detection logic (prevent double capture)
                         current_qr_set = set(qr['data'] for qr in self.latest_qr_results) if self.latest_qr_results else set()
                         should_capture = True
                         
                         if current_qr_set and self.last_captured_qr_set:
                             overlap = len(current_qr_set & self.last_captured_qr_set)
                             if overlap > 0 and overlap >= len(current_qr_set) * 0.5:
                                 should_capture = False
                                 self.duplicate_warning.text = "Wait for new pallet..."
                                 self.duplicate_warning.color = hex_color(COLOR_STATUS_ORANGE)
                         
                         if should_capture:
                             self.duplicate_warning.text = ""
                             self.add_log("Auto-triggering capture...")
                             self.last_captured_qr_set = current_qr_set
                             self.trigger_capture(frame_resized) # Capture directly
                             self.stability_counter = 0 # Reset
                else:
                    self.stability_counter = 0 # Reset if count doesn't match

    
    
    def trigger_capture(self, frame):
        """Trigger batch capture"""
        main_logger.info("=" * 60)
        main_logger.info("TRIGGER_CAPTURE STARTED")
        main_logger.info("=" * 60)
        
        if self.processing:
            main_logger.warning("BLOCKED: Already processing")
            return
        
        # Clear any previous status display
        self.clear_status_display()
        
        beer_name = self.beer_display.text
        # Resolve ID from name
        beer_id = self.beer_type_map.get(beer_name, beer_name)
        
        batch = self.batch_display.text
        
        # Store user's batch for display in callbacks
        self.current_batch_number = batch
        
        main_logger.info(f"Beer Name: {beer_name}")
        main_logger.info(f"Beer ID: {beer_id}")
        main_logger.info(f"Batch: {batch}")
        main_logger.debug(f"Required Keg Count: {self.required_keg_count}")
        main_logger.debug(f"Current Count: {self.current_count}")
        
        if not batch:
            main_logger.error("No batch number entered!")
            self.add_log("Please enter batch number!")
            return
        
        main_logger.info(f"[CAPTURE] Triggered! Beer: {beer_name} (ID: {beer_id}) | Batch: {batch}")
        self.processing = True
        self.add_log(f"Capturing {self.current_count} kegs...")
        
        # Save frame
        image_name = f"batch_{create_timestamp()}.jpg"
        frame_path = str(SAVE_FOLDER / image_name)
        main_logger.debug(f"Saving frame to: {frame_path}")
        cv2.imwrite(frame_path, frame)
        main_logger.debug("Frame saved successfully")
        
        # Generate session ID
        session_id = f"BATCH_{self.database.get_next_batch_number():04d}"
        main_logger.info(f"Session ID: {session_id}")
        
        # Capture exact filling timestamp
        filling_timestamp = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        
        # Store capture data - ready for sending
        self._pending_capture = {
            'frame_path': frame_path,
            'image_name': image_name,
            'session_id': session_id,
            'beer_id': beer_id,
            'batch': batch,
            'filling_timestamp': filling_timestamp
        }
        
        # Mark data as ready and enable SEND button
        self.data_ready_to_send = True
        self.processing = False  # Not processing anymore, waiting for user to click SEND
        self._enable_send_button()
        self.update_process_status('ready', f'Ready to send {self.required_keg_count} kegs', f'Batch: {batch} | Beer: {beer_name}')
    
    
    def _submit_pending_capture(self):
        """Submit the pending capture to API (after user confirmation)"""
        capture_data = self._pending_capture
        if not capture_data:
            main_logger.error("No pending capture data!")
            self.processing = False
            return
        
        def process_capture():
            main_logger.info("=" * 60)
            main_logger.info("PROCESS_CAPTURE THREAD STARTED (After Confirmation)")
            main_logger.info("=" * 60)
            
            try:
                submit_batch(
                    capture_data['frame_path'], 
                    capture_data['image_name'], 
                    capture_data['session_id'], 
                    self.required_keg_count,
                    beer_type=capture_data['beer_id'], 
                    batch=capture_data['batch'],
                    filling_date=capture_data['filling_timestamp']
                )
                main_logger.info("submit_batch completed")
                Clock.schedule_once(lambda dt: self.complete_capture(capture_data['session_id']), 3)
            except Exception as e:
                main_logger.error(f"ERROR in submit_batch: {e}")
                import traceback
                traceback.print_exc()
                self.add_log(f"Capture error: {str(e)[:50]}")
                Clock.schedule_once(lambda dt: self.capture_failed(), 0)
        
        main_logger.info("Starting process_capture thread...")
        threading.Thread(target=process_capture, daemon=True).start()
    
    def _trigger_recapture(self):
        """Trigger a recapture after cancel (auto-mode only)"""
        if not self.is_auto_mode:
            return
        if self.processing or self.auto_confirm_pending:
            return
        main_logger.info("Auto-recapture triggered after 10s delay")
        self.add_log("Auto-recapturing...")
        self.stability_counter = 0  # Reset to start detection fresh
    
    def complete_capture(self, session_id):
        """Handle successful capture completion"""
        self.processing = False
        self.stability_counter = 0
        self.data_ready_to_send = False
        self._disable_send_button()
        
        # Use user's batch number for display
        user_batch = getattr(self, 'current_batch_number', session_id)
        self.show_status_message(f"Sent: {user_batch}", "success")
        self.show_toast(f"Batch {user_batch} submitted successfully!", "success")
        
        # Update process status
        self.process_status_label.text = '✓ Sent Successfully!'
        self.process_status_label.color = (0.2, 0.9, 0.3, 1)
        self.process_detail_label.text = f'Batch: {user_batch}'
        self._last_process_update = 'success'
        
        # Reset process status after 5 seconds
        Clock.schedule_once(lambda dt: self._reset_process_status(), 5)
        
        self.add_log(f"Batch {user_batch} submitted")
        Clock.schedule_once(lambda dt: self.check_active_batch_status(session_id, user_batch), 1)
    
    def capture_failed(self):
        """Handle capture failure"""
        self.processing = False
        self.stability_counter = 0
        self.data_ready_to_send = False
        self._disable_send_button()
        
        self.add_log("Capture failed - check logs")
        self.show_toast("Capture Failed! Check logs for details.", "error")
        
        # Update process status
        self.process_status_label.text = '❌ Send Failed!'
        self.process_status_label.color = (0.9, 0.3, 0.3, 1)
        self.process_detail_label.text = 'Check logs for details'
        self._last_process_update = 'error'
        
        # Reset process status after 5 seconds
        Clock.schedule_once(lambda dt: self._reset_process_status(), 5)
    
    def _reset_process_status(self):
        """Reset process status to waiting state"""
        self.process_status_label.text = '⏳ Waiting for kegs...'
        self.process_status_label.color = (0.5, 0.8, 1, 1)
        self.process_detail_label.text = 'Place kegs under camera'
        self._last_process_update = 'waiting'
        self.duplicate_warning.text = ''
    
    def check_active_batch_status(self, session_id, user_batch=None, attempts=0):
        """Poll batch status and show pallet confirmation when ready"""
        # Use user_batch for display, session_id for DB lookup
        display_batch = user_batch or session_id
        # print(f"\n[CHECK_STATUS] Checking status for: {session_id} (Attempt {attempts+1})")
        
        status = self.database.get_batch_status(session_id)
        # print(f"[CHECK_STATUS] Batch status: {status}")
        
        if status == 'api_sent':
            # SUCCESS!
            response_json = self.database.get_batch_response(session_id)
            print(f"[CHECK_STATUS] API Response from DB: {response_json}")
            
            pallet_id = "UNKNOWN"
            if response_json:
                try:
                    # Parse JSON response to get pallet ID
                    import json
                    resp_data = json.loads(response_json)
                    # Adjust based on your actual API response structure
                    # Example: {"palletId": "PAL-123", ...}
                    pallet_id = resp_data.get('palletId', display_batch)
                except:
                    pallet_id = display_batch
            
            self.show_pallet_created(pallet_id, display_batch)
            self.add_log(f"Batch {display_batch} sent successfully!")
            return

        elif status == 'api_failed':
            # FAILURE
            self.add_log(f"Batch {display_batch} failed to send")
            self.show_status_message(f"Batch Failed: {display_batch}", "error")
            self.show_toast(f"API Error: Batch {display_batch} failed!", "error", 5)
            # Fetch error details if possible
            # error_msg = self.database.get_last_error(session_id)
            return

        elif attempts > 60: # 30 seconds timeout (0.5s interval)
            # TIMEOUT
            self.add_log(f"Batch {display_batch} timed out (still processing)")
            self.show_toast(f"Processing timed out for {display_batch}", "error")
            return

        else:
            # STILL PROCESSING -> Poll again
            Clock.schedule_once(lambda dt: self.check_active_batch_status(session_id, user_batch, attempts + 1), 0.5)
            
            # Auto-increment batch number for next run
            self.increment_batch_number()
            
            return

    
    def force_capture(self, instance):
        """Manual capture trigger with confirmation dialog"""
        main_logger.info("=" * 60)
        main_logger.info("CAPTURE BUTTON CLICKED")
        main_logger.info("=" * 60)
        main_logger.debug(f"  Processing: {self.processing}")
        main_logger.debug(f"  Current Count: {self.current_count}")
        main_logger.debug(f"  Required Count: {self.required_keg_count}")
        
        # Validate inputs first
        if self.processing:
            self.show_toast("Already processing a batch. Please wait.", "warning")
            return
        
        if self.current_count <= 0:
            self.show_toast("No kegs detected! Cannot capture.", "error")
            return
        
        # Validate beer type
        if self.beer_display.text in ['Loading...', '', None]:
            self.show_toast("Please select a beer type first!", "error")
            return
        
        # Validate batch number
        batch = self.batch_display.text.strip()
        if not batch:
            self.show_toast("Please enter a valid batch number!", "error")
            return
        
        # Show confirmation dialog
        # Check for duplicate batch number
        is_duplicate, prev_session = self.database.is_batch_number_sent(batch)
        duplicate_warning = ""
        if is_duplicate:
            duplicate_warning = "\n\n⚠ WARNING: This batch was already sent!"
            self.duplicate_warning.text = f"⚠ DUPLICATE: {batch}"
            self.duplicate_warning.color = hex_color(COLOR_STATUS_ORANGE)
        else:
            self.duplicate_warning.text = ''
        
        # Use target count (user-entered) for confirmation, not detected count
        beer_type = self.beer_display.text
        message = f"Capture {self.required_keg_count} kegs?\n\nBeer Type: {beer_type}\nBatch: {batch}{duplicate_warning}"
        
        def on_confirm():
            main_logger.info("Capture confirmed, getting frame...")
            ret, frame = self.camera.get_frame()
            main_logger.debug(f"Frame acquired: ret={ret}, frame shape={frame.shape if frame is not None else 'None'}")
            if ret and frame is not None:
                main_logger.info("Triggering capture process...")
                self.show_toast("Capturing kegs...", "info")
                self.trigger_capture(frame)
            else:
                main_logger.error("Failed to get frame from camera")
                self.show_toast("Camera error! Could not get frame.", "error")
        
        modal = ConfirmationModal(
            title="Confirm Capture",
            message=message,
            on_confirm=on_confirm
        )
        modal.open()
    
    def show_toast(self, message, msg_type="info", duration=3):
        """Show a toast notification to the user"""
        toast = ToastMessage(message, msg_type, duration)
        toast.open()
    
    def check_network(self, dt):
        """Check network connectivity via API Sender"""
        if self.api_sender:
            is_online = self.api_sender.get_network_status()
            self._update_network_status(is_online)

    def _update_network_status(self, is_online):
        if is_online:
            self.network_status.text = 'ONLINE'
            self.network_status.color = hex_color(COLOR_STATUS_GREEN)
        else:
            self.network_status.text = 'OFFLINE'
            self.network_status.color = hex_color(COLOR_ALERT_RED)
    
    def recover_batches(self, dt):
        """Recover stuck batches"""
        try:
            stuck = self.database.get_stuck_batches(timeout_minutes=5)
            if stuck:
                self.add_log(f"Recovered {len(stuck)} stuck batches")
        except:
            pass
    
    def start_detection_system(self):
        """Initialize camera system"""
        try:
            self.camera = CameraManager(CAMERA_CONFIG)
            if self.camera.start():
                self.detection_active = True
                self.add_log("Camera started successfully")
            else:
                self.add_log("Camera failed to start")
        except Exception as e:
            self.add_log(f"Camera error: {str(e)[:50]}")
    
    def add_log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.workflow_logs.append(log_entry)
        
        # Keep last 10 logs
        if len(self.workflow_logs) > 10:
            self.workflow_logs = self.workflow_logs[-10:]
        
        # Update system logs (optional - can be viewed in logs popup)
    
    def show_logs(self, instance):
        """Show detailed logs popup"""
        from kivy.uix.popup import Popup
        
        popup = Popup(title='System Logs', size_hint=(0.8, 0.6))
        
        scroll = ScrollView()
        content = BoxLayout(orientation='vertical', size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        for log_entry in reversed(self.workflow_logs):
            label = Label(
                text=log_entry,
                font_size='10sp',
                color=hex_color(COLOR_TEXT_LIGHT),
                size_hint_y=None,
                height=25,
                halign='left',
                text_size=(400, None)
            )
            content.add_widget(label)
        
        scroll.add_widget(content)
        popup.content = scroll
        popup.open()
    
    def confirm_exit(self, instance):
        """Show confirmation dialog before exiting"""
        def do_exit():
            App.get_running_app().stop()
        
        modal = ConfirmationModal(
            title="Exit Application",
            message="Are you sure you want to exit?",
            on_confirm=do_exit
        )
        modal.open()
    
    def on_stop(self):
        """Cleanup on application stop"""
        if self.camera:
            self.camera.stop()
        if hasattr(self, 'api_sender'):
            self.api_sender.stop_retry_monitor()
            self.api_sender.close()

class SimpleKegApp(App):
    def build(self):
        self.title = 'Keg Counting System'
        Window.clearcolor = COLOR_BG_DARK
        # Removed window size to allow full screen
        # Window.size = (1024, 600)  
        return SimpleKegHMI()
    
    def on_stop(self):
        if hasattr(self, 'root') and self.root:
            self.root.on_stop()

if __name__ == '__main__':
    SimpleKegApp().run()