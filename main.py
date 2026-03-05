#!/usr/bin/env python3
import os
import sys

# GPU Detection moved to after config load

# --- CONFIGURATION FOR FULL SCREEN HMI ---
# Must be done before other Kivy imports
from kivy.config import Config

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
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.metrics import dp

# KivyMD Imports
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDRaisedButton, MDIconButton, MDFillRoundFlatButton, MDRectangleFlatIconButton, MDFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, IconRightWidget, IRightBodyTouch
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.spinner import MDSpinner
from kivymd.theming import ThemableBehavior

# Custom Modules
from modules.camera import CameraManager
from modules.database import DatabaseManager
from modules.api_sender import APISender
from modules.process_worker import submit_batch, shutdown
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
                def detect_and_decode(self, *args):
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
print("   PALLETIZATION SYSTEM (KivyMD)")
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

# Helper for custom list item with delete button
class DeleteListItem(OneLineAvatarIconListItem):
    pass

class DeleteIconWidget(IRightBodyTouch, MDIconButton):
    pass


class SimpleKegHMI(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 0
        self.spacing = 0
        
        main_logger.info("Initializing SimpleKegHMI (Material Design)...")
        
        # Initialize components
        self.camera = None
        self.database = DatabaseManager()
        self.api_sender = APISender()
        # Initialize QR Detector (lazy loaded)
        qr_detector_class = get_qr_detector_class()
        self.qr_detector = qr_detector_class()
        
        # Async Detection Setup
        self.detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LiveDetector")
        self.current_detection_task = None
        self.latest_qr_results = []
        
        # Track last captured QR codes to detect same kegs
        self.last_captured_qr_set = set()
        self.captured_qr_codes = []
        
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
        self.menu_beer = None # Dropdown menu ref

        # Dialog refs
        self.confirm_dialog = None
        self.count_dialog = None
        
        # === PERFORMANCE OPTIMIZATIONS ===
        self._preview_texture = None
        self._texture_size = (0, 0)
        self.data_ready_to_send = False
        self._pending_capture = None
        self._frame_times = []
        self._last_perf_log = time.time()
        self._detection_resize = None
        
        # Load last batch number
        self.last_batch_number = load_last_batch()
        
        # Logs

        
        # Build UI
        self.build_ui()
        
        # Start systems
        self.start_detection_system()
        
        # Schedule tasks
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        Clock.schedule_interval(self.check_network, 30)
        Clock.schedule_interval(self.update_filling_date, 60)
        Clock.schedule_once(self.recover_batches, 1)
        Clock.schedule_once(lambda dt: self.sync_cloud(), 2)
        Clock.schedule_once(self.fetch_beer_types, 3)
    
    def build_ui(self):
        """Build Material Design UI layout"""
        
        # 1. Top Bar
        self.top_bar = MDTopAppBar(
            title="Keg Counting System",
            anchor_title="left",
            elevation=2
        )
        self.top_bar.right_action_items = [["circle", lambda x: None, "System Ready"]] # Status indicator
        self.add_widget(self.top_bar)
        
        # 2. Main Content
        main_content = MDBoxLayout(orientation='horizontal', spacing=dp(10), padding=dp(10))
        
        # --- LEFT PANEL: CAMERA ---
        left_panel = MDCard(
            orientation='vertical',
            size_hint_x=0.6,
            elevation=1,
            padding=dp(2),
            radius=[8]
        )
        
        # Camera Container
        cam_container = MDBoxLayout(size_hint_y=1)
        self.preview_image = Image(allow_stretch=True, keep_ratio=True)
        cam_container.add_widget(self.preview_image)
        left_panel.add_widget(cam_container)
        
        # Overlay Stats (Footer of camera)
        stats_box = MDBoxLayout(size_hint_y=None, height=dp(50), padding=[dp(10), 0])
        
        # Target Count Display
        stats_box.add_widget(MDLabel(
            text="TARGET:",
            theme_text_color="Secondary",
            halign="left",
            size_hint_x=None,
            width=dp(80)
        ))
        self.target_display = MDLabel(
            text=str(self.required_keg_count),
            font_style="H4",
            theme_text_color="Primary",
            halign="left",
            bold=True
        )
        stats_box.add_widget(self.target_display)
        
        # Current status (Detection)
        self.current_display = MDLabel(
            text="0",
            font_style="H4",
            theme_text_color="Custom",
            text_color=(1, 0.6, 0, 1), # Orange
            halign="right",
            bold=True,
            opacity=0 # Hidden initially
        )
        stats_box.add_widget(self.current_display)
        
        left_panel.add_widget(stats_box)
        main_content.add_widget(left_panel)
        
        # --- RIGHT PANEL: CONTROLS ---
        right_panel = MDBoxLayout(orientation='vertical', size_hint_x=0.4, spacing=dp(10))
        
        # A. QR List Card
        qr_card = MDCard(orientation='vertical', size_hint_y=0.4, elevation=1, radius=[8], padding=dp(5))
        
        qr_header = MDBoxLayout(size_hint_y=None, height=dp(40), padding=[dp(10), 0])
        self.qr_list_title = MDLabel(
            text='Captured QR Codes (0)',
            font_style='Subtitle1',
            theme_text_color='Primary',
            bold=True
        )
        clear_btn = MDIconButton(icon='delete-sweep', theme_text_color="Error")
        clear_btn.bind(on_press=self.clear_all_qr_codes)
        
        qr_header.add_widget(self.qr_list_title)
        qr_header.add_widget(clear_btn)
        qr_card.add_widget(qr_header)
        
        # Scrollable List
        qr_scroll = ScrollView()
        self.qr_list_layout = MDList()
        qr_scroll.add_widget(self.qr_list_layout)
        qr_card.add_widget(qr_scroll)
        
        right_panel.add_widget(qr_card)
        
        # B. Operations Card
        ops_card = MDCard(orientation='vertical', size_hint_y=0.6, elevation=1, radius=[8], padding=dp(10), spacing=dp(10))
        
        # 1. Mode Switch
        mode_box = MDBoxLayout(size_hint_y=None, height=dp(40), spacing=dp(10))
        mode_box.add_widget(MDLabel(text="MODE:", theme_text_color="Secondary", size_hint_x=0.3))
        
        self.auto_btn = MDRectangleFlatIconButton(
            text="AUTO",
            icon="autorenew",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            line_color=(0, 0, 0, 0),
            md_bg_color=self.theme_cls.primary_color,
            size_hint_x=0.35
        )
        self.auto_btn.bind(on_press=self.set_auto_mode)
        
        self.manual_btn = MDRectangleFlatIconButton(
            text="MANUAL",
            icon="hand-back-right",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
            line_color=self.theme_cls.primary_color,
            md_bg_color=(0, 0, 0, 0),
            size_hint_x=0.35
        )
        self.manual_btn.bind(on_press=self.set_manual_mode)
        
        mode_box.add_widget(self.auto_btn)
        mode_box.add_widget(self.manual_btn)
        ops_card.add_widget(mode_box)
        
        # 2. Config Inputs
        # Beer Type
        self.beer_dropdown_btn = MDRectangleFlatIconButton(
            text="Select Beer Type",
            icon="beer",
            size_hint_x=0.5,
            pos_hint={'center_x': 0.5},
            size_hint_y=None, height=dp(40)
        )
        self.beer_dropdown_btn.bind(on_release=self.open_beer_menu)
        ops_card.add_widget(self.beer_dropdown_btn)
        
        # Batch No
        self.batch_field = MDTextField(
            text=self.last_batch_number,
            hint_text="Batch Number",
            helper_text="Format: BATCH-001",
            helper_text_mode="on_focus",
            icon_right="clipboard-text"
        )
        self.batch_field.bind(text=self.on_batch_text_change)
        ops_card.add_widget(self.batch_field)
        
        # Target Count Button
        count_box = MDBoxLayout(size_hint_y=None, height=dp(40))
        count_box.add_widget(MDLabel(text="Target Count:", theme_text_color="Secondary"))
        count_btn = MDRaisedButton(text=str(self.required_keg_count), on_press=self.change_count)
        self.count_button = count_btn # Keep ref
        count_box.add_widget(count_btn)
        ops_card.add_widget(count_box)
        
        # 3. Process Status
        self.process_card = MDCard(
            orientation='vertical',
            size_hint_y=None,
            height=dp(70),
            padding=dp(8),
            md_bg_color=(0.1, 0.1, 0.1, 0.05) # Light grey
        )
        self.process_status_label = MDLabel(
            text="Waiting for kegs...",
            halign="center",
            theme_text_color="Primary",
            bold=True
        )
        self.process_detail_label = MDLabel(
            text="Place kegs under camera",
            halign="center",
            theme_text_color="Secondary",
            font_style="Caption"
        )
        self.process_card.add_widget(self.process_status_label)
        self.process_card.add_widget(self.process_detail_label)
        ops_card.add_widget(self.process_card)
        
        # 4. Action Buttons
        self.send_btn = MDRaisedButton(
            text="SEND TO SERVER",
            icon="cloud-upload",
            size_hint_x=1,
            size_hint_y=None,
            height=dp(50),
            disabled=True,
            md_bg_color=(0.2, 0.2, 0.2, 0.2)
        )
        self.send_btn.bind(on_press=self.send_to_server)
        ops_card.add_widget(self.send_btn)
        
        # Footer Actions
        footer_actions = MDBoxLayout(spacing=dp(5), size_hint_y=None, height=dp(40))
        
        self.capture_btn = MDRaisedButton(
            text="CAPTURE",
            md_bg_color=self.theme_cls.accent_color,
            on_press=self.force_capture
        )
        
        sync_btn = MDIconButton(icon="sync", on_press=lambda x: self.sync_cloud())
        exit_btn = MDIconButton(icon="power", theme_text_color="Error", on_press=self.confirm_exit)
        
        footer_actions.add_widget(sync_btn)
        # footer_actions.add_widget(logs_btn) # Removed
        footer_actions.add_widget(Widget()) # Spacer
        footer_actions.add_widget(self.capture_btn)
        footer_actions.add_widget(exit_btn)
        
        ops_card.add_widget(footer_actions)
        
        right_panel.add_widget(ops_card)
        main_content.add_widget(right_panel)
        
        self.add_widget(main_content)
        
        # Add initial log
        self.add_log("System started (KivyMD) - Manual mode")

    # --- BEER MENU LOGIC ---
    def open_beer_menu(self, item):
        menu_items = [
            {
                "text": name,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=name: self.set_beer_type(x),
            } for name in self.beer_types
        ]
        self.menu_beer = MDDropdownMenu(
            caller=item,
            items=menu_items,
            width_mult=4,
        )
        self.menu_beer.open()

    def set_beer_type(self, text_item):
        self.beer_dropdown_btn.text = text_item
        self.menu_beer.dismiss()
        self.add_log(f"Beer type selected: {text_item}")
        
    def on_batch_text_change(self, instance, text):
        self.last_batch_number = text
        save_last_batch(text)

    # --- LOGIC METHODS (ADAPTED) ---
    def add_qr_to_list(self, qr_data):
        """Add a QR code to the MDList"""
        if qr_data not in self.captured_qr_codes:
            self.captured_qr_codes.append(qr_data)
            self.update_qr_list_display()
            return True
        return False
    
    def remove_qr_from_list(self, qr_data):
        if qr_data in self.captured_qr_codes:
            self.captured_qr_codes.remove(qr_data)
            self.update_qr_list_display()
            self.current_count = len(self.captured_qr_codes)

    def clear_all_qr_codes(self, instance=None):
        self.captured_qr_codes = []
        self.update_qr_list_display()
        self.current_count = 0
        self.add_log("QR list cleared")

    def update_qr_list_display(self):
        self.qr_list_layout.clear_widgets()
        count = len(self.captured_qr_codes)
        self.qr_list_title.text = f'Captured QR Codes ({count})'
        
        for qr_data in self.captured_qr_codes:
            item = OneLineAvatarIconListItem(text=qr_data, on_release=lambda x: None)
            delete_icon = IconRightWidget(icon="close-circle", theme_text_color="Error")
            delete_icon.bind(on_press=lambda x, d=qr_data: self.remove_qr_from_list(d))
            item.add_widget(delete_icon)
            self.qr_list_layout.add_widget(item)

    # --- DIALOGS ---
    def show_toast(self, message, msg_type="info", duration=3):
        MDSnackbar(
            MDLabel(text=message, theme_text_color="Custom", text_color=(1,1,1,1)),
            md_bg_color=(0.2, 0.2, 0.2, 1) if msg_type=="info" else (0.8, 0, 0, 1),
            duration=duration
        ).open()

    def confirm_exit(self, instance):
        self.show_confirmation_dialog("Exit Application", "Are you sure you want to exit?", lambda x: App.get_running_app().stop())

    def show_confirmation_dialog(self, title, text, on_confirm):
        if not self.confirm_dialog:
            self.confirm_dialog = MDDialog(
                title=title,
                text=text,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda x: self.confirm_dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="CONFIRM",
                        theme_text_color="Custom",
                        text_color=(1, 1, 1, 1),
                        on_release=lambda x: [self.confirm_dialog.dismiss(), on_confirm(x)]
                    ),
                ],
            )
        else:
             self.confirm_dialog.title = title
             self.confirm_dialog.text = text
             # Update callback (this is a bit hacky, cleaner to create new dialog or manage callbacks properly)
             # For simplicity, creating new one for different actions is safer, but reuse is better for memory.
             # Re-creating for safety in this demo
             self.confirm_dialog = MDDialog(
                title=title,
                text=text,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda x: self.confirm_dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="CONFIRM",
                        theme_text_color="Custom",
                        text_color=(1, 1, 1, 1),
                        on_release=lambda x: [self.confirm_dialog.dismiss(), on_confirm(x)]
                    ),
                ],
            )
        self.confirm_dialog.open()

    def change_count(self, instance):
        # Implementation for MDDialog with custom content for keypad? 
        # For now, let's just cycle 1-5 or use simple inputs. 
        # A full keypad dialog in MD is custom.
        # I will keep it simple: Use a text input dialog
        self.dialog = MDDialog(
            title="Set Target Count",
            type="custom",
            content_cls=MDTextField(hint_text="Enter count (e.g. 5)", text=str(self.required_keg_count)),
            buttons=[
                MDFlatButton(text="CANCEL", on_release=lambda x: self.dialog.dismiss()),
                MDRaisedButton(text="OK", on_release=lambda x: self.set_count_from_dialog(self.dialog.content_cls.text))
            ],
        )
        self.dialog.open()

    def set_count_from_dialog(self, text):
        try:
            val = int(text)
            if val > 0:
                self.required_keg_count = val
                self.count_button.text = str(val)
                self.target_display.text = str(val)
                self.dialog.dismiss()
            else:
                self.show_toast("Invalid number", "error")
        except ValueError:
             self.show_toast("Must be a number", "error")

    # --- EXISTING LOGIC HANDLERS (Simplified for brevity but kept functional) ---
    def update_frame(self, dt):
        if not self.detection_active or not self.camera:
            return

        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return
        
        vis_frame = frame.copy()
        
        if self.processing:
            self._draw_qr_overlays(vis_frame)
            self._update_preview_texture(vis_frame)
            return

        self._handle_detection_process(frame)
        self._draw_qr_overlays(vis_frame)
        self._update_preview_texture(vis_frame)
        self.process_frame(frame)

    def _draw_qr_overlays(self, vis_frame):
        try:
            for qr in self.latest_qr_results:
                x1, y1, x2, y2 = qr['bbox']
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if 'data' in qr:
                    label = qr['data'][:10]
                    cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass

    def _handle_detection_process(self, frame):
        try:
            if self.current_detection_task and self.current_detection_task.done():
                self._process_detection_results(frame)
            
            if self.current_detection_task is None:
                self.current_detection_task = self.detector_executor.submit(
                    self.qr_detector.detect_and_decode, frame, False
                )
        except Exception:
            pass

    def _process_detection_results(self, frame):
        try:
            results = self.current_detection_task.result()
            if results:
                qr_list = results[0]
                self.latest_qr_results = qr_list
                if qr_list:
                    self.sync_qr_list_with_detection(qr_list)
                    self._check_auto_trigger(qr_list, frame)
        except Exception as e:
            main_logger.warning(f"Detection error: {e}")
        finally:
            self.current_detection_task = None

    def _check_auto_trigger(self, qr_list, frame):
        if not (self.is_auto_mode and not self.processing and not self.auto_confirm_pending):
            return

        if len(qr_list) == self.required_keg_count:
            self.stability_counter += 1
            if self.stability_counter >= 5:
                self.trigger_capture(frame)
                self.stability_counter = 0
        else:
            self.stability_counter = 0

    def _update_preview_texture(self, frame):
        try:
            h, w = frame.shape[:2]
            texture = Texture.create(size=(w, h), colorfmt='bgr')
            flipped = cv2.flip(frame, 0)
            texture.blit_buffer(flipped.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.preview_image.texture = texture
            self.preview_image.canvas.ask_update()
        except Exception:
            pass

    def process_frame(self, frame):
        # Simplified process frame logic for UI update
        qr_count = len(self.latest_qr_results)
        self.current_display.text = str(qr_count)
        
        # Status update
        if self.processing:
            self.update_status("Processing batch...", "primary")
        elif self.data_ready_to_send:
             self.update_status("Ready to Send!", "accent")
        elif qr_count == self.required_keg_count:
             self.update_status("Target Achieved", "primary")
        else:
             self.update_status("Waiting...", "secondary")

    def update_status(self, text, color_theme):
        self.process_status_label.text = text
        # color_theme logic handled by KivyMD theme colors, can just set text
    
    def sync_qr_list_with_detection(self, qr_results):
        for qr in qr_results:
            self.add_qr_to_list(qr['data'])

    # --- CAPTURE & SERVER LOGIC (Mirrors original) ---
    def trigger_capture(self, frame):
        if self.processing: return
        
        batch = self.batch_field.text
        if not batch or batch == "BATCH-":
             self.show_toast("Enter Batch Number!", "error")
             return

        # Validate Beer Type
        selected_beer_name = self.beer_dropdown_btn.text
        if selected_beer_name == "Select Beer Type" or selected_beer_name == "Loading...":
             self.show_toast("Select a Beer Type!", "error")
             return
             
        # Get Beer ID
        beer_id = self.beer_type_map.get(selected_beer_name, selected_beer_name)
        
        self.processing = True
        self.update_status("Capturing...", "accent")
        
        # Save Frame Logic
        image_name = f"batch_{create_timestamp()}.jpg"
        frame_path = str(SAVE_FOLDER / image_name)
        cv2.imwrite(frame_path, frame)
        
        # Fake Deep Scan (async)
        def deep_scan():
            time.sleep(0.5) # Simulate processing
            _, count = self.latest_qr_results, len(self.latest_qr_results)
            session_id = f"BATCH_{self.database.get_next_batch_number():04d}"
            
            Clock.schedule_once(lambda dt: self._finalize_capture(
                frame_path, image_name, session_id, beer_id, selected_beer_name, batch, datetime.now().strftime('%d-%m-%Y %H:%M:%S'), count
            ))
            
        threading.Thread(target=deep_scan, daemon=True).start()

    def _finalize_capture(self, frame_path, image_name, session_id, beer_id, beer_name, batch, filling_timestamp, final_count):
        self._pending_capture = {
            'frame_path': frame_path, 'image_name': image_name, 'session_id': session_id,
            'beer_id': beer_id, 'batch': batch, 'filling_timestamp': filling_timestamp, 'detected_count': final_count
        }
        self.data_ready_to_send = True
        self.processing = False
        self.send_btn.disabled = False
        self.send_btn.md_bg_color = (0.2, 0.8, 0.2, 1) # Green
        self.show_toast("Capture Complete! Ready to Send.", "info")

    def send_to_server(self, instance):
        if not self._pending_capture: return
        self.send_btn.disabled = True
        
        capture_data = self._pending_capture
        def process():
             try:
                 future = submit_batch(
                    capture_data['frame_path'], capture_data['image_name'], capture_data['session_id'], 
                    self.required_keg_count, beer_type=capture_data['beer_id'], batch=capture_data['batch'],
                    filling_date=capture_data['filling_timestamp']
                 )
                 # Wait for processing to complete
                 future.result()
                 
                 # Fetch response to get Pallet ID
                 pallet_id = None
                 try:
                     response_str = self.database.get_batch_response(capture_data['session_id'])
                     if response_str:
                         resp = json.loads(response_str)
                         pallet_id = resp.get('paletteId') or resp.get('palletId') or resp.get('id')
                 except Exception as e:
                     main_logger.warning(f"Failed to parse response: {e}")

                 Clock.schedule_once(lambda dt: self.complete_capture(capture_data['session_id'], pallet_id))
             except Exception as e:
                 Clock.schedule_once(lambda dt: self.show_toast(f"Error: {e}", "error"))
        
        threading.Thread(target=process, daemon=True).start()

    def complete_capture(self, session_id, pallet_id=None):
        if pallet_id:
            self.show_toast(f"Pallet {pallet_id} Created!", "success")
            self.add_log(f"SUCCESS: Pallet {pallet_id} Created")
        else:
            self.show_toast("Batch Sent!", "success")
            self.add_log("SUCCESS: Batch Sent")
            
        # Reset UI and Detection State
        self.clear_all_qr_codes()
        self.latest_qr_results = []
        self.last_captured_qr_set = set()
        self.data_ready_to_send = False
        self.send_btn.disabled = True
        self.send_btn.md_bg_color = (0.2, 0.2, 0.2, 0.2)
        
        # If auto mode, we might want to pause briefly or just let it continue
        if self.is_auto_mode:
            self.add_log("Auto-resetting for next batch...")

    # --- UTILS ---
    def set_auto_mode(self, instance):
        self.is_auto_mode = True
        self.auto_btn.md_bg_color = self.theme_cls.primary_color
        self.auto_btn.text_color = (1,1,1,1)
        self.manual_btn.md_bg_color = (0,0,0,0)
        self.manual_btn.text_color = self.theme_cls.primary_color
        self.add_log("Switched to AUTO")
        
    def set_manual_mode(self, instance):
        self.is_auto_mode = False
        self.manual_btn.md_bg_color = self.theme_cls.primary_color
        self.manual_btn.text_color = (1,1,1,1)
        self.auto_btn.md_bg_color = (0,0,0,0)
        self.auto_btn.text_color = self.theme_cls.primary_color
        self.add_log("Switched to MANUAL")

    def force_capture(self, instance):
        self.show_confirmation_dialog("Confirm Capture", "Capture current batch?", 
                                      lambda x: self.trigger_capture_manual())

    def trigger_capture_manual(self):
        ret, frame = self.camera.get_frame()
        if ret: self.trigger_capture(frame)

    def check_network(self, dt):
        if self.api_sender.get_network_status():
            self.top_bar.right_action_items = [["wifi", lambda x: None, "Online"]]
        else:
             self.top_bar.right_action_items = [["wifi-off", lambda x: None, "Offline"]]

    def recover_batches(self, dt):
        """Recover stuck batches"""
        try:
            stuck = self.database.get_stuck_batches(timeout_minutes=5)
            if stuck:
                self.add_log(f"Recovered {len(stuck)} stuck batches")
        except Exception:
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
    
    def sync_cloud(self):
        """Sync configuration with cloud (Non-blocking)"""
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
            mac_address = CAMERA_MAC_ID
            if not mac_address or mac_address == "3C:6D:66:01:5A:F0":
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                               for elements in range(0, 2*6, 2)][::-1])
                mac_address = mac.upper()
            
            payload = {"macId": mac_address}
            endpoints = [CLOUD_CONFIG_ENDPOINT]
            
            success = False
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint, json=payload, timeout=5, verify=True
                    )
                    if response.status_code == 200:
                        data = response.json()
                        keg_type = data.get("keg_type", "30L")
                        count = data.get("keg_count", DEFAULT_KEG_COUNT)
                        Clock.schedule_once(lambda dt, count=count, keg_type=keg_type: self._apply_sync_success(count, keg_type))
                        success = True
                        break
                except Exception:
                    continue
            
            if not success:
                 Clock.schedule_once(lambda dt: self._apply_sync_fail("Cloud sync failed: No valid response"))
        
        except Exception as e:
            Clock.schedule_once(lambda dt: self._apply_sync_fail(f"Sync error: {str(e)[:50]}"))
        finally:
            self.syncing = False

    def _apply_sync_success(self, count, keg_type):
        self.required_keg_count = count
        self.count_button.text = str(count)
        self.target_display.text = str(count)
        self.add_log(f"Cloud sync: {count} {keg_type} kegs")
        self.show_toast(f"Synced: {count} kegs", "success")

    def _apply_sync_fail(self, error_msg):
        self.add_log(error_msg)

    def fetch_beer_types(self, dt=None):
        """Fetch beer types from cloud API"""
        threading.Thread(target=self._fetch_beer_types_thread, daemon=True).start()

    def _fetch_beer_types_thread(self):
        try:
            beer_types = self.api_sender.get_beer_types()
            if beer_types:
                Clock.schedule_once(lambda dt: self._update_beer_types(beer_types))
            else:
                self.add_log("Using default beer types")
                Clock.schedule_once(lambda dt: self._update_beer_types(None))
        except Exception as e:
            self.add_log(f"Failed to fetch beer types: {str(e)[:50]}")
            Clock.schedule_once(lambda dt: self._update_beer_types(None))

    def _update_beer_types(self, beer_types_data):
        self.beer_type_map = {}
        names = []
        
        if beer_types_data:
            for item in beer_types_data:
                if isinstance(item, dict):
                    name = item.get('name', 'Unknown')
                    bid = item.get('_id', item.get('id', name))
                    self.beer_type_map[name] = bid
                    names.append(name)
                else:
                    name = str(item)
                    self.beer_type_map[name] = name
                    names.append(name)
        
        self.beer_types = names if names else ["Lager", "Ale", "Stout", "IPA"]
        self.add_log(f"Beer types loaded: {len(self.beer_types)}")




    def add_log(self, message):
        print(message)
    


    def update_filling_date(self, dt):
        # Method intentionally left empty. 
        # Future implementation will handle updating filling date logic here.
        pass


class SimpleKegApp(MDApp):
    def build(self):
        self.title = 'Keg Counting System'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Teal"
        
        return SimpleKegHMI()
        
    def on_stop(self):
        # Shutdown Process Worker (background threads)
        try:
            shutdown()
        except Exception as e:
            print(f"Error shutting down worker: {e}")

        if hasattr(self, 'root') and self.root:
            # Shutdown Camera
            if hasattr(self.root, 'camera') and self.root.camera:
                self.root.camera.stop()
            
            # Shutdown API Sender (UI instance)
            if hasattr(self.root, 'api_sender') and self.root.api_sender:
                self.root.api_sender.close()

if __name__ == '__main__':
    SimpleKegApp().run()
