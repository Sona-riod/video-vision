#!/usr/bin/env python3
"""
Keg Counting System HMI
Dark industrial UI matching the new design.
All original logic preserved exactly.
"""
import os
import sys

# --- CONFIGURATION FOR FULL SCREEN HMI ---
from kivy.config import Config
# Config.set('graphics', 'show_cursor', '1')
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

# -- Logging --------------------------------------------------
import logging
main_logger = logging.getLogger("MainApp")
main_logger.setLevel(logging.DEBUG)
if not main_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    main_logger.addHandler(_h)

# -- Kivy core ------------------------------------------------
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
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner

# -- KivyMD ---------------------------------------------------
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import (MDRaisedButton, MDIconButton,
                                MDFillRoundFlatButton,
                                MDRectangleFlatIconButton, MDFlatButton)
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.list import (MDList, OneLineAvatarIconListItem,
                               IconRightWidget, IRightBodyTouch)
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.spinner import MDSpinner
from kivymd.theming import ThemableBehavior

# -- Custom Modules -------------------------------------------
from modules.camera import CameraManager
from modules.database import DatabaseManager
from modules.api_sender import APISender
import modules.process_worker as process_worker
from modules.process_worker import submit_batch, shutdown
from modules.session_store import SessionStore, ScanState
from modules.utils import setup_logging, create_timestamp, save_last_batch, load_last_batch
# --- ADDED PRINTER MODULE ---
from modules.printer import ZebraPrinter  


_QRDetector = None

def get_qr_detector_class():
    global _QRDetector
    if _QRDetector is None:
        try:
            from modules.detector import QRDetector
            _QRDetector = QRDetector
            print("[MAIN] QRDetector loaded successfully")
        except Exception as e:
            print(f"[WARNING] QRDetector import failed: {e}")
            class DummyQRDetector:
                def __init__(self):
                    print("[WARNING] Using dummy QR detector")
                def detect_and_decode(self, *args):
                    return [], 0
            _QRDetector = DummyQRDetector
    return _QRDetector

from config import (
    CAMERA_CONFIG, DEFAULT_KEG_COUNT, MAX_KEG_COUNT, SAVE_FOLDER,
    MIN_KEG_COUNT, STABILITY_THRESHOLD,
    CLOUD_CONFIG_ENDPOINT, CLOUD_SYNC_INTERVAL, CAMERA_MAC_ID, GPU_CONFIG
)

# -- GPU Detection ---------------------------------------------
print("\n" + "=" * 60)
print("   PALLETIZATION SYSTEM Dark HMI")
print("=" * 60 + "\n")

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
        GPU_STATUS = {'cuda_available': False}
except Exception as e:
    print(f"GPU Detection skipped: {e}")
    GPU_STATUS = {'cuda_available': False}

logger = setup_logging()


# -------------------------------------------------------------
#  COLOUR PALETTE  — defined in modules/theme.py, imported here
# -------------------------------------------------------------
from modules.theme import C

# Convenience helpers
def _c(key):          return C[key]
def _hex(r,g,b,a=1):  return (r, g, b, a)

WAITING_FOR_KEGS_TEXT = 'Waiting for kegs...'
PLACE_KEGS_TEXT = 'Place kegs under camera'
SEND_TO_SERVER_TEXT = 'SEND TO SERVER'


# -------------------------------------------------------------
#  HELPER WIDGETS
# -------------------------------------------------------------
class DarkCard(MDBoxLayout):
    """A dark card with border and optional radius."""
    def __init__(self, radius=8, border=True, bg_key='card', **kwargs):
        super().__init__(**kwargs)
        self._bg_key  = bg_key
        self._radius  = radius
        self._border  = border
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            # background
            Color(*C[self._bg_key])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self._radius])
            # border
            if self._border:
                Color(*C['border'])
                Line(rounded_rectangle=(self.x, self.y, self.width, self.height,
                                        self._radius), width=1)


class SectionHeader(MDBoxLayout):
    """Dark section header bar with title + right widget slot."""
    def __init__(self, title='', **kwargs):
        super().__init__(
            orientation='horizontal',
            size_hint_y=None, height=dp(42),
            padding=[dp(14), 0],
            spacing=dp(8),
            **kwargs
        )
        self.bind(pos=self._draw_bg, size=self._draw_bg)
        self._title_lbl = MDLabel(
            text=title,
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text2'],
            bold=True,
            halign='left',
            valign='center',
        )
        self._title_lbl.bind(size=self._title_lbl.setter('text_size'))
        self.add_widget(self._title_lbl)

    def _draw_bg(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*C['panel'])
            Rectangle(pos=self.pos, size=self.size)
            Color(*C['border'])
            Line(points=[self.x, self.y, self.right, self.y], width=1)

    @property
    def title_label(self):
        return self._title_lbl


class DarkButton(Button):
    """Flat dark button with colour presets."""
    PRESETS = {
        'primary':      {'bg': C['accent'],      'fg': (1, 1, 1, 1)},
        'orange':       {'bg': C['orange'],      'fg': (1, 1, 1, 1)},
        'amber':        {'bg': C['amber'],       'fg': (0.1, 0.07, 0, 1)},
        'green':        {'bg': C['green'],       'fg': (0.05, 0.15, 0.07, 1)},
        'ghost':        {'bg': C['card2'],       'fg': C['text2']},
        'ghost_auto':   {'bg': C['accent_dim'],  'fg': C['text2']},
        'ghost_manual': {'bg': C['amber_dim'],   'fg': C['text3']},
        'danger':       {'bg': C['red'],         'fg': (1, 1, 1, 1)},
        'dim':          {'bg': (0.10, 0.12, 0.17, 1), 'fg': C['text3']},
    }

    def __init__(self, preset='primary', radius=6, **kwargs):
        p = self.PRESETS.get(preset, self.PRESETS['primary'])
        # CRITICAL: set background_color=(0,0,0,0) so Kivy's own rectangular
        # fill does NOT paint over our RoundedRectangle in canvas.before.
        super().__init__(
            background_color=(0, 0, 0, 0),
            background_normal='',
            background_down='',
            color=p['fg'],
            **kwargs
        )
        self._radius = radius
        self._bg_color = list(p['bg'])
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg_color)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self._radius])

    def set_color(self, rgba):
        self._bg_color = list(rgba)
        self._draw()

    def set_preset(self, preset):
        p = self.PRESETS.get(preset, self.PRESETS['primary'])
        self._bg_color = list(p['bg'])
        self.color = p['fg']
        # Schedule on next frame so Kivy layout is settled before redraw
        Clock.schedule_once(lambda dt: self._draw(), 0)


class StatCell(MDBoxLayout):
    """A stat block: label on top, big number below."""
    def __init__(self, label='', value='0', value_color_key='accent', **kwargs):
        super().__init__(orientation='vertical', padding=[0, dp(6)], **kwargs)
        self._lbl = MDLabel(
            text=label.upper(),
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text3'],
            halign='center', valign='center',
            size_hint_y=None, height=dp(14),
        )
        self._lbl.bind(size=self._lbl.setter('text_size'))
        self._val = MDLabel(
            text=value,
            font_style='H4',
            theme_text_color='Custom',
            text_color=C[value_color_key],
            halign='center', valign='center',
            bold=True,
        )
        self._val.bind(size=self._val.setter('text_size'))
        self.add_widget(self._lbl)
        self.add_widget(self._val)

    @property
    def value_label(self):
        return self._val


class QRListItem(MDBoxLayout):
    """One row in the QR list."""
    def __init__(self, code, index, on_remove, **kwargs):
        super().__init__(
            orientation='horizontal',
            size_hint_y=None, height=dp(36),
            padding=[dp(14), 0, dp(8), 0],
            spacing=dp(8),
            **kwargs
        )
        self.bind(pos=self._draw, size=self._draw)

        # green dot
        dot_wrap = MDBoxLayout(size_hint=(None, None), size=(dp(10), dp(10)),
                               pos_hint={'center_y': .5})
        with dot_wrap.canvas:
            Color(*C['green'])
            RoundedRectangle(pos=dot_wrap.pos, size=dot_wrap.size, radius=[5])
        dot_wrap.bind(pos=lambda *_: self._redraw_dot(dot_wrap),
                      size=lambda *_: self._redraw_dot(dot_wrap))
        self.add_widget(dot_wrap)

        code_lbl = MDLabel(
            text=code,
            theme_text_color='Custom',
            text_color=C['text1'],
            font_style='Caption',
            halign='left', valign='center',
        )
        code_lbl.bind(size=code_lbl.setter('text_size'))
        self.add_widget(code_lbl)

        idx_lbl = MDLabel(
            text=f'#{index}',
            theme_text_color='Custom',
            text_color=C['text3'],
            font_style='Caption',
            halign='right', valign='center',
            size_hint_x=None, width=dp(30),
        )
        self.add_widget(idx_lbl)

        rm_btn = MDIconButton(
            icon='close-circle',
            theme_text_color='Custom',
            icon_color=C['text3'],
            size_hint_x=None, width=dp(32),
        )
        rm_btn.bind(on_press=lambda _: on_remove(code))
        self.add_widget(rm_btn)

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*C['card'])
            Rectangle(pos=self.pos, size=self.size)
            Color(*C['border'])
            Line(points=[self.x, self.y, self.right, self.y], width=.8)

    def _redraw_dot(self, dot):
        dot.canvas.clear()
        with dot.canvas:
            Color(*C['green'])
            RoundedRectangle(pos=dot.pos, size=dot.size, radius=[5])


# SplashScreen — root-widget splash (modules/splash.py + modules/theme.py)
from modules.splash import SplashScreen


# -------------------------------------------------------------
#  MAIN HMI WIDGET
# -------------------------------------------------------------
class SimpleKegHMI(MDBoxLayout):
    def __init__(self, splash, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 0
        self.spacing = 0

        # Keep reference to splash (owned by App) for status updates
        self.splash = splash

        main_logger.info("Initializing SimpleKegHMI (Dark HMI UI)...")

        # -- Component placeholders -----------------------------------
        self.camera      = None
        self.database    = None
        self.api_sender  = None
        self.printer     = None
        self.qr_detector = None
        self.detector_executor = None

        # ── In-memory session state machine (replaces 8 scattered flags) ──
        self.session = SessionStore(target_count=DEFAULT_KEG_COUNT)

        # QR Tracking & Detection
        self.latest_qr_results  = []
        self.detection_active   = False
        self.is_auto_mode       = True
        self.beer_types         = ["Loading..."]
        self.beer_type_map      = {}
        self.menu_beer          = None
        self.confirm_dialog     = None
        self.count_dialog       = None
        self.last_batch_number  = load_last_batch()

        # Performance & Preview
        self._preview_texture = None
        self._texture_size    = (0, 0)
        self._frame_times     = []
        self._last_perf_log   = time.time()
        self._no_detect_secs  = 0
        self.processing       = False

        # Keep these for backward compat with display helpers
        self.required_keg_count = DEFAULT_KEG_COUNT
        self.current_count      = 0

        # Build UI layout (synchronous)
        self.build_ui()

        # Schedule heavy init — App will call this after splash is shown
        Clock.schedule_once(self.deferred_init, 0.1)

    def deferred_init(self, dt):
        """Heavy initialization that happens after the Splash Screen is drawn."""
        main_logger.info("Starting deferred initialization...")

        self.splash.update_status("Initialising Database & API...")
        self.database   = DatabaseManager()
        self.api_sender = APISender()          # single instance — one retry-monitor thread
        process_worker.set_api_sender(self.api_sender)  # inject; kills dual-singleton bug
        self.printer    = ZebraPrinter()  # --- INITIALIZE PRINTER ---
        
        self.splash.update_status("Loading AI Detection Models...")
        qr_cls           = get_qr_detector_class()
        self.qr_detector = qr_cls()

        # Async detection executor
        self.detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LiveDetector")
        self.current_detection_task = None

        self.splash.update_status("Starting Camera Hardware...")
        self.start_detection_system()

        # Update initialization progress
        self.splash.update_status("Syncing Cloud Config...")
        Clock.schedule_once(lambda dt: self.sync_cloud(), 0.5)
        
        self.splash.update_status("Fetching Beer Types...")
        Clock.schedule_once(self.fetch_beer_types, 1.0)

        # Start background update loops
        Clock.schedule_interval(self.update_frame,        1.0 / 30.0)
        Clock.schedule_interval(self.check_network,       30)
        Clock.schedule_interval(self.update_filling_date, 60)
        Clock.schedule_interval(self._tick_clock,         1)
        Clock.schedule_once(self.recover_batches,         1)

        # Final — swap splash for this HMI in the app root
        self.splash.update_status("System Ready!")
        Clock.schedule_once(lambda dt: MDApp.get_running_app().show_hmi(self), 1.5)

    # 
    #  BUILD UI
    # 
    def build_ui(self):
        # -- 1. TOP BAR --------------------------------------
        topbar = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=dp(52),
            padding=[dp(16), 0],
            spacing=dp(14),
        )
        with topbar.canvas.before:
            Color(*C['panel'])
            self._topbar_rect = Rectangle(pos=topbar.pos, size=topbar.size)
            Color(*C['border'])
            self._topbar_line = Line(
                points=[topbar.x, topbar.y, topbar.right, topbar.y], width=1.2)
        topbar.bind(pos=self._redraw_topbar, size=self._redraw_topbar)

        # Logo box — slightly larger, more visible
        logo_box = MDBoxLayout(size_hint=(None, None), size=(dp(34), dp(34)),
                               pos_hint={'center_y': .5})
        with logo_box.canvas:
            Color(*C['accent'])
            RoundedRectangle(pos=logo_box.pos, size=logo_box.size, radius=[7])
        logo_box.bind(pos=lambda *_: self._redraw_logo(logo_box),
                      size=lambda *_: self._redraw_logo(logo_box))
        logo_lbl = MDLabel(text='KC', font_style='Subtitle1', bold=True,
                           theme_text_color='Custom', text_color=(1,1,1,1),
                           halign='center', valign='center')
        logo_box.add_widget(logo_lbl)
        topbar.add_widget(logo_box)

        # Title
        title_lbl = MDLabel(
            text='[b]KEG[/b] [color=#1e90fe]Counting[/color] System',
            markup=True,
            font_style='H6',
            theme_text_color='Custom',
            text_color=C['text1'],
            halign='left', valign='center',
            size_hint_x=None, width=dp(260),
        )
        title_lbl.bind(size=title_lbl.setter('text_size'))
        topbar.add_widget(title_lbl)

        # Vertical divider
        _vbar1 = Widget(size_hint=(None, 0.5), width=dp(1), pos_hint={'center_y': .5})
        with _vbar1.canvas:
            Color(*C['border'])
            Rectangle(pos=_vbar1.pos, size=_vbar1.size)
        _vbar1.bind(pos=lambda w, *_: self._redraw_div(w), size=lambda w, *_: self._redraw_div(w))
        topbar.add_widget(_vbar1)

        # Network chip
        self.net_chip = MDLabel(
            text='ONLINE',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['green'],
            halign='left', valign='center',
            size_hint_x=None, width=dp(90),
            bold=True,
        )
        topbar.add_widget(self.net_chip)

        # Camera info chip
        cam_chip = MDLabel(
            text='Camera: ICAM-540  |  1408×1080  |  30fps',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text2'],
            halign='left', valign='center',
        )
        topbar.add_widget(cam_chip)

        topbar.add_widget(Widget())  # spacer

        # Mode indicator chip in topbar (shows current mode at a glance)
        self.topbar_mode_chip = MDLabel(
            text='[b]AUTO MODE[/b]',
            markup=True,
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='center', valign='center',
            size_hint_x=None, width=dp(110),
            bold=True,
        )
        topbar.add_widget(self.topbar_mode_chip)

        # Clock
        self.clock_lbl = MDLabel(
            text=datetime.now().strftime('%H:%M:%S'),
            font_style='Subtitle2',
            theme_text_color='Custom',
            text_color=C['text2'],
            halign='right', valign='center',
            size_hint_x=None, width=dp(90),
            bold=True,
        )
        topbar.add_widget(self.clock_lbl)

        exit_btn = MDIconButton(
            icon='power',
            theme_text_color='Custom',
            icon_color=C['red'],
        )
        exit_btn.bind(on_press=self.confirm_exit)
        topbar.add_widget(exit_btn)

        self.add_widget(topbar)

        # -- 2. WARNING BANNER (hidden by default) ------------
        self.warn_banner = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=0,           # hidden: height=0
            padding=[dp(16), 0],
            spacing=dp(8),
            opacity=0,
        )
        with self.warn_banner.canvas.before:
            Color(0.941, 0.533, 0.243, 0.10)
            self._banner_bg = Rectangle(pos=self.warn_banner.pos,
                                        size=self.warn_banner.size)
            Color(0.941, 0.533, 0.243, 0.25)
            self._banner_line = Line(
                points=[self.warn_banner.x, self.warn_banner.y,
                        self.warn_banner.right, self.warn_banner.y], width=1)
        self.warn_banner.bind(pos=self._redraw_banner, size=self._redraw_banner)
        self.warn_lbl = MDLabel(
            text='QR codes not detected  switched to MANUAL MODE. '
                 'Position kegs and press Capture.',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['orange'],
            halign='left', valign='center',
        )
        self.warn_lbl.bind(size=self.warn_lbl.setter('text_size'))
        self.warn_banner.add_widget(self.warn_lbl)
        self.add_widget(self.warn_banner)

        # -- 2b. IGNORED KEGS INFO BANNER (hidden by default) --
        # Shows when kegs from previous pallet(s) are still visible
        # but being ignored (continuous-flow mode)
        self.dup_banner = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=0,
            padding=[dp(16), 0],
            spacing=dp(8),
            opacity=0,
        )
        with self.dup_banner.canvas.before:
            Color(0.118, 0.565, 1.0, 0.10)       # soft blue tint
            self._dup_banner_bg = Rectangle(pos=self.dup_banner.pos,
                                            size=self.dup_banner.size)
            Color(0.118, 0.565, 1.0, 0.30)
            self._dup_banner_line = Line(
                points=[self.dup_banner.x, self.dup_banner.y,
                        self.dup_banner.right, self.dup_banner.y], width=1.2)
        self.dup_banner.bind(pos=self._redraw_dup_banner, size=self._redraw_dup_banner)

        dup_icon = MDLabel(
            text='ℹ',
            font_style='H6',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='center', valign='center',
            size_hint_x=None, width=dp(30),
        )
        self.dup_banner.add_widget(dup_icon)

        self.dup_warn_lbl = MDLabel(
            text='Previous kegs ignored — detecting new kegs only.',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='left', valign='center',
            bold=True,
        )
        self.dup_warn_lbl.bind(size=self.dup_warn_lbl.setter('text_size'))
        self.dup_banner.add_widget(self.dup_warn_lbl)

        # -- 3. MAIN SPLIT ------------------------------------
        split = MDBoxLayout(orientation='horizontal', spacing=0)

        # -- LEFT: CAMERA PANE (61%) --------------------------
        cam_pane = MDBoxLayout(
            orientation='vertical',
            size_hint_x=0.61,
        )
        with cam_pane.canvas.before:
            Color(*C['border'])
            self._cam_border = Line(
                points=[cam_pane.right, cam_pane.y,
                        cam_pane.right, cam_pane.top], width=1)
        cam_pane.bind(pos=self._redraw_cam_border,
                      size=self._redraw_cam_border)

        # Camera sub-header
        cam_header = SectionHeader(title='Live Camera Feed')
        cam_header.add_widget(Widget())
        self.live_badge = MDLabel(
            text='LIVE',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['green'],
            halign='right', valign='center',
            size_hint_x=None, width=dp(60),
        )
        cam_header.add_widget(self.live_badge)
        cam_pane.add_widget(cam_header)

        # Camera feed area
        cam_feed_wrap = MDBoxLayout()
        with cam_feed_wrap.canvas.before:
            Color(*C['bg'])
            self._feed_bg = Rectangle(pos=cam_feed_wrap.pos,
                                      size=cam_feed_wrap.size)
        cam_feed_wrap.bind(pos=lambda *_: setattr(self._feed_bg, 'pos', cam_feed_wrap.pos),
                           size=lambda *_: setattr(self._feed_bg, 'size', cam_feed_wrap.size))
        self.preview_image = Image(allow_stretch=True, keep_ratio=True)
        cam_feed_wrap.add_widget(self.preview_image)
        cam_pane.add_widget(cam_feed_wrap)

        # Camera footer stats row
        cam_footer = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=dp(56),
            spacing=0,
        )
        with cam_footer.canvas.before:
            Color(*C['panel'])
            self._footer_bg = Rectangle(pos=cam_footer.pos, size=cam_footer.size)
            Color(*C['border'])
            self._footer_line = Line(
                points=[cam_footer.x, cam_footer.top,
                        cam_footer.right, cam_footer.top], width=1)
        cam_footer.bind(pos=self._redraw_cam_footer, size=self._redraw_cam_footer)

        self.stat_target    = StatCell(label='Target',    value=str(self.required_keg_count),
                                       value_color_key='accent')
        self.stat_detected  = StatCell(label='Detected',  value='0',
                                       value_color_key='orange')
        self.stat_stability = StatCell(label='Stability', value='0',
                                       value_color_key='green')

        # dividers
        def _vdiv():
            d = Widget(size_hint=(None, 1), width=dp(1))
            with d.canvas:
                Color(*C['border'])
                Rectangle(pos=d.pos, size=d.size)
            d.bind(pos=lambda w,*_: self._redraw_div(w),
                   size=lambda w,*_: self._redraw_div(w))
            return d

        cam_footer.add_widget(self.stat_target)
        cam_footer.add_widget(_vdiv())
        cam_footer.add_widget(self.stat_detected)
        cam_footer.add_widget(_vdiv())
        cam_footer.add_widget(self.stat_stability)
        cam_footer.add_widget(_vdiv())

        # status badge cell
        badge_cell = MDBoxLayout(padding=[dp(12), 0])
        self.cam_status_badge = MDLabel(
            text='Auto Mode',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='center', valign='center',
        )
        self.cam_status_badge.bind(size=self.cam_status_badge.setter('text_size'))
        badge_cell.add_widget(self.cam_status_badge)
        cam_footer.add_widget(badge_cell)

        cam_pane.add_widget(cam_footer)
        split.add_widget(cam_pane)

        # -- RIGHT: CONTROL PANE ------------------------------
        ctrl_pane = MDBoxLayout(
            orientation='vertical',
            size_hint_x=0.39,
        )
        with ctrl_pane.canvas.before:
            Color(*C['bg'])
            self._ctrl_bg = Rectangle(pos=ctrl_pane.pos, size=ctrl_pane.size)
        ctrl_pane.bind(pos=lambda *_: setattr(self._ctrl_bg, 'pos', ctrl_pane.pos),
                       size=lambda *_: setattr(self._ctrl_bg, 'size', ctrl_pane.size))

        # -- QR LIST SECTION ----------------------------------
        qr_section = MDBoxLayout(
            orientation='vertical',
            size_hint_y=0.38,
        )

        qr_hdr = SectionHeader(title='Scanned QR Codes')
        self.qr_count_lbl = MDLabel(
            text='(0)',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='left', valign='center',
            size_hint_x=None, width=dp(30),
        )
        qr_hdr.add_widget(self.qr_count_lbl)
        qr_hdr.add_widget(Widget())
        clear_btn = MDIconButton(
            icon='delete-sweep',
            theme_text_color='Custom',
            icon_color=C['red'],
        )
        clear_btn.bind(on_press=self.clear_all_qr_codes)
        qr_hdr.add_widget(clear_btn)
        qr_section.add_widget(qr_hdr)

        qr_scroll = ScrollView()
        self.qr_list_layout = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=0,
        )
        self.qr_list_layout.bind(minimum_height=self.qr_list_layout.setter('height'))

        self.qr_empty_lbl = MDLabel(
            text='No QR codes scanned yet',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text3'],
            halign='center', valign='center',
            size_hint_y=None, height=dp(80),
        )
        self.qr_list_layout.add_widget(self.qr_empty_lbl)
        qr_scroll.add_widget(self.qr_list_layout)
        qr_section.add_widget(qr_scroll)
        ctrl_pane.add_widget(qr_section)

        # -- BATCH CONFIG SECTION ----------------------------
        batch_section = MDBoxLayout(
            orientation='vertical',
            size_hint_y=0.62,
            padding=[dp(14), dp(10)],
            spacing=dp(8),
        )
        with batch_section.canvas.before:
            Color(*C['bg'])
            self._batch_bg = Rectangle(pos=batch_section.pos, size=batch_section.size)
            Color(*C['border'])
            self._batch_top = Line(
                points=[batch_section.x, batch_section.top,
                        batch_section.right, batch_section.top], width=1)
        batch_section.bind(pos=self._redraw_batch, size=self._redraw_batch)

        # Section title
        batch_title = MDLabel(
            text='Batch Configuration',
            font_style='H6',
            theme_text_color='Custom',
            text_color=C['text1'],
            bold=True,
            halign='left', valign='center',
            size_hint_y=None, height=dp(28),
        )
        batch_title.bind(size=batch_title.setter('text_size'))
        batch_section.add_widget(batch_title)

        # Status card
        self.status_card = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=dp(52),
            padding=[dp(12), dp(6)],
            spacing=dp(10),
        )
        with self.status_card.canvas.before:
            Color(*C['card'])
            self._sc_bg = RoundedRectangle(pos=self.status_card.pos,
                                           size=self.status_card.size, radius=[7])
            Color(*C['border'])
            self._sc_bd = Line(rounded_rectangle=(
                self.status_card.x, self.status_card.y,
                self.status_card.width, self.status_card.height, 7), width=1)
        self.status_card.bind(pos=self._redraw_sc, size=self._redraw_sc)

        self.status_icon_lbl = MDLabel(
            text='[W]',
            font_style='H6',
            theme_text_color='Custom',
            text_color=C['text2'],
            halign='center', valign='center',
            size_hint=(None, 1), width=dp(36),
        )
        self.status_card.add_widget(self.status_icon_lbl)

        status_texts = MDBoxLayout(orientation='vertical', spacing=0)
        self.process_status_label = MDLabel(
            text=WAITING_FOR_KEGS_TEXT,
            font_style='Subtitle2',
            theme_text_color='Custom',
            text_color=C['text1'],
            bold=True,
            halign='left', valign='bottom',
        )
        self.process_status_label.bind(size=self.process_status_label.setter('text_size'))
        self.process_detail_label = MDLabel(
            text=PLACE_KEGS_TEXT,
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text2'],
            halign='left', valign='top',
        )
        self.process_detail_label.bind(size=self.process_detail_label.setter('text_size'))
        status_texts.add_widget(self.process_status_label)
        status_texts.add_widget(self.process_detail_label)
        self.status_card.add_widget(status_texts)
        batch_section.add_widget(self.status_card)

        # ── MODE SELECTOR ─────────────────────────────────────
        # Full-width two-button toggle; active side glows with its signature colour
        mode_row = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None, height=dp(64),
            spacing=dp(4),
        )

        # Label above
        mode_lbl = MDLabel(
            text='OPERATION MODE',
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['text3'],
            halign='left', valign='center',
            size_hint_y=None, height=dp(16),
            bold=True,
        )
        mode_lbl.bind(size=mode_lbl.setter('text_size'))
        mode_row.add_widget(mode_lbl)

        # Toggle pill container
        toggle_wrap = MDBoxLayout(
            orientation='horizontal',
            spacing=dp(4),
            padding=[dp(4), dp(4)],
            size_hint_y=None, height=dp(44),
        )
        with toggle_wrap.canvas.before:
            Color(*C['card'])
            self._toggle_bg = RoundedRectangle(pos=toggle_wrap.pos,
                                               size=toggle_wrap.size, radius=[8])
            Color(*C['border'])
            self._toggle_bd = Line(rounded_rectangle=(
                toggle_wrap.x, toggle_wrap.y,
                toggle_wrap.width, toggle_wrap.height, 8), width=1.2)
        toggle_wrap.bind(pos=self._redraw_toggle, size=self._redraw_toggle)

        self.auto_btn = DarkButton(
            text='AUTO',
            preset='primary',
            radius=6,
            font_size=dp(13),
            bold=True,
        )
        self.auto_btn.bind(on_press=self.set_auto_mode)

        self.manual_btn = DarkButton(
            text='MANUAL',
            preset='ghost',
            radius=6,
            font_size=dp(13),
            bold=True,
        )
        self.manual_btn.bind(on_press=self.set_manual_mode)

        toggle_wrap.add_widget(self.auto_btn)
        toggle_wrap.add_widget(self.manual_btn)
        mode_row.add_widget(toggle_wrap)
        batch_section.add_widget(mode_row)

        # Product type + keg count row
        product_row = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=dp(38),
            spacing=dp(8),
        )

        # Beer type dropdown button
        self.beer_dropdown_btn = DarkButton(
            text='Select Product Type',
            preset='ghost',
            radius=6,
            font_size=dp(12),
        )
        with self.beer_dropdown_btn.canvas.before:
            Color(*C['card'])
            self._beer_bg = RoundedRectangle(
                pos=self.beer_dropdown_btn.pos,
                size=self.beer_dropdown_btn.size, radius=[6])
            Color(*C['border'])
            self._beer_bd = Line(rounded_rectangle=(
                self.beer_dropdown_btn.x, self.beer_dropdown_btn.y,
                self.beer_dropdown_btn.width, self.beer_dropdown_btn.height, 6), width=1)
        self.beer_dropdown_btn.bind(pos=self._redraw_beer, size=self._redraw_beer,
                                    on_release=self.open_beer_menu)
        product_row.add_widget(self.beer_dropdown_btn)

        # Keg count box (only show the count button)
        keg_count_box = MDBoxLayout(
            orientation='horizontal',
            size_hint=(None, 1), width=dp(60),
            spacing=0,
        )
        with keg_count_box.canvas.before:
            Color(*C['card'])
            self._kc_bg = RoundedRectangle(pos=keg_count_box.pos,
                                           size=keg_count_box.size, radius=[6])
            Color(*C['border'])
            self._kc_bd = Line(rounded_rectangle=(
                keg_count_box.x, keg_count_box.y,
                keg_count_box.width, keg_count_box.height, 6), width=1)
        keg_count_box.bind(pos=self._redraw_kc, size=self._redraw_kc)

        self.count_button = Button(
            text=str(self.required_keg_count),
            font_size=dp(20),
            color=C['text1'],
            bold=True,
            background_color=(0, 0, 0, 0),
            background_normal='',
        )
        self.count_button.bind(on_release=self.change_count)

        keg_count_box.add_widget(self.count_button)
        product_row.add_widget(keg_count_box)
        batch_section.add_widget(product_row)

        # Batch number field (MDTextField kept for compatibility)
        self.batch_field = MDTextField(
            text=self.last_batch_number,
            hint_text='Batch Number',
            helper_text='Format: BATCH-001',
            helper_text_mode='on_focus',
            icon_right='clipboard-text',
            size_hint_y=None, height=dp(48),
            mode='rectangle',
        )
        self.batch_field.bind(text=self.on_batch_text_change)
        batch_section.add_widget(self.batch_field)

        # -- CAPTURE BUTTON — always shown; dims in AUTO, glows amber in MANUAL
        self.capture_btn_wrap = MDBoxLayout(
            size_hint_y=None, height=dp(52),
            padding=[0, dp(4)],
        )
        self.capture_btn = DarkButton(
            text='[ CAPTURE ]',
            preset='dim',   # starts dim (auto mode is default)
            radius=8,
            font_size=dp(15),
            bold=True,
            disabled=True,  # disabled in auto mode
        )
        self.capture_btn.bind(on_press=self.force_capture)
        self.capture_btn_wrap.add_widget(self.capture_btn)
        batch_section.add_widget(self.capture_btn_wrap)

        # -- SEND TO SERVER ------------------------------------
        self.send_btn = DarkButton(
            text=SEND_TO_SERVER_TEXT,
            preset='dim',
            radius=8,
            font_size=dp(14),
            bold=True,
            size_hint_y=None, height=dp(48),
            disabled=True,
        )
        self.send_btn.bind(on_press=self.send_to_server)
        batch_section.add_widget(self.send_btn)

        # -- CLEAR KEG HISTORY BUTTON -------------------------
        self.clear_history_btn = DarkButton(
            text='CLEAR KEG HISTORY',
            preset='ghost',
            radius=6,
            font_size=dp(12),
            bold=True,
            size_hint_y=None, height=dp(36),
        )
        self.clear_history_btn.bind(on_press=self._clear_keg_history)
        batch_section.add_widget(self.clear_history_btn)

        # -- FOOTER UTIL ROW ----------------------------------
        util_row = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None, height=dp(42),
            padding=[0, dp(4)],
            spacing=dp(6),
        )
        with util_row.canvas.before:
            Color(*C['panel'])
            self._util_bg = Rectangle(pos=util_row.pos, size=util_row.size)
            Color(*C['border'])
            self._util_line = Line(
                points=[util_row.x, util_row.top,
                        util_row.right, util_row.top], width=1)
        util_row.bind(pos=self._redraw_util, size=self._redraw_util)

        self.auto_active_lbl = MDLabel(
            text='[b]AUTO[/b]  Auto-capture active — triggers at target count',
            markup=True,
            font_style='Caption',
            theme_text_color='Custom',
            text_color=C['accent'],
            halign='left', valign='center',
        )
        self.auto_active_lbl.bind(size=self.auto_active_lbl.setter('text_size'))
        util_row.add_widget(self.auto_active_lbl)
        util_row.add_widget(Widget())

        sync_btn = MDIconButton(
            icon='sync',
            theme_text_color='Custom',
            icon_color=C['text2'],
        )
        sync_btn.bind(on_press=lambda _: self.sync_cloud())
        util_row.add_widget(sync_btn)
        # Exit button moved to topbar

        batch_section.add_widget(util_row)
        ctrl_pane.add_widget(batch_section)
        split.add_widget(ctrl_pane)
        self.add_widget(split)

        self.add_log("System started (Dark HMI)")

    # --------------------------------------------------------
    #  CANVAS REDRAW HELPERS
    # --------------------------------------------------------
    def _redraw_topbar(self, w, *_):
        self._topbar_rect.pos  = w.pos
        self._topbar_rect.size = w.size
        self._topbar_line.points = [w.x, w.y, w.right, w.y]

    def _redraw_banner(self, w, *_):
        self._banner_bg.pos   = w.pos
        self._banner_bg.size  = w.size
        self._banner_line.points = [w.x, w.y, w.right, w.y]

    def _redraw_dup_banner(self, w, *_):
        self._dup_banner_bg.pos   = w.pos
        self._dup_banner_bg.size  = w.size
        self._dup_banner_line.points = [w.x, w.y, w.right, w.y]

    def _redraw_cam_border(self, w, *_):
        self._cam_border.points = [w.right, w.y, w.right, w.top]

    def _redraw_cam_footer(self, w, *_):
        self._footer_bg.pos   = w.pos
        self._footer_bg.size  = w.size
        self._footer_line.points = [w.x, w.top, w.right, w.top]

    def _redraw_div(self, w):
        w.canvas.clear()
        with w.canvas:
            Color(*C['border'])
            Rectangle(pos=w.pos, size=w.size)

    def _redraw_sc(self, w, *_):
        self._sc_bg.pos  = w.pos
        self._sc_bg.size = w.size
        self._sc_bd.rounded_rectangle = (w.x, w.y, w.width, w.height, 7)

    def _redraw_toggle(self, w, *_):
        self._toggle_bg.pos  = w.pos
        self._toggle_bg.size = w.size
        self._toggle_bd.rounded_rectangle = (w.x, w.y, w.width, w.height, 6)

    def _redraw_batch(self, w, *_):
        self._batch_bg.pos   = w.pos
        self._batch_bg.size  = w.size
        self._batch_top.points = [w.x, w.top, w.right, w.top]

    def _redraw_util(self, w, *_):
        self._util_bg.pos   = w.pos
        self._util_bg.size  = w.size
        self._util_line.points = [w.x, w.top, w.right, w.top]

    def _redraw_logo(self, w):
        w.canvas.clear()
        with w.canvas:
            Color(*C['accent'])
            RoundedRectangle(pos=w.pos, size=w.size, radius=[5])

    def _redraw_beer(self, w, *_):
        self._beer_bg.pos  = w.pos
        self._beer_bg.size = w.size
        self._beer_bd.rounded_rectangle = (w.x, w.y, w.width, w.height, 6)

    def _redraw_kc(self, w, *_):
        self._kc_bg.pos  = w.pos
        self._kc_bg.size = w.size
        self._kc_bd.rounded_rectangle = (w.x, w.y, w.width, w.height, 6)

    # --------------------------------------------------------
    #  CLOCK TICK
    # --------------------------------------------------------
    def _tick_clock(self, dt):
        self.clock_lbl.text = datetime.now().strftime('%H:%M:%S')

    # --------------------------------------------------------
    #  BEER MENU  (unchanged logic)
    # --------------------------------------------------------
    def open_beer_menu(self, item):
        menu_items = [
            {
                'text': name,
                'viewclass': 'OneLineListItem',
                'on_release': lambda x=name: self.set_beer_type(x),
            } for name in self.beer_types
        ]
        self.menu_beer = MDDropdownMenu(caller=item, items=menu_items, width_mult=4)
        self.menu_beer.open()

    def set_beer_type(self, text_item):
        self.beer_dropdown_btn.text = f'{text_item}'
        if self.menu_beer:
            self.menu_beer.dismiss()
        self.add_log(f"Beer type selected: {text_item}")

    def on_batch_text_change(self, instance, text):
        self.last_batch_number = text
        save_last_batch(text)

    # --------------------------------------------------------
    #  QR LIST  (unchanged logic, new widget)
    # --------------------------------------------------------
    def add_qr_to_list(self, qr_data):
        if self.session.add_qr(qr_data):
            self.update_qr_list_display()
            return True
        return False

    def remove_qr_from_list(self, qr_data):
        self.session.qr_codes.discard(qr_data)
        self.update_qr_list_display()
        self.current_count = self.session.count()

    def clear_all_qr_codes(self, instance=None):
        self.session.qr_codes.clear()
        self.update_qr_list_display()
        self.current_count = 0
        self.add_log("QR list cleared")

    def update_qr_list_display(self):
        self.qr_list_layout.clear_widgets()
        qr_list = self.session.qr_list()
        count   = len(qr_list)
        self.qr_count_lbl.text = f'({count})'

        if count == 0:
            self.qr_list_layout.add_widget(self.qr_empty_lbl)
        else:
            for i, qr_data in enumerate(qr_list, 1):
                item = QRListItem(
                    code=qr_data,
                    index=i,
                    on_remove=self.remove_qr_from_list,
                )
                self.qr_list_layout.add_widget(item)

    # --------------------------------------------------------
    #  STATUS CARD  helper
    # --------------------------------------------------------
    def _set_status(self, icon, main, sub, badge_text=None, badge_color=None):
        self.status_icon_lbl.text         = icon
        self.process_status_label.text    = main
        self.process_detail_label.text    = sub
        if badge_text:
            self.cam_status_badge.text       = badge_text
            self.cam_status_badge.text_color = badge_color or C['accent']

    # --------------------------------------------------------
    #  TOASTS & DIALOGS  (unchanged logic)
    # --------------------------------------------------------
    def show_toast(self, message, msg_type='info', duration=3):
        color = {
            'success': (0.1, 0.5, 0.2, 1),
            'error':   (0.6, 0.1, 0.1, 1),
            'warning': (0.5, 0.3, 0.0, 1),
        }.get(msg_type, (0.15, 0.18, 0.25, 1))
        MDSnackbar(
            MDLabel(text=message, theme_text_color='Custom', text_color=(1,1,1,1)),
            md_bg_color=color,
            duration=duration,
        ).open()

    def confirm_exit(self, instance):
        self.show_confirmation_dialog(
            'Exit Application',
            'Are you sure you want to exit?',
            lambda x: App.get_running_app().stop()
        )

    def show_confirmation_dialog(self, title, text, on_confirm):
        self.confirm_dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    theme_text_color='Custom',
                    text_color=C['accent'],
                    on_release=lambda x: self.confirm_dialog.dismiss()
                ),
                MDRaisedButton(
                    text='CONFIRM',
                    theme_text_color='Custom',
                    text_color=(1,1,1,1),
                    on_release=lambda x: [self.confirm_dialog.dismiss(), on_confirm(x)]
                ),
            ],
        )
        self.confirm_dialog.open()

    def change_count(self, instance):
        """Open dialog to set target count."""
        self.count_dialog = MDDialog(
            title='Set Target Count',
            type='custom',
            content_cls=MDTextField(
                hint_text='Enter count (e.g. 5)',
                text=str(self.required_keg_count),
                input_filter='int',
                mode='rectangle'
            ),
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    theme_text_color='Custom',
                    text_color=C['accent'],
                    on_release=lambda x: self.count_dialog.dismiss()
                ),
                MDRaisedButton(
                    text='OK',
                    on_release=lambda x: self.set_count_from_dialog(
                        self.count_dialog.content_cls.text)
                ),
            ],
        )
        self.count_dialog.open()

    def set_count_from_dialog(self, text):
        if not text:
            self.show_toast('Please enter a value', 'error')
            return

        try:
            val = int(text)
            if val > 0:
                self.required_keg_count = val
                self.count_button.text = str(val)
                self.stat_target.value_label.text = str(val)
                if self.count_dialog:
                    self.count_dialog.dismiss()
                self.add_log(f"Target count updated to: {val}")
            else:
                self.show_toast('Count must be above 0', 'error')
        except ValueError:
            self.show_toast('Please enter a valid number', 'error')

    def _adj_count(self, delta):
        """+/- keg count buttons."""
        new_val = max(MIN_KEG_COUNT,
                      min(MAX_KEG_COUNT, self.required_keg_count + delta))
        self.required_keg_count   = new_val
        self.count_button.text    = str(new_val)
        self.stat_target.value_label.text = str(new_val)

    # --------------------------------------------------------
    #  FRAME UPDATE  (unchanged logic)
    # --------------------------------------------------------
    def update_frame(self, dt):
        if not self.detection_active or not self.camera:
            return
        ret, frame = self.camera.get_frame()
        if not ret or frame is None:
            return
        vis_frame = frame.copy()
        # Don't run new detection while sending (SENDING/SENT) — just show live feed
        if self.session.state in (ScanState.SENDING, ScanState.SENT):
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
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (63, 185, 80), 3)
                if 'data' in qr:
                    label = qr['data'][:10]
                    cv2.putText(vis_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (63, 185, 80), 2)
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
                    # Reset no-detect watchdog on success
                    self._no_detect_secs = 0
                else:
                    # Count seconds of no detection (called ~30fps so scale)
                    self._no_detect_secs += 1
                    if (self.is_auto_mode
                            and not self.processing
                            and self._no_detect_secs > 7 * 30):   # 7 s x 30 fps
                        self._trigger_no_detect_switch()
        except Exception as e:
            main_logger.warning(f"Detection error: {e}")
        finally:
            self.current_detection_task = None

    def _trigger_no_detect_switch(self):
        """Auto-switch to manual when QR codes absent for too long."""
        self._no_detect_secs = 0
        self.add_log("No QR detected  switching to Manual mode")
        Clock.schedule_once(lambda dt: self._apply_no_detect_ui(), 0)

    def _apply_no_detect_ui(self):
        # Show warning banner
        self.warn_banner.height  = dp(32)
        self.warn_banner.opacity = 1
        # Switch mode
        self.set_manual_mode(None, auto_triggered=True)
        # Hide banner after 6 s
        Clock.schedule_once(lambda dt: self._hide_banner(), 6)

    def _hide_banner(self):
        self.warn_banner.height  = 0
        self.warn_banner.opacity = 0

    def _check_auto_trigger(self, qr_list, frame):
        """Continuous auto-pallet flow.

        Only NEW kegs (not in sent_qr_history) are counted.  When the
        target count of NEW kegs is reached and stable, the system
        auto-captures → auto-sends → auto-prints with zero operator
        interaction.
        """
        if not self.is_auto_mode:
            return
        if self.session.state != ScanState.SCANNING:
            return

        # session.count() already returns only NEW kegs because
        # add_qr() filters out sent_qr_history entries.

        # Stability check: NEW keg count must be at target for
        # STABILITY_THRESHOLD consecutive frames.
        if self.session.target_reached():
            self.session.stability_counter += 1
            if self.session.stability_counter >= STABILITY_THRESHOLD:
                self.session.stability_counter = 0
                self.trigger_capture(frame)
                # Fully automatic: send immediately after capture
                Clock.schedule_once(lambda dt: self._auto_send_and_print(), 0.5)
        else:
            self.session.stability_counter = 0

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
        qr_count = self.session.count()           # NEW kegs only
        ignored  = self.session.ignored_count()    # kegs from previous pallets
        self.stat_detected.value_label.text  = str(qr_count)
        self.stat_stability.value_label.text = str(self.session.stability_counter)

        # ── IGNORED KEG INFO BANNER (continuous-flow) ──────────
        has_ignored, ign_count, ign_msg = self.session.get_ignored_info()
        if has_ignored:
            self._show_info_banner(ign_msg)
        elif self.dup_banner.opacity > 0:
            self._hide_info_banner()

        state = self.session.state
        if state == ScanState.SENDING:
            self._set_status('', 'Sending to server...', 'Please wait',
                             'Sending...', C['orange'])
        elif state == ScanState.READY:
            self._set_status('', 'Ready to Send!', 'Tap Send to upload',
                             'Ready', C['green'])
        elif self.session.over_target():
            over = qr_count - self.required_keg_count
            self._set_status('', f'OVER TARGET — {qr_count} new ({over} extra)',
                             f'Remove {over} keg(s) to match target {self.required_keg_count}',
                             'Over', C['red'])
        elif qr_count == self.required_keg_count:
            self._set_status('', 'Target Achieved',
                             f'{qr_count} new kegs detected' + (f' ({ignored} ignored)' if ignored else ''),
                             'Target Met', C['green'])
        elif qr_count > 0:
            remaining = self.required_keg_count - qr_count
            self._set_status('', f'Detecting — {qr_count} new of {self.required_keg_count}',
                             f'{remaining} more needed' + (f' ({ignored} ignored)' if ignored else ''),
                             'Detecting', C['accent'])
        else:
            self._set_status('', WAITING_FOR_KEGS_TEXT, PLACE_KEGS_TEXT,
                             'Scanning...', C['accent'])

    def update_status(self, text, color_theme):
        self.process_status_label.text = text

    def sync_qr_list_with_detection(self, qr_results):
        for qr in qr_results:
            self.add_qr_to_list(qr['data'])

    # --------------------------------------------------------
    #  IGNORED KEGS INFO BANNER HELPERS
    # --------------------------------------------------------
    def _show_info_banner(self, message):
        """Show the informational ignored-kegs banner (blue tint)."""
        self.dup_warn_lbl.text       = message
        self.dup_warn_lbl.text_color = C['accent']
        self.dup_banner.height  = dp(32)
        self.dup_banner.opacity = 1

    def _hide_info_banner(self):
        """Hide the ignored-kegs info banner."""
        self.dup_banner.height  = 0
        self.dup_banner.opacity = 0

    # --------------------------------------------------------
    #  MODE SWITCHING
    # --------------------------------------------------------
    def set_auto_mode(self, instance):
        self.is_auto_mode = True
        self._no_detect_secs          = 0
        self.session.stability_counter = 0

        # Toggle button styles — AUTO glows blue, MANUAL goes to dim ghost
        self.auto_btn.set_preset('primary')
        self.manual_btn.set_preset('ghost')

        # Disable capture in auto (visible but greyed, so user knows it exists)
        self.capture_btn.set_preset('dim')
        self.capture_btn.disabled = True

        # Footer label
        self.auto_active_lbl.text       = '[b]AUTO[/b]  Auto-capture active — triggers at target count'
        self.auto_active_lbl.text_color = C['accent']

        # Top-bar mode chip
        if hasattr(self, 'topbar_mode_chip'):
            self.topbar_mode_chip.text       = '[b]AUTO MODE[/b]'
            self.topbar_mode_chip.text_color = C['accent']

        # Camera feed badge
        self.cam_status_badge.text       = 'Auto Mode'
        self.cam_status_badge.text_color = C['accent']

        self.add_log("Switched to AUTO")

    def set_manual_mode(self, instance, auto_triggered=False):
        self.is_auto_mode              = False
        self.session.stability_counter = 0

        # Toggle button styles — MANUAL glows amber, AUTO goes to dim ghost
        self.manual_btn.set_preset('amber')
        self.auto_btn.set_preset('ghost')

        # Enable capture — amber glow signals "action required"
        self.capture_btn.set_preset('amber')
        self.capture_btn.disabled = False

        # Footer label
        self.auto_active_lbl.text       = '[b]MANUAL[/b]  Position kegs and tap CAPTURE'
        self.auto_active_lbl.text_color = C['amber']

        # Top-bar mode chip
        if hasattr(self, 'topbar_mode_chip'):
            self.topbar_mode_chip.text       = '[b]MANUAL MODE[/b]'
            self.topbar_mode_chip.text_color = C['amber']

        # Camera feed badge
        self.cam_status_badge.text       = 'Manual'
        self.cam_status_badge.text_color = C['amber']

        self._set_status('', 'Manual mode active',
                         'Position kegs, then tap CAPTURE')

        if not auto_triggered:
            self.add_log("Switched to MANUAL")
        else:
            self.add_log("Auto-switched to MANUAL (no QR detected)")

    # --------------------------------------------------------
    #  CAPTURE
    # --------------------------------------------------------
    def force_capture(self, instance):
        self.show_confirmation_dialog(
            'Confirm Capture',
            'Capture current batch?',
            lambda x: self.trigger_capture_manual()
        )

    def trigger_capture_manual(self):
        ret, frame = self.camera.get_frame()
        if ret:
            self.trigger_capture(frame)

    def trigger_capture(self, frame):
        # Guard: only capture when actively scanning
        if self.session.state != ScanState.SCANNING:
            return

        batch = self.batch_field.text
        if not batch or batch == 'BATCH-':
            self.show_toast('Enter Batch Number!', 'error')
            return

        selected_beer_name = self.beer_dropdown_btn.text.replace('[P]  ', '')
        if selected_beer_name in ('Select Product Type', 'Loading...'):
            self.show_toast('Select a Product Type!', 'error')
            return

        beer_id = self.beer_type_map.get(selected_beer_name, selected_beer_name)

        # ── Generate session_id ONCE here (timestamp-based, no DB call = no race) ──
        session_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save frame to disk
        image_name = f"batch_{create_timestamp()}.jpg"
        frame_path = str(SAVE_FOLDER / image_name)
        cv2.imwrite(frame_path, frame)

        # Populate session store
        self.session.beer_type     = selected_beer_name
        self.session.beer_type_id  = beer_id
        self.session.batch_number  = batch
        self.session.frame_path    = frame_path
        self.session.image_name    = image_name
        self.session.captured_at   = datetime.now()
        self.session.session_id    = session_id
        self.session.target_count  = self.required_keg_count
        self.session.state         = ScanState.READY

        # UI feedback
        self.send_btn.disabled = False
        self.send_btn.set_preset('green')
        self.send_btn.text = SEND_TO_SERVER_TEXT
        self._set_status('', 'Capture Complete! Ready to Send.',
                         f'{self.session.count()} QR codes recorded',
                         'Ready', C['green'])
        self.show_toast('Capture Complete! Ready to Send.', 'success')
        self.add_log(f"Captured session {session_id} | {self.session.count()} QRs")

    # --------------------------------------------------------
    #  SEND TO SERVER  (WITH PRINTER INTEGRATION)
    # --------------------------------------------------------
    def send_to_server(self, instance):
        # Guard: only send when in READY state
        if self.session.state != ScanState.READY:
            return

        # Guard: count must match target exactly
        qr_count = self.session.count()
        if qr_count != self.session.target_count:
            if qr_count > self.session.target_count:
                self.show_toast(f'Too many QRs ({qr_count}). Target is {self.session.target_count}. Remove extras.', 'error')
            else:
                self.show_toast(f'Only {qr_count} QRs. Need {self.session.target_count}.', 'error')
            return

        self.session.state = ScanState.SENDING
        self.send_btn.disabled = True
        self.send_btn.set_preset('dim')
        self._set_status('', 'Sending to server...', 'Uploading batch data',
                         'Sending', C['orange'])

        # Snapshot all needed data before spawning the thread
        s = self.session

        def process():
            try:
                future = submit_batch(
                    frame_path    = s.frame_path,
                    image_name    = s.image_name,
                    session_id    = s.session_id,
                    qr_codes      = s.qr_list(),
                    required_count= s.target_count,
                    beer_type     = s.beer_type_id,
                    batch         = s.batch_number,
                    filling_date  = s.captured_at.isoformat() if s.captured_at else None,
                )
                result = future.result()   # dict: {success, pallet_id, error, qr_count}
                Clock.schedule_once(lambda dt: self._on_send_complete(result))
            except Exception as e:
                Clock.schedule_once(
                    lambda dt: self._on_send_complete({'success': False, 'error': str(e)}))

        threading.Thread(target=process, daemon=True).start()

    def _on_send_complete(self, result):
        """Called on the Kivy main thread after the background send finishes."""
        if result.get('success'):
            pallet_id = result.get('pallet_id')
            if pallet_id:
                self.show_toast(f"Pallet {pallet_id} Created!", 'success')
                self.add_log(f"SUCCESS: Pallet {pallet_id} Created")
                # Print label
                if self.printer:
                    self.show_toast("Printing Pallet Label...", "success")
                    ok, err = self.printer.print_pallet_qr(pallet_id)
                    if ok:
                        self.show_toast("Pallet Label Printed!", "success")
                    else:
                        self.show_toast(f"Printer Error: {err}", "error")
            else:
                self.show_toast('Batch Sent!', 'success')
                self.add_log("SUCCESS: Batch Sent")

            self.session.mark_sent()   # records last_sent_qr_set + sent_qr_history, sets state=SENT
            # Faster reset for continuous flow (was 2.0s)
            Clock.schedule_once(lambda dt: self._reset_session(), 1.0)
        else:
            error = result.get('error', 'Unknown error')
            self.show_toast('Send failed — queued for retry', 'error')
            self.add_log(f"FAILED: {error}")
            # Roll back to READY so user can retry
            self.session.state = ScanState.READY
            self.send_btn.disabled = False
            self.send_btn.set_preset('green')
            self._set_status('', 'Send Failed — Retry?',
                             'Tap Send to try again', 'Ready', C['green'])

    def _reset_session(self):
        """Reset to SCANNING state after a successful send.

        Continuous-flow: sent_qr_history is PRESERVED across resets
        so kegs that remain physically under the camera are ignored.
        The operator can clear history manually via the CLEAR KEG HISTORY button.
        """
        self.session.reset()   # clears qr_codes but keeps sent_qr_history
        self.update_qr_list_display()
        self.send_btn.disabled = True
        self.send_btn.set_preset('dim')
        self.send_btn.text = SEND_TO_SERVER_TEXT
        self._set_status('', WAITING_FOR_KEGS_TEXT, PLACE_KEGS_TEXT,
                         'Scanning...', C['accent'])
        ignored = self.session.ignored_count()
        if ignored > 0:
            self.add_log(f"Ready for next pallet ({ignored} previous keg(s) will be ignored)")
        if self.is_auto_mode:
            self.add_log("Auto-resetting for next batch...")

    def _auto_send_and_print(self):
        """Fully automatic: send to server after auto-capture.

        Called by _check_auto_trigger() after a successful capture.
        The _on_send_complete() callback handles printing and reset.
        """
        if self.session.state != ScanState.READY:
            return
        self.add_log("Auto-sending to server...")
        self.send_to_server(None)

    def _clear_keg_history(self, instance=None):
        """Operator clears the sent-keg ignore list.

        Use when all kegs have been physically removed from under the camera
        and a completely fresh pallet load is starting.
        """
        self.show_confirmation_dialog(
            'Clear Keg History',
            f'{self.session.ignored_count()} keg(s) are being ignored from previous pallets.\n'
            'Clear the list so all kegs are detected again?',
            lambda x: self._do_clear_history()
        )

    def _do_clear_history(self):
        self.session.clear_history()
        self._hide_info_banner()
        self.add_log("Keg history cleared — all kegs will be detected")
        self.show_toast('Keg history cleared!', 'success')


    def check_network(self, dt):
        if self.api_sender.get_network_status():
            self.net_chip.text       = '[A]  ONLINE'
            self.net_chip.text_color = C['green']
        else:
            self.net_chip.text       = '[X]  OFFLINE'
            self.net_chip.text_color = C['red']

    def recover_batches(self, dt):
        try:
            stuck = self.database.get_stuck_batches(timeout_minutes=5)
            if stuck:
                self.add_log(f"Recovered {len(stuck)} stuck batches")
        except Exception:
            pass

    def start_detection_system(self):
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
        if hasattr(self, 'syncing') and self.syncing:
            self.add_log("Sync already in progress...")
            return
        if self.is_auto_mode:
            self.syncing = True
            self.add_log("Syncing with cloud...")
            threading.Thread(target=self._sync_thread, daemon=True).start()
        else:
            self.add_log("Manual mode  using local config")

    def _sync_thread(self):
        try:
            mac_address = CAMERA_MAC_ID
            if not mac_address or mac_address == "3C:6D:66:01:5A:F0":
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> e) & 0xff)
                                for e in range(0, 2*6, 2)][::-1])
                mac_address = mac.upper()

            payload   = {"macId": mac_address}
            endpoints = [CLOUD_CONFIG_ENDPOINT]
            success   = False

            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint, json=payload, timeout=5, verify=True)
                    if response.status_code == 200:
                        data     = response.json()
                        keg_type = data.get("keg_type", "30L")
                        count    = data.get("keg_count", DEFAULT_KEG_COUNT)
                        Clock.schedule_once(
                            lambda dt, c=count, k=keg_type:
                            self._apply_sync_success(c, k))
                        success = True
                        break
                except Exception:
                    continue

            if not success:
                Clock.schedule_once(
                    lambda dt: self._apply_sync_fail(
                        "Cloud sync failed: No valid response"))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: self._apply_sync_fail(f"Sync error: {str(e)[:50]}"))
        finally:
            self.syncing = False

    def _apply_sync_success(self, count, keg_type):
        self.required_keg_count                  = count
        self.count_button.text                   = str(count)
        self.stat_target.value_label.text        = str(count)
        self.add_log(f"Cloud sync: {count} {keg_type} kegs")
        self.show_toast(f"Synced: {count} kegs", 'success')

    def _apply_sync_fail(self, error_msg):
        self.add_log(error_msg)

    def fetch_beer_types(self, dt=None):
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
                    bid  = item.get('_id', item.get('id', name))
                    self.beer_type_map[name] = bid
                    names.append(name)
                else:
                    name = str(item)
                    self.beer_type_map[name] = name
                    names.append(name)
        self.beer_types = names if names else []
        self.add_log(f"Beer types loaded: {len(self.beer_types)}")

    def add_log(self, message):
        print(message)

    def update_filling_date(self, dt):
        pass


# -------------------------------------------------------------
#  APP
# -------------------------------------------------------------
class SimpleKegApp(MDApp):

    def build(self):
        self.title = 'Keg Counting System'
        self.theme_cls.theme_style     = "Dark"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette  = "Teal"
        Window.clearcolor = C['bg']

        # ── Show splash as the REAL root widget ───────────────
        # No ModalView race — the splash IS the window content.
        self.splash = SplashScreen(app=self)
        # Kick off HMI init on the first rendered frame
        Clock.schedule_once(self.splash.start_init, 0.5)
        return self.splash

    def init_hmi(self):
        """
        Called by SplashScreen.start_init() on the first rendered frame.
        Builds SimpleKegHMI (which runs deferred_nit in background).
        The HMI holds a reference to the splash for stiatus updates.
        """
        self._hmi = SimpleKegHMI(splash=self.splash)

    def show_hmi(self, hmi):
        """
        Swap the splash out and replace the root widget with the HMI.
        Called by SimpleKegHMI.deferred_init() when init is complete.
        """
        self.root_window.remove_widget(self.root)
        self.root_window.add_widget(hmi)

    def on_stop(self):
        if getattr(self, '_stopped', False):
            return
        self._stopped = True
        try:
            shutdown()
        except Exception as e:
            print(f"Error shutting down worker: {e}")
        hmi = getattr(self, '_hmi', None)
        if hmi:
            if getattr(hmi, 'qr_detector', None):
                hmi.qr_detector.shutdown()
            if getattr(hmi, 'camera', None):
                hmi.camera.stop()
            if getattr(hmi, 'api_sender', None):
                hmi.api_sender.close()


if __name__ == '__main__':
    import signal

    def _signal_handler(sig, frame):
        app = MDApp.get_running_app()
        if app:
            app.stop()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    SimpleKegApp().run()