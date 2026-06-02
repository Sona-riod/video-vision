# modules/camera_init_splash.py
"""
CameraInitSplash — the FIRST screen shown on startup (when CAMERA_INIT_ENABLED).

It runs the Advantech camera REST configuration sequence (modules.camera_init)
on a background thread and shows per-step progress. On success it advances to the
original SplashScreen via app.on_camera_init_done(). On failure it shows the failing
step in red with large RETRY / EXIT buttons (operator cannot enter the app until the
camera is configured).

Styled to match modules/splash.py so the two splashes feel like one product.
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.metrics import dp
from kivy.clock import Clock
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.progressbar import MDProgressBar

from modules.theme import C
from modules.camera_init import CameraInitializer


class CameraInitSplash(FloatLayout):
    """Full-screen camera-initialization splash rendered as the root widget."""

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self._app = app
        self._initializer = None
        self._draw_bg()
        self._build_ui()

    # ----------------------------------------------------------
    # Background
    # ----------------------------------------------------------
    def _draw_bg(self):
        with self.canvas.before:
            Color(*C['bg'])
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_bg, size=self._sync_bg)

    def _sync_bg(self, *_):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------
    def _build_ui(self):
        col = BoxLayout(
            orientation='vertical',
            spacing=dp(16),
            size_hint=(None, None),
            size=(dp(440), dp(400)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
        )

        # ── Logo badge ────────────────────────────────────────
        badge_wrap = BoxLayout(
            size_hint=(None, None),
            size=(dp(88), dp(88)),
            pos_hint={'center_x': 0.5},
        )
        with badge_wrap.canvas:
            Color(*C['accent'])
            self._badge_rect = RoundedRectangle(
                pos=badge_wrap.pos, size=badge_wrap.size, radius=[18])
        badge_wrap.bind(pos=self._sync_badge, size=self._sync_badge)
        badge_wrap.add_widget(MDLabel(
            text='KC', font_style='H4', bold=True,
            theme_text_color='Custom', text_color=(1, 1, 1, 1),
            halign='center', valign='center',
        ))
        col.add_widget(badge_wrap)

        # ── Title ─────────────────────────────────────────────
        col.add_widget(MDLabel(
            text='CAMERA INITIALIZATION',
            font_style='H5', bold=True,
            theme_text_color='Custom', text_color=C['text1'],
            halign='center',
            size_hint_y=None, height=dp(36),
        ))

        # ── Sub-title ─────────────────────────────────────────
        col.add_widget(MDLabel(
            text='Configuring Advantech camera hardware',
            font_style='Subtitle2',
            theme_text_color='Custom', text_color=C['text2'],
            halign='center',
            size_hint_y=None, height=dp(26),
        ))

        # ── Spinner ───────────────────────────────────────────
        spinner_wrap = BoxLayout(
            size_hint=(None, None), size=(dp(52), dp(52)),
            pos_hint={'center_x': 0.5},
        )
        self.spinner = MDSpinner(
            size_hint=(None, None), size=(dp(48), dp(48)),
            active=True,
            palette=[C['accent'], C['green'], C['orange']],
        )
        spinner_wrap.add_widget(self.spinner)
        col.add_widget(spinner_wrap)

        # ── Progress bar ──────────────────────────────────────
        self.progress = MDProgressBar(
            type="determinate", value=0, max=100,
            size_hint_y=None, height=dp(6),
            color=C['accent'],
        )
        col.add_widget(self.progress)

        # ── Step / status label ───────────────────────────────
        self.status_label = MDLabel(
            text='Connecting to camera service…',
            font_style='Subtitle2',
            theme_text_color='Custom', text_color=C['text2'],
            halign='center',
            size_hint_y=None, height=dp(34),
        )
        col.add_widget(self.status_label)

        # ── Action buttons (hidden until error) ───────────────
        self.btn_row = BoxLayout(
            orientation='horizontal',
            spacing=dp(16),
            size_hint=(None, None),
            size=(dp(360), dp(56)),
            pos_hint={'center_x': 0.5},
            opacity=0,
            disabled=True,
        )
        self.retry_btn = MDRaisedButton(
            text='RETRY', md_bg_color=C['accent'],
            size_hint=(0.5, 1), font_size='18sp',
            on_release=lambda *_: self._on_retry(),
        )
        self.exit_btn = MDRaisedButton(
            text='EXIT', md_bg_color=C['red'],
            size_hint=(0.5, 1), font_size='18sp',
            on_release=lambda *_: self._on_exit(),
        )
        self.btn_row.add_widget(self.retry_btn)
        self.btn_row.add_widget(self.exit_btn)
        col.add_widget(self.btn_row)

        self.add_widget(col)

    def _sync_badge(self, instance, _):
        self._badge_rect.pos = instance.pos
        self._badge_rect.size = instance.size

    # ----------------------------------------------------------
    # Thread-safe UI callbacks (invoked from the worker thread)
    # ----------------------------------------------------------
    def update_status(self, text, color=None):
        def _apply(_dt):
            self.status_label.text = text
            self.status_label.text_color = color or C['text2']
        Clock.schedule_once(_apply, 0)

    def _on_step(self, index, total):
        Clock.schedule_once(
            lambda _dt: setattr(self.progress, 'value', (index / total) * 100), 0)
        self.update_status(f"Step {index} of {total}")

    def _on_status(self, text):
        # Keep the step counter readable: append the action under it.
        self.update_status(text)

    def _on_done(self, success, error):
        if success:
            Clock.schedule_once(
                lambda _dt: setattr(self.progress, 'value', 100), 0)
            self.update_status('Camera ready!', C['green'])
            # Brief pause so the operator sees the success state.
            Clock.schedule_once(lambda _dt: self._app.on_camera_init_done(), 0.8)
        else:
            Clock.schedule_once(lambda _dt: self._show_error(error), 0)

    # ----------------------------------------------------------
    # Error state
    # ----------------------------------------------------------
    def _show_error(self, message):
        self.spinner.active = False
        self.spinner.opacity = 0
        self.status_label.text = message
        self.status_label.text_color = C['red']
        self.btn_row.opacity = 1
        self.btn_row.disabled = False

    def _hide_error(self):
        self.spinner.opacity = 1
        self.spinner.active = True
        self.btn_row.opacity = 0
        self.btn_row.disabled = True
        self.status_label.text_color = C['text2']

    def _on_retry(self):
        self._hide_error()
        self.progress.value = 0
        self.update_status('Retrying camera initialization…')
        self._run_sequence()

    def _on_exit(self):
        app = MDApp.get_running_app()
        if app:
            app.stop()

    # ----------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------
    def start(self, dt):
        """Called by the App after the first frame — kicks off the sequence."""
        self._run_sequence()

    def _run_sequence(self):
        self._initializer = CameraInitializer(
            status_cb=self._on_status,
            step_cb=self._on_step,
            done_cb=self._on_done,
        )
        self._initializer.run_async()
