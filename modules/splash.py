# modules/splash.py
"""
SplashScreen — shown as the ROOT widget on startup (not a ModalView).

The App.build() returns this widget first. Once init is done,
call splash.dismiss_and_load(app) to swap it for SimpleKegHMI.

No ModalView race conditions — this IS the window content.
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.metrics import dp
from kivy.clock import Clock
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner

from modules.theme import C


class SplashScreen(FloatLayout):
    """
    Full-screen splash rendered as the real root widget.

    Usage (in SimpleKegApp):
        def build(self):
            self.splash = SplashScreen(app=self)
            Clock.schedule_once(self.splash.start_init, 0.1)
            return self.splash

    The splash calls app.load_hmi() when init is complete.
    """

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self._app = app
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
        self._bg_rect.pos  = self.pos
        self._bg_rect.size = self.size

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------
    def _build_ui(self):
        # Centred card column
        col = BoxLayout(
            orientation='vertical',
            spacing=dp(16),
            size_hint=(None, None),
            size=(dp(400), dp(340)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
        )

        # ── Logo badge ────────────────────────────────────────
        badge_wrap = BoxLayout(
            size_hint=(None, None),
            size=(dp(88), dp(88)),
            pos_hint={'center_x': 0.5},
        )
        with badge_wrap.canvas:
            self._badge_color = Color(*C['accent'])
            self._badge_rect  = RoundedRectangle(
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
            text='PALLETIZATION SYSTEM',
            font_style='H5', bold=True,
            theme_text_color='Custom', text_color=C['text1'],
            halign='center',
            size_hint_y=None, height=dp(36),
        ))

        # ── Sub-title ─────────────────────────────────────────
        col.add_widget(MDLabel(
            text='Keg Counting & QR Detection HMI',
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

        # ── Status label ──────────────────────────────────────
        self.status_label = MDLabel(
            text='Starting up...',
            font_style='Subtitle2',
            theme_text_color='Custom', text_color=C['text2'],
            halign='center',
            size_hint_y=None, height=dp(34),
        )
        col.add_widget(self.status_label)

        # ── Copyright ─────────────────────────────────────────
        col.add_widget(MDLabel(
            text='v2.0  ·  © 2025 Palletization Systems',
            font_style='Caption',
            theme_text_color='Custom', text_color=C['text3'],
            halign='center',
            size_hint_y=None, height=dp(22),
        ))

        self.add_widget(col)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _sync_badge(self, instance, _):
        self._badge_rect.pos  = instance.pos
        self._badge_rect.size = instance.size

    def update_status(self, text: str):
        """Thread-safe status update."""
        Clock.schedule_once(
            lambda dt: setattr(self.status_label, 'text', text), 0)

    def start_init(self, dt):
        """Called by the App after the first frame — triggers HMI init."""
        self._app.init_hmi()