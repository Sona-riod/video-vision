# modules/theme.py
"""
Shared colour palette for the Palletization HMI.
Import this in any module that needs colours — no circular imports.

Usage:
    from modules.theme import C
"""

C = {
    # Backgrounds
    'bg':        (0.039, 0.047, 0.063, 1),   # #0a0c10  deepest bg
    'panel':     (0.071, 0.082, 0.106, 1),   # #12151b  header/footer panels
    'card':      (0.098, 0.114, 0.149, 1),   # #191d26  card surfaces
    'card2':     (0.122, 0.141, 0.184, 1),   # #1f242f  slightly lighter card
    'border':    (1, 1, 1, 0.09),            # subtle white border

    # Accents
    'accent':    (0.118, 0.565, 0.996, 1),   # #1e90fe  vivid blue  — AUTO active
    'accent_dim':(0.071, 0.239, 0.490, 1),   # #12407d  muted blue
    'amber':     (1.000, 0.671, 0.000, 1),   # #ffab00  amber       — MANUAL active
    'amber_dim': (0.471, 0.314, 0.000, 1),   # #785000  muted amber

    # Status
    'green':     (0.157, 0.839, 0.380, 1),   # #28d661  success / online
    'orange':    (1.000, 0.459, 0.102, 1),   # #ff751a  processing / warning
    'red':       (0.961, 0.255, 0.235, 1),   # #f5413c  error / offline

    # Text hierarchy
    'text1':     (0.929, 0.945, 0.961, 1),   # #edf1f5  primary text
    'text2':     (0.553, 0.588, 0.639, 1),   # #8d96a3  secondary / labels
    'text3':     (0.255, 0.282, 0.322, 1),   # #414852  disabled / placeholder
    'transp':    (0, 0, 0, 0),
}