"""Shared CT volume rendering transfer functions for pvbatch scripts."""


def apply_ct_transfer_function(ct_lut, ct_pwf, preset="bone", ct_opacity=1.0):
    """Configure CT color and opacity transfer functions.

    Args:
        ct_lut: ParaView GetColorTransferFunction("HU")
        ct_pwf: ParaView GetOpacityTransferFunction("HU")
        preset: "bone" or "cardiac"
        ct_opacity: global opacity multiplier (default 1.0)
    """
    op = ct_opacity

    if preset == "cardiac":
        ct_lut.RGBPoints = [
            -1028.0,  0.0,   0.0,   0.0,     # air: black
            0.0,      0.0,   0.0,   0.0,     # lumen interior: black
            50.0,     0.55,  0.35,  0.30,    # myocardium/wall: warm brown-red
            120.0,    0.65,  0.45,  0.40,    # dense soft tissue: lighter warm
            200.0,    0.3,   0.3,   0.3,     # transition to grayscale
            400.0,    0.6,   0.6,   0.6,     # calcification start: mid gray
            700.0,    0.85,  0.85,  0.85,    # dense calcification: light gray
            1200.0,   1.0,   1.0,   1.0,     # cortical bone: white
            3073.0,   1.0,   1.0,   1.0,     # max HU: white
        ]
        # Opacity bump at 40-120 HU (wall/myocardium), transparent at 200-400 HU (lumen)
        ct_pwf.Points = [
            -1028.0,  0.0,              0.5, 0.0,   # air: transparent
            -50.0,    0.0,              0.5, 0.0,   # fat: transparent
            40.0,     0.0,              0.5, 0.0,   # below myocardium: transparent
            70.0,     0.02 * op,        0.5, 0.0,   # myocardium/aortic wall
            120.0,    0.012 * op,       0.5, 0.0,   # trailing edge of wall
            200.0,    0.0,              0.5, 0.0,   # contrast blood: transparent
            400.0,    0.0,              0.5, 0.0,   # contrast blood: still transparent
            500.0,    0.003 * op,       0.5, 0.0,   # calcification hint
            700.0,    0.01 * op,        0.5, 0.0,   # calcification
            1000.0,   0.035 * op,       0.5, 0.0,   # cortical bone
            1500.0,   0.07 * op,        0.5, 0.0,   # dense bone
            3073.0,   0.10 * op,        0.5, 0.0,   # max density
        ]
    else:
        # "bone" — skeleton and calcification only (default)
        ct_lut.RGBPoints = [
            -1028.0,  0.0,  0.0,  0.0,      # air: black
            0.0,      0.0,  0.0,  0.0,      # lumen: black
            200.0,    0.3,  0.3,  0.3,      # soft tissue: dark gray
            400.0,    0.6,  0.6,  0.6,      # calcification start: mid gray
            700.0,    0.85, 0.85, 0.85,     # dense calcification: light gray
            1200.0,   1.0,  1.0,  1.0,      # cortical bone: white
            3073.0,   1.0,  1.0,  1.0,      # max HU: white
        ]
        ct_pwf.Points = [
            -1028.0,  0.0,              0.5, 0.0,   # air: transparent
            0.0,      0.0,              0.5, 0.0,   # lumen: transparent
            300.0,    0.0,              0.5, 0.0,   # soft tissue: transparent
            500.0,    0.003 * op,       0.5, 0.0,   # calcification hint
            700.0,    0.01 * op,        0.5, 0.0,   # calcification
            1000.0,   0.035 * op,       0.5, 0.0,   # cortical bone
            1500.0,   0.07 * op,        0.5, 0.0,   # dense bone
            3073.0,   0.10 * op,        0.5, 0.0,   # max density
        ]

    ct_lut.ColorSpace = "RGB"
    ct_lut.AutomaticRescaleRangeMode = "Never"
    ct_lut.RescaleTransferFunction(-1028.0, 3073.0)
