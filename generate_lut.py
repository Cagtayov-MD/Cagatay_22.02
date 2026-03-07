"""
Green Cast Fix LUT Generator
Sahne: Metal cati / kapali alan - yesil-olive renk kaymasini duzeltir
Cikti: green_cast_fix.cube (33x33x33)
Kullanim: DaVinci'de Color > LUTs > Import LUT
"""

import numpy as np

LUT_SIZE = 33

def sigmoid_curve(x, amount, center=0.5):
    """Midtone odakli egri ayari"""
    mask = np.sin(np.pi * np.clip(x, 0, 1))
    return np.clip(x + amount * mask, 0, 1)

def apply_corrections(r, g, b):
    # 1. Kontrast
    contrast = 0.12
    pivot = 0.45
    r = (r - pivot) * (1 + contrast) + pivot
    g = (g - pivot) * (1 + contrast) + pivot
    b = (b - pivot) * (1 + contrast) + pivot

    # 2. RGB Curves - midtone kaymasi
    r = sigmoid_curve(r,  0.031)   # Red  +8/255
    g = sigmoid_curve(g, -0.039)   # Green -10/255
    b = sigmoid_curve(b,  0.020)   # Blue  +5/255

    # 3. Gamma (midtone): yesili dustur, magenta/warm ekle
    luma = (r + g + b) / 3.0
    gamma_mask = np.sin(np.pi * np.clip(luma, 0, 1))
    r = r + 0.015 * gamma_mask
    g = g - 0.038 * gamma_mask
    b = b + 0.012 * gamma_mask

    # 4. Gain (highlight)
    r = r * 1.018
    b = b * 1.010

    # 5. Lift (shadow) - hafif magenta
    shadow_mask = np.clip(1.0 - luma * 2.5, 0, 1)
    r = r + 0.008 * shadow_mask
    b = b + 0.006 * shadow_mask

    # 6. Yesil-sari doygunluk azalt (arka plan metal cati icin)
    gray = (r + g + b) / 3.0
    # Yesil baskın pikseller: g en buyuk VE g-r ve g-b farki yeterli
    green_dom = (g > r) & (g > b) & ((g - r) > 0.05) & ((g - b) > 0.02)
    blend = np.where(green_dom, 0.38, 0.0)
    r = r + (gray - r) * blend
    g = g + (gray - g) * blend
    b = b + (gray - b) * blend

    return (
        np.clip(r, 0, 1),
        np.clip(g, 0, 1),
        np.clip(b, 0, 1)
    )

def generate_lut():
    idx = np.linspace(0, 1, LUT_SIZE)
    r_vals, g_vals, b_vals = np.meshgrid(idx, idx, idx, indexing='ij')

    ro, go, bo = apply_corrections(r_vals, g_vals, b_vals)

    output_path = "green_cast_fix.cube"
    with open(output_path, 'w') as f:
        f.write('TITLE "Green Cast Fix - Metal Ceiling Scene"\n')
        f.write(f'LUT_3D_SIZE {LUT_SIZE}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')

        for b_i in range(LUT_SIZE):
            for g_i in range(LUT_SIZE):
                for r_i in range(LUT_SIZE):
                    f.write(f'{ro[r_i, g_i, b_i]:.6f} {go[r_i, g_i, b_i]:.6f} {bo[r_i, g_i, b_i]:.6f}\n')

    print(f"LUT olusturuldu: {output_path}")
    print(f"Boyut: {LUT_SIZE}x{LUT_SIZE}x{LUT_SIZE}")
    print()
    print("DaVinci Resolve'da kullanim:")
    print("  Color sayfasi > LUTs paneli > sag tik > Refresh")
    print("  Veya: Clip uzerine sur-birak")
    print("  Veya: Color > Apply LUT")

if __name__ == "__main__":
    generate_lut()
