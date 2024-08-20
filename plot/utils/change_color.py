import colorsys

def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Convert RGB color back to hex."""
    return '#' + ''.join(f'{int(x):02x}' for x in rgb_color)

def adjust_saturation(colors):
    adjusted_colors = []
    for color in colors:
        # Convert hex to RGB
        rgb = hex_to_rgb(color)
        # Convert RGB to HSV
        hsv = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        # Increase saturation by 0.1, ensuring it does not exceed 1
        new_saturation = min(hsv[1] + 0.08, 1)
        new_value = min(hsv[2] - 0.05, 1)
        print(new_saturation, new_value)
        # Convert back to RGB
        new_rgb = colorsys.hsv_to_rgb(hsv[0], new_saturation, new_value)
        # Scale RGB back to 0-255 range and convert to integer
        new_rgb_scaled = tuple(int(x * 255) for x in new_rgb)
        # Convert RGB back to hex
        new_hex = rgb_to_hex(new_rgb_scaled)
        adjusted_colors.append(new_hex)
    return adjusted_colors

# Sample input
# https://colorkit.co/palettes/9-colors/
input_colors =  ["#d6e6ff","#d7f9f8","#ffffea","#fff0d4","#fbe0e0","#e5d4ef"]
adjusted_colors = adjust_saturation(input_colors)

print(adjusted_colors)