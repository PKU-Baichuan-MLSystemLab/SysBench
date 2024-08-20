import colorsys
import random
import argparse

def generate_n_colors(n, 
                      hue_offset_ratio=0.2,
                      saturation_mean=0.6,
                      saturation_bias=0.1,
                      brightness_mean=0.75,
                      brightness_bias=0.15,
                      seed=None,
                      shuffle='none'):
    """Generate n colors with random hue, saturation, and brightness."""
    
    assert shuffle in ['none', 'interleave', 'random', 'max_adjacent'], \
        'Invalid value for shuffle'
    
    if seed is not None:
        random.seed(seed)
    
    # random starting hue in [0, 1/n)
    start_hue = random.random() / n
    
    colors = []
    for i in range(n):
        hue = (start_hue + i / n + (2*random.random() - 1) * hue_offset_ratio / n) % 1
        saturation = max(0, min(1, saturation_mean + (2*random.random() - 1) * saturation_bias))
        brightness = max(0, min(1, brightness_mean + (2*random.random() - 1) * brightness_bias))
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        rgb_scaled = tuple(int(x * 255) for x in rgb)
        hex_color = '#' + ''.join(f'{int(x):02x}' for x in rgb_scaled)
        colors.append(hex_color)
    
    if shuffle == 'interleave':
        colors = colors[::2] + colors[1::2]
    elif shuffle == 'random':
        random.shuffle(colors)
    elif shuffle == 'max_adjacent':
        half_n = n // 2
        colors = [colors[i // 2] if i % 2 == 0 else colors[half_n + i // 2] for i in range(n)]
    
    return colors

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-n', type=int, help='Number of colors to generate', default=9)
    args = arg_parser.parse_args()
    
    n = args.n
    colors = generate_n_colors(n)
    print(colors)
    
    import matplotlib.pyplot as plt
    
    # Plot the colors in patches
    fig, ax = plt.subplots(1, 1, figsize=(n / 4, 1), dpi=300, tight_layout=True)
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i / n, 0), 0.95/n, 0.95, color=color))
    ax.axis('off')
    plt.show()