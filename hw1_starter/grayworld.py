import numpy as np

def grayworld(I):
    """
    Function that takes an image I and returns the
    light L and color image C using the above calculations.
    """
    L = np.zeros(3)
    C = np.zeros_like(I)

    avg = np.average(I, axis=(0,1))

    for i in range(len(avg)):
        L[i] = avg[i] / 0.5

    for i in range(len(I)):
        for j in range(len(I[0])):
            pixel_color = np.zeros(3)
            for color in range(len(I[0][0])):
                pixel_color[color] = I[i][j][color]  * 0.5 / avg[color]
            C[i][j] = pixel_color
    
    return L, C

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import PIL
    im = np.array(PIL.Image.open("./wb_sardmen-incorrect.jpg"))
    L, C = grayworld(im / 255.0)
    print("Light color:", L)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im)
    ax[0].axis('off')
    ax[1].imshow(C)
    ax[1].axis('off')
    plt.show()
