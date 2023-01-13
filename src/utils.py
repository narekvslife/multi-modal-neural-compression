import matplotlib.pyplot as plt


def show_images(images: list):
    fig, axs = plt.subplots(1, len(images))

    for i in range(len(images)):
        axs[i].imshow(images[i])

    plt.show()
