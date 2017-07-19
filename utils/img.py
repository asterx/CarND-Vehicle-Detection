# coding=utf-8
import matplotlib.pyplot as plt


def show_hog_imgs(img, hog_image, title, output):
    fig = plt.figure()

    plt.subplot(1, len(hog_image) + 1, 1)
    plt.imshow(img)
    plt.title(title)

    for idx, hog_img in enumerate(hog_image):
        plt.subplot(1, len(hog_image) + 1, idx+2)
        plt.imshow(hog_img, cmap='gray')
        plt.title('HOG')

    fig.savefig(output, bbox_inches='tight')
