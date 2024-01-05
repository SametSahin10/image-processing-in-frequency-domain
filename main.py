import cv2
import numpy as np
from scipy import signal


def slice_matrix(matrix, row_start, col_start, size):
    col_end = col_start + size
    row_end = row_start + size
    return [row[col_start:col_end] for row in matrix[row_start:row_end]]


def print_matrix(matrix):
    for row in matrix:
        for element in row:
            print(element, end=' ')
    print("\n")


image = cv2.imread('assets/house_pixel_art.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Shape of the image:", gray_image.shape)
image_height, image_width = gray_image.shape[:2]

kernel = np.array(
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]]
)

output_image = np.zeros_like(gray_image)

for i in range(1, image_height - 2):
    for j in range(1, image_width - 2):
        # print("i:", i)
        # print("j:", j)

        sliced_matrix = slice_matrix(
            matrix=gray_image, row_start=i, col_start=j, size=3
        )

        sliced_matrix_as_np_array = np.array(sliced_matrix)

        # TODO: Try to implement this by yourself.
        result_of_convolution = signal.convolve2d(
            kernel, sliced_matrix_as_np_array, mode="valid"
        )

        output_image[i + 1][j + 1] = result_of_convolution[0, 0]

# Beyhan - FFT uygulayarak frekans uzayina gecip, konvolusyonu uygulayip tekrar zaman uzayina geri donecegiz.
# Samet signal.convolve2d yerine kendi fonksiyonumuzu yazmaliyiz.
# Cumartesi 23 gibi toplanti yapacagiz.

cv2.imshow("After", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
