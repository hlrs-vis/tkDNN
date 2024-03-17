import cv2
import numpy as np

def main():
    image_path = 'cow.png'
    
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return

    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)

    gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

    gray_image = gpu_gray.download()
    while True:
        cv2.imshow('Original Image', image)
        cv2.imshow('Grayscale Image', gray_image)
        cv2.waitKey(0)
        print("test")
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
