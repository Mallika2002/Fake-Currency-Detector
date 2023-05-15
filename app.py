from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
import pickle

app = Flask(__name__)


@app.route('/')
def do():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # Handle file upload and processing
        file1 = request.files['img1']
        file2 = request.files['img2']
        img2 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img3 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
        res3 = cv2.resize(img2, (800, 300))
        res4 = cv2.resize(img3, (800, 300))
        image2 = np.concatenate((res3, res4), axis=0)
        img_median1 = cv2.medianBlur(image2, 3)
        gray1 = cv2.cvtColor(img_median1, cv2.COLOR_BGR2GRAY)
        edges1 = cv2.Canny(gray1, 60, 180)
        th21 = cv2.adaptiveThreshold(edges1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

        cv2.imwrite('static/test_selected_image.jpg', image2)  # original images
        cv2.imwrite('static/test_noise_filtered.jpg', img_median1)  # filtered images
        cv2.imwrite('static/test_gray_scale.jpg', gray1)  # gray scale
        cv2.imwrite('static/test_edge_detected.jpg', edges1)  # edge detected
        cv2.imwrite('static/test_segmented.jpg', th21)  # edge detected

        # comparing saved image and test image of currency
        infile = open('saved', 'rb')
        th2 = pickle.load(infile)
        infile.close()
        (score, diff) = ssim(th2, th21, full=True)
        diff = (diff * 255).astype("uint8")
        cv2.imwrite('static/diff.jpg', diff)
        if score == 1:
            is_genuine = True
        else:
            is_genuine = False
        return render_template('result.html', is_genuine=is_genuine,ssim_score=score, image="diff.jpg")


if __name__ == '__main__':
    app.run()
