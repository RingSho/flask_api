#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:40:50 2019

@author: hayashi
"""

# 必要なモジュールの読み込み
from flask import Flask, jsonify, make_response, request
import numpy as np
from PIL import Image
import models
import io

HEIGHT = 144
WIDTH = 256
n_class = 12

fish_list = ["akagarei",
             "hatahata",
             "etegarei",
             "kawahagi",
             "nodoguro",
             "aji"]

ebikani_list = ["kurozakoebi",
                "matsubagani"]

ika_list = ["kouika",
            "surumeika"]

kai_list = ["baigai",
            "sazae"]

all_list = fish_list + ebikani_list + ika_list + kai_list


# Flaskクラスのインスタンスを作成
# __name__は現在のファイルのモジュール名
api = Flask(__name__)


# GETの実装
@api.route('/get', methods=['GET'])
def get():
    return make_response(jsonify(all_list))

@api.route('/upload', methods=['POST'])
def upload():
    if request.files and 'image' in request.files:
        img = request.files['image'].read()
        img = Image.open(io.BytesIO(img))
        img = img.resize((256,144))
        img_array = np.array(img)
        
        X_test = []
        X_test.append(img_array)
        X_test = np.asarray(X_test) / 255.
        model = models.model_vgg(n_class, HEIGHT, WIDTH)
        model.load_weights("all_weights12_25.h5")
        Yp = model.predict(X_test)
        predict = np.argmax(Yp)
        name = all_list[predict]
        data = dict(predict=str(predict), name=str(name))
        
        #return make_response(jsonify(all_list))
        return jsonify(data)

    return 'Picture info did not get.'

# エラーハンドリング
@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# ファイルをスクリプトとして実行した際に
# ホスト0.0.0.0, ポート3001番でサーバーを起動
if __name__ == '__main__':
    api.debug = True
    api.run()