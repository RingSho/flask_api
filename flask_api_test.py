#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:40:50 2019

@author: hayashi
"""

# 必要なモジュールの読み込み
from bottle import route, run
from flask import Flask, jsonify, make_response, request
import numpy as np
import os
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
#api = Flask(__name__)



# GETの実装
@route('/get', methods=['GET'])
def get():
#    return make_response(jsonify(all_list))
    return print(all_list)

@route('/hello', methods=['GET'])
def hello():
    return "hello"

@route('/upload', methods=['POST'])
def upload():
    if request.files and 'image' in request.files:
        import tensorflow as tf
        from keras import backend as K
    
        num_cores = 4
        GPU = False
        CPU = True
    
        if CPU:
            num_CPU = 4
            num_GPU = 0
    
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
        session = tf.Session(config=config)
        K.set_session(session)
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
#        data = dict(predict=str(predict), name=str(name))
#        
        #return make_response(jsonify(all_list))
        return print(name)

    return 'Picture info did not get.'

# エラーハンドリング
#@errorhandler(404)
#def not_found(error):
#    return make_response(jsonify({'error': 'Not found'}), 404)

run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
# ファイルをスクリプトとして実行した際に
# ホスト0.0.0.0, ポート3001番でサーバーを起動
#if __name__ == '__main__':
#    api.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))