# upi-engine

[![Build Status](https://travis-ci.org/TukamotoRyuzo/upi-engine.svg?branch=develop)](https://travis-ci.org/TukamotoRyuzo/upi-engine)
  
upiプロトコル準拠の思考エンジンです。  

## 動作環境

- python3.6以降

## 使い方

- upi.batを、upiプロトコル準拠のGUIに登録してください。
    - upiプロトコル準拠のGUIとして、[upi-gui](https://github.com/TukamotoRyuzo/upi-gui)を使用することができます。

## 主要なファイル

- upi.py
    - pythonで書かれたupiエンジン本体です。
- upi.bat
    - upiプロトコル準拠のGUIに登録するためのバッチファイルです。

## 開発者向け

- pipenvを用いてパッケージ管理しています。`git clone`後、`pipenv install`をして仮想環境を構築してください。
