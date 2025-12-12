import numpy as np
import pytest
from mnist import load_mnist 

def test_data_shape():
    """データロード後の訓練画像とラベルの形状を検証する"""
    # flatten=True がデフォルトの前提
    (x_train, y_train), _ = load_mnist(flatten=True) 
    
    # 訓練画像が (60000, 784) であるか確認
    assert x_train.shape == (60000, 784) 
    # 訓練ラベルが (60000,) であるか確認
    assert y_train.shape == (60000,)

def test_normalization_option():
    """正規化オプションが正しく適用されているか検証する"""
    # 正規化 (normalize=True) してデータを読み込む
    (x_train, _), _ = load_mnist(normalize=True)
    
    # ピクセル値の最大値が 1.0 になっているか検証
    assert np.max(x_train) <= 1.0
    assert np.min(x_train) >= 0.0

def test_one_hot_encoding_option():
    """ワンホットエンコーディングが正しく適用されているか検証する"""
    
    # ワンホットエンコーディング (one_hot_label=True) してデータを読み込む
    (_, y_train), (_, y_test) = load_mnist(one_hot_label=True)
    
    # 1. 訓練ラベルの形状を確認: (サンプル数, クラス数) となっているか？
    # MNISTは10クラスなので、(60000, 10) になっているはず
    assert y_train.shape == (60000, 10)
    
    # 2. 値の範囲を確認: 0 または 1 の値だけを持つか？
    # ワンホット化されたラベルの最大値は 1、最小値は 0 のはず
    assert np.max(y_train) == 1
    assert np.min(y_train) == 0
    
    # 3. 各行の合計値を確認: 各行（各ラベル）の合計は必ず 1 になるはず
    # T = np.zeros((X.size, 10)) の処理を検証
    assert np.all(np.sum(y_train, axis=1) == 1)