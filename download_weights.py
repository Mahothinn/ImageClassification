#!/usr/bin/env python3
"""
EfficientNetの事前学習済み重みを事前にダウンロードするスクリプト
"""

import os
import ssl
import urllib.request
import sys

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

if __name__ == "__main__":
    print("EfficientNet-B0の事前学習済み重みをダウンロード中...")
    
    # SSL証明書の問題を回避するための設定
    print("SSL証明書の検証を無効にしています（開発環境のみ）...")
    
    # SSL設定用の変数を初期化
    original_https_context = None
    original_opener = None
    original_ssl_verify = None
    
    try:
        # urllibのSSL検証を無効化
        original_https_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # urllib.requestのopenerも設定
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        # 既存のopenerを保存（Noneの可能性がある）
        original_opener = getattr(urllib.request, '_opener', None)
        urllib.request.install_opener(opener)
        
        # 環境変数も設定（torchvisionが使用する可能性がある）
        original_ssl_verify = os.environ.get('PYTHONHTTPSVERIFY', '')
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            print("ダウンロードが完了しました！")
        finally:
            # SSL設定を元に戻す
            if original_https_context is not None:
                ssl._create_default_https_context = original_https_context
            if original_opener is not None:
                urllib.request.install_opener(original_opener)
            else:
                # デフォルトのopenerを再構築
                urllib.request._opener = None
            if original_ssl_verify:
                os.environ['PYTHONHTTPSVERIFY'] = original_ssl_verify
            elif 'PYTHONHTTPSVERIFY' in os.environ:
                del os.environ['PYTHONHTTPSVERIFY']
    except Exception as e:
        print(f"ダウンロードに失敗しました: {e}")
        print("ネットワーク接続を確認してください。")
        sys.exit(1)

