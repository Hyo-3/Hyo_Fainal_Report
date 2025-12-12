# Hyo_Fainal_Report
火曜3限期末レポート

#　Python3xを使用

#　仮想環境pytorchを作成有効化し、その中にPytorch lightningをインストール
  $ uv venv pytorch
  $ pytorch\Scripts\activate
  $ uv pip install lightning

#　MNISTの元データをGitHubからダウンロード
  # deep-learning-from-scratch/dataset/mnist.py at master · oreilly-japan/deep-learning-from-scratch
  # 上記URLからmnist.pyをダウンロード

#　requirements.txtというディレクトリを作成
 # requirements.txt内に以下の5項目を記述
  # torch
  # torchvision
  # lightning
  # pytest       # テストのために追加
  # mypy         # 型ヒントのチェックのために追加
  
#　必要なモジュールをインストール
  $　uv pip install -r requirements.txt　　

#　正常に作動するか確認
  $　python mnist.py


#　修正内容：mnist.py内のデータ処理関数に対し、numpy.ndarray、str、boolなどの型ヒントを厳密に追加した。これにより、関数のインターフェイスが明確化され、静的解析ツールによるチェックが可能になった。

#　以下修正点
11行目に追加　from typing import Dict, Tuple

32行目
修正前　def _download(file_name):	
修正後　def _download(file_name: str) -> None:

50行目
修正前　def _load_label(file_name):
修正後　def _load_label(file_name: str) -> np.ndarray:

71行目
修正前　def _convert_numpy():
修正後　def _convert_numpy() -> Dict[str, np.ndarray]:

96行目
修正前　def load_mnist(normalize=True, flatten=True, one_hot_label=False):
修正後　def load_mnist(normalize: bool =True, flatten: bool =True, one_hot_label: bool=False) -> Tuple[Tuple[np.ndarray, np.ndarray],Tuple[np.ndarray, np.ndarray]]:

#　pytestによるテストの導入
#　tests/test_data_loader.pyというディレクトリを作成
#　テストの内容はデータロードと正規化のオプションを正しく処理するかどうかを見る
#　pytestを実行
　$ pytest

#　補足：キャッシュファイルのmnist.pklは50MBを超えるため、.gitignoreに記述しGitの管理から外した。





