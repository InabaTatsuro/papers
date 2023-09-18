# articles

## 最適輸送と自然言語処理
参考リンク: [[横井さん, NLP22](https://www.youtube.com/watch?v=vlCjNbVSyOc&t=1616s)]
### メモ
- 入力: Before の重み分布, After の重み分布, コスト行列
- 重み，輸送コストは直感・ドメイン知識等を使って決めて良い
- 単語ベクトル間の距離を撮る際はL2距離よりもよりもコサイン距離の方が安定しがち
- そのままでは NN に組み込んで組み込んで損失として利用不可
  - エントロピー正則化項，シンクホーンダイバージェンス，近接勾配法（IPOT）


### 微分可能な最適輸送コストの利用例
- テキスト生成の損失として利用
  - 生成文とリファレンス間の Word Mover's Distance
- Word Mover's Distance のための単語重みと単語間距離を教師ありで更新
  - 単語間距離の更新: 元の輸送コスト行列に行列をかけることで更新
- 各単語表現としてベクトル集合を学習
  - word2vec ではなく word2vec集合
  - 多義性を考慮した表現が学習可能

## GNN
参考リンク: [GNN](https://buildersbox.corp-sansan.com/entry/2021/02/19/114000),[GCN HackMD](https://hackmd.io/@kkume/rkK3tmpHd), [GNN HackMD](https://hackmd.io/0IwDJxeITPGLyq40EfvT1g)
### 2種類のタスク
- ノード単位のタスク
  - グラフフィルタと活性化層を積み重ねて特徴抽出
- グラフ単位のタスク
  - グラフフィルタと活性化層に加えてグラフプーリングを重ねることでグラフを粗く（小さく

### グラフフィルタ
- spatial-based グラフフィルタ: ノード間の接続を明示的に利用してグラフ構造のまま特徴抽出
  - 最初の GNN フィルタ [Scarselli et al. (2005, 2008)]
    - 隣接したノードの特徴量を集約する
    - 層を積み重ねると全てのノードの特徴が似通う
  - GraphSAGE [Hamilton et al. (2017) ]
    - 隣接したノードの特徴量を集約するが，ランダムに選択
    - 重み付けには，mean aggregator, LSTM aggregator, pooling operator 
  - GAT [Velckovic et al. (2017)]
    - 隣接したノードの特徴量を集約するが，Attention 機構により重み付け
- spectral-based グラフフィルタ: スペクトル領域でフィルタリング
  - Graph Spectral Filtering
    -  グラフフーリエ変換により周波数領域に変換後に操作を行う．その後逆変換によりグラフ構造に戻す．
    -  学習パラメータ数=グラフのノード数になり，大規模グラフには適用できない
  -  Defferrard et al. (2016)
    -  係数を多項式近似で表現することで，学習パラメータ数を多項式の次数数 k に依存するように減らす（ノード数に依存しない）
    -  その後，直交基底の集合をもつ Chebyshev 多項式近似
    -  k-hop の近傍のみを考慮することに相当？
  -  GCN
    -  多項式の次元数を k = 1
    -  各ノードの最近房最近房のみを考慮することに相当？
      -  *本当にこれで表現力が足りる？*

