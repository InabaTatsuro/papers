# articles

## 最適輸送と自然言語処理
参考リンク: [[横井さん, NLP22](https://www.youtube.com/watch?v=vlCjNbVSyOc&t=1616s)]

## GNN
参考リンク: [GNN](https://buildersbox.corp-sansan.com/entry/2021/02/19/114000),[GCN HackMD](https://hackmd.io/@kkume/rkK3tmpHd), [GNN HackMD](https://hackmd.io/0IwDJxeITPGLyq40EfvT1g)
- 2種類のタスク
  - ノード単位のタスク
    - グラフフィルタと活性化層を積み重ねて特徴抽出
  - グラフ単位のタスク
    - グラフフィルタと活性化層に加えてグラフプーリングを重ねることでグラフを粗く（小さく）
- グラフフィルタ
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

