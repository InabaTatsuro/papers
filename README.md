# papers

## MIR
### Music Generation
MELONS: generating melody with long-term structure using transformers and structure graph [[ICASSP22](https://arxiv.org/abs/2110.05020)][[#22](https://github.com/InabaTatsuro/papers/issues/18)]
- factor melody generation into 2 sub-problems: structure generation and structure conditional melody generation
- these sub-problems are solved by the linear transformer

Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions [[ACM20](https://arxiv.org/abs/2002.00212)][[#20](https://github.com/InabaTatsuro/papers/issues/20)]
- propose REMI, new event representation of beat-based music
- use Auto Transcription to get enough training data

A Survey of AI Music Generation Tools and Models [[Zhu+, 23/8](https://arxiv.org/abs/2308.12982)][[#12](https://github.com/InabaTatsuro/papers/issues/12)]

Museformer [[Yu+ NeurIPS22](https://arxiv.org/abs/2210.10349)][[#11](https://github.com/InabaTatsuro/papers/issues/11)]
- Use fine- and coarse-grained attention for music generation
- fine-grained attention captures the tokens in the most relevant measure (the previous 1,2,4,8...)
- coarse-grained attention captures the summarization of the other measure, which reduces the computational cost

Music Transformer [[Huang+, ICLR19](https://arxiv.org/abs/1809.04281)][[#10](https://github.com/InabaTatsuro/papers/issues/10)]
- Generate symbolic music by transformers with relative position-based attention
- reduce the memory requirements in relative position-based attention by "skewing"

Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance [[Jeong+, ICML19](https://proceedings.mlr.press/v97/jeong19a.html)][[[#6](https://github.com/InabaTatsuro/papers/issues/6)]]
- Use GNN and LSTM with Hierarchical Attention Network to generate expressive piano performance
- GNN captures the nodes information in a measure and LSTM w/HAN captures the measures information
- Let the node have nodes information in other measures by updating iteratively

- Graph-based Polyphonic Multitrack Music Generation [[Cosenza+, 23/7](https://arxiv.org/abs/2307.14928)][[#1](https://github.com/InabaTatsuro/papers/issues/1)]
- Use graph to represent multitrack music score
- Train GCN and VAE to generate graph (music)
- Not good performance



### Read Later
Modeling temporal tonal relations in polyphonic music through deep networks with a novel image-based representation [[Chuan+, AAAI18](https://ojs.aaai.org/index.php/AAAI/article/view/11880)]
- mugic generation, convolution

Counterpoint by Convolution [[Huang+, ISMIR17](https://arxiv.org/abs/1903.07227)]
- music generation, convolution, inpainting, Gibbs sampling

Musicaiz: A python library for symbolic music generation, analysis and visualization [[Olivan+, 23](https://carlosholivan.github.io/musicaiz/)]
- Python の Symbolic Music Generation 用ライブラリ

Cadence Detection in Symbolic Classical Music using Graph Neural Networks [[Karystinaios+, ISMIR22](https://arxiv.org/abs/2208.14819)]

VirtuosoTune: Hierarchical Melody Language Model [[Jeong, 23](http://ieiespc.org/AURIC_OPEN_temp/RDOC/ieie03/ieietspc_202308_006.pdf)]

Noise2Music: Text-conditioned Music Generation with Diffusion Models [[[Huang+, 23/2](https://arxiv.org/abs/2302.03917)]]

Transformer vae: A hierarchical model for structure-aware and interpretable music representation learning [[Jiang+, ICASSP20](https://ieeexplore.ieee.org/document/9054554)]

- Music VAE
- NSynth
- MuseNet
- SDMuse





## NLP
### Architecture
Hierarchical Attention Networks for Document Classification [[Yang+, 16](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)][[#8](https://github.com/InabaTatsuro/papers/issues/8)]
- HAN can capture the insights of the hierarchical structure (words form sentences, sentences form a document) and the difference in importance of each word and sentence

### Sentence Embedding
WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings [[Gao+, ACL23](https://arxiv.org/abs/2104.08821)][[#5](https://github.com/InabaTatsuro/papers/issues/5)]
- SimCSE + whitening
- whitening means the transformation of the data to have a mean of zero and a covariance matrix of the identity matrix

### Read Later
Locating and Editing Factual Associations in GPT [[Meng+, NeurIPS22](https://openreview.net/forum?id=-h6WAS6eE4)]

Mass-Editing Memory in a Transformer [[Meng+, ICLR23](https://openreview.net/forum?id=MkbcAHIYgyS)]
- the research after "Locating and Editing Factual Associations in GPT"

## ML
### Attention, graph
Do Transformers Really Perform Bad for Graph Representation? [[NeurIPS21](https://arxiv.org/abs/2106.05234)][[#21](https://github.com/InabaTatsuro/papers/issues/21)]
- propose Graphormer, which is based on the standard Transformer and can utilize the structural information of a graph
- similar to [#18](https://github.com/InabaTatsuro/papers/issues/18)

GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs [[Zhang+, 18](https://arxiv.org/abs/1803.07294)] [[#19](https://github.com/InabaTatsuro/papers/issues/19)]
- use a convolutional sub-network to control each attention head's weight

Global Self-Attention as a Replacement for Graph Convolution [[KDD22](https://arxiv.org/abs/2108.03348)][[#18](https://github.com/InabaTatsuro/papers/issues/18)]
- add edge information to the self-attention calculation

Graph Transformer Networks [[Yun+, NeurIPS19](https://arxiv.org/abs/1911.06455)][[#17](https://github.com/InabaTatsuro/papers/issues/17)]
- generate new meta-path which represents multi-hop relations and new-graph structure by multiplication of different adjacency matrix
- can be applied to heterogeneous graph

Graph Transformer [[Li+, 18](https://openreview.net/forum?id=HJei-2RcK7)][[#16](https://github.com/InabaTatsuro/papers/issues/16)]
- evolve target graph by recurrently applying source-attention from source graph and self-attention from target graph
- they can have different structure

Attention Guided Graph Convolutional Networks for Relation Extraction [[Guo+, ACL19](https://aclanthology.org/P19-1024/)][[#15](https://github.com/InabaTatsuro/papers/issues/15)]
- transform the original graph to a fully connected edge-weighted graph by self-attention


Multi-hop Attention Graph Neural Networks [[Wang+, IJCAI21](https://www.ijcai.org/proceedings/2021/425)][[#14](https://github.com/InabaTatsuro/papers/issues/14)]
- compute attention weights on edges, then compute self-attention weight between disconnected nodes
- can capture the long-range dependencies


On the Global Self-attention Mechanism for Graph Convolutional Networks[[Wang+, IEEE20](https://arxiv.org/abs/2010.10711)][[#13](https://github.com/InabaTatsuro/papers/issues/13)]
- Apply Global self-attention (GSA) to GCNs
- GSA allows GCNs to capture feature-based vertex relations regardless of edge connections


Self-Attention with Relative Position Representations [[Shaw, NAACL18](https://arxiv.org/abs/1803.02155)][[#9](https://github.com/InabaTatsuro/papers/issues/9)]
- extend self-attention to consider the pairwise relationships between each elements
- Add the trainable relative position representations, and add them to key and query vectors


Graph Attention Networks [[Velickovic+, ICLR18](https://arxiv.org/abs/1710.10903)][[#4](https://github.com/InabaTatsuro/papers/issues/4)]
- train weight matrix which represents the relation between nodes
- it can be seen as self attention with artificially created mask

## CV
High-Resolution Image Synthesis with Latent Diffusion Models [[Rombach+, CVPR22](https://arxiv.org/abs/2112.10752)]
- LDM performs semactic compression and AE + GAN performs perceptual compression
- the original paper of Stable Diffusion

### Read Later

PeerNets: Exploiting Peer Wisdom Against Adversarial Attacks [[Svoboda+, 18](https://arxiv.org/abs/1806.00088)]