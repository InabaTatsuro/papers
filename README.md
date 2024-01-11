# MIR
## Music Generation/Understanding
### Symbolic Music Generation
Compose & Embellish: Well-Structured Piano Performance Generation via A Two-Stage Approach [ICASSP23, [#54](https://github.com/InabaTatsuro/papers/issues/54)]
- a transformer model generates a lead sheet and then another transformer model generates all sequences
- a lead sheet consists of melody, chords, and structure information(like ABA*B*) of each bar

Melody Infilling with User-Provided Structural Context [ISMIR22, [#52](https://github.com/InabaTatsuro/papers/issues/52)]
- encoder-decoder transformer model for structure-aware conditioning infilling
- use the bar-count-down technique and order embeddings to control the length and attention-selecting module that allows the model to access multiple structural contexts while infilling

Variable-Length Music Score Infilling via XLNet and Musically Specialized Positional Encoding [ISMIR21, [#51](https://github.com/InabaTatsuro/papers/issues/51)]
- can infill a variable number of notes (up to 128) for different time spans
- use CP, XLNet, relative bar encoding, and look-ahead onset prediction

Anticipation-RNN: Enforcing unary constraints in sequence generation, with application to interactive music generation [NCAA18, https://github.com/InabaTatsuro/papers/issues/50]
- anticipation-RNN can enforce user-defined unary constraints
- one RNN is used to encode the constraints and the other one is used to generate the sequence with the constraint information
- the constraints are fed reversely into RNNs, so the decoder (the other one) can use the future constraint information

The Piano Inpainting Application [Sony21, [#49](https://github.com/InabaTatsuro/papers/issues/49)]
- Structured MIDI Encoding is proposed and used to train Linear Transformer for infilling(inpainting)
- use time-shift tokens instead of note-on/off or duration tokens

Composer's Assistant: An Interactive Transformer for Multi-Track MIDI Infilling [ISMIR23, [#44](https://github.com/InabaTatsuro/papers/issues/44)]
- Train T5-like model to infill multi-track MIDI whose arbitrary track-measures have been deleted
- The model can be used in the REAPER digital audio workstation(DAW)

Infilling Piano Performances [[NeurIPS workshop18, [#40](https://github.com/InabaTatsuro/papers/issues/40)]
- infill deleted section of MIDI by using the transformer
- give {left context + special token + right context} to the model and generate the blanks

FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control [ICLR23, [#39](https://github.com/InabaTatsuro/papers/issues/39)]
- FIne-grained music Generation via Attention-based, RObust control (FIGARO) by applying description-to-sequence modelling
- combine learned high-level features with domain knowledge which acts as a strong inductive bias

Exploring the Efficacy of Pre-trained Checkpoints in Text-to-Music Generation Task [AAAI23 workshop, [#38](https://github.com/InabaTatsuro/papers/issues/38)]
- given text, generate symbolic music using pre-trained language models like BERT, GPT-2, and BART
- the model that initializes the parameters by BART outperformed the model that initializes parameters randomly

Mubert [[github](https://github.com/MubertAI/Mubert-Text-to-Music)]
- generate music from a free text prompt
- this is not the actual generation, but just combines the pre-composed music according to the rule

Vector Quantized Contrastive Predictive Coding for Template-based Music Generation [20, [#36](https://github.com/InabaTatsuro/papers/issues/36)]
- given a template sequence, generate novel sequences sharing perceptible similarities with the original template
- encode and quantize the template sequence followed by decoding to generate variations

MMM : Exploring Conditional Multi-Track Music Generation [ArXiv20, [#34](https://github.com/InabaTatsuro/papers/issues/34)]
- Multitrack inpainting, using event-level token representation and transformer
- replace the sequences of the bar representation which we want to predict into the token, and add the token to the last
- no quantitative results, only method and demo

Multitrack Music Transformer [ICASSP23][[#29](https://github.com/InabaTatsuro/papers/issues/29)]
- propose MMT, multitrack music representation for symbolic music generation can reduce memory usage
- multitrack symbolic music generation by using MMT
  - Continuation, scratch generation, instrument informed generation

Compound word transformer: Learning to compose full-song music over dynamic directed hypergraphs [AAAI21][[#27](https://github.com/InabaTatsuro/papers/issues/27)]
- group consecutive and related tokens into compound words to capture the co-occurrence relationship
- 5-10 times faster at training with comparable quality
- CP transformer can be seen as the hyperedge prediction

PopMAG: Pop Music Accompaniment Generation [[ACM20](https://arxiv.org/abs/2008.07703)][[#26](https://github.com/InabaTatsuro/papers/issues/26)]
- Multitrack-MIDI representation (MuMIDI) enables simultaneous multi-track generation in a single sequence
- model multiple note attributes of a musical note in one step and use the architecture of transformerXL to capture the long-term dependencies

Controllable deep melody generation via hierarchical music structure representation [[ISMIR21](https://arxiv.org/abs/2109.00663)][[#24](https://github.com/InabaTatsuro/papers/issues/24)]
- music framework generates rhythm and basic melody using two separate transformer-based networks
- then, generate the melody conditioned on the basic melody, rhythm, and chords

A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music [[ICML18](https://arxiv.org/abs/1803.05428)][[#23](https://github.com/InabaTatsuro/papers/issues/23)]
- hierarchical decoder which first outputs embeddings for sub-sequences and then uses these embeddings to generate each subsequence
- propose MusicVAE which uses the hierarchical latent vector model

PopMNet: Generating structured pop music melodies using neural networks [[Wu+, 19](https://www.sciencedirect.com/science/article/abs/pii/S000437022030062X)]
- CNN generates melody structure which is defined by pairwise relations, specifically the sequence between all bars in a melody
- RNN generates melodies conditioned on the structure and chord progression

MELONS: generating melody with long-term structure using transformers and structure graph [[ICASSP22](https://arxiv.org/abs/2110.05020)][[#22](https://github.com/InabaTatsuro/papers/issues/18)]
- factor melody generation into 2 sub-problems: structure generation and structure conditional melody generation
- these sub-problems are solved by the linear transformer

Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions [[ACM20](https://arxiv.org/abs/2002.00212)][[#20](https://github.com/InabaTatsuro/papers/issues/20)]
- propose REMI, a new event representation of beat-based music
- use Auto Transcription to get enough training data

Museformer [[Yu+ NeurIPS22](https://arxiv.org/abs/2210.10349)][[#11](https://github.com/InabaTatsuro/papers/issues/11)]
- Use fine- and coarse-grained attention for music generation
- fine-grained attention captures the tokens in the most relevant measure (the previous 1,2,4,8...)
- coarse-grained attention captures the summarization of the other measure, which reduces the computational cost

Music Transformer [[Huang+, ICLR19](https://arxiv.org/abs/1809.04281)][[#10](https://github.com/InabaTatsuro/papers/issues/10)]
- Generate symbolic music by transformers with relative position-based attention
- reduce the memory requirements in relative position-based attention by "skewing"

Graph-based Polyphonic Multitrack Music Generation [[Cosenza+, 23/7](https://arxiv.org/abs/2307.14928)][[#1](https://github.com/InabaTatsuro/papers/issues/1)]
- Use a graph to represent the multitrack music score
- Train GCN and VAE to generate graph (music)
- Not good performance

### Symbolic Music Understanding
Impact of time and note duration tokenizations on deep learning symbolic music modeling [ISMIR23, [#48](https://github.com/InabaTatsuro/papers/issues/48)]
- analyze the common tokenization methods and especially experiment with time and note duration representations
- demonstrate that explicit information leads to better results depending on the tasks

Multimodal Multifaceted Music Emotion Recognition Based on Self-Attentive Fusion of Psychology-Inspired Symbolic and Acoustic Features[APSIPA23, [#43](https://github.com/InabaTatsuro/papers/issues/43)]
- Multimodal multifaceted MER method that uses features from MIDI and audio data based on musical psychology.
- Self-attention mechanism can learn the complicated relationships between different features and fuse the

PiRhDy: Learning Pitch-, Rhythm-, and Dynamics-aware Embeddings for Symbolic Music[ACM20][[#30](https://github.com/InabaTatsuro/papers/issues/30)]
- generate music note embeddings
- (1) token modeling: separately represents pitch, rhythm, and dynamics and integrates them into a single token embedding
- (2) context modeling: use melodic and harmonic embedding to train the token embedding

MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training [[ACL finding21](https://aclanthology.org/2021.findings-acl.70/)][[#28](https://github.com/InabaTatsuro/papers/issues/28)]
- Pre-train BERT with 1.5M MIDI files which are private and created by Microsoft Research Asia
- use OctupleMIDI encoding and bar-level masking strategy to enhance symbolic music data

Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance [[Jeong+, ICML19](https://proceedings.mlr.press/v97/jeong19a.html)][[[#6](https://github.com/InabaTatsuro/papers/issues/6)]]
- Use GNN and LSTM with Hierarchical Attention Network to generate expressive piano performance
- GNN captures the node information in a measure and LSTM w/HAN captures the measure information
- Let the node have node information in other measures by updating iteratively

### Audio Music Generation
ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models [AACLdemo23, [#37](https://github.com/InabaTatsuro/papers/issues/37)]
- use the diffusion model to generate music conditioned by free-form text
- outperform the previous models in terms of text-music relevance and music quality, as judged by human

### Survey
Data Collection in Music Generation Training Sets: A Critical Analysis [ISMIR23, [#45](https://github.com/InabaTatsuro/papers/issues/45)]
- Analysis of all datasets used to train Automatic Music Generation (AMG) models presented at the last 10 editions of ISMIR
- Discussed ethics and suggested the way to collect or use the dataset for AMG training

A Survey on Deep Learning for Symbolic Music Generation [[ACM23/8](https://dl.acm.org/doi/abs/10.1145/3597493)][[#25](https://github.com/InabaTatsuro/papers/issues/25)]

A Survey of AI Music Generation Tools and Models [[Zhu+, 23/8](https://arxiv.org/abs/2308.12982)][[#12](https://github.com/InabaTatsuro/papers/issues/12)]


### Read Later
Modeling temporal tonal relations in polyphonic music through deep networks with a novel image-based representation [[Chuan+, AAAI18](https://ojs.aaai.org/index.php/AAAI/article/view/11880)]
- music generation, convolution

Counterpoint by Convolution [[Huang+, ISMIR17](https://arxiv.org/abs/1903.07227)]
- music generation, convolution, inpainting, Gibbs sampling

Musicaiz: A Python library for symbolic music generation, analysis, and visualization [[Olivan+, 23](https://carlosholivan.github.io/musicaiz/)]
- Python の Symbolic Music Generation 用ライブラリ

Cadence Detection in Symbolic Classical Music using Graph Neural Networks [[Karystinaios+, ISMIR22](https://arxiv.org/abs/2208.14819)]

VirtuosoTune: Hierarchical Melody Language Model [[Jeong, 23](http://ieiespc.org/AURIC_OPEN_temp/RDOC/ieie03/ieietspc_202308_006.pdf)]

Noise2Music: Text-conditioned Music Generation with Diffusion Models [[[Huang+, 23/2](https://arxiv.org/abs/2302.03917)]]

Transformer vae: A hierarchical model for structure-aware and interpretable music representation learning [[Jiang+, ICASSP20](https://ieeexplore.ieee.org/document/9054554)]

- Music VAE
- NSynth
- MuseNet
- SDMuse

## ASR
Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding [ICASSP22, [#53](https://github.com/InabaTatsuro/papers/issues/53)]
- improve the ASR robustness by contrastive pretraining of RoBERTa
- fine-tuning with self-distillation to reduce the label noises due to ASR errors

Transformer ASR with Contextual Block Processing [ASRU19, [#47](https://github.com/InabaTatsuro/papers/issues/47)]
- to capture global information (long dependency), introduce a context-aware inheritance mechanism into the transformer encoder
- also introduce a noble mask technique to implement the above mechanism

## Beat Tracking
BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer [ICASSP24, [#46](https://github.com/InabaTatsuro/papers/issues/46)]
- Beat tracking streaming transformer (BEAST) is for online beat tracking and has a transformer-encoder with relative positional encoding
- improvement of 5% in beat and 13% in downbeat over the SOTA model

# NLP
## Text Generation
Locally Typical Sampling [TACL22, [#42](https://github.com/InabaTatsuro/papers/issues/42)]
- in each time t, create the local typical set, consisting of the words that have a probability close to the entropy of the predicted distribution
- random sample from the local typical set

Contrastive Decoding: Open-ended Text Generation as Optimization[ACl23, [#41](https://github.com/InabaTatsuro/papers/issues/41)]
- decoding with maximum probability often results in short and repetitive text and sampling can often produce incoherent text
- Contrastive Decoding (CD) is a reliable approach that optimizes a contrastive objective subject to a plausibility constraint

InCoder: A Generative Model for Code Infilling and Synthesis [ICLR23][[#32](https://github.com/InabaTatsuro/papers/issues/32)]
- InCoder can infill the program via left-to-right generation
- train by maximizing logP([left; <mask>; right; <mask>; span; <EOM>]) and inference by sampling tokens autoregressively from the distributions P(・| [left; <mask>; right; <mask>])

## Text Classification
Hierarchical Attention Networks for Document Classification [[Yang+, 16](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)][[#8](https://github.com/InabaTatsuro/papers/issues/8)]
- HAN can capture the insights of the hierarchical structure (words form sentences, sentences form a document) and the difference in importance of each word and sentence

## Sentence Embedding
WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings [[Gao+, ACL23](https://arxiv.org/abs/2104.08821)][[#5](https://github.com/InabaTatsuro/papers/issues/5)]
- SimCSE + whitening
- whitening means the transformation of the data to have a mean of zero and a covariance matrix of the identity matrix

## Positional embedding
On Positional embeddings in BERT [ICLR21][[#31](https://github.com/InabaTatsuro/papers/issues/31)]
- analyze Positional Embeddings (PEs) based on 3 properties, translational invariance, monotonicity, and symmetry
- breaking translational invariance and monotonicity degrades downstream task performance while breaking symmetry improves downstream task performance
- fully learnable absolute position embedding generally improves performance on the classification task, while relative position embedding improves performance on the span prediction task

## Read Later
Locating and Editing Factual Associations in GPT [[Meng+, NeurIPS22](https://openreview.net/forum?id=-h6WAS6eE4)]

Mass-Editing Memory in a Transformer [[Meng+, ICLR23](https://openreview.net/forum?id=MkbcAHIYgyS)]
- the research after "Locating and Editing Factual Associations in GPT"

# ML
## Attention, graph
Do Transformers Really Perform Bad for Graph Representation? [[NeurIPS21](https://arxiv.org/abs/2106.05234)][[#21](https://github.com/InabaTatsuro/papers/issues/21)]
- propose Graphormer, which is based on the standard Transformer and can utilize the structural information of a graph
- similar to [#18](https://github.com/InabaTatsuro/papers/issues/18)

GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs [[Zhang+, 18](https://arxiv.org/abs/1803.07294)] [[#19](https://github.com/InabaTatsuro/papers/issues/19)]
- use a convolutional sub-network to control each attention head's weight

Global Self-Attention as a Replacement for Graph Convolution [[KDD22](https://arxiv.org/abs/2108.03348)][[#18](https://github.com/InabaTatsuro/papers/issues/18)]
- add edge information to the self-attention calculation

Graph Transformer Networks [[Yun+, NeurIPS19](https://arxiv.org/abs/1911.06455)][[#17](https://github.com/InabaTatsuro/papers/issues/17)]
- generate a new meta-path that represents multi-hop relations and new-graph structure by multiplication of different adjacency matrix
- can be applied to heterogeneous graph

Graph Transformer [[Li+, 18](https://openreview.net/forum?id=HJei-2RcK7)][[#16](https://github.com/InabaTatsuro/papers/issues/16)]
- evolve the target graph by recurrently applying source-attention from the source graph and self-attention from the target graph
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
- extend self-attention to consider the pairwise relationships between each element
- Add the trainable relative position representations, and add them to key and query vectors

Graph Attention Networks [[Velickovic+, ICLR18](https://arxiv.org/abs/1710.10903)][[#4](https://github.com/InabaTatsuro/papers/issues/4)]
- train weight matrix which represents the relation between nodes
- it can be seen as self-attention with an artificially created mask

# CV
GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration [ICML23, [#35](https://github.com/InabaTatsuro/papers/issues/35)]
- solve linear inverse problems using denoising diffusion restoration
- it can be used in cases where the linear operator is unknown

## Read Later

PeerNets: Exploiting Peer Wisdom Against Adversarial Attacks [[Svoboda+, 18](https://arxiv.org/abs/1806.00088)]
