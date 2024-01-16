## link
- [ISMIR23-intro1](https://docs.google.com/presentation/d/1AB27cUQzXaKbZlh3xtOCTWHR-PdWWXrF8K9xROFoNvk/edit?usp=sharing), [ISMIR23-intro2](https://docs.google.com/presentation/d/1qPPOfn8YDNIvpnGZFvx8UVYLuA8iXiHKK8OJgq7RSiM/edit?usp=sharing)

## Representation
### Piano-roll like
2-dimensional matrix, where one dimension represents time and the other represents pitch

### Event-level
<img src="https://github.com/InabaTatsuro/papers/assets/102784221/fe2e60d0-3397-47da-a6e8-f3c7d53f068a" width="400">


- MIDI-like(Music transformer)
  - including note-on, note-off, velocity, pitch, etc.
- REMI(Pop Music Transformer)
  - improves the MIDI-like representation by using duration, bar, chord, and tempo
- Structured MIDI
  - use time-shift tokens instead of note-on/off or duration tokens
 
### Note-level
- MuMIDI(PopMAG), CP(Compound word transformer), OctupleMIDI(MusicBERT), Multitrack Music Transformer
  - compress the attributes of a note, including pitch, duration, and velocity into one symbol

## Generation metrics (General)
### Negative log-likelihood (NLL)
<img width="400" alt="Screenshot 2024-01-09 at 1 03 18" src="https://github.com/InabaTatsuro/papers/assets/102784221/08362a12-48bc-4dfd-8b6e-a134a112218b">

- smaller values are better
- e.g., Music Transformer

### perplexity
<img width="400" alt="Screenshot 2024-01-09 at 1 04 41" src="https://github.com/InabaTatsuro/papers/assets/102784221/e4a2685b-60bc-4d1a-83fe-42c29027cb8c">

- smaller values are better
- e.g., Museformer

### empty bars (EB)
- the ratio of empty bars
- values closer to the original data are better
- e.g., MuseGAN

### used pitch classes (UPC)
- the number of used pitch classes per bar
- values closer to the original data are better
- e.g., MuseGAN

### qualified notes (QN)
- the ratio of qualified notes that are no shorter than a time step(i.e. a 32th note)
- values closer to the original data are better
- e.g., MuseGAN

### Drum Pattern (DP)
- the ratio of notes in 8- or 16-beat patterns
- values closer to the original data are better
- e.g., MuseGAN

### Tonal Distance (TD)
- the harmonicity between a pair of tracks
- smaller values are better
- e.g., MuseGAN

### Beat/Downbeat STD
- convert the generated symbolic music into audio music and evaluate by beat tracking model
- values closer to the original data are better
- e.g., Pop music transformer

### Note F1
- match if generated onset, measure, pitch, track match exactly those of original
- higher values are better
- e.g., MT3, Composer Assistant

### Pitch class histogram entropy difference
- Drums are ignored
- lower values are better
- e.g., Composer Assistant, XLNet

### Groove similarity
- higher values are better

### MIREX-like prediction test(≒PPL, NLL)
- choose the correct answer from 4 choices by calculating the average probability of generating the event
- e.g., Jazz transformer, XLNet Piano Infilling

### Human Eval
- number of wins
  - e.g., Music Transformer
- harmonious/rhythmic/musically structured/coherent/overall rating
  - e.g., MuseGAN
- musicality/short-term structure/long-term structure/overall/overall score/preference
  - e.g., Museformer
- structure/richness/pleasure/overall
  - e.g., MELON
- coherence/richness/arrangement/overall
  - e.g., MMT
- average rank, p-value
  - e.g., composer's assistant

- distinguish pro and non-pro, p-value
  - e.g., Pop Music Transformer
 
### Similarity Error (SE)
- the error between the similarity distribution of original data and generated music
- smaller values are better
- e.g., Museformer

### Inference speed
- notes per second
- e.g., MMT

## Generation Analysis
### similarity distribution
- to analyze the repetition
- e.g., Museformer

### Attention gain
- e.g., MMT

