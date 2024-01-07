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
