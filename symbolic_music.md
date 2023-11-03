## Representation
### Piano-roll like
2-dimensional matrix, where one dimension represents time and the other represents pitch

### Event-based
- MIDI-like
  - including note-on, note-off, velocity, pitch, etc.
- REMI
  - improves the MIDI-like representation by using note-duration, bar, chord, and tempo
- MuMIDI(PopMAG), CP(Compound word transformer), OctupleMIDI(MusicBERT)
  - compress the attributes of a note, including pitch, duration, and velocity into one symbol
