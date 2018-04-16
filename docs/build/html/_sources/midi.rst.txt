Midi
====

Beatsearch provides functionality to import polyphonic rhythm patterns from MIDI files. The MIDI files should comply to
the `General MIDI Level 1 Percussion Key Map <https://www.midi.org/specifications/item/gm-level-1-sound-set>`_. Custom
MIDI drum mappings are also allowed, but require a bit more work. The MIDI import/export functionality is found in the
:class:`beatsearch.rhythm.MidiRhythm` class. For example, to import a MIDI file called `rumba.mid`:

.. code-block:: python

   from beatsearch.rhythm import MidiRhythm

   rhythm = MidiRhythm("rumba.mid")


MIDI drum mappings
__________________

A drum mapping is represented as a :class:`beatsearch.rhythm.MidiDrumMapping` object, which is essentially a collection
of :class:`beatsearch.rhythm.MidiDrumKey` objects. :class:`beatsearch.rhythm.MidiDrumKey` is a struct-like class which
holds information about a single key within a MIDI drum mapping. Each drum key holds information about its
frequency-band, the decay-time and the MIDI pitch. To load a MIDI drum loop with a custom mapping, you could do:

.. code-block:: python

   from beatsearch.rhythm import MidiRhythm, MidiDrumKey, FrequencyBand, DecayTime, create_drum_mapping

   CustomMapping = create_drum_mapping("CustomMapping", [
      MidiDrumKey(60, FrequencyBand.LOW, DecayTime.NORMAL, "Kick", key_id="kck"),
      MidiDrumKey(62, FrequencyBand.MID, DecayTime.NORMAL, "Snare", key_id="snr"),
      MidiDrumKey(64, FrequencyBand.MID, DecayTime.NORMAL, "Tom", key_id="tom"),
      MidiDrumKey(66, FrequencyBand.HIGH, DecayTime.SHORT, "Hi-hat", key_id="hht"),
      MidiDrumKey(70, FrequencyBand.HIGH, DecayTime.LONG, "Crash", key_id="crs")
   ])

   rhythm = MidiRhythm("loop.mid", midi_mapping=CustomMapping)


Instrumentation reduction
_________________________

For analytical purposes it's sometimes useful to reduce the instrument count of the drum patterns. This can be
done setting the ``MidiRhythm`` constructor's  `midi_mapping_reducer_cls` parameter. The MIDI mapping reducer must
be a subclass of ``MidiDrumMappingReducer`` or ``None`` for no instrumentation reduction. Beatsearch provides the
following instrumentation reducers:

FrequencyBandMidiDrumMappingReducer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When this reducer is applied, the instrumentation will be reduced down to three streams, based on the frequency-band
of the MIDI drum keys.

- LOW
- MID
- HIGH

DecayTimeMidiDrumMappingReducer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When this reducer is applied, the instrumentation will be reduced down to three streams, based on the decay-time of the
MIDI drum keys.

- SHORT
- NORMAL
- LONG

UniquePropertyComboMidiDrumMappingReducer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When this reducer is applied, the instrumentation will be reduced down to nine streams. One stream per unique
[frequency-band, decay-time] combination.

- LOW.SHORT
- LOW.NORMAL
- LOW.LONG
- MID.SHORT
- MID.NORMAL
- MID.LONG
- LONG.SHORT
- LONG.NORMAL
- LONG.LONG

Example
~~~~~~~

To load a MIDI rhythm and reduce it down to three instruments: `LOW`, `MID` and `HIGH`, you could do:

.. code-block:: python

   from beatsearch.rhythm import MidiRhythm, FrequencyBandMidiDrumMappingReducer

   rhythm = MidiRhythm(
       "./rumba.mid",
       midi_mapping_reducer_cls=FrequencyBandMidiDrumMappingReducer
   )


Rhythm corpus
_____________

We can use the :class:`beatsearch.rhythm.MidiRhythmCorpus` class to load multiple MIDI files. For example, load all the
MIDI files in a directory called `LOOPS`, you could do:

.. code-block:: python

   from beatsearch.rhythm import MidiRhythmCorpus

   rhythms = MidiRhythmCorpus("./LOOPS")

:class:`beatsearch.rhythm.MidiRhythmCorpus` also provides functionality to export its rhythms as MIDI files to a given
directory with the :meth:`beatsearch.rhythm.MidiRhythmCorpus.export_as_midi_files` method. This can be useful, for
example, to reduce the instrumentation of all the MIDI files in a particular directory.

.. code-block:: python

   from beatsearch.rhythm import MidiRhythmCorpus, FrequencyBandMidiDrumMappingReducer

   loops = MidiRhythmCorpus(
       "./LOOPS",
       midi_mapping_reducer=FrequencyBandMidiDrumMappingReducer
   )

   loops.export_as_midi_files("./LOOPS/reduced")
