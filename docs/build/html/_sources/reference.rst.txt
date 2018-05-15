Reference
=========

Rhythm
______

Rhythm
~~~~~~

.. autoclass:: beatsearch.rhythm.Rhythm
   :members:

MonophonicRhythm
~~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.MonophonicRhythm
   :show-inheritance:
   :members:

PolyphonicRhythm
~~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.PolyphonicRhythm
   :show-inheritance:
   :members:


Feature Extraction
__________________

Monophonic Feature Extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.feature_extraction.BinaryOnsetVector
   :members:
.. autoclass:: beatsearch.feature_extraction.NoteVector
   :members:
.. autoclass:: beatsearch.feature_extraction.BinarySchillingerChain
   :members:
.. autoclass:: beatsearch.feature_extraction.ChronotonicChain
   :members:
.. autoclass:: beatsearch.feature_extraction.OnsetDensity
   :members:
.. autoclass:: beatsearch.feature_extraction.OnsetPositionVector
   :members:
.. autoclass:: beatsearch.feature_extraction.MonophonicOnsetLikelihoodVector
   :members:
.. autoclass:: beatsearch.feature_extraction.MonophonicVariabilityVector
   :members:
.. autoclass:: beatsearch.feature_extraction.MonophonicSyncopationVector
   :members:
.. autoclass:: beatsearch.feature_extraction.SyncopatedOnsetRatio
   :members:
.. autoclass:: beatsearch.feature_extraction.MeanSyncopationStrength
   :members:
.. autoclass:: beatsearch.feature_extraction.MonophonicMetricalTensionVector
   :members:
.. autoclass:: beatsearch.feature_extraction.MonophonicMetricalTensionMagnitude
   :members:
.. autoclass:: beatsearch.feature_extraction.IOIVector
   :members:
.. autoclass:: beatsearch.feature_extraction.IOIDifferenceVector
   :members:
.. autoclass:: beatsearch.feature_extraction.IOIHistogram
   :members:


Polyphonic Feature Extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.feature_extraction.MultiTrackMonoFeature
   :members:
.. autoclass:: beatsearch.feature_extraction.PolyphonicMetricalTensionVector
   :members:
.. autoclass:: beatsearch.feature_extraction.PolyphonicMetricalTensionMagnitude
   :members:
.. autoclass:: beatsearch.feature_extraction.PolyphonicSyncopationVector
   :members:
.. autoclass:: beatsearch.feature_extraction.PolyphonicSyncopationVectorWitek
   :members:


Midi
____


MidiRhythm
~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.MidiRhythm
   :show-inheritance:
   :members:

MidiRhythmCorpus
~~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.MidiRhythmCorpus
   :members:

MidiDrumMapping
~~~~~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.MidiDrumMapping
   :members:

create_drum_mapping
~~~~~~~~~~~~~~~~~~~

.. autofunction:: beatsearch.rhythm.create_drum_mapping

MidiDrumKey
~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.MidiDrumKey
   :members:

FrequencyBand
~~~~~~~~~~~~~~

.. autoclass:: beatsearch.rhythm.FrequencyBand
   :members:
   :undoc-members:

DecayTime
~~~~~~~~~

.. autoclass:: beatsearch.rhythm.DecayTime
   :members:
   :undoc-members:
