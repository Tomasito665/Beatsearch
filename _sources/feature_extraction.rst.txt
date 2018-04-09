Feature extraction
==================

There are feature extractors available for three kinds of features: monophonic, polyphonic and generic.

.. graphviz::
   :align: center

   digraph "Feature extraction structure" {
      generic [label="Generic features"]
      mono [label="Monophonic features"];
      poly [label="Polyphonic features"];

      generic -> mono;
      generic -> poly;
   }

All feature extractor classes and function can be found in the ``beatsearch.feature_extraction`` package. Features are
extracted with implementations of the ``FeatureExtractor`` interface.

.. autoclass:: beatsearch.feature_extraction.FeatureExtractor
   :members:

Generic rhythm features
-----------------------

Generic rhythm features are based on generic rhythm properties and apply to both monophonic and polyphonic rhythms.
Feature extractors for these type of features are implemented as ``RhythmFeatureExtractor`` subclasses.

There hasn't yet been implemented any generic rhythm feature extractor.

Monophonic rhythm features
--------------------------

Monophonic rhythm features are based on monophonic rhythm properties and are implemented as
``MonophonicRhythmFeatureExtractor`` subclasses. For example, to compute the onset density of a rhythm:

.. code-block:: python

   from beatsearch.rhythm import MonophonicRhythm
   from beatsearch.feature_extraction import OnsetDensity

   rhythm = MonophonicRhythm.create.from_string("x--x--x---x-x---")
   onset_density = OnsetDensity().process(rhythm)

**Implementations**

* :class:`beatsearch.feature_extraction.BinaryOnsetVector`
* :class:`beatsearch.feature_extraction.NoteVector`
* :class:`beatsearch.feature_extraction.IOIVector`
* :class:`beatsearch.feature_extraction.IOIHistogram`
* :class:`beatsearch.feature_extraction.BinarySchillingerChain`
* :class:`beatsearch.feature_extraction.ChronotonicChain`
* :class:`beatsearch.feature_extraction.IOIDifferenceVector`
* :class:`beatsearch.feature_extraction.OnsetPositionVector`
* :class:`beatsearch.feature_extraction.SyncopationVector`
* :class:`beatsearch.feature_extraction.SyncopatedOnsetRatio`
* :class:`beatsearch.feature_extraction.MeanSyncopationStrength`
* :class:`beatsearch.feature_extraction.OnsetDensity`

Polyphonic rhythm features
--------------------------

Polyphonic rhythm features are based on polyphonic rhythm properties and are implemented as
``PolyphonicRhythmFeatureExtractor`` subclasses.

**Implementations**

* :class:`beatsearch.feature_extraction.MultiChannelMonophonicRhythmFeatureVector`
* :class:`beatsearch.feature_extraction.PolyphonicSyncopationVector`
