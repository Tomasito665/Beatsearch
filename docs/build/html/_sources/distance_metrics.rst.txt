Distance metrics
================

Beatsearch provides distance metrics for both monophonic and polyphonic rhythms, which are implemented
as subclasses of ``MonophonicRhythmDistanceMeasure`` and ``PolyphonicRhythmDistanceMeasure`` respectively.
Both of these implement the ``DistanceMeasure`` interface and are found in the ``beatsearch.metrics`` module.

.. autoclass:: beatsearch.metrics.DistanceMeasure
    :members:



Monophonic rhythms
------------------

The distance between two monophonic rhythms can be computed with one of the ``MonophonicRhythmDistanceMeasure``
implementations. For example, to compute the hamming distance between the clave 23 and clave rumba:

.. code-block:: python

   from beatsearch.rhythm import MonophonicRhythm
   from beatsearch.metrics import HammingDistanceMeasure

   clave_23    = MonophonicRhythm.create.from_string("--x-x---x--x--x-")
   clave_rumba = MonophonicRhythm.create.from_string("--x-x---x--x---x")

   measure = HammingDistanceMeasure()
   distance = measure.get_distance(clave_23, clave_rumba)

Individual tracks of polyphonic rhythms are monophonic, which allows for distance computation between instruments:

.. code-block:: python

   from textwrap import dedent
   from beatsearch.rhythm import PolyphonicRhythm
   from beatsearch.metrics import HammingDistanceMeasure

   rhythm = PolyphonicRhythm.create.from_string(dedent("""
       hi-hat: x-xxx-xxx-xxx-xx
       kick:   ---x--x----x--x-
   """))

   measure = HammingDistanceMeasure()
   distance = measure.get_distance(
       rhythm.get_track_by_name('hi-hat'),
       rhythm.get_track_by_name('kick')
   )


Polyphonic rhythms
------------------

The distance measure between two polyphonic rhythms can be computed with one of the ``PolyphonicRhythmDistanceMeasure``
implementations.

.. todo:: add example
