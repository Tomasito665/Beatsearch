Rhythm model
============

The main building block of Beatsearch is the rhythm model, which defines
two types of rhythms: monophonic and polyphonic rhythms.

.. graphviz::
   :align: center

   digraph Rhythm {
      rhythm [label="Rhythm"];
      mono [label="Monophonic rhythm"];
      poly [label="Polyphonic rhythm"];

      rhythm -> mono;
      rhythm -> poly;
   }

Monophonic and polyphonic rhythms have common properties, such as tempo,
duration and time signature. In the diagram displayed above, these properties
live in the `Rhythm` node.

Monophonic rhythm
-----------------

A monophonic rhythm consists of one single track (one instrument). To create a
simple monophonic rhythm from a string:

.. code-block:: python

   from beatsearch.rhythm import MonophonicRhythm
   clave_23 = MonophonicRhythm.create.from_string("--x-x---x--x--x-")

Polyphonic rhythm
-----------------

A polyphonic rhythm consists of multiple tracks (multiple instruments). To create a
simple polyphonic rhythm from a string:

.. code-block:: python

   from textwrap import dedent
   from beatsearch.rhythm import PolyphonicRhythm

   cascara = PolyphonicRhythm.create.from_string(dedent("""
       clave:       --x-x---x--x--x-
       timbales:    x-x-xx-xx-xx-x-x
       kick:        ---x------------
       side stick:  --x------x------
       toms:        ------xx------xx
   """))
