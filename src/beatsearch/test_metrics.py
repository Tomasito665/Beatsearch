import inspect
import typing as tp
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from unittest import TestCase, main
from unittest.mock import patch, MagicMock, PropertyMock, call
from beatsearch.rhythm import MonophonicRhythm, PolyphonicRhythm

# miscellaneous
from beatsearch.utils import friendly_named_class, QuantizableMixin
from beatsearch.rhythm import Unit

# feature extractors
from beatsearch.feature_extraction import BinaryOnsetVector, IOIVector, IOIDifferenceVector, OnsetPositionVector

# rhythm distance measure abstract base classes
from beatsearch.metrics import MonophonicRhythmDistanceMeasure, PolyphonicRhythmDistanceMeasure

# concrete monophonic rhythm distance measures
from beatsearch.metrics import (
    HammingDistanceMeasure,
    EuclideanIntervalVectorDistanceMeasure,
    IntervalDifferenceVectorDistanceMeasure,
    SwapDistanceMeasure,
    ChronotonicDistanceMeasure,
)


@friendly_named_class("Fake measure friendly name")
class FakeMonophonicRhythmDistanceMeasureImpl(MonophonicRhythmDistanceMeasure):
    def __get_iterable__(self, rhythm: MonophonicRhythm):
        return tuple()

    @staticmethod
    def __compute_distance__(max_len, iterable_a, iterable_b, cookie_a, cookie_b):
        return 0


def get_concrete_monophonic_measure_implementations():
    subclasses = MonophonicRhythmDistanceMeasure.__subclasses__()
    for sub in subclasses:
        if inspect.isabstract(sub):
            continue
        yield sub


class TestMonophonicRhythmDistanceMeasureBase(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, MonophonicRhythmDistanceMeasure)

    def test_all_concrete_subclasses_are_friendly_named(self):
        for clazz in get_concrete_monophonic_measure_implementations():
            with self.subTest(msg=clazz.__name__):
                try:
                    # noinspection PyUnresolvedReferences
                    clazz.__friendly_name__
                except AttributeError:
                    self.assertTrue(False, "no __friendly_name__ found in '%s'" % clazz.__name__)

    @patch.object(Unit, "get")
    def test_all_concrete_subclasses_have_unit_and_length_policy_as_first_arguments_and_set_the_props_accordingly(
            self, _):
        def patch_prop(prop_name):
            return patch.object(MonophonicRhythmDistanceMeasure, prop_name, new_callable=PropertyMock)

        for clazz in get_concrete_monophonic_measure_implementations():
            with patch_prop("length_policy") as length_policy_mock, patch.object(clazz, "set_unit") as set_unit_mock:
                clazz(unit="fake-unit", length_policy="fake-length-policy")
                with self.subTest(msg=clazz.__name__):
                    length_policy_mock.assert_called_once_with("fake-length-policy")
                    set_unit_mock.assert_called_once_with("fake-unit")

    def test_all_quantizable_concrete_subclasses_have_quantize_named_argument_and_set_the_prop_accordingly(self):
        for clazz in get_concrete_monophonic_measure_implementations():
            if not issubclass(clazz, QuantizableMixin):
                continue
            with patch.object(clazz, "quantize", new_callable=PropertyMock) as quantize_mock:
                clazz(quantize="fake-quantize")
                with self.subTest(msg=clazz.__name__):
                    quantize_mock.assert_called_once_with("fake-quantize")

    def test_get_measure_by_friendly_name(self):
        friendly_name = FakeMonophonicRhythmDistanceMeasureImpl.__friendly_name__
        measure = MonophonicRhythmDistanceMeasure.get_measure_by_name(friendly_name)
        self.assertIs(measure, FakeMonophonicRhythmDistanceMeasureImpl)

    def test_get_measure_by_class_name(self):
        name = FakeMonophonicRhythmDistanceMeasureImpl.__name__
        measure = MonophonicRhythmDistanceMeasure.get_measure_by_name(name)
        self.assertIs(measure, FakeMonophonicRhythmDistanceMeasureImpl)

    @staticmethod
    def get_distance_measure_type():
        return FakeMonophonicRhythmDistanceMeasureImpl

    @classmethod
    def create_measure(cls, *args, **kwargs):
        measure_type = cls.get_distance_measure_type()
        return measure_type(*args, **kwargs)

    def test_length_policy_set_by_constructor_arg(self):
        for lp in MonophonicRhythmDistanceMeasure.LENGTH_POLICIES:
            m = self.create_measure(length_policy=lp)
            with self.subTest(length_policy=lp):
                self.assertEqual(m.length_policy, lp)

    def test_length_policy_prop(self):
        m = self.create_measure()
        for lp in MonophonicRhythmDistanceMeasure.LENGTH_POLICIES:
            m.length_policy = lp
            with self.subTest(length_policy=lp):
                self.assertEqual(m.length_policy, lp)

    def test_internal_unit(self):
        m = self.create_measure(unit=Unit.SIXTEENTH)
        self.assertEqual(m.unit, Unit.SIXTEENTH)

    @patch.object(MonophonicRhythmDistanceMeasure, "LENGTH_POLICIES", tuple())
    def test_length_policy_setter_raises_unknown_length_policy_exception(self):
        with patch.object(MonophonicRhythmDistanceMeasure, "length_policy"):
            m = self.create_measure()
        self.assertRaises(MonophonicRhythmDistanceMeasure.UnknownLengthPolicy,
                          setattr, m, "length_policy", "foo")

    def test_get_distance_raises_value_error_when_rhythms_have_different_resolutions(self):
        rhythm_a = MagicMock(MonophonicRhythm)
        rhythm_b = MagicMock(MonophonicRhythm)

        rhythm_a.get_resolution = MagicMock(return_value=123)
        rhythm_b.get_resolution = MagicMock(return_value=321)

        m = self.create_measure()
        self.assertRaises(ValueError, m.get_distance, rhythm_a, rhythm_b)

    def test_get_distance(self):
        ppq = 123
        distance = 321

        rhythm_a = MagicMock(PolyphonicRhythm)  # type: MonophonicRhythm
        rhythm_b = MagicMock(PolyphonicRhythm)  # type: MonophonicRhythm

        for r in [rhythm_a, rhythm_b]:
            r.get_resolution = MagicMock(return_value=ppq)

        m = self.create_measure(unit=None)
        m.__get_iterable__ = MagicMock(side_effect=["iterable_a", "iterable_b"])  # second is longer, len = 14
        m.__get_cookie__ = MagicMock(side_effect=["cookie_a", "cookie_b"])
        m.__compute_distance__ = MagicMock(return_value=distance)

        with patch.object(MonophonicRhythmDistanceMeasure, "check_if_iterables_meet_len_policy", return_value=True):
            result = m.get_distance(rhythm_a, rhythm_b)

        self.assertEqual(result, distance)

        m.__get_iterable__.assert_has_calls([call(rhythm_a), call(rhythm_b)])
        m.__get_cookie__.assert_has_calls([call(rhythm_a), call(rhythm_b)])
        m.__compute_distance__.assert_called_once_with(10, "iterable_a", "iterable_b", "cookie_a", "cookie_b")

    def test_iterables_meet_len_policy_check(self):
        def get_fake_iterable(length):
            iterable_mock = MagicMock(tuple)
            iterable_mock.__len__.return_value = length
            return iterable_mock

        def do_test(policy, len_a, len_b, expected):
            iterable_a = get_fake_iterable(len_a)
            iterable_b = get_fake_iterable(len_b)
            args = [policy, iterable_a, iterable_b]
            result = MonophonicRhythmDistanceMeasure.check_if_iterables_meet_len_policy(*args)
            with self.subTest(args=args, expected=expected):
                self.assertEqual(result, expected)

        test_info = [
            ("exact", 2, 2, True),
            ("exact", 2, 4, False),
            ("multiple", 2, 4, True),
            ("multiple", 4, 2, True),
            ("multiple", 3, 4, False),
            ("fill", 3, 4, True),
            ("fill", 4, 3, True),
            ("fill", 0, 2, False),
            ("fill", 2, 0, False)
        ]

        for info in test_info:
            do_test(*info)

    def test_iterables_meet_len_policy_check_raises_unknown_length_policy_error(self):
        length_policy = "illegal_length_policy"
        assert length_policy not in MonophonicRhythmDistanceMeasure.LENGTH_POLICIES
        self.assertRaises(
            MonophonicRhythmDistanceMeasure.UnknownLengthPolicy,
            MonophonicRhythmDistanceMeasure.check_if_iterables_meet_len_policy,
            length_policy, "iter_a", "iter_b"
        )


MonophonicDistanceTestInfo = namedtuple("MonophonicDistanceTestInfo", [
    "main",
    "secondary_exact",
    "secondary_multiple",
    "secondary_fill"
])


class TestMonophonicRhythmDistanceMeasureImplementation(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        raise NotImplementedError

    @classmethod
    def get_test_cookies(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(None, None, None, None)

    def test_compute_distance(self):
        measure_class = self.get_distance_measure_class()

        test_iterables = self.get_test_iterables()
        test_cookies = self.get_test_cookies()
        expected_distances = self.get_expected_distances_to_main()

        assert isinstance(self, TestCase), "this mixin should be used on a TestCase class"
        zipped = zip(test_iterables, test_cookies, expected_distances, MonophonicDistanceTestInfo._fields)

        for secondary_iterable, secondary_cookie, expected_d, secondary_name in zipped:
            max_len = max(len(it) for it in (secondary_iterable, test_iterables.main))
            actual_d = measure_class.__compute_distance__(max_len, test_iterables.main, secondary_iterable,
                                                          test_cookies.main, secondary_cookie)

            with self.subTest(msg="main_to_%s" % secondary_name):
                self.assertEqual(actual_d, expected_d)

    @abstractmethod
    def test_get_iterable(self, *args):
        raise NotImplementedError


class TestHammingDistanceMeasure(TestCase, TestMonophonicRhythmDistanceMeasureImplementation):
    # noinspection SpellCheckingInspection,PyShadowingNames
    @classmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        # binary chains
        main = "1001000100101000"  # 1 0 0 1 0 0 0 1 0 0 1 0 1 0 0 0 -> distance to main = 0
        exct = "1001001000101001"  # 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 -> distance to main = 3
        mult = "10010001"          # 1 0 0 1 0 0 0 1/1 0 0 1 0 0 0 1 -> distance to main = 5
        fill = "10010001001010"    # 1 0 0 1 0 0 0 1 0 0 1 0 1 0/1 0 -> distance to main = 1

        return MonophonicDistanceTestInfo(
            main=tuple(int(c) for c in main),
            secondary_exact=tuple(int(c) for c in exct),
            secondary_multiple=tuple(int(c) for c in mult),
            secondary_fill=tuple(int(c) for c in fill)
        )

    @classmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(
            main=0,
            secondary_exact=3,
            secondary_multiple=5,
            secondary_fill=1
        )

    @classmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        return HammingDistanceMeasure

    @patch.object(BinaryOnsetVector, "process", return_value="fake-binary-onset-vector")
    def test_get_iterable(self, mock_binary_onset_vector_process, *args):
        rhythm = MagicMock(MonophonicRhythm)  # type: MonophonicRhythm
        measure = HammingDistanceMeasure()
        result = measure.__get_iterable__(rhythm)

        self.assertEqual(result, "fake-binary-onset-vector")
        mock_binary_onset_vector_process.assert_called_once_with(rhythm)


class TestEuclideanIntervalVectorDistanceMeasure(TestCase, TestMonophonicRhythmDistanceMeasureImplementation):
    # noinspection SpellCheckingInspection,PyShadowingNames
    @classmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        # inter-onset intervals (post-note)
        main = "343222"
        exct = "334231"
        mult = "844"
        fill = "343"

        return MonophonicDistanceTestInfo(
            main=tuple(int(c) for c in main),
            secondary_exact=tuple(int(c) for c in exct),
            secondary_multiple=tuple(int(c) for c in mult),
            secondary_fill=tuple(int(c) for c in fill)
        )

    @classmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(
            main=0.0,
            secondary_exact=2.0,
            secondary_multiple=8.366600265340756,
            secondary_fill=2.449489742783178
        )

    @classmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        return EuclideanIntervalVectorDistanceMeasure

    @patch.object(IOIVector, "process", return_value="fake-ioi-vector")
    def test_get_iterable(self, mock_ioi_vector_process):
        rhythm = MagicMock(MonophonicRhythm)  # type: MonophonicRhythm
        measure = EuclideanIntervalVectorDistanceMeasure()
        result = measure.__get_iterable__(rhythm)

        self.assertEqual(result, "fake-ioi-vector")
        mock_ioi_vector_process.assert_called_once_with(rhythm)


class TestIntervalDifferenceVectorDistanceMeasure(TestCase, TestMonophonicRhythmDistanceMeasureImplementation):
    # noinspection SpellCheckingInspection,PyShadowingNames
    @classmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        # interval difference vectors
        main = [1.33, 0.75, 0.67, 1.00, 1.00, 1.50]  # ioi = 343222
        exct = [1.00, 1.33, 0.50, 1.50, 0.33, 3.00]  # ioi = 334231
        mult = [0.50, 2.00, 2.00]                    # ioi = 844
        fill = [1.33, 0.75, 1.00]                    # ioi = 343
        return MonophonicDistanceTestInfo(main=main, secondary_exact=exct, secondary_multiple=mult, secondary_fill=fill)

    @classmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(
            main=0.0,
            secondary_exact=4.973636363636363,
            secondary_multiple=7.6450746268656715,
            secondary_fill=1.6558706467661688
        )

    @classmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        return IntervalDifferenceVectorDistanceMeasure

    @patch.object(IOIDifferenceVector, "process", return_value="fake-ioi-diff-vector")
    def test_get_iterable(self, mock_ioi_diff_vector_process):
        rhythm = MagicMock(MonophonicRhythm)  # type: MonophonicRhythm
        measure = IntervalDifferenceVectorDistanceMeasure()
        result = measure.__get_iterable__(rhythm)

        self.assertEqual(result, "fake-ioi-diff-vector")
        mock_ioi_diff_vector_process.assert_called_once_with(rhythm)


class TestSwapDistanceMeasure(TestCase, TestMonophonicRhythmDistanceMeasureImplementation):
    # noinspection SpellCheckingInspection,PyShadowingNames
    @classmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        # absolute onset times
        main = [0, 3, 7, 10, 12, 14]  # ioi = 343222 (duration = 16)
        exct = [0, 3, 6, 10, 12, 15]  # ioi = 334231 (duration = 16)
        mult = [0, 3, 7]              # ioi = 341    (duration = 8)
        fill = [0, 2, 5]              # ioi = 233    (duration = 8)

        return MonophonicDistanceTestInfo(
            main=tuple(int(c) for c in main),
            secondary_exact=tuple(int(c) for c in exct),
            secondary_multiple=tuple(int(c) for c in mult),
            secondary_fill=tuple(int(c) for c in fill)
        )

    @classmethod
    def get_test_cookies(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(main=16, secondary_exact=16, secondary_multiple=8, secondary_fill=8)

    @classmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(
            main=0,
            secondary_exact=2,
            secondary_multiple=4,
            secondary_fill=8  # TODO re-check this, is it really correct?
        )

    @classmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        return SwapDistanceMeasure

    @patch.object(OnsetPositionVector, "process", return_value="fake-onset-pos-vector")
    def test_get_iterable(self, mock_onset_pos_vector_process):
        rhythm = MagicMock(MonophonicRhythm)  # type: MonophonicRhythm
        measure = SwapDistanceMeasure()
        result = measure.__get_iterable__(rhythm)
        self.assertEqual(result, "fake-onset-pos-vector")
        mock_onset_pos_vector_process.assert_called_once_with(rhythm)


class TestChronotonicDistanceMeasure(TestCase, TestMonophonicRhythmDistanceMeasureImplementation):
    # noinspection SpellCheckingInspection,PyShadowingNames
    @classmethod
    def get_test_iterables(cls) -> MonophonicDistanceTestInfo:
        # chronotonic chains
        main = "3334444333224444"
        exct = "3333334444223331"
        mult = "33344441"
        fill = "33344443332222"

        return MonophonicDistanceTestInfo(
            main=tuple(int(c) for c in main),
            secondary_exact=tuple(int(c) for c in exct),
            secondary_multiple=tuple(int(c) for c in mult),
            secondary_fill=tuple(int(c) for c in fill)
        )

    @classmethod
    def get_expected_distances_to_main(cls) -> MonophonicDistanceTestInfo:
        return MonophonicDistanceTestInfo(
            main=0,
            secondary_exact=12,
            secondary_multiple=8,
            secondary_fill=6
        )

    @classmethod
    def get_distance_measure_class(cls) -> tp.Type[MonophonicRhythmDistanceMeasure]:
        return ChronotonicDistanceMeasure

    @patch.object(OnsetPositionVector, "process", return_value="fake-onset-position-vector")
    def test_get_iterable(self, mock_onset_position_vector_process):
        rhythm = MagicMock(MonophonicRhythm)  # type: MonophonicRhythm
        measure = SwapDistanceMeasure()
        result = measure.__get_iterable__(rhythm)
        mock_onset_position_vector_process.assert_called_once_with(rhythm)
        self.assertEqual(result, "fake-onset-position-vector")


# TODO Add tests for EuclideanMonophonicMTVDistance
# TODO Add tests for EuclideanPolyphonicMTVDistance


class TestPolyphonicRhythmDistanceMeasureBase(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, PolyphonicRhythmDistanceMeasure)


if __name__ == "__main__":
    main()
