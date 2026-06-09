from __future__ import annotations

from common.singleton import Singleton
from common.testing import UnitTest


class TestSingleton(UnitTest):
    COVERAGE = ["common.singleton"]

    def test_same_instance_returned(self):
        """Two calls to the same Singleton subclass return the identical object."""

        class MyService(Singleton):
            pass

        a = MyService()
        b = MyService()
        self.assertIs(a, b)

    def test_different_subclasses_are_independent(self):
        """Distinct subclasses each maintain their own singleton instance."""

        class Alpha(Singleton):
            pass

        class Beta(Singleton):
            pass

        self.assertIsNot(Alpha(), Beta())
        self.assertIs(Alpha(), Alpha())
        self.assertIs(Beta(), Beta())
