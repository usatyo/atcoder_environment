from random import randint
from answer import main
from honesty import honesty
from generator import generator
import unittest


LOOP = 100


class Test(unittest.TestCase):
    def _test_single_case(self):
        gen = generator()
        gen.generate()
        file = open("input.txt", "r")
        self.assertEqual(main(lambda: file.readline().rstrip()), honesty())
        file.close()

    def test_multiple_cases(self):
        gen = generator()
        for _ in range(LOOP):
            gen.generate()
            file = open("input.txt", "r")
            self.assertEqual(main(lambda: file.readline().rstrip()), honesty())
            file.close()

    def _test_satisfy_conditions(self):
        gen = generator()
        for _ in range(LOOP):
            gen.generate()
            file = open("input.txt", "r")

            # edit here
            n, k, *ans = main(lambda: file.readline().rstrip())

            file.close()


if __name__ == "__main__":
    unittest.main(warnings="ignore", verbosity=2)
