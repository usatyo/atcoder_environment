from random import randint
from answer import main
from honesty import honesty
from generator import generator
import unittest


class Test(unittest.TestCase):
    def test_single_case(self):
        gen = generator()
        gen.generate()
        file = open("input.txt", "r")
        self.assertEqual(main(file.readline), honesty())
        file.close()

    def test_multiple_cases(self):
        gen = generator()
        for _ in range(10**2):
            gen.generate()
            file = open("input.txt", "r")
            self.assertEqual(main(file.readline), honesty())
            file.close()

    def _test_satisfy_conditions(self):
        gen = generator()
        for _ in range(10**2):
            gen.generate()
            file = open("input.txt", "r")

            # edit here
            self.assertTrue(main(file.readline) == 0)

            file.close()


if __name__ == "__main__":
    unittest.main()
