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
        ans = main(lambda: file.readline().rstrip())
        honest = honesty()
        self.assertEqual(ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}")
        file.close()

    def test_multiple_cases(self):
        gen = generator()
        for _ in range(LOOP):
            gen.generate()
            file = open("input.txt", "r")
            ans = main(lambda: file.readline().rstrip())
            honest = honesty()
            self.assertEqual(
                ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}"
            )
            file.close()

    def _test_satisfy_conditions(self):
        gen = generator()
        for _ in range(LOOP):
            gen.generate()
            file = open("input.txt", "r")

            # edit here
            self.assertTrue(main(lambda: file.readline().rstrip()))

            file.close()


if __name__ == "__main__":
    unittest.main(warnings="ignore", verbosity=2)
