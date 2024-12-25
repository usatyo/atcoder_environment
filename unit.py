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
        file.close()
        honest = honesty()
        self.assertEqual(ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}")

    def test_multiple_cases(self):
        gen = generator()
        for _ in range(LOOP):
            info = gen.generate()
            file = open("input.txt", "r")
            ans = main(lambda: file.readline().rstrip())
            file.close()
            honest = honesty()
            with self.subTest(info=info):
                self.assertEqual(
                    ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}"
                )

    def _test_satisfy_conditions(self):
        gen = generator()
        for _ in range(LOOP):
            info = gen.generate()
            file = open("input.txt", "r")
            ans = main(lambda: file.readline().rstrip())
            file.close()

            file = open("input.txt", "r")
            r, x, y = map(int, file.readline().split())
            file.close()

            with self.subTest(info=info):
                self.assertTrue(x == y)


if __name__ == "__main__":
    unittest.main(warnings="ignore", verbosity=2)
