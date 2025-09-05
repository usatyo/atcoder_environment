from answer import main
from honesty import honesty
from generator import Generator
import unittest
from tqdm import tqdm

LOOP = 100


class Test(unittest.TestCase):
    def _test_single_case(self):
        gen = Generator()
        gen.generate()
        file = open("./texts/input.txt", "r")
        ans = main(lambda: file.readline().rstrip())
        file.close()
        honest = honesty()
        self.assertEqual(ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}")

    def test_multiple_cases(self):
        gen = Generator()
        for _ in tqdm(range(LOOP)):
            info = gen.generate()
            file = open("./texts/input.txt", "r")
            ans = main(lambda: file.readline().rstrip())
            file.close()
            honest = honesty()
            with self.subTest(info=info):
                self.assertEqual(
                    ans, honest, msg=f"\nYour Value: {ans}\nTrue Value: {honest}"
                )


if __name__ == "__main__":
    unittest.main(warnings="ignore", verbosity=2)
