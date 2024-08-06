from random import randint
from answer import main
import unittest

from honesty import main as honesty


class Test(unittest.TestCase):
    def _generate_input(self):
        file = open("input.txt", "w")
        n = randint(1, 10**5)
        m = randint(1, 10**5)
        file.write(f"{n} {m}\n")
        file.close()

    def test_single_case(self):
        self._generate_input()
        file = open("input.txt", "r")
        self.assertEqual(main(file.readline), honesty(file.readline))
        file.close()




unittest.main()
