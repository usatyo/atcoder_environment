if [ $# = 0 ]; then
    # use stdin readline
    pypy3 answer.py < input.txt
elif [ $1 = "f" ]; then
    pypy3 answer.py < input.txt > output.txt
elif [ $1 = "m" ]; then
    # use file readline
    pypy3 randomtest.py
else
    echo "Usage: sh test.sh [m]"
fi
