if [ $# = 0 ]; then
    # use stdin readline
    python3 answer.py < input.txt
elif [ $1 = "m" ]; then
    # use file readline
    python3 unit.py
else
    echo "Usage: sh test.sh [m]"
fi
