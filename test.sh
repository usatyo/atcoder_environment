if [ $# = 0 ]; then
    # use stdin readline
    python3 ./python/answer.py < ./texts/input.txt
elif [ $1 = "f" ]; then
    python3 ./python/answer.py < ./texts/input.txt > ./texts/output.txt
elif [ $1 = "m" ]; then
    # use file readline
    python3 ./python/randomtest.py
else
    echo "Usage: sh test.sh [m|f]"
fi
