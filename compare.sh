
python3 generator.py
while [ true ]
do
python3 generator.py
ANSWER=`python3 answer.py < input.txt`
# echo "\nreal answer"
HONEST=`python3 honesty.py < input.txt`
if [ "$ANSWER" != "$HONEST" ]
then
    echo "$ANSWER"
    echo "$HONEST"
    exit 0
fi
done
