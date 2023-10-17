for file in `ls in`
do
    python3 "./answer.py" < "./in/${file}" > "./out/${file}"
done