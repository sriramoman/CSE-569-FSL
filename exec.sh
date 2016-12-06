b="$1"
if [ "$b" == "hinge" ];then
a="src/Hinge.py"
else
a="src/LogLikelihood.py"
fi
python $a $2
