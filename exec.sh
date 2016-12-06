b="$1"
if [ "$b" == "hinge" ];then
a="src/Hinge.py"
else
a="src/LogLikelihood.py"
fi
python2.7 $a $2
