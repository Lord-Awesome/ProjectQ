if [[ $# -ne 1 ]] ;
	then
	echo "Need exactly one argument: the name of the python script you want to profile"
	exit 1
fi
python3.8 -m cProfile -o test_profile.txt $1
python3.8 -c "import pstats; p = pstats.Stats('test_profile.txt'); p.sort_stats('tottime'); p.print_stats()" > test_stats.txt
