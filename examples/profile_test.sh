python -m cProfile -o test_profile.txt $1
python -c "import pstats; p = pstats.Stats('test_profile.txt'); p.print_stats()" > test_stats.txt
