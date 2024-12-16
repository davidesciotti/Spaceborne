### Steps to save benchmark files and run the tests:
1. Set some cfg file, possibly with a fast runtime, and set
    cfg['misc']['save_output_as_benchmark'] = True
    cfg['misc']['bench_filename']: /home/davide/Documenti/Lavoro/Programmi Spaceborne_bench/<descriptive_name> 
2. Run SB as usual
3. Make sure that in the main we take the cfg path as command-line input
3. Run tests/test_outputs.py 
    