
# Steps to save benchmark files

1. Set some cfg file, possibly with a fast runtime, and set

    ````[Python]
    cfg['misc']['save_output_as_benchmark']: True
    cfg['misc']['bench_filename']: /home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/<descriptive_name> 
    ````

2. Run SB as usual

# Steps to run the tests

1. Set `cfg['misc']['save_output_as_benchmark']: False`
2. In main.py, make sure that we take the cfg path as command-line input (i.e., unceomment the argparse bit at the beginning)
3. Run tests/test_outputs.py
