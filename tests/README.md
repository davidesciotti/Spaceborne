
# Steps to save benchmark files

1. Set some cfg file, possibly with a fast runtime, and at the beginning of main.py set

    ````[Python]
    cfg['misc']['save_output_as_benchmark']: True
    cfg['misc']['bench_filename']: /home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/<descriptive_name> 
    ````

Note: The code is very fast if you set something like

    ````[YAML]
    # k grid used for power spectrum and trispectrum computation
    k_steps: 20 
    z_steps: 200
    z_steps_trisp: 10 
    ````

2. Run SB as usual

# Steps to run the tests

1. In main.py, set `cfg['misc']['save_output_as_benchmark']: False`
2. In main.py, make sure that we take the cfg path as command-line input (i.e., uncomment the argparse bit at the beginning)
3. In tests/test_outputs.py, set the desired bench filename
4. Run tests/test_outputs.py
