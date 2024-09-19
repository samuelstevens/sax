# Technical Debt

Just some things that we probably want to fix at some point.

1. When using slurm/submitit, I want to see progress, either on the launcher script's stdout, or in the .err/.out files.
2. I want to resize the array based on the number of images actually embedded.
3. I (might) want to shard the array so that during SAE training, multiple processes can each read a separate file.
